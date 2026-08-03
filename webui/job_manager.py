from __future__ import annotations

import copy
import json
import os
import queue
import subprocess
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from .process_guard import ProcessJob


EVENT_PREFIX = "H3EVENT "
TERMINAL_STATES = {"completed", "failed", "cancelled", "interrupted"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    """Runs one GPU generation job at a time and persists its visible state."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.data_dir = self.root / "webui_data"
        self.jobs_dir = self.data_dir / "jobs"
        self.outputs_dir = self.root / "outputs" / "webui"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: dict[str, dict[str, Any]] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.RLock()
        self._runner: threading.Thread | None = None
        self._stop = threading.Event()
        self._current_job_id: str | None = None
        self._current_process: subprocess.Popen[str] | None = None
        self._current_process_job: ProcessJob | None = None
        self._engine_variant: str | None = None
        self._load_existing_jobs()

    def _load_existing_jobs(self) -> None:
        for metadata_path in self.jobs_dir.glob("*/job.json"):
            try:
                job = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("status") not in TERMINAL_STATES:
                job.update(
                    status="interrupted",
                    phase="停止しました",
                    message="Web UIの前回終了により処理が中断されました。もう一度生成してください。",
                    finished_at=utc_now(),
                    progress_updated_at=utc_now(),
                )
                self._save_job(job)
            self._jobs[job["id"]] = job

    def start(self) -> None:
        with self._lock:
            if self._runner and self._runner.is_alive():
                return
            self._stop.clear()
            self._runner = threading.Thread(target=self._run_loop, name="h3-job-runner", daemon=True)
            self._runner.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put("")
        self._terminate_current_process()

    def submit(self, job: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._jobs[job["id"]] = job
            self._save_job(job)
            self._queue.put(job["id"])
            return copy.deepcopy(job)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [copy.deepcopy(job) for job in self._jobs.values()]
        return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def current_job_id(self) -> str | None:
        with self._lock:
            return self._current_job_id

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        owned_process: tuple[subprocess.Popen[str] | None, ProcessJob | None] = (None, None)
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.get("status") in TERMINAL_STATES:
                return copy.deepcopy(job) if job else None
            job.update(
                status="cancelled",
                phase="キャンセルしました",
                message="生成をキャンセルしました。モデルや重みは変更されていません。",
                finished_at=utc_now(),
                progress_updated_at=utc_now(),
            )
            self._save_job(job)
            is_current = self._current_job_id == job_id
            if is_current:
                # Claim this exact engine in the same critical section as the
                # job-id comparison. A completed old cancel must never kill a
                # newly installed engine for the next queued job.
                owned_process = self._claim_current_process_locked()
        self._cleanup_owned_process(*owned_process)
        return self.get_job(job_id)

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            job_id = self._queue.get()
            if not job_id or self._stop.is_set():
                continue
            with self._lock:
                job = self._jobs.get(job_id)
                if not job or job.get("status") == "cancelled":
                    continue
                self._current_job_id = job_id
                job.update(
                    status="running",
                    phase="生成を開始しています",
                    message="ローカル生成エンジンを起動しています。",
                    progress=1,
                    started_at=utc_now(),
                    progress_updated_at=utc_now(),
                )
                self._save_job(job)

            request_path = self.jobs_dir / job_id / "request.json"
            try:
                requested_variant = "ref2va" if job.get("mode") == "omni" else "fl2va"
                process = self._ensure_engine(requested_variant)
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write(json.dumps({"request": os.fspath(request_path)}, ensure_ascii=False) + "\n")
                process.stdin.flush()
                terminal_status = None
                while not self._stop.is_set():
                    raw_line = process.stdout.readline()
                    if not raw_line:
                        return_code = process.poll()
                        raise RuntimeError(f"生成エンジンが終了コード {return_code} で停止しました。")
                    line = raw_line.rstrip("\r\n")
                    if line.startswith(EVENT_PREFIX):
                        payload = line[len(EVENT_PREFIX) :]
                        self._handle_event(job_id, payload)
                        try:
                            event = json.loads(payload)
                            if event.get("status") in TERMINAL_STATES:
                                terminal_status = event["status"]
                                break
                        except json.JSONDecodeError:
                            pass
                    elif line.strip():
                        self._append_log(job_id, line.strip())
                if terminal_status != "completed":
                    self._terminate_current_process()
            except Exception as exc:
                with self._lock:
                    current = self._jobs.get(job_id)
                    if current and current.get("status") not in TERMINAL_STATES:
                        current.update(
                            status="failed",
                            phase="生成に失敗しました",
                            message=f"生成エンジンを起動できませんでした: {exc}",
                            finished_at=utc_now(),
                            progress_updated_at=utc_now(),
                        )
                        self._save_job(current)
                self._terminate_current_process()
            finally:
                owned_process: tuple[subprocess.Popen[str] | None, ProcessJob | None] = (None, None)
                with self._lock:
                    if self._current_process and self._current_process.poll() is not None:
                        owned_process = self._claim_current_process_locked()
                    self._current_job_id = None
                self._cleanup_owned_process(*owned_process)

    def _ensure_engine(self, requested_variant: str) -> subprocess.Popen[str]:
        with self._lock:
            process = self._current_process
            if process and process.poll() is None and self._engine_variant == requested_variant:
                if self._current_process_job is not None:
                    self._current_process_job.check()
                return process
            if process and process.poll() is not None:
                # The kernel Job may still own a redirector child or helper
                # even though the root Popen has already exited.
                self._terminate_current_process()
                process = None
            if process and process.poll() is None:
                # A fresh process is intentional here. PyTorch's CPU allocator
                # can retain tens of GiB after deleting a 33B model, so an
                # in-process FL2VA/Ref2VA swap risks paging both checkpoints.
                self._terminate_current_process()
                process = None
            if process:
                if process.stdin:
                    process.stdin.close()
                if process.stdout:
                    process.stdout.close()
            python = self.root / ".comfy-venv" / "Scripts" / "python.exe"
            if not python.is_file():
                raise FileNotFoundError(f"ComfyUI用Python環境が見つかりません: {python}")
            command = [os.fspath(python), "-m", "webui.comfy_engine_worker", "--serve"]
            env = os.environ.copy()
            env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            process_job = ProcessJob()
            process: subprocess.Popen[str] | None = None
            try:
                process = subprocess.Popen(
                    command,
                    cwd=self.root,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env,
                    creationflags=creation_flags,
                )
                process_job.attach(process)
            except Exception:
                process_job.terminate()
                self._cleanup_owned_process(process, None)
                raise
            self._current_process = process
            self._current_process_job = process_job
            self._engine_variant = requested_variant
            return process

    def _handle_event(self, job_id: str, payload: str) -> None:
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            self._append_log(job_id, payload)
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.get("status") == "cancelled":
                return
            for key in (
                "status",
                "phase",
                "message",
                "progress",
                "step",
                "total_steps",
                "result",
                "preview",
                "backend",
                "attention_backend",
                "acceleration",
                "media",
            ):
                if key in event:
                    job[key] = event[key]
            for key in ("cache", "timings"):
                if key not in event:
                    continue
                value = event[key]
                current = job.get(key)
                if isinstance(current, Mapping) and isinstance(value, Mapping):
                    job[key] = {**current, **value}
                else:
                    job[key] = value
            if event.get("status") in TERMINAL_STATES:
                job["finished_at"] = utc_now()
            if any(key in event for key in ("progress", "phase", "message")):
                job["progress_updated_at"] = utc_now()
            self._save_job(job)

    def _append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            logs = job.setdefault("logs", [])
            logs.append({"time": utc_now(), "text": line[-2000:]})
            del logs[:-160]
            self._save_job(job)

    def _save_job(self, job: dict[str, Any]) -> None:
        job_dir = self.jobs_dir / job["id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        target = job_dir / "job.json"
        temporary = job_dir / "job.json.tmp"
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    def _claim_current_process_locked(
        self,
    ) -> tuple[subprocess.Popen[str] | None, ProcessJob | None]:
        """Atomically detach the current engine; caller must hold `_lock`."""

        process = self._current_process
        process_job = self._current_process_job
        self._current_process = None
        self._current_process_job = None
        self._engine_variant = None
        return process, process_job

    def _terminate_current_process(self) -> None:
        with self._lock:
            owned_process = self._claim_current_process_locked()
        self._cleanup_owned_process(*owned_process)

    @staticmethod
    def _cleanup_owned_process(
        process: subprocess.Popen[str] | None,
        process_job: ProcessJob | None,
    ) -> None:
        if not process and process_job is None:
            return
        try:
            if process is not None:
                try:
                    parent = psutil.Process(process.pid)
                    children = parent.children(recursive=True)
                    for child in reversed(children):
                        child.terminate()
                    parent.terminate()
                    _, alive = psutil.wait_procs([*children, parent], timeout=5)
                    for remaining in alive:
                        remaining.kill()
                except (psutil.Error, OSError):
                    try:
                        process.terminate()
                    except OSError:
                        pass
        finally:
            # The kernel Job is authoritative even if the root Popen exited
            # before psutil could enumerate an orphaned redirector/helper.
            if process_job is not None:
                process_job.terminate()
            if process is not None:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
