from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.context_ir import compile_request as real_compile_request
from webui.job_manager import (
    EVENT_PREFIX,
    JobManager,
    _validate_compiled_prompt,
    _validate_direct_reference_tags,
)


TRANSLATOR_REPO = "LiquidAI/LFM2-350M-ENJP-MT"
TRANSLATOR_REVISION = "80367784d525777ad7565b24534ba5810eeac59f"


def _install_fake_runtime(root: Path) -> Path:
    python = root / ".comfy-venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b"test executable marker")
    return python


def _install_fake_translator(root: Path) -> Path:
    python = _install_fake_runtime(root)
    model_file = root / "models" / "prompt_translator" / "model.safetensors"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_bytes(b"pinned model fixture")
    lock = {
        "schema_version": 1,
        "source": {
            "repo_id": TRANSLATOR_REPO,
            "revision": TRANSLATOR_REVISION,
        },
        "verification": {"total_bytes": model_file.stat().st_size},
        "files": [
            {
                "path": "models/prompt_translator/model.safetensors",
                "size": model_file.stat().st_size,
                "sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / "prompt_translator.lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return python


def _official_six_section_prompt(*, dialogue: str = "こんにちは。") -> str:
    speech = (
        f"Cut 2: Her lips move once in sync with <d>[Japanese] {dialogue}</d> "
        "and then remain closed.\n\n"
        if dialogue
        else "Cut 2: She lowers her hand and looks toward the horizon.\n\n"
    )
    soundscape = (
        "Gentle ocean surf accompanies the one brief natural greeting."
        if dialogue
        else "Gentle ocean surf and a soft breeze move through the scene."
    )
    return (
        "subject_definitions:\n"
        "<Picture 1>: The sole visual identity and summer outfit reference for the woman.\n\n"
        "summary:\n"
        "A woman in the referenced summer outfit greets the viewer beside the sea.\n\n"
        "retention_analysis:\n"
        "Retain her face, body proportions, hairstyle, glasses, and exact outfit from <Picture 1>.\n\n"
        "detailed_description:\n"
        "Cut 1: A stable medium shot frames her beside the sea. She lifts one hand once.\n"
        f"{speech}"
        "overall_soundscape:\n"
        f"{soundscape}\n\n"
        "non_diegetic_music:\n"
        "N/A"
    )


def _official_three_section_prompt(*, dialogue: str = "", detail: str = "") -> str:
    speech = (
        f" Her lips move exactly once in sync with <d>[Japanese] {dialogue}</d>, "
        "then remain closed."
        if dialogue
        else ""
    )
    visual = detail or (
        "[Shot 1] A stable medium shot frames a woman beside the sea. "
        "She lifts one hand once."
    )
    soundscape = (
        "Gentle ocean surf accompanies the one brief natural greeting."
        if dialogue
        else "Gentle ocean surf and a soft breeze move through the scene."
    )
    return (
        "integrated_multimodal_description:\n"
        f"{visual}{speech}\n\n"
        "overall_soundscape:\n"
        f"{soundscape}\n\n"
        "non_diegetic_music:\n"
        "N/A"
    )


def _worker_success(compiled_prompt: str, **metadata: object) -> tuple[int, str, str]:
    return (
        0,
        json.dumps(
            {
                "ok": True,
                "compiled_prompt": compiled_prompt,
                "compiler_metadata": {
                    "compiler": "fixture-compiler",
                    "visible_text_literals": [],
                    "visible_text_literal_sha256": [],
                    **metadata,
                },
                "diagnostics": [],
            },
            ensure_ascii=False,
        ),
        "",
    )


def _request(**overrides: object) -> dict[str, object]:
    request: dict[str, object] = {
        "mode": "t2v",
        "prompt": "A paper kite rises in a summer breeze.",
        "num_frames": 121,
        "style": "natural",
        "dialogue": "",
        "soundscape": "Wind moves through grass.",
        "music_policy": "none",
        "references": [],
        "embedded_video_audio_policy": "auto",
        "prompt_processing_mode": "direct",
        "prompt_processing": {"mode": "context_ir"},
    }
    request.update(overrides)
    return request


class _FakeCompletedEngine:
    def __init__(self) -> None:
        event = {
            "status": "completed",
            "phase": "completed",
            "message": "done",
            "progress": 100,
        }
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(
            EVENT_PREFIX + json.dumps(event, ensure_ascii=True) + "\n"
        )

    def poll(self) -> None:
        return None


class _BlockingTranslatorProcess:
    """Small Popen double whose communicate call ends only after termination."""

    pid = 2_147_483_646
    stdin = None
    stdout = None
    stderr = None

    def __init__(self, response: tuple[int, str, str]) -> None:
        self._response = response
        self.returncode: int | None = None
        self.started = threading.Event()
        self.released = threading.Event()
        self.communication_finished = threading.Event()
        self.terminated = threading.Event()
        self.killed = threading.Event()
        self.reaped = threading.Event()
        self.input_text: str | None = None

    def communicate(self, input: str | None = None) -> tuple[str, str]:
        self.input_text = input
        self.started.set()
        if not self.released.wait(5):
            raise TimeoutError("test translator was not released")
        if self.returncode is None:
            self.returncode = self._response[0]
        self.communication_finished.set()
        return self._response[1], self._response[2]

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated.set()
        self.returncode = -15
        self.released.set()

    def kill(self) -> None:
        self.killed.set()
        self.returncode = -9
        self.released.set()

    def wait(self, timeout: float | None = None) -> int:
        if not self.released.wait(timeout):
            raise subprocess.TimeoutExpired("fixture-translator", timeout)
        self.reaped.set()
        return int(self.returncode if self.returncode is not None else 0)


class _FakeProcessJob:
    def __init__(self) -> None:
        self.attached: object | None = None
        self.terminated = threading.Event()

    def attach(self, process: object) -> None:
        self.attached = process

    def terminate(self) -> None:
        self.terminated.set()


class JobManagerContextIRTests(unittest.TestCase):
    def _persist_request(
        self,
        manager: JobManager,
        job_id: str,
        request: dict[str, object],
    ) -> tuple[Path, bytes]:
        manager.submit(
            {
                "id": job_id,
                "mode": request["mode"],
                "status": "queued",
                "phase": "queued",
                "message": "waiting",
                "progress": 0,
                "created_at": "2026-08-04T00:00:00+00:00",
            }
        )
        request_path = manager.jobs_dir / job_id / "request.json"
        request_path.write_text(
            json.dumps(request, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return request_path, request_path.read_bytes()

    def _wait_for_terminal(self, manager: JobManager, job_id: str) -> dict[str, object]:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            job = manager.get_job(job_id)
            if job is not None and job.get("status") in {
                "completed",
                "failed",
                "cancelled",
                "interrupted",
            }:
                return job
            time.sleep(0.01)
        self.fail(f"job {job_id!r} did not reach a terminal state")

    def _stop_runner(self, manager: JobManager) -> None:
        manager.stop()
        runner = manager._runner  # noqa: SLF001 - deterministic runner test cleanup
        if runner is not None:
            runner.join(timeout=2.0)
            self.assertFalse(runner.is_alive())

    def test_success_keeps_request_immutable_and_persists_effective_ir_and_audit_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = JobManager(Path(temporary))
            request = _request(
                mode="omni",
                prompt="Use <Video 1> for motion and its audio only as a voice reference.",
                references=[
                    {
                        "kind": "video",
                        "stored_path": "uploads/reference.mp4",
                        "has_audio": True,
                    }
                ],
                embedded_video_audio_policy="reference",
            )
            request_path, original_bytes = self._persist_request(
                manager, "compile-success", request
            )

            execution_path = manager._prepare_effective_prompt(  # noqa: SLF001
                "compile-success", request_path
            )

            self.assertIsNotNone(execution_path)
            assert execution_path is not None
            self.assertNotEqual(execution_path, request_path)
            self.assertEqual(request_path.read_bytes(), original_bytes)

            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            self.assertEqual(execution["effective_prompt_source"], "context_ir")
            self.assertNotEqual(execution["effective_prompt"], request["prompt"])
            self.assertEqual(execution["embedded_video_audio_policy"], "reference")
            self.assertEqual(execution["embedded_video_audio_indices"], [0])
            self.assertEqual(execution["compiler"]["status"], "degraded")
            self.assertFalse(execution["compiler"]["fatal"])
            self.assertFalse(execution["compiler"]["model_inference"])
            self.assertIn("elapsed_ms", execution["compiler"])

            saved = manager.get_job("compile-success")
            assert saved is not None
            self.assertEqual(saved["effective_prompt"], execution["effective_prompt"])
            self.assertEqual(saved["compiler"], execution["compiler"])
            self.assertEqual(
                saved["diagnostics"], execution["compiler"]["diagnostics"]
            )

            records = execution["compiler"]["artifacts"]
            self.assertGreaterEqual(len(records), 4)
            relative_paths = {record["relative_path"] for record in records}
            self.assertEqual(
                relative_paths,
                {
                    "context_ir/diagnostics.json",
                    "context_ir/provenance.json",
                    "context_ir/document.json",
                    "context_ir/final_ir.txt",
                },
            )
            for record in records:
                artifact = request_path.parent / Path(record["relative_path"])
                payload = artifact.read_bytes()
                self.assertEqual(record["bytes"], len(payload))
                self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())

            diagnostics = json.loads(
                (request_path.parent / "context_ir" / "diagnostics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(diagnostics["status"], "degraded")
            self.assertIsInstance(diagnostics["diagnostics"], list)

    def test_english_raw_with_japanese_dialogue_uses_deterministic_compiler_without_mt_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _install_fake_runtime(root)
            manager = JobManager(root)
            original_prompt = (
                "海辺でジュースを飲む。\n"
                "セリフ以外のカットではしゃべらないこと。\n"
                "こちらを見てキャラクターのセリフ。"
            )
            effective_prompt = (
                "A woman drinks juice beside the sea and looks toward the camera.\n"
                "Cut 2: She says <d>[Japanese] 暑いね。</d> exactly once.\n\n"
                "Audio: Gentle ocean surf and one brief straw sip."
            )
            request = _request(
                prompt=original_prompt,
                effective_prompt=effective_prompt,
                prompt_processing_mode="official_en",
                prompt_processing={
                    "mode": "raw_guarded",
                    "context_ir": False,
                    "dialogue_policy": "inline_h3_native",
                    "dialogue_count": 1,
                    "dialogue_events": [
                        {
                            "source": "prompt",
                            "target_cut": 2,
                            "language": "Japanese",
                            "original_text": "暑いね。",
                            "effective_text": "暑いね。",
                        }
                    ],
                    "auto_adjustments": ["DIALOGUE_PUNCTUATION_NORMALIZED"],
                    "diagnostics": ["DIALOGUE_TARGET_DEFAULTED"],
                    "removed_speech_cues": [
                        "セリフ以外のカットではしゃべらないこと。"
                    ],
                },
            )
            request_path, original_bytes = self._persist_request(
                manager, "raw-success", request
            )
            compiled = _official_three_section_prompt(dialogue="暑いね。")

            with (
                patch("webui.context_ir.compile_request") as compile_ir,
                patch.object(
                    manager,
                    "_run_tracked_prompt_translator",
                    return_value=_worker_success(compiled),
                ) as compiler_run,
            ):
                execution_path = manager._prepare_effective_prompt(  # noqa: SLF001
                    "raw-success", request_path
                )

            compile_ir.assert_not_called()
            compiler_run.assert_called_once()
            self.assertIsNotNone(execution_path)
            assert execution_path is not None
            self.assertEqual(request_path.read_bytes(), original_bytes)
            self.assertIn("キャラクターのセリフ", original_prompt)

            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            self.assertEqual(execution["effective_prompt_source"], "compiled_guarded")
            self.assertEqual(execution["effective_prompt"], compiled)
            self.assertFalse(execution["prompt_processing"]["context_ir"])
            self.assertEqual(execution["prompt_processing"]["status"], "compiled_guarded")
            self.assertFalse(execution["prompt_processing"]["translation_required"])
            self.assertFalse(execution["prompt_processing"]["model_inference"])
            self.assertEqual(execution["prompt_processing"]["dialogue_count"], 1)
            self.assertEqual(
                execution["prompt_processing"]["dialogue_events"][0]["target_cut"],
                2,
            )
            self.assertEqual(execution["embedded_video_audio_policy"], "ignore")
            self.assertEqual(execution["embedded_video_audio_indices"], [])
            self.assertFalse((request_path.parent / "context_ir").exists())
            self.assertEqual(
                (request_path.parent / "prompt_processing" / "final_prompt.txt").read_text(
                    encoding="utf-8"
                ),
                compiled,
            )
            report = json.loads(
                (request_path.parent / "prompt_processing" / "report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                report["output_sha256"],
                hashlib.sha256(compiled.encode("utf-8")).hexdigest(),
            )
            compiler_payload = json.loads(compiler_run.call_args.args[2])
            self.assertNotIn("model_path", compiler_payload)
            self.assertNotIn("model_id", compiler_payload)
            self.assertNotIn("revision", compiler_payload)
            self.assertAlmostEqual(compiler_payload["duration_seconds"], 121 / 24)
            self.assertTrue(
                any(
                    isinstance(item, dict)
                    and item["code"] == "RAW_SPEECH_CUES_REMOVED"
                    and item["severity"] == "info"
                    and "メインプロンプト" in item["message"]
                    and "台詞・声質" not in item["message"]
                    for item in report["diagnostics"]
                )
            )
            self.assertIn("DIALOGUE_PUNCTUATION_NORMALIZED", report["auto_adjustments"])
            self.assertIn("DIALOGUE_TARGET_DEFAULTED", report["diagnostics"])
            self.assertTrue(
                any(
                    isinstance(item, dict)
                    and "対象Cut内に保持" in item["message"]
                    for item in report["auto_adjustments"]
                )
            )

    def test_japanese_raw_guarded_prompt_uses_compiled_worker_output_and_keeps_request_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = _install_fake_translator(root)
            manager = JobManager(root)
            source_prompt = (
                "<Picture 1>の女性が海辺でこちらを向く。\n"
                "顔、体型、眼鏡、夏服は参照画像から正確に保持する。\n"
                "Cut 2で一度だけ <d>[Japanese] こんにちは。</d> と言う。"
            )
            request = _request(
                mode="omni",
                prompt=source_prompt,
                effective_prompt=source_prompt,
                prompt_processing_mode="official_en",
                music_policy="none",
                references=[
                    {
                        "kind": "image",
                        "stored_path": "inputs/reference.png",
                        "original_name": "reference.png",
                    }
                ],
                prompt_processing={
                    "mode": "raw_guarded",
                    "context_ir": False,
                    "dialogue_policy": "inline_h3_native",
                    "dialogue_count": 1,
                    "dialogue_events": [
                        {
                            "language": "Japanese",
                            "effective_text": "こんにちは。",
                            "target_cut": 2,
                        }
                    ],
                    "auto_adjustments": [],
                    "diagnostics": [],
                },
            )
            request_path, original_bytes = self._persist_request(
                manager, "translated-success", request
            )
            compiled = _official_six_section_prompt()
            worker_result = {
                "ok": True,
                "compiled_prompt": compiled,
                "compiler_metadata": {
                    "compiler": "fixture-six-section-compiler",
                    "schema_version": 1,
                    "model_repo": TRANSLATOR_REPO,
                    "model_revision": TRANSLATOR_REVISION,
                    "local_path": r"C:\private\must-not-leak",
                },
                "diagnostics": [
                    {
                        "severity": "info",
                        "code": "TRANSLATION_COMPLETE",
                        "message": "compiled",
                        "fatal": False,
                    }
                ],
            }
            tracked_result = (
                0,
                json.dumps(worker_result, ensure_ascii=False),
                "",
            )

            with patch.object(
                manager,
                "_run_tracked_prompt_translator",
                return_value=tracked_result,
            ) as run:
                execution_path = manager._prepare_effective_prompt(  # noqa: SLF001
                    "translated-success", request_path
                )

            self.assertIsNotNone(execution_path)
            assert execution_path is not None
            self.assertEqual(request_path.read_bytes(), original_bytes)
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            self.assertEqual(execution["effective_prompt"], compiled)
            self.assertEqual(execution["effective_prompt_source"], "translated_guarded")
            processing = execution["prompt_processing"]
            self.assertEqual(processing["status"], "translated_guarded")
            self.assertEqual(processing["mode"], "translated_guarded")
            self.assertTrue(processing["translation_required"])
            self.assertTrue(processing["model_inference"])
            self.assertTrue(processing["local_only"])
            self.assertEqual(processing["model_repo"], TRANSLATOR_REPO)
            self.assertEqual(processing["model_revision"], TRANSLATOR_REVISION)
            self.assertEqual(
                processing["source_sha256"],
                hashlib.sha256(source_prompt.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(
                processing["output_sha256"],
                hashlib.sha256(compiled.encode("utf-8")).hexdigest(),
            )
            self.assertNotIn("local_path", processing["compiler_metadata"])

            run.assert_called_once()
            command = run.call_args.args[1]
            self.assertEqual(
                command,
                [os.fspath(python), "-m", "webui.prompt_translation_worker"],
            )
            kwargs = run.call_args.kwargs
            payload = json.loads(run.call_args.args[2])
            self.assertEqual(payload["prompt"], source_prompt)
            self.assertEqual(payload["mode"], "omni")
            self.assertEqual(payload["music_policy"], "none")
            self.assertEqual(
                payload["references"],
                [
                    {
                        "kind": "image",
                        "index": 1,
                        "source_index": 0,
                        "tag": "<Picture 1>",
                    }
                ],
            )
            self.assertEqual(
                payload["dialogue_events"],
                request["prompt_processing"]["dialogue_events"],
            )
            self.assertEqual(payload["model_id"], TRANSLATOR_REPO)
            self.assertEqual(payload["revision"], TRANSLATOR_REVISION)
            self.assertEqual(
                Path(payload["model_path"]),
                (root / "models" / "prompt_translator").resolve(),
            )
            env = run.call_args.args[3]
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(kwargs["timeout"], 180)
            self.assertAlmostEqual(payload["duration_seconds"], 121 / 24)

            artifact_dir = request_path.parent / "prompt_processing"
            self.assertEqual(
                (artifact_dir / "original_prompt.txt").read_text(encoding="utf-8"),
                source_prompt,
            )
            self.assertEqual(
                (artifact_dir / "source_prompt.txt").read_text(encoding="utf-8"),
                source_prompt,
            )
            self.assertEqual(
                (artifact_dir / "final_prompt.txt").read_text(encoding="utf-8"),
                compiled,
            )

    def test_translation_worker_failure_is_fail_closed_and_never_launches_engine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _install_fake_translator(root)
            manager = JobManager(root)
            source_prompt = "海辺で女性が手を振る。"
            request = _request(
                prompt=source_prompt,
                effective_prompt=source_prompt,
                prompt_processing_mode="official_en",
                prompt_processing={"mode": "raw_guarded", "diagnostics": []},
            )
            request_path, original_bytes = self._persist_request(
                manager, "translated-worker-failure", request
            )
            response = (
                1,
                json.dumps(
                    {
                        "ok": False,
                        "error": "synthetic worker failure",
                        "code": "TRANSLATION_FAILED",
                    }
                ),
                "",
            )

            with (
                patch.object(
                    manager,
                    "_run_tracked_prompt_translator",
                    return_value=response,
                ) as run,
                patch.object(manager, "_ensure_engine") as ensure_engine,
            ):
                manager.start()
                saved = self._wait_for_terminal(manager, "translated-worker-failure")
                self._stop_runner(manager)

            run.assert_called_once()
            ensure_engine.assert_not_called()
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["compiler"]["error_code"], "TRANSLATION_FAILED")
            self.assertEqual(request_path.read_bytes(), original_bytes)
            self.assertFalse((request_path.parent / "execution_request.json").exists())
            self.assertFalse(
                (request_path.parent / "prompt_processing" / "final_prompt.txt").exists()
            )

    def test_invalid_compiled_prompt_is_fail_closed_and_never_launches_engine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _install_fake_translator(root)
            manager = JobManager(root)
            source_prompt = "海辺で女性が一度だけ手を振る。"
            request = _request(
                prompt=source_prompt,
                effective_prompt=source_prompt,
                prompt_processing_mode="official_en",
                prompt_processing={"mode": "raw_guarded", "diagnostics": []},
            )
            request_path, original_bytes = self._persist_request(
                manager, "translated-invalid-output", request
            )
            invalid_compiled = _official_three_section_prompt(
                detail="[Shot 1] 安定した medium shot frames the woman by the sea."
            )
            response = (
                0,
                json.dumps(
                    {"ok": True, "compiled_prompt": invalid_compiled, "diagnostics": []},
                    ensure_ascii=False,
                ),
                "",
            )

            with (
                patch.object(
                    manager,
                    "_run_tracked_prompt_translator",
                    return_value=response,
                ),
                patch.object(manager, "_ensure_engine") as ensure_engine,
            ):
                manager.start()
                saved = self._wait_for_terminal(manager, "translated-invalid-output")
                self._stop_runner(manager)

            ensure_engine.assert_not_called()
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(
                saved["compiler"]["error_code"], "TRANSLATOR_VALIDATION_FAILED"
            )
            self.assertTrue(
                any(
                    isinstance(item, dict)
                    and item.get("code") == "TRANSLATOR_CJK_OUTSIDE_DIALOGUE"
                    for item in saved["diagnostics"]
                )
            )
            self.assertEqual(request_path.read_bytes(), original_bytes)
            self.assertFalse((request_path.parent / "execution_request.json").exists())

    def test_missing_optional_translator_blocks_only_japanese_jobs_before_engine_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = JobManager(root)
            source_prompt = "夏の海辺をゆっくり歩く。"
            request = _request(
                prompt=source_prompt,
                effective_prompt=source_prompt,
                prompt_processing_mode="official_en",
                prompt_processing={"mode": "raw_guarded", "diagnostics": []},
            )
            request_path, original_bytes = self._persist_request(
                manager, "translator-missing", request
            )

            with (
                patch.object(manager, "_run_tracked_prompt_translator") as run,
                patch.object(manager, "_ensure_engine") as ensure_engine,
            ):
                manager.start()
                saved = self._wait_for_terminal(manager, "translator-missing")
                self._stop_runner(manager)

            run.assert_not_called()
            ensure_engine.assert_not_called()
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(
                saved["compiler"]["error_code"], "TRANSLATOR_MODEL_NOT_READY"
            )
            self.assertEqual(request_path.read_bytes(), original_bytes)
            self.assertFalse((request_path.parent / "execution_request.json").exists())

    def test_mode_schema_accept_reject_matrix_is_strict(self) -> None:
        metadata = {
            "visible_text_literals": [],
            "visible_text_literal_sha256": [],
        }
        six = _official_six_section_prompt(dialogue="")
        three = _official_three_section_prompt()
        omni_references = [{"kind": "image"}]

        self.assertEqual(
            _validate_compiled_prompt(
                six,
                six,
                mode="omni",
                references=omni_references,
                compiler_metadata=metadata,
            ),
            [],
        )
        self.assertTrue(
            any(
                item["code"] == "TRANSLATOR_INVALID_SECTIONS"
                for item in _validate_compiled_prompt(
                    three,
                    three,
                    mode="omni",
                    references=[],
                    compiler_metadata=metadata,
                )
            )
        )
        for mode in ("t2v", "i2v", "first_last"):
            with self.subTest(mode=mode, accepted="three"):
                self.assertEqual(
                    _validate_compiled_prompt(
                        three,
                        three,
                        mode=mode,
                        references=[],
                        compiler_metadata=metadata,
                    ),
                    [],
                )
            with self.subTest(mode=mode, rejected="six"):
                self.assertTrue(
                    any(
                        item["code"] == "TRANSLATOR_INVALID_SECTIONS"
                        for item in _validate_compiled_prompt(
                            six,
                            six,
                            mode=mode,
                            references=[],
                            compiler_metadata=metadata,
                        )
                    )
                )

    def test_direct_default_preserves_japanese_visual_camera_ambient_and_skips_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = JobManager(Path(temporary))
            source = (
                "Cut 1: 白い夏服の女性が海辺でカメラへゆっくり手を振る。\n"
                "Cut 2: The woman says exactly once: "
                "<d>[Japanese] 暑いね。</d>\n"
                "Audio: 穏やかな波音と海風、グラスを置く小さな音。"
            )
            request_path, original = self._persist_request(
                manager,
                "direct-japanese",
                _request(
                    prompt=source,
                    effective_prompt=source,
                    prompt_processing_mode="direct",
                    prompt_processing={
                        "mode": "raw_guarded",
                        "dialogue_count": 1,
                        "diagnostics": [],
                    },
                ),
            )

            with patch.object(manager, "_run_tracked_prompt_translator") as worker:
                execution_path = manager._prepare_effective_prompt(  # noqa: SLF001
                    "direct-japanese", request_path
                )

            worker.assert_not_called()
            self.assertIsNotNone(execution_path)
            assert execution_path is not None
            self.assertEqual(request_path.read_bytes(), original)
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            self.assertEqual(execution["effective_prompt"], source)
            self.assertEqual(execution["effective_prompt_source"], "direct")
            processing = execution["prompt_processing"]
            self.assertEqual(processing["status"], "direct")
            self.assertEqual(processing["processing_mode_requested"], "direct")
            self.assertEqual(processing["processing_mode_effective"], "direct")
            self.assertFalse(processing["translation_required"])
            self.assertFalse(processing["model_inference"])
            self.assertIn("白い夏服", execution["effective_prompt"])
            self.assertIn("カメラ", execution["effective_prompt"])
            self.assertIn("波音と海風", execution["effective_prompt"])

    def test_direct_reference_tag_allowlist_matches_generation_inputs(self) -> None:
        cases = (
            ("t2v", "A scene without references.", [], True),
            ("t2v", "Use <Picture 1>.", [], False),
            ("i2v", "Animate <Picture 1> and <Subject 1>.", [], True),
            ("i2v", "Animate <picture 1>.", [], False),
            ("i2v", "Animate <Picture   1>.", [], False),
            ("i2v", "Animate <Picture1>.", [], False),
            ("i2v", "Animate <Picture 0>.", [], False),
            ("i2v", "Animate <<Picture 1>>.", [], False),
            ("i2v", "Animate <Unknown 1>.", [], False),
            ("i2v", "Animate <|cutoff|>.", [], False),
            ("i2v", "Animate <Picture 2>.", [], False),
            (
                "first_last",
                "Transition from <Picture 1> to <Picture 2>; retain <Subject 1>.",
                [],
                True,
            ),
            ("first_last", "Use <Picture 3>.", [], False),
            (
                "omni",
                "Use <Picture 1>, <Subject 1>, <Video 1>, and <Audio 1>.",
                [{"kind": "image"}, {"kind": "video"}, {"kind": "audio"}],
                True,
            ),
            (
                "omni",
                "Use <Picture 2> and <Audio 9>.",
                [{"kind": "image"}, {"kind": "audio"}],
                False,
            ),
        )
        for mode, prompt, references, accepted in cases:
            with self.subTest(mode=mode, prompt=prompt):
                diagnostics = _validate_direct_reference_tags(
                    prompt,
                    mode=mode,
                    references=references,
                )
                self.assertEqual(not diagnostics, accepted)

    def test_direct_invalid_reference_fails_before_engine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = JobManager(Path(temporary))
            source = "Animate the unattached <Picture 99>."
            self._persist_request(
                manager,
                "direct-invalid-reference",
                _request(
                    prompt=source,
                    effective_prompt=source,
                    prompt_processing_mode="direct",
                    prompt_processing={"mode": "raw_guarded", "diagnostics": []},
                ),
            )

            with (
                patch.object(manager, "_run_tracked_prompt_translator") as worker,
                patch.object(manager, "_ensure_engine") as engine,
            ):
                manager.start()
                saved = self._wait_for_terminal(manager, "direct-invalid-reference")
                self._stop_runner(manager)

            worker.assert_not_called()
            engine.assert_not_called()
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(
                saved["compiler"]["error_code"],
                "PROMPT_REFERENCE_OUT_OF_RANGE",
            )

    def test_direct_noncanonical_control_tag_fails_before_engine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = JobManager(Path(temporary))
            source = "Animate the opening image <picture   1>."
            self._persist_request(
                manager,
                "direct-invalid-control",
                _request(
                    mode="i2v",
                    prompt=source,
                    effective_prompt=source,
                    prompt_processing_mode="direct",
                    prompt_processing={"mode": "raw_guarded", "diagnostics": []},
                ),
            )

            with (
                patch.object(manager, "_run_tracked_prompt_translator") as worker,
                patch.object(manager, "_ensure_engine") as engine,
            ):
                manager.start()
                saved = self._wait_for_terminal(manager, "direct-invalid-control")
                self._stop_runner(manager)

            worker.assert_not_called()
            engine.assert_not_called()
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(
                saved["compiler"]["error_code"],
                "PROMPT_INVALID_CONTROL_TAG",
            )
            self.assertTrue(
                any(
                    item.get("code") == "PROMPT_INVALID_CONTROL_TAG"
                    for item in saved["diagnostics"]
                    if isinstance(item, dict)
                )
            )

    def test_dialogue_free_soundscape_rejects_positive_or_negative_speech_cues(self) -> None:
        source = "A quiet empty beach with waves and wind."
        metadata = {
            "visible_text_literals": [],
            "visible_text_literal_sha256": [],
        }
        bad_soundscapes = (
            "The soundtrack contains continuous spoken Chinese narration.",
            "Waves and wind are heard; no human voice or dialogue.",
        )

        for soundscape in bad_soundscapes:
            with self.subTest(soundscape=soundscape):
                compiled = _official_three_section_prompt().replace(
                    "Gentle ocean surf and a soft breeze move through the scene.",
                    soundscape,
                )
                diagnostics = _validate_compiled_prompt(
                    source,
                    compiled,
                    mode="t2v",
                    references=[],
                    compiler_metadata=metadata,
                )
                self.assertTrue(
                    any(
                        item["code"] == "TRANSLATOR_UNREQUESTED_SPEECH_AUDIO"
                        for item in diagnostics
                    )
                )

    def test_compiler_verified_visible_japanese_literal_is_preserved_without_mt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _install_fake_runtime(root)
            manager = JobManager(root)
            literal = "営業中"
            source = f'A storefront sign visibly reads "{literal}".'
            compiled = _official_three_section_prompt(
                detail=f'[Shot 1] A storefront sign visibly reads "{literal}".'
            )
            request_path, _ = self._persist_request(
                manager,
                "visible-text-success",
                _request(
                    prompt=source,
                    effective_prompt=source,
                    prompt_processing_mode="official_en",
                    prompt_processing={"mode": "raw_guarded", "diagnostics": []},
                ),
            )
            worker_result = _worker_success(
                compiled,
                visible_text_literals=[literal],
                visible_text_literal_sha256=[
                    hashlib.sha256(literal.encode("utf-8")).hexdigest()
                ],
            )

            with patch.object(
                manager,
                "_run_tracked_prompt_translator",
                return_value=worker_result,
            ) as run:
                execution_path = manager._prepare_effective_prompt(  # noqa: SLF001
                    "visible-text-success", request_path
                )

            self.assertIsNotNone(execution_path)
            assert execution_path is not None
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            self.assertIn(f'"{literal}"', execution["effective_prompt"])
            self.assertFalse(execution["prompt_processing"]["model_inference"])
            payload = json.loads(run.call_args.args[2])
            self.assertNotIn("model_path", payload)

    def test_official_prompt_locally_authorizes_visible_literal_and_bypasses_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = JobManager(Path(temporary))
            literal = "営業中"
            official = _official_three_section_prompt(
                detail=f'[Shot 1] A storefront sign visibly reads "{literal}".'
            )
            request_path, _ = self._persist_request(
                manager,
                "official-visible-text",
                _request(
                    prompt=official,
                    effective_prompt=official,
                    prompt_processing={"mode": "raw_guarded", "diagnostics": []},
                ),
            )

            with patch.object(manager, "_run_tracked_prompt_translator") as worker:
                execution_path = manager._prepare_effective_prompt(  # noqa: SLF001
                    "official-visible-text", request_path
                )

            worker.assert_not_called()
            self.assertIsNotNone(execution_path)
            assert execution_path is not None
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            self.assertEqual(execution["effective_prompt"], official)
            self.assertEqual(execution["effective_prompt_source"], "official_prompt")
            self.assertEqual(
                execution["prompt_processing"]["compiler_metadata"],
                {
                    "visible_text_literals": [literal],
                    "visible_text_literal_sha256": [
                        hashlib.sha256(literal.encode("utf-8")).hexdigest()
                    ],
                },
            )

    def test_official_prompt_rejects_unclassified_quoted_japanese(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = JobManager(Path(temporary))
            official = _official_three_section_prompt(
                detail='[Shot 1] A woman says "こんにちは" while facing the camera.'
            )
            request_path, _ = self._persist_request(
                manager,
                "official-unclassified-cjk",
                _request(
                    prompt=official,
                    effective_prompt=official,
                    prompt_processing={"mode": "raw_guarded", "diagnostics": []},
                ),
            )

            with patch.object(manager, "_run_tracked_prompt_translator") as worker:
                execution_path = manager._prepare_effective_prompt(  # noqa: SLF001
                    "official-unclassified-cjk", request_path
                )

            worker.assert_not_called()
            self.assertIsNone(execution_path)
            saved = manager.get_job("official-unclassified-cjk")
            assert saved is not None
            self.assertEqual(
                saved["compiler"]["error_code"],
                "UNCLASSIFIED_CJK_QUOTED_TEXT",
            )

    def test_visible_japanese_literal_with_bad_hash_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _install_fake_runtime(root)
            manager = JobManager(root)
            literal = "営業中"
            source = f'A storefront sign visibly reads "{literal}".'
            compiled = _official_three_section_prompt(
                detail=f'[Shot 1] A storefront sign visibly reads "{literal}".'
            )
            request_path, _ = self._persist_request(
                manager,
                "visible-text-bad-hash",
                _request(
                    prompt=source,
                    effective_prompt=source,
                    prompt_processing_mode="official_en",
                    prompt_processing={"mode": "raw_guarded", "diagnostics": []},
                ),
            )

            with patch.object(
                manager,
                "_run_tracked_prompt_translator",
                return_value=_worker_success(
                    compiled,
                    visible_text_literals=[literal],
                    visible_text_literal_sha256=["0" * 64],
                ),
            ):
                execution_path = manager._prepare_effective_prompt(  # noqa: SLF001
                    "visible-text-bad-hash", request_path
                )

            self.assertIsNone(execution_path)
            saved = manager.get_job("visible-text-bad-hash")
            assert saved is not None
            self.assertEqual(saved["status"], "failed")
            self.assertTrue(
                any(
                    item.get("code") == "TRANSLATOR_VISIBLE_TEXT_METADATA_INVALID"
                    for item in saved["diagnostics"]
                    if isinstance(item, dict)
                )
            )
            self.assertFalse((request_path.parent / "execution_request.json").exists())

    def test_malformed_native_dialogue_fails_before_worker_or_engine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = JobManager(Path(temporary))
            source = "A woman says <d>[Japanese] 閉じていない。"
            request_path, _ = self._persist_request(
                manager,
                "malformed-dialogue",
                _request(
                    prompt=source,
                    effective_prompt=source,
                    prompt_processing={"mode": "raw_guarded", "diagnostics": []},
                ),
            )

            with (
                patch.object(manager, "_run_tracked_prompt_translator") as worker,
                patch.object(manager, "_ensure_engine") as engine,
            ):
                execution_path = manager._prepare_effective_prompt(  # noqa: SLF001
                    "malformed-dialogue", request_path
                )

            self.assertIsNone(execution_path)
            worker.assert_not_called()
            engine.assert_not_called()
            saved = manager.get_job("malformed-dialogue")
            assert saved is not None
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["compiler"]["error_code"], "MALFORMED_DIALOGUE_TAG")
            self.assertFalse((request_path.parent / "execution_request.json").exists())

    def test_worker_paths_stay_private_and_browser_diagnostics_are_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _install_fake_translator(root)
            manager = JobManager(root)
            source = "海辺で女性が手を振る。"
            request_path, _ = self._persist_request(
                manager,
                "private-worker-path",
                _request(
                    prompt=source,
                    effective_prompt=source,
                    prompt_processing_mode="official_en",
                    prompt_processing={"mode": "raw_guarded", "diagnostics": []},
                ),
            )
            private_path = r"C:\Users\Secret\private-model\weights.bin"
            response = (
                1,
                json.dumps(
                    {
                        "ok": False,
                        "code": "TRANSLATION_FAILED",
                        "error": f"failed while loading {private_path}",
                        "diagnostics": [
                            {
                                "severity": "error",
                                "code": "TRANSLATION_FAILED",
                                "message": f"trace at {private_path}",
                                "fatal": True,
                            }
                        ],
                    }
                ),
                f"decoder replacement: \ufffd; local file {private_path}",
            )

            with patch.object(
                manager,
                "_run_tracked_prompt_translator",
                return_value=response,
            ):
                self.assertIsNone(
                    manager._prepare_effective_prompt(  # noqa: SLF001
                        "private-worker-path", request_path
                    )
                )

            saved = manager.get_job("private-worker-path")
            assert saved is not None
            public_json = json.dumps(saved, ensure_ascii=False)
            self.assertNotIn(private_path, public_json)
            self.assertNotIn("private-model", public_json)
            self.assertIn("[local path redacted]", public_json)
            artifact_dir = request_path.parent / "prompt_processing"
            self.assertIn(
                private_path,
                (artifact_dir / "translator.stderr.log").read_text(encoding="utf-8"),
            )
            raw_response = json.loads(
                (artifact_dir / "translator_response.json").read_text(encoding="utf-8")
            )
            self.assertIn(private_path, raw_response["error"])
            raw_failure = json.loads(
                (artifact_dir / "translator_failure.private.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn(private_path, raw_failure["detail"])

    def test_cancel_during_translation_reaps_exact_worker_then_starts_next_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _install_fake_translator(root)
            manager = JobManager(root)
            first_source = "海辺で女性がゆっくり手を振る。"
            self._persist_request(
                manager,
                "cancel-translator",
                _request(
                    prompt=first_source,
                    effective_prompt=first_source,
                    prompt_processing_mode="official_en",
                    prompt_processing={"mode": "raw_guarded", "diagnostics": []},
                ),
            )
            official = _official_three_section_prompt()
            self._persist_request(
                manager,
                "next-official",
                _request(
                    prompt=official,
                    effective_prompt=official,
                    prompt_processing={"mode": "raw_guarded", "diagnostics": []},
                ),
            )
            process = _BlockingTranslatorProcess(_worker_success(official))
            process_job = _FakeProcessJob()
            engine = _FakeCompletedEngine()

            def ensure_engine(_variant: str) -> _FakeCompletedEngine:
                self.assertTrue(process.communication_finished.is_set())
                return engine

            with (
                patch("webui.job_manager.subprocess.Popen", return_value=process) as popen,
                patch("webui.job_manager.ProcessJob", return_value=process_job),
                patch.object(manager, "_ensure_engine", side_effect=ensure_engine) as ensure,
            ):
                manager.start()
                self.assertTrue(process.started.wait(2), "translator did not start")
                cancelled = manager.cancel("cancel-translator")
                self.assertIsNotNone(cancelled)
                next_job = self._wait_for_terminal(manager, "next-official")
                self._stop_runner(manager)

            self.assertEqual(cancelled["status"], "cancelled")
            self.assertEqual(next_job["status"], "completed")
            self.assertTrue(process.terminated.is_set())
            self.assertTrue(process.communication_finished.is_set())
            self.assertTrue(process.reaped.is_set())
            self.assertTrue(process_job.terminated.is_set())
            self.assertIsNone(manager._current_translation_process)  # noqa: SLF001
            self.assertIsNone(manager._current_translation_job_id)  # noqa: SLF001
            ensure.assert_called_once_with("fl2va")
            kwargs = popen.call_args.kwargs
            self.assertEqual(kwargs["encoding"], "utf-8")
            self.assertEqual(kwargs["errors"], "replace")
            self.assertTrue(kwargs["text"])

    def test_stop_during_translation_terminates_and_reaps_without_timeout_wait(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _install_fake_translator(root)
            manager = JobManager(root)
            source = "海辺で女性が歩く。"
            self._persist_request(
                manager,
                "stop-translator",
                _request(
                    prompt=source,
                    effective_prompt=source,
                    prompt_processing_mode="official_en",
                    prompt_processing={"mode": "raw_guarded", "diagnostics": []},
                ),
            )
            process = _BlockingTranslatorProcess(
                _worker_success(_official_three_section_prompt())
            )
            process_job = _FakeProcessJob()

            with (
                patch("webui.job_manager.subprocess.Popen", return_value=process),
                patch("webui.job_manager.ProcessJob", return_value=process_job),
                patch.object(manager, "_ensure_engine") as ensure_engine,
            ):
                manager.start()
                self.assertTrue(process.started.wait(2), "translator did not start")
                started = time.monotonic()
                manager.stop()
                runner = manager._runner  # noqa: SLF001
                assert runner is not None
                runner.join(timeout=2)
                elapsed = time.monotonic() - started

            self.assertFalse(runner.is_alive())
            self.assertLess(elapsed, 2)
            self.assertTrue(process.terminated.is_set())
            self.assertTrue(process.communication_finished.is_set())
            self.assertTrue(process.reaped.is_set())
            self.assertTrue(process_job.terminated.is_set())
            self.assertIsNone(manager._current_translation_process)  # noqa: SLF001
            ensure_engine.assert_not_called()

    def test_fatal_compilation_fails_before_engine_launch_even_if_ir_text_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = JobManager(Path(temporary))
            request = _request(prompt="")
            request_path, original_bytes = self._persist_request(
                manager, "compile-fatal", request
            )
            blocked = real_compile_request(request)
            self.assertTrue(blocked.fatal)
            fatal_with_text = dataclasses.replace(
                blocked,
                ir_text="This text must never be sent to the engine.",
            )

            with (
                patch(
                    "webui.context_ir.compile_request",
                    return_value=fatal_with_text,
                ),
                patch.object(manager, "_ensure_engine") as ensure_engine,
            ):
                manager.start()
                saved = self._wait_for_terminal(manager, "compile-fatal")
                self._stop_runner(manager)

            ensure_engine.assert_not_called()
            self.assertEqual(saved["status"], "failed")
            self.assertTrue(saved["compiler"]["fatal"])
            self.assertEqual(saved["compiler"]["status"], "blocked")
            self.assertEqual(request_path.read_bytes(), original_bytes)
            self.assertFalse((request_path.parent / "execution_request.json").exists())
            self.assertTrue(
                (request_path.parent / "context_ir" / "diagnostics.json").is_file()
            )
            self.assertTrue(
                any(item["fatal"] for item in saved["diagnostics"])
            )

    def test_compiler_exception_fails_open_and_worker_receives_legacy_execution_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = JobManager(Path(temporary))
            request = _request(
                prompt="Keep this exact conservative prompt.",
                effective_prompt="Previously validated conservative prompt.",
            )
            request_path, original_bytes = self._persist_request(
                manager, "compile-fallback", request
            )
            engine = _FakeCompletedEngine()

            with (
                patch(
                    "webui.context_ir.compile_request",
                    side_effect=RuntimeError("synthetic compiler failure"),
                ),
                patch.object(manager, "_ensure_engine", return_value=engine) as ensure_engine,
            ):
                manager.start()
                saved = self._wait_for_terminal(manager, "compile-fallback")
                self._stop_runner(manager)

            ensure_engine.assert_called_once_with("fl2va")
            self.assertEqual(saved["status"], "completed")
            self.assertEqual(request_path.read_bytes(), original_bytes)

            command = json.loads(engine.stdin.getvalue().strip())
            execution_path = Path(command["request"])
            self.assertNotEqual(os.fspath(execution_path), os.fspath(request_path))
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            self.assertEqual(
                execution["effective_prompt"],
                "Previously validated conservative prompt.",
            )
            self.assertEqual(execution["effective_prompt_source"], "legacy_fallback")
            self.assertEqual(execution["embedded_video_audio_policy"], "ignore")
            self.assertEqual(execution["compiler"]["status"], "degraded_fallback")
            self.assertEqual(saved["compiler"], execution["compiler"])

            compiler_dir = request_path.parent / "compiler"
            audit = json.loads(
                (compiler_dir / "compiler_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(audit["exception_type"], "RuntimeError")
            self.assertEqual(audit["exception"], "synthetic compiler failure")
            self.assertEqual(
                (compiler_dir / "final_ir.txt").read_text(encoding="utf-8"),
                "Previously validated conservative prompt.",
            )
            self.assertTrue(
                any(
                    item["code"] == "COMPILER_FALLBACK"
                    for item in saved["diagnostics"]
                )
            )


if __name__ == "__main__":
    unittest.main()
