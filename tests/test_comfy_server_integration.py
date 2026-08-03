from __future__ import annotations

import copy
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.job_manager import JobManager  # noqa: E402
import webui.job_manager as job_manager_module  # noqa: E402


class IsolatedManager:
    """JobManager-shaped test double that never starts a runner thread."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.jobs_dir = self.root / "webui_data" / "jobs"
        self.outputs_dir = self.root / "outputs" / "webui"
        self.jobs_dir.mkdir(parents=True)
        self.outputs_dir.mkdir(parents=True)
        self.submitted: list[dict] = []

    def submit(self, job: dict) -> dict:
        saved = copy.deepcopy(job)
        self.submitted.append(saved)
        job_dir = self.jobs_dir / saved["id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(
            json.dumps(saved, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return copy.deepcopy(saved)

    def start(self) -> None:
        raise AssertionError("the integration test must not start a JobManager thread")

    def stop(self) -> None:
        raise AssertionError("the integration test must not stop a runner that was never started")

    def list_jobs(self) -> list[dict]:
        return copy.deepcopy(self.submitted)

    def current_job_id(self) -> None:
        return None


class ComfyServerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.test_root = Path(cls.temporary.name)
        cls.manager = IsolatedManager(cls.test_root)

        # webui.server constructs its global manager at import time.  Replace
        # that constructor during import so merely collecting these tests never
        # scans or updates the repository's real webui_data/jobs directory.
        sys.modules.pop("webui.server", None)
        with patch.object(job_manager_module, "JobManager", return_value=cls.manager):
            cls.server = importlib.import_module("webui.server")

        cls.original_model_root = cls.server.COMFY_MODEL_ROOT
        cls.original_model_files = cls.server.COMFY_MODEL_FILES
        cls.model_root = cls.test_root / "models" / "comfy"
        model_files = {
            "fl2va": cls.model_root / "diffusion_models" / "fl2va.safetensors",
            "ref2va": cls.model_root / "diffusion_models" / "ref2va.safetensors",
            "text_encoder": cls.model_root / "text_encoders" / "qwen.safetensors",
            "video_vae": cls.model_root / "vae" / "video_vae.safetensors",
            "audio_vae": cls.model_root / "vae" / "audio_vae.safetensors",
        }
        for model_path in model_files.values():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"test-model-marker")
        cls.server.COMFY_MODEL_ROOT = cls.model_root
        cls.server.COMFY_MODEL_FILES = model_files

        # Do not use TestClient as a context manager: that would deliberately
        # run FastAPI's lifespan and start the real background worker contract.
        cls.client = TestClient(cls.server.app, base_url="http://127.0.0.1")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        cls.server.COMFY_MODEL_ROOT = cls.original_model_root
        cls.server.COMFY_MODEL_FILES = cls.original_model_files
        sys.modules.pop("webui.server", None)
        cls.temporary.cleanup()

    def setUp(self) -> None:
        self.manager.submitted.clear()

    @staticmethod
    def _t2v_form(*, steps: int = 20, acceleration: str = "balanced", ref_image_size: str = "max") -> dict[str, str]:
        return {
            "mode": "t2v",
            "style": "natural",
            "prompt": "A paper airplane circles through warm afternoon sunlight.",
            "width": "640",
            "height": "384",
            "num_frames": "243",
            "steps": str(steps),
            "seed": "424242",
            "acceleration": acceleration,
            "ref_image_size": ref_image_size,
            "audio_preset": "ambience",
            "dialogue": "",
            "soundscape": "Soft wind and paper flutter.",
            "music_policy": "none",
            "audio_gain_db": "-2",
        }

    def _post_multipart_t2v(self, **overrides: str):
        form = self._t2v_form()
        form.update(overrides)
        # A valid optional image makes httpx build a real multipart request;
        # T2V does not require or condition on it, but the upload boundary is
        # exercised without relying on private TestClient encoders.
        files = {"first_image": ("reference.png", b"\x89PNG\r\n\x1a\nfixture", "image/png")}
        return self.client.post(
            "/api/jobs",
            data=form,
            files=files,
            headers={self.server.LOCAL_MUTATION_HEADER: self.server.LOCAL_MUTATION_VALUE},
        )

    def test_capabilities_report_all_five_comfy_models_ready(self):
        capabilities = self.server.capabilities()

        self.assertEqual(capabilities["backend"], "comfy")
        self.assertTrue(capabilities["ready"])
        self.assertEqual(len(capabilities["model_files"]), 5)
        self.assertTrue(all(capabilities["model_files"].values()))
        self.assertTrue(all(capabilities["modes"].values()))
        self.assertEqual(capabilities["acceleration"]["minimum_steps"], 12)
        self.assertEqual(capabilities["acceleration"]["default"], "off")
        self.assertEqual(capabilities["attention"]["backend"], "sage")

    def test_local_browser_boundary_blocks_rebinding_and_simple_cross_site_posts(self):
        normal = self.client.get("/")
        rebound = self.client.get("/api/state", headers={"host": "evil.example"})
        missing_header = self.client.post("/api/jobs", data=self._t2v_form())
        cross_site = self.client.get("/api/state", headers={"sec-fetch-site": "cross-site"})
        wrong_origin = self.client.post(
            "/api/jobs",
            data=self._t2v_form(),
            headers={
                self.server.LOCAL_MUTATION_HEADER: self.server.LOCAL_MUTATION_VALUE,
                "origin": "https://evil.example",
            },
        )

        self.assertEqual(normal.status_code, 200)
        self.assertIn("frame-ancestors 'none'", normal.headers["content-security-policy"])
        self.assertEqual(normal.headers["x-frame-options"], "DENY")
        self.assertEqual(rebound.status_code, 421)
        self.assertEqual(missing_header.status_code, 403)
        self.assertEqual(cross_site.status_code, 403)
        self.assertEqual(wrong_origin.status_code, 403)
        self.assertEqual(self.manager.submitted, [])

    def test_declared_aggregate_upload_limit_is_rejected_before_multipart_parsing(self):
        before = set(self.manager.jobs_dir.iterdir())
        response = self.client.post(
            "/api/jobs",
            content=b"not parsed",
            headers={
                self.server.LOCAL_MUTATION_HEADER: self.server.LOCAL_MUTATION_VALUE,
                "content-type": "multipart/form-data; boundary=unused",
                "content-length": str(self.server.MAX_REQUEST_BYTES + 1),
            },
        )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.manager.submitted, [])
        self.assertEqual(set(self.manager.jobs_dir.iterdir()), before)

    def test_omni_reference_count_is_rejected_before_job_directory_creation(self):
        before = set(self.manager.jobs_dir.iterdir())
        form = self._t2v_form()
        form["mode"] = "omni"
        files = [
            ("references", (f"reference-{index}.png", b"tiny", "image/png"))
            for index in range(13)
        ]
        response = self.client.post(
            "/api/jobs",
            data=form,
            files=files,
            headers={self.server.LOCAL_MUTATION_HEADER: self.server.LOCAL_MUTATION_VALUE},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.manager.submitted, [])
        self.assertEqual(set(self.manager.jobs_dir.iterdir()), before)

    def test_multipart_t2v_persists_backend_cache_reference_policy_and_model_root(self):
        response = self._post_multipart_t2v()

        self.assertEqual(response.status_code, 200, response.text)
        public_job = response.json()
        self.assertEqual(len(self.manager.submitted), 1)
        submitted_job = self.manager.submitted[0]
        job_id = submitted_job["id"]
        job_dir = self.manager.jobs_dir / job_id
        request = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
        persisted_job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))

        for saved in (request, submitted_job, persisted_job, public_job):
            self.assertEqual(saved["backend"], "comfy")
            self.assertEqual(saved["acceleration"], "balanced")
            self.assertEqual(saved["ref_image_size"], "max")
            self.assertTrue(saved["cache"]["enabled"])
            self.assertEqual(saved["cache"]["requested"], "balanced")
            self.assertEqual(saved["cache"]["effective"], "balanced")
            self.assertEqual(saved["cache"]["reuse_threshold"], 0.30)

        self.assertEqual(Path(request["model_path"]), self.model_root.resolve())
        self.assertEqual(request["mode"], "t2v")
        self.assertEqual(request["steps"], 20)
        self.assertEqual(request["audio_gain_db"], -2.0)
        self.assertEqual(len(request["attachments"]), 1)
        self.assertTrue(Path(request["attachments"][0]["stored_path"]).is_file())

        # Private filesystem locations must not leak back through the public API.
        self.assertNotIn("output_path", public_job)
        self.assertNotIn("preview_path", public_job)

    def test_draft_steps_automatically_disable_easycache_but_keep_requested_preset(self):
        response = self._post_multipart_t2v(steps="7", acceleration="balanced")

        self.assertEqual(response.status_code, 200, response.text)
        job = self.manager.submitted[0]
        request = json.loads(
            (self.manager.jobs_dir / job["id"] / "request.json").read_text(encoding="utf-8")
        )
        for saved in (job, request):
            self.assertEqual(saved["acceleration"], "balanced")
            self.assertEqual(saved["cache"]["requested"], "balanced")
            self.assertEqual(saved["cache"]["effective"], "off")
            self.assertFalse(saved["cache"]["enabled"])
            self.assertEqual(saved["cache"]["reason"], "steps_below_12")
            self.assertIsNone(saved["cache"]["reuse_threshold"])

    def test_invalid_acceleration_and_reference_image_size_are_rejected_before_job_creation(self):
        before = set(self.manager.jobs_dir.iterdir())
        bad_acceleration = self._post_multipart_t2v(acceleration="turbo")
        bad_reference_policy = self._post_multipart_t2v(ref_image_size="original")

        self.assertEqual(bad_acceleration.status_code, 400)
        self.assertEqual(bad_reference_policy.status_code, 400)
        self.assertEqual(self.manager.submitted, [])
        self.assertEqual(set(self.manager.jobs_dir.iterdir()), before)


class JobManagerComfyBoundaryTests(unittest.TestCase):
    def test_h3event_merges_cache_and_timing_updates_instead_of_erasing_initial_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager = JobManager(Path(temporary))
            manager.submit(
                {
                    "id": "merge-test",
                    "status": "running",
                    "cache": {
                        "requested": "balanced",
                        "effective": "balanced",
                        "reuse_threshold": 0.30,
                        "skipped_steps": 0,
                    },
                    "timings": {"queued_seconds": 1.25},
                }
            )

            manager._handle_event(  # noqa: SLF001 - intentional process-boundary test
                "merge-test",
                json.dumps(
                    {
                        "status": "completed",
                        "backend": "comfy",
                        "acceleration": "balanced",
                        "cache": {"skipped_steps": 7, "speedup": 1.35},
                        "timings": {"generation_seconds": 870.0},
                    }
                ),
            )
            saved = manager.get_job("merge-test")

            assert saved is not None
            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["backend"], "comfy")
            self.assertEqual(saved["cache"]["requested"], "balanced")
            self.assertEqual(saved["cache"]["reuse_threshold"], 0.30)
            self.assertEqual(saved["cache"]["skipped_steps"], 7)
            self.assertEqual(saved["cache"]["speedup"], 1.35)
            self.assertEqual(saved["timings"], {"queued_seconds": 1.25, "generation_seconds": 870.0})
            self.assertIn("finished_at", saved)

    def test_engine_command_uses_dedicated_comfy_venv_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / ".comfy-venv" / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"test executable marker")
            manager = JobManager(root)
            process = Mock()
            process.poll.return_value = None

            process_job = Mock()
            with (
                patch.object(job_manager_module.subprocess, "Popen", return_value=process) as popen,
                patch.object(job_manager_module, "ProcessJob", return_value=process_job),
            ):
                returned = manager._ensure_engine("fl2va")  # noqa: SLF001 - command contract test

            self.assertIs(returned, process)
            process_job.attach.assert_called_once_with(process)
            command = popen.call_args.args[0]
            self.assertEqual(
                command,
                [os.fspath(python), "-m", "webui.comfy_engine_worker", "--serve"],
            )
            self.assertEqual(popen.call_args.kwargs["cwd"], root.resolve())
            self.assertEqual(popen.call_args.kwargs["env"]["PYTHONIOENCODING"], "utf-8")
            self.assertEqual(popen.call_args.kwargs["env"]["PYTHONUNBUFFERED"], "1")

    def test_job_object_creation_failure_never_starts_an_uncontained_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / ".comfy-venv" / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"test executable marker")
            manager = JobManager(root)

            with (
                patch.object(job_manager_module, "ProcessJob", side_effect=OSError("job failed")),
                patch.object(job_manager_module.subprocess, "Popen") as popen,
            ):
                with self.assertRaisesRegex(OSError, "job failed"):
                    manager._ensure_engine("fl2va")  # noqa: SLF001

            popen.assert_not_called()
            self.assertIsNone(manager._current_process)  # noqa: SLF001
            self.assertIsNone(manager._current_process_job)  # noqa: SLF001

    def test_cancel_claims_the_exact_current_engine_before_releasing_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager = JobManager(Path(temporary))
            old_process = Mock(pid=101)
            old_process.poll.return_value = None
            old_job = Mock()
            new_process = Mock(pid=202)
            new_process.poll.return_value = None
            new_job = Mock()
            manager._jobs["old"] = {"id": "old", "status": "running"}  # noqa: SLF001
            manager._current_job_id = "old"  # noqa: SLF001
            manager._current_process = old_process  # noqa: SLF001
            manager._current_process_job = old_job  # noqa: SLF001

            def cleanup(process, process_job):
                self.assertIs(process, old_process)
                self.assertIs(process_job, old_job)
                with manager._lock:  # noqa: SLF001
                    manager._current_process = new_process  # noqa: SLF001
                    manager._current_process_job = new_job  # noqa: SLF001

            with patch.object(manager, "_cleanup_owned_process", side_effect=cleanup):
                cancelled = manager.cancel("old")

            assert cancelled is not None
            self.assertEqual(cancelled["status"], "cancelled")
            self.assertIs(manager._current_process, new_process)  # noqa: SLF001
            self.assertIs(manager._current_process_job, new_job)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
