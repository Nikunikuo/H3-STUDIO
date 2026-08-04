from __future__ import annotations

import copy
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

import webui.job_manager as job_manager_module  # noqa: E402
from webui import comfy_engine_worker as worker  # noqa: E402
from webui.job_manager import _translator_reference_manifest  # noqa: E402
from webui.prompt_translation import translate_and_compile_prompt  # noqa: E402


class _IsolatedManager:
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
        return copy.deepcopy(saved)

    def list_jobs(self) -> list[dict]:
        return copy.deepcopy(self.submitted)

    def current_job_id(self) -> None:
        return None


class StandaloneAudioPolicyTests(unittest.TestCase):
    DIALOGUE_PROMPT = (
        'Cut 1\nThe woman shown in <Picture 1> says "Hello." using <Audio 1>.\n'
        "She closes the door while ocean surf continues."
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.test_root = Path(cls.temporary.name).resolve()
        cls.manager = _IsolatedManager(cls.test_root)
        sys.modules.pop("webui.server", None)
        with patch.object(job_manager_module, "JobManager", return_value=cls.manager):
            cls.server = importlib.import_module("webui.server")
        cls.original_model_root = cls.server.COMFY_MODEL_ROOT
        cls.original_model_files = cls.server.COMFY_MODEL_FILES
        model_root = cls.test_root / "models" / "comfy"
        model_files = {
            "fl2va": model_root / "diffusion_models" / "fl2va.safetensors",
            "ref2va": model_root / "diffusion_models" / "ref2va.safetensors",
            "text_encoder": model_root / "text_encoders" / "qwen.safetensors",
            "video_vae": model_root / "vae" / "video_vae.safetensors",
            "audio_vae": model_root / "vae" / "audio_vae.safetensors",
        }
        for path in model_files.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"model-marker")
        cls.server.COMFY_MODEL_ROOT = model_root
        cls.server.COMFY_MODEL_FILES = model_files
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

    def _post(self, *, prompt: str, policy: str | None, audio_count: int = 1):
        form = {
            "mode": "omni", "style": "natural", "prompt": prompt,
            "width": "640", "height": "384", "num_frames": "124",
            "steps": "7", "seed": "424242", "acceleration": "off",
            "ref_image_size": "match", "prompt_processing_mode": "official_en",
            "audio_preset": "auto", "dialogue": "", "soundscape": "",
            "music_policy": "auto", "audio_gain_db": "0",
        }
        if policy is not None:
            form["standalone_audio_policy"] = policy
        files = [("references", ("character.png", b"image", "image/png"))]
        files += [
            ("references", (f"voice-{n}.wav", b"audio", "audio/wav"))
            for n in range(1, audio_count + 1)
        ]
        return self.client.post(
            "/api/jobs", data=form, files=files,
            headers={self.server.LOCAL_MUTATION_HEADER: self.server.LOCAL_MUTATION_VALUE},
        )

    def _request(self) -> dict:
        self.assertEqual(len(self.manager.submitted), 1)
        job_id = self.manager.submitted[0]["id"]
        return json.loads(
            (self.manager.jobs_dir / job_id / "request.json").read_text(encoding="utf-8")
        )

    def _staged(self, request: dict) -> dict:
        paths = worker.make_runtime_paths(self.test_root, request["id"])
        staged, _ = worker.stage_request_inputs(request, paths, self.test_root)
        return staged

    def test_dialogue_priority_excludes_audio_from_worker_and_official_compile(self) -> None:
        response = self._post(
            prompt=self.DIALOGUE_PROMPT, policy="dialogue_priority", audio_count=2
        )
        self.assertEqual(response.status_code, 200, response.text)
        request = self._request()
        self.assertEqual(
            [item["kind"] for item in request["uploaded_references"]],
            ["image", "audio", "audio"],
        )
        self.assertEqual([item["kind"] for item in request["references"]], ["image"])
        self.assertEqual(request["excluded_audio_reference_ids"], [1, 2])
        self.assertFalse(request["standalone_audio_conditioning"])
        self.assertNotIn("<Audio 1>", request["effective_prompt"])
        self.assertEqual(request["effective_prompt"].count("<d>[English] Hello.</d>"), 1)
        event = request["prompt_processing"]["dialogue_events"][0]
        self.assertEqual(event["audio_reference_id_requested"], 1)
        self.assertIsNone(event["audio_reference_id_effective"])
        self.assertTrue(event["audio_reference_suppressed"])
        self.assertEqual(self._staged(request)["reference_audio_filenames"], [])

        compiled = translate_and_compile_prompt(
            request["effective_prompt"],
            dialogue_events=request["prompt_processing"]["dialogue_events"],
            reference_inventory=_translator_reference_manifest(request["references"]),
            mode="omni",
        ).compiled_prompt
        self.assertNotIn("<Audio 1>", compiled)
        self.assertNotIn("audio reference", compiled.lower())
        self.assertEqual(compiled.count("<d>[English] Hello.</d>"), 1)

    def test_full_content_keeps_audio_conditioning_and_warning(self) -> None:
        response = self._post(prompt=self.DIALOGUE_PROMPT, policy="full_content")
        self.assertEqual(response.status_code, 200, response.text)
        request = self._request()
        self.assertEqual(
            [item["kind"] for item in request["references"]], ["image", "audio"]
        )
        self.assertTrue(request["standalone_audio_conditioning"])
        self.assertIn("<Audio 1>", request["effective_prompt"])
        event = request["prompt_processing"]["dialogue_events"][0]
        self.assertEqual(event["audio_reference_id_effective"], 1)
        self.assertFalse(event["audio_reference_suppressed"])
        self.assertIn(
            "FULL_CONTENT_AUDIO_MAY_OVERRIDE_DIALOGUE",
            request["prompt_processing"]["diagnostics"],
        )
        self.assertEqual(len(self._staged(request)["reference_audio_filenames"]), 1)

    def test_conflict_requires_an_explicit_valid_policy(self) -> None:
        before = set(self.manager.jobs_dir.iterdir())
        missing = self._post(prompt=self.DIALOGUE_PROMPT, policy=None)
        invalid = self._post(prompt=self.DIALOGUE_PROMPT, policy="voice_only")
        self.assertEqual(missing.status_code, 409)
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(self.manager.submitted, [])
        self.assertEqual(set(self.manager.jobs_dir.iterdir()), before)

    def test_suppression_cannot_hide_an_out_of_range_audio_reference(self) -> None:
        response = self._post(
            prompt=self.DIALOGUE_PROMPT.replace("<Audio 1>", "<Audio 2>"),
            policy="dialogue_priority",
        )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.manager.submitted, [])

    def test_formatter_cannot_hide_malformed_audio_like_tokens(self) -> None:
        for token in ("<Audio 01>", "<Audio 0>", "<Audio x>"):
            with self.subTest(token=token):
                before = set(self.manager.jobs_dir.iterdir())
                response = self._post(
                    prompt=self.DIALOGUE_PROMPT.replace("<Audio 1>", token),
                    policy="full_content",
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(self.manager.submitted, [])
                self.assertEqual(set(self.manager.jobs_dir.iterdir()), before)

    def test_no_dialogue_legacy_request_preserves_full_audio_behavior(self) -> None:
        response = self._post(
            prompt="Cut 1\n<Picture 1> stands beside a quiet shoreline.", policy=None
        )
        self.assertEqual(response.status_code, 200, response.text)
        request = self._request()
        self.assertEqual(request["standalone_audio_policy_effective"], "legacy_full_content")
        self.assertTrue(request["standalone_audio_conditioning"])
        self.assertEqual(len(self._staged(request)["reference_audio_filenames"]), 1)


if __name__ == "__main__":
    unittest.main()
