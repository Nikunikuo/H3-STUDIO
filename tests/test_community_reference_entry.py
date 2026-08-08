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
from webui.community_prompt_planner import (  # noqa: E402
    CommunityPromptPlannerError,
    preflight_source_references,
)


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


class CommunityReferenceEntryTests(unittest.TestCase):
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

    def _post(
        self,
        *,
        prompt: str,
        references: tuple[tuple[str, str], ...] = (("character.png", "image/png"),),
        prompt_processing_mode: str = "community",
        policy: str | None = None,
        dialogue: str = "",
    ):
        form = {
            "mode": "omni",
            "style": "natural",
            "prompt": prompt,
            "width": "640",
            "height": "384",
            "num_frames": "124",
            "steps": "7",
            "seed": "424242",
            "acceleration": "off",
            "ref_image_size": "match",
            "prompt_processing_mode": prompt_processing_mode,
            "audio_preset": "auto",
            "dialogue": dialogue,
            "soundscape": "",
            "music_policy": "auto",
            "audio_gain_db": "0",
        }
        if policy is not None:
            form["standalone_audio_policy"] = policy
        files = [
            ("references", (filename, b"reference-bytes", content_type))
            for filename, content_type in references
        ]
        return self.client.post(
            "/api/jobs",
            data=form,
            files=files,
            headers={
                self.server.LOCAL_MUTATION_HEADER: self.server.LOCAL_MUTATION_VALUE
            },
        )

    def _request(self) -> dict:
        self.assertEqual(len(self.manager.submitted), 1)
        job_id = self.manager.submitted[0]["id"]
        return json.loads(
            (self.manager.jobs_dir / job_id / "request.json").read_text(
                encoding="utf-8"
            )
        )

    def test_shared_preflight_normalizes_authoring_variants_and_keeps_duplicates(self) -> None:
        inventory = [{"kind": "image"}]
        for token in (
            "<Picture01>",
            "<picture 01>",
            "＜Ｐｉｃｔｕｒｅ０１＞",
            "< Picture 1 >",
        ):
            with self.subTest(token=token):
                result = preflight_source_references(
                    f"Cut 1\n{token} appears twice: {token}.",
                    reference_inventory=inventory,
                    mode="omni",
                )
                self.assertEqual(
                    result.canonical_tags,
                    ("<Picture 1>", "<Picture 1>"),
                )
                self.assertEqual(len(result.warnings), 2)

    def test_shared_preflight_rejects_invalid_or_missing_inventory_reference(self) -> None:
        for token, expected_code in (
            ("<Picture 0>", "SOURCE_REFERENCE_TAG_UNSUPPORTED"),
            ("<Picture X>", "SOURCE_REFERENCE_TAG_UNSUPPORTED"),
            ("<Picture 2>", "SOURCE_REFERENCE_NOT_IN_INVENTORY"),
        ):
            with self.subTest(token=token):
                with self.assertRaises(CommunityPromptPlannerError) as raised:
                    preflight_source_references(
                        f"Cut 1\n{token} appears.",
                        reference_inventory=[{"kind": "image"}],
                        mode="omni",
                    )
                self.assertEqual(raised.exception.code, expected_code)

    def test_community_entry_accepts_variant_and_persists_source_verbatim(self) -> None:
        source = "Cut 1\n＜Ｐｉｃｔｕｒｅ０１＞ faces the camera."
        response = self._post(prompt=source)

        self.assertEqual(response.status_code, 200, response.text)
        request = self._request()
        self.assertEqual(request["prompt"], source)
        self.assertEqual(request["effective_prompt"], source)
        self.assertEqual(request["prompt_processing_mode"], "community")

    def test_community_entry_rejects_invalid_and_missing_references_before_submit(self) -> None:
        for token, expected_status, expected_detail in (
            (
                "<Picture 0>",
                400,
                "対応していない参照タグの綴りまたは番号が含まれています。",
            ),
            (
                "<Picture X>",
                400,
                "対応していない参照タグの綴りまたは番号が含まれています。",
            ),
            (
                "<Picture 2>",
                409,
                "プロンプトの参照タグに対応する素材が添付されていません。",
            ),
        ):
            with self.subTest(token=token):
                response = self._post(prompt=f"Cut 1\n{token} appears.")
                self.assertEqual(response.status_code, expected_status, response.text)
                self.assertEqual(response.json()["detail"], expected_detail)
                self.assertEqual(self.manager.submitted, [])

    def test_community_entry_rejects_kind_inventory_mismatch(self) -> None:
        response = self._post(prompt="Cut 1\n<Audio 1> is audible.")

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(
            response.json()["detail"],
            "プロンプトの参照タグに対応する素材が添付されていません。",
        )
        self.assertEqual(self.manager.submitted, [])

    def test_community_entry_keeps_non_reference_control_gate(self) -> None:
        response = self._post(
            prompt="Cut 1\n<Picture01> faces the camera. <Unknown 1> is forbidden."
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.manager.submitted, [])

    def test_community_orphan_audio_check_uses_canonical_variant(self) -> None:
        response = self._post(
            prompt="Cut 1\n<Picture 1> uses <Audio01>.",
            references=(
                ("character.png", "image/png"),
                ("voice.wav", "audio/wav"),
            ),
            policy="dialogue_priority",
            dialogue="Hello.",
        )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.manager.submitted, [])

    def test_raw_en_keeps_strict_reference_boundary(self) -> None:
        response = self._post(
            prompt="Cut 1\n<Picture01> faces the camera.",
            prompt_processing_mode="raw_en",
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.manager.submitted, [])


if __name__ == "__main__":
    unittest.main()
