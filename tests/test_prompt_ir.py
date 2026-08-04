from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.prompt_ir import (
    OMNI_SECTION_HEADERS,
    PromptIRRequest,
    compile_prompt_ir,
    compile_prompt_ir_result,
    is_h3_context_ir,
)


class PromptIRCompatibilityTests(unittest.TestCase):
    def test_request_canonicalizes_reference_labels(self) -> None:
        request = PromptIRRequest(
            mode="omni",
            prompt="Use the reference.",
            duration_seconds=5,
            reference_labels=("picture 2", "<AUDIO 1>"),
        )
        self.assertEqual(request.reference_labels, ("<Picture 2>", "<Audio 1>"))

    def test_legacy_string_api_uses_official_shot_format(self) -> None:
        text = compile_prompt_ir(
            PromptIRRequest(
                mode="t2v",
                prompt="Cut1\nfirst\nCut2\nsecond",
                duration_seconds=8,
                music_policy="none",
            )
        )
        self.assertIn("[Shot 1]\n", text)
        self.assertNotIn("[Shot 1] At", text)
        self.assertIn("[Shot 2] At 00:04.000", text)

    def test_legacy_result_api_exposes_diagnostics(self) -> None:
        result = compile_prompt_ir_result(
            PromptIRRequest(
                mode="omni",
                prompt="<Picture 1> turns.",
                duration_seconds=5,
                reference_labels=("<Picture 1>",),
            )
        )
        self.assertFalse(result.fatal)
        self.assertTrue(result.to_public_dict()["degraded"])

    def test_dialogue_and_japanese_are_preserved(self) -> None:
        text = compile_prompt_ir(
            PromptIRRequest(
                mode="t2v",
                prompt="日本のセルアニメ。",
                duration_seconds=5,
                dialogue="「こんにちは。」",
            )
        )
        self.assertIn("日本のセルアニメ。", text)
        self.assertEqual(text.count("<d>[Japanese] こんにちは。</d>"), 1)

    def test_raw_ir_detection_remains_available(self) -> None:
        raw = "\n\n".join(f"{header}:\nvalue" for header in OMNI_SECTION_HEADERS)
        self.assertTrue(is_h3_context_ir(raw))


if __name__ == "__main__":
    unittest.main()
