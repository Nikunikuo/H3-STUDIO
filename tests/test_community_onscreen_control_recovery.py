"""Regression tests for recoverable model on-screen-text control clauses."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from webui.community_prompt_planner import (  # noqa: E402
    PLAN_SCHEMA_VERSION,
    compile_model_result,
    prepare_planner_input,
)


QWEN_CONTRADICTION_SCENE = (
    "A static subject waits. The scene transitions to a crowded market where "
    "the protagonist runs. A subtitle appears, but subtitles are prohibited."
)


def _model_json(*, scene: str, action: str) -> str:
    return json.dumps(
        {
            "schema_version": PLAN_SCHEMA_VERSION,
            "style": "Cinematic animation with clear physical motion.",
            "scene": scene,
            "shots": [
                {
                    "number": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 5.0,
                    "framing": "A readable medium-wide composition.",
                    "camera": "The camera follows the physical action.",
                    "action": action,
                }
            ],
            "ambient": ["Quiet room tone."],
            "foley": ["Physical movement and material contact sounds."],
            "music": "N/A",
            "dialogue_delivery": [],
        },
        ensure_ascii=False,
    )


class CommunityOnscreenControlRecoveryTests(unittest.TestCase):
    def _prepared(self):
        return prepare_planner_input(
            "Cut 1\nThe physical scene continues without spoken dialogue.",
            duration_seconds=5.0,
        )

    def test_saved_qwen_contradiction_scene_preserves_physical_beats(self) -> None:
        compiled = compile_model_result(
            _model_json(
                scene=QWEN_CONTRADICTION_SCENE,
                action="The protagonist keeps running through the market.",
            ),
            self._prepared(),
        )

        scene = compiled.plan.scene
        self.assertNotEqual(
            scene,
            "The authored environment and requested subjects continue coherently.",
        )
        self.assertIn("A static subject waits", scene)
        self.assertIn("crowded market", scene)
        self.assertIn("the protagonist runs", scene)
        self.assertNotIn("subtitle", scene.casefold())
        self.assertNotIn("subtitles", scene.casefold())
        self.assertTrue(
            any(
                warning.code == "MODEL_ONSCREEN_TEXT_CONTROL_REMOVED"
                and "scene" in warning.message
                for warning in compiled.diagnostics()
            )
        )

    def test_mixed_action_removes_only_readable_caption_clause(self) -> None:
        compiled = compile_model_result(
            _model_json(
                scene="A workshop surrounds the protagonist.",
                action=(
                    "The protagonist swings the hammer; "
                    "a readable caption appears below."
                ),
            ),
            self._prepared(),
        )

        action = compiled.plan.shots[0].action
        self.assertIn("The protagonist swings the hammer", action)
        self.assertNotIn("caption", action.casefold())
        self.assertTrue(
            any(
                warning.code == "MODEL_ONSCREEN_TEXT_CONTROL_REMOVED"
                and "action" in warning.message
                for warning in compiled.diagnostics()
            )
        )
        self.assertNotIn(
            "MODEL_ONSCREEN_TEXT_REPAIRED",
            {warning.code for warning in compiled.diagnostics()},
        )

    def test_on_screen_text_only_fields_keep_the_existing_generic_default(self) -> None:
        compiled = compile_model_result(
            _model_json(
                scene="A subtitle appears.",
                action="A caption appears below.",
            ),
            self._prepared(),
        )

        self.assertEqual(
            compiled.plan.scene,
            "The authored environment and requested subjects continue coherently.",
        )
        self.assertEqual(
            compiled.plan.shots[0].action,
            "The visible action continues naturally with clear physical cause and effect.",
        )
        warnings = compiled.diagnostics()
        self.assertGreaterEqual(
            sum(warning.code == "MODEL_ONSCREEN_TEXT_REPAIRED" for warning in warnings),
            2,
        )
        self.assertNotIn(
            "MODEL_ONSCREEN_TEXT_CONTROL_REMOVED",
            {warning.code for warning in warnings},
        )

    def test_source_authorized_reference_title_survives_partial_cleanup(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n"
            "参考画像に存在するタイトル文字「ファイナリーちゃん」が暗闇から浮かび上がる。",
            reference_inventory=[{"kind": "image", "index": 1}],
            duration_seconds=5.0,
        )
        compiled = compile_model_result(
            _model_json(
                scene=(
                    "The exact visible title lettering from the supplied reference image "
                    "appears in the center; a subtitle appears below; the emblem glows."
                ),
                action=(
                    "The title 'ファイナリーちゃん' appears as engraved lettering; "
                    "a caption appears below; the emblem emerges from darkness."
                ),
            ),
            prepared,
        )

        self.assertIn(
            "the exact visible title lettering from the supplied reference image",
            compiled.plan.scene.casefold(),
        )
        self.assertIn(
            "the exact visible title lettering from the supplied reference image",
            compiled.plan.shots[0].action.casefold(),
        )
        self.assertIn("the emblem glows", compiled.plan.scene)
        self.assertIn("the emblem emerges from darkness", compiled.plan.shots[0].action)
        self.assertNotIn("ファイナリーちゃん", compiled.prompt)
        self.assertNotIn("subtitle", compiled.prompt.casefold())
        self.assertNotIn("caption", compiled.prompt.casefold())

    def test_other_control_gates_still_run_after_on_screen_partial_cleanup(self) -> None:
        cases = (
            (
                "unknown Japanese",
                "The protagonist swings the hammer; a subtitle appears; 日本語の制御文",
                "MODEL_CONTROL_TEXT_REPAIRED",
                False,
            ),
            (
                "quote",
                'The protagonist swings the hammer; a caption appears; "invented text"',
                "MODEL_CONTROL_TEXT_REPAIRED",
                False,
            ),
            (
                "reference tag",
                "The protagonist swings the hammer; a caption appears; <Picture 99>",
                "MODEL_CONTROL_TAGS_REMOVED",
                True,
            ),
            (
                "speech",
                "The protagonist swings the hammer; a caption appears; the pilot says hello.",
                "MODEL_SPEECH_CONTROL_REMOVED",
                True,
            ),
        )
        for name, action, expected_warning, keeps_motion in cases:
            with self.subTest(name=name):
                compiled = compile_model_result(
                    _model_json(
                        scene="A workshop surrounds the protagonist.",
                        action=action,
                    ),
                    self._prepared(),
                )
                effective_action = compiled.plan.shots[0].action
                warning_codes = {warning.code for warning in compiled.diagnostics()}
                self.assertIn(expected_warning, warning_codes)
                self.assertNotIn("日本語", compiled.prompt)
                self.assertNotIn('"invented text"', compiled.prompt)
                self.assertNotIn("<Picture 99>", compiled.prompt)
                self.assertNotIn("caption", compiled.prompt.casefold())
                self.assertNotIn("subtitle", compiled.prompt.casefold())
                self.assertNotIn("says", compiled.prompt.casefold())
                if keeps_motion:
                    self.assertIn("The protagonist swings the hammer", effective_action)
                else:
                    self.assertNotIn("The protagonist swings the hammer", effective_action)


if __name__ == "__main__":
    unittest.main()
