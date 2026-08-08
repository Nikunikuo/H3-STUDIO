"""Regression tests for recoverable model speech-control clauses."""

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
            "foley": ["Mechanical contact sounds."],
            "music": "N/A",
            "dialogue_delivery": [],
        },
        ensure_ascii=False,
    )


class CommunitySpeechControlRecoveryTests(unittest.TestCase):
    def test_fixture_preserves_both_physical_scene_beats_and_exact_dialogue(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n整備ロボットが格納庫の扉を押し、操縦士が「開けて」と明確に言う。",
            duration_seconds=5.0,
        )
        compiled = compile_model_result(
            _model_json(
                scene=(
                    "A maintenance robot pushes open a storage cabinet door; "
                    "the pilot speaks clearly in response. After a pause, "
                    "a hydraulic arm extends and slides the door horizontally."
                ),
                action=(
                    "The maintenance robot pushes open the storage cabinet door; "
                    "the pilot speaks clearly"
                ),
            ),
            prepared,
        )

        scene = compiled.plan.scene
        action = compiled.plan.shots[0].action
        self.assertIn("robot pushes open a storage cabinet door", scene)
        self.assertIn("hydraulic arm extends and slides the door horizontally", scene)
        self.assertIn("robot pushes open the storage cabinet door", action)
        self.assertNotRegex(scene.casefold(), r"\b(?:speaks?|says?)\b")
        self.assertNotRegex(action.casefold(), r"\b(?:speaks?|says?)\b")
        self.assertEqual(compiled.prompt.count('"開けて"'), 1)

        speech_warnings = [
            warning
            for warning in compiled.diagnostics()
            if warning.code == "MODEL_SPEECH_CONTROL_REMOVED"
        ]
        self.assertTrue(any("scene" in warning.message for warning in speech_warnings))
        self.assertTrue(any("action" in warning.message for warning in speech_warnings))

    def test_mixed_no_source_dialogue_keeps_opening_and_removes_speech_clause(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n人物が扉の前に立つ。",
            duration_seconds=5.0,
        )
        compiled = compile_model_result(
            _model_json(
                scene="A person waits beside a closed door.",
                action="The person opens the door; then says hello.",
            ),
            prepared,
        )

        action = compiled.plan.shots[0].action
        self.assertIn("opens the door", action)
        self.assertNotIn("says", action.casefold())
        self.assertNotIn("hello", action.casefold())
        self.assertTrue(
            any(
                warning.code == "MODEL_SPEECH_CONTROL_REMOVED"
                and "action" in warning.message
                for warning in compiled.diagnostics()
            )
        )

    def test_speech_only_still_uses_the_existing_generic_action_default(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n人物が静かに歩く。",
            duration_seconds=5.0,
        )
        compiled = compile_model_result(
            _model_json(
                scene="A quiet corridor surrounds the subject.",
                action="The pilot speaks clearly and narrates the scene.",
            ),
            prepared,
        )

        self.assertEqual(
            compiled.plan.shots[0].action,
            "The visible action continues naturally with clear physical cause and effect.",
        )
        self.assertNotIn("speaks", compiled.plan.shots[0].action.casefold())
        self.assertTrue(
            any(
                warning.code == "MODEL_SPEECH_CONTROL_REMOVED"
                and "action" in warning.message
                for warning in compiled.diagnostics()
            )
        )

    def test_non_english_quotes_subtitles_and_model_tags_cannot_bypass_control_gates(self) -> None:
        cases = (
            (
                "japanese",
                "The person opens the door; 日本語の制御文",
                "MODEL_CONTROL_TEXT_REPAIRED",
                False,
            ),
            (
                "quote",
                'The person opens the door; "hello"',
                "MODEL_CONTROL_TEXT_REPAIRED",
                False,
            ),
            (
                "subtitle",
                "The person opens the door; a readable subtitle appears",
                "MODEL_ONSCREEN_TEXT_CONTROL_REMOVED",
                True,
            ),
            (
                "reference tag",
                "The person opens the door; <Picture 99>",
                "MODEL_CONTROL_TAGS_REMOVED",
                True,
            ),
        )
        for name, action, expected_warning, keeps_motion in cases:
            with self.subTest(name=name):
                prepared = prepare_planner_input(
                    "Cut 1\n人物が扉の前に立つ。",
                    duration_seconds=5.0,
                )
                compiled = compile_model_result(
                    _model_json(
                        scene="A person waits beside a closed door.",
                        action=action,
                    ),
                    prepared,
                )
                effective_action = compiled.plan.shots[0].action
                warning_codes = {warning.code for warning in compiled.diagnostics()}
                self.assertIn(expected_warning, warning_codes)
                self.assertNotIn("<Picture 99>", compiled.prompt)
                self.assertNotIn("日本語", compiled.prompt)
                self.assertNotIn('"hello"', compiled.prompt)
                self.assertNotIn("subtitle", compiled.prompt.casefold())
                if keeps_motion:
                    self.assertIn("opens the door", effective_action)
                else:
                    self.assertNotIn("opens the door", effective_action)


if __name__ == "__main__":
    unittest.main()
