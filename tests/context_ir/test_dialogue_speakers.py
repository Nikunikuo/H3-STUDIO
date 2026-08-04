from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.context_ir import compile_request


class DialogueSpeakerBindingTests(unittest.TestCase):
    def test_explicit_picture_tag_is_not_rewritten_to_subject_one(self) -> None:
        result = compile_request(
            {
                "mode": "omni",
                "prompt": (
                    "<Picture 1> and <Picture 2> are separate character identity references.\n"
                    "Cut1\n二人が歩く。\nCut2\n二人が振り向く。"
                ),
                "dialogue": "Cut2 <Picture 2>が明るい声で「こんにちは。」",
                "dialogue_language": "Japanese",
                "num_frames": 124,
                "style": "natural",
                "music_policy": "none",
                "references": [{"kind": "image"}, {"kind": "image"}],
            }
        )

        self.assertFalse(result.fatal)
        assert result.ir_text is not None
        speaker = "The visible character identified by <Picture 2>"
        self.assertEqual(result.ir_text.count(speaker), 2)
        self.assertNotIn("<Subject 1> (S1) says", result.ir_text)
        assert result.document is not None
        event = result.document.shots[1].dialogue_events[0]
        self.assertEqual(event.speaker_id, 2)
        self.assertEqual(event.speaker_reference.text, "<Picture 2>")

    def test_plain_japanese_reference_image_number_binds_the_speaker(self) -> None:
        result = compile_request(
            {
                "mode": "omni",
                "prompt": "参照画像1は衣装だけ、参照画像2はキャラクター本人の参照。",
                "dialogue": "参照画像2が「了解です。」",
                "dialogue_language": "Japanese",
                "num_frames": 124,
                "style": "natural",
                "music_policy": "none",
                "references": [{"kind": "image"}, {"kind": "image"}],
            }
        )

        self.assertFalse(result.fatal)
        assert result.ir_text is not None
        self.assertIn(
            "The visible character identified by <Picture 2> says: "
            "<d>[Japanese] 了解です。</d>",
            result.ir_text,
        )
        self.assertNotIn("(S2)", result.ir_text)

    def test_dialogue_speakers_promote_neutral_picture_roles_to_characters(self) -> None:
        result = compile_request(
            {
                "mode": "omni",
                "prompt": "Two girls sit across a table and talk.",
                "dialogue": '<Picture 1> 「Hello.」 <Picture 2> 「Hi.」',
                "dialogue_language": "English",
                "num_frames": 124,
                "style": "natural",
                "music_policy": "none",
                "references": [{"kind": "image"}, {"kind": "image"}],
            }
        )

        self.assertFalse(result.fatal)
        assert result.document is not None
        character_sources = [
            item.source_label.text
            for item in result.document.subject_definitions
            if item.source_label is not None and item.subject_index is not None
        ]
        self.assertEqual(character_sources, ["<Picture 1>", "<Picture 2>"])
        events = result.document.shots[0].dialogue_events
        self.assertEqual(
            [item.speaker_reference.text for item in events],
            ["<Picture 1>", "<Picture 2>"],
        )
        role_messages = [
            item.message
            for item in result.auto_adjustments
            if item.code == "PICTURE_ROLE_RESOLVED"
        ]
        self.assertEqual(len(role_messages), 2)
        self.assertTrue(all("as character" in message for message in role_messages))
        self.assertTrue(
            any(
                item.code == "SOURCE_OUTFIT_OVERRIDE_POLICY"
                for item in result.auto_adjustments
            )
        )

    def test_speaker_and_audio_tags_do_not_leak_into_voice_direction(self) -> None:
        result = compile_request(
            {
                "mode": "omni",
                "prompt": "A character speaks.",
                "dialogue": "<Picture 2> <Audio 1> （明るい声で）「Hello.」",
                "dialogue_language": "English",
                "num_frames": 124,
                "style": "natural",
                "music_policy": "none",
                "references": [
                    {"kind": "image"},
                    {"kind": "image"},
                    {"kind": "audio"},
                ],
            }
        )

        self.assertFalse(result.fatal)
        assert result.document is not None
        event = result.document.shots[0].dialogue_events[0]
        self.assertEqual(event.speaker_reference.text, "<Picture 2>")
        self.assertEqual(event.audio_reference.text, "<Audio 1>")
        self.assertEqual(event.voice_direction, "（明るい声で）")
        assert result.ir_text is not None
        self.assertNotIn(
            "BEGIN H3 STUDIO VERBATIM TEXT ---\n<Picture 2>",
            result.ir_text,
        )

    def test_explicit_subject_switch_clears_a_stale_picture_reference(self) -> None:
        result = compile_request(
            {
                "mode": "omni",
                "prompt": "<Picture 1> and <Picture 2> are character identity references.",
                "dialogue": '<Picture 1> 「one」 Subject 2 「two」',
                "dialogue_language": "English",
                "num_frames": 124,
                "style": "natural",
                "music_policy": "none",
                "references": [{"kind": "image"}, {"kind": "image"}],
            }
        )
        self.assertFalse(result.fatal)
        assert result.document is not None
        events = result.document.shots[0].dialogue_events
        self.assertEqual(events[0].speaker_reference.text, "<Picture 1>")
        self.assertIsNone(events[1].speaker_reference)
        self.assertEqual(events[1].speaker_id, 2)
        self.assertEqual(events[1].voice_direction, "")
        assert result.ir_text is not None
        self.assertIn("<Subject 2> (S2) says: <d>[English] two</d>", result.ir_text)

    def test_video_dialogue_speaker_gets_an_identity_definition(self) -> None:
        result = compile_request(
            {
                "mode": "omni",
                "prompt": "Use <Video 1> for the scene.",
                "dialogue": '<Video 1> 「Hello.」',
                "dialogue_language": "English",
                "num_frames": 124,
                "style": "natural",
                "music_policy": "none",
                "references": [{"kind": "video"}],
            }
        )
        self.assertFalse(result.fatal)
        assert result.document is not None
        video_definition = next(
            item
            for item in result.document.subject_definitions
            if item.source_label is not None
            and item.source_label.text == "<Video 1>"
        )
        self.assertIn("visual identity and appearance", video_definition.text)
        self.assertFalse(
            any(
                item.subject_index == 1 and item.source_label is None
                for item in result.document.subject_definitions
            )
        )
        assert result.ir_text is not None
        self.assertIn(
            "The visible character identified by <Video 1> says: "
            "<d>[English] Hello.</d>",
            result.ir_text,
        )

    def test_audio_timbre_definition_targets_the_paired_picture_speaker(self) -> None:
        result = compile_request(
            {
                "mode": "omni",
                "prompt": "<Picture 1> and <Picture 2> are character identity references.",
                "dialogue": '<Picture 2> <Audio 1> 「Hello.」',
                "dialogue_language": "English",
                "num_frames": 124,
                "style": "natural",
                "music_policy": "none",
                "references": [
                    {"kind": "image"},
                    {"kind": "image"},
                    {"kind": "audio"},
                ],
            }
        )
        self.assertFalse(result.fatal)
        assert result.document is not None
        audio_definition = next(
            item.text
            for item in result.document.subject_definitions
            if item.source_label is not None
            and item.source_label.text == "<Audio 1>"
        )
        self.assertIn("identified by <Picture 2>", audio_definition)
        self.assertNotIn("<Subject 1> (S1)", audio_definition)


if __name__ == "__main__":
    unittest.main()
