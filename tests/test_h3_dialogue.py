from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.h3_dialogue import (  # noqa: E402
    DialogueOverrideError,
    format_inline_dialogue,
)


class InlineH3DialogueTests(unittest.TestCase):
    def test_visible_text_and_sound_effect_quotes_are_not_spoken(self) -> None:
        prompt = (
            "Cut1\n看板に「営業中」と書かれている。\n"
            "グラスが「カラン」と鳴る。\n効果音「ドン」。"
        )

        result = format_inline_dialogue(prompt)

        self.assertEqual(result.text, prompt)
        self.assertEqual(result.events, ())
        self.assertNotIn("<d>", result.text)

    def test_multiple_cut_dialogue_stays_in_each_original_cut(self) -> None:
        prompt = (
            "Cut1\n彼女は「おはよう。」と言う。\n"
            "Cut2\n彼女は小声で「またね。」と話す。\n"
            "Cut3\n手を振る。"
        )

        result = format_inline_dialogue(prompt)

        first = "<d>[Japanese] おはよう。</d>"
        second = "<d>[Japanese] またね。</d>"
        self.assertEqual(result.text.count("<d>"), 2)
        self.assertNotIn("visible speaking character", result.text.lower())
        self.assertNotIn("jaw stops", result.text.lower())
        self.assertLess(result.text.index("Cut1"), result.text.index(first))
        self.assertLess(result.text.index(first), result.text.index("Cut2"))
        self.assertLess(result.text.index("Cut2"), result.text.index(second))
        self.assertLess(result.text.index(second), result.text.index("Cut3"))
        self.assertEqual([event.target_cut for event in result.events], [1, 2])

    def test_existing_native_tag_is_idempotent(self) -> None:
        prompt = "Cut1\nThe woman (S2) says: <d>[English] Hello.</d>"

        first = format_inline_dialogue(prompt)
        second = format_inline_dialogue(first.text)

        self.assertEqual(first.text, prompt)
        self.assertEqual(second.text, prompt)
        self.assertEqual(second.events[0].speaker_id, 2)

    def test_override_replaces_only_the_requested_cut(self) -> None:
        prompt = (
            "Cut1\n彼女は「最初。」と言う。\n"
            "Cut2\n彼女は「古い台詞。」と言う。\n"
            "Cut3\n終わる。"
        )

        result = format_inline_dialogue(
            prompt,
            "Cut2\n（低い声）\n「新しい台詞。」",
        )

        self.assertIn("<d>[Japanese] 最初。</d>", result.text)
        self.assertIn("<d>[Japanese] 新しい台詞。</d>", result.text)
        self.assertNotIn("古い台詞", result.text)
        self.assertLess(result.text.index("Cut2"), result.text.index("新しい台詞"))
        self.assertLess(result.text.index("新しい台詞"), result.text.index("Cut3"))
        self.assertEqual(result.source, "mixed")

    def test_legacy_override_is_inserted_into_target_cut(self) -> None:
        prompt = "Cut1\n海を見る。\nCut4\nカメラを見る。"

        result = format_inline_dialogue(
            prompt,
            "Cut4\n（低くくぐもった女性の声）\n「はぁ～～あっつい・・・。」",
            normalize_decorative=True,
        )

        tag = "<d>[Japanese] はぁ、あっつい。</d>"
        self.assertEqual(result.text.count(tag), 1)
        self.assertLess(result.text.index("Cut4"), result.text.index(tag))
        self.assertNotIn("低くくぐもった女性の声", result.text)
        self.assertIn("low-pitched, muffled female voice", result.text)
        inside = result.text.split("<d>", 1)[1].split("</d>", 1)[0]
        self.assertNotIn("Cut4", inside)
        self.assertNotIn("声", inside)
        self.assertEqual(result.events[0].original_text, "はぁ～～あっつい・・・。")
        self.assertEqual(result.events[0].effective_text, "はぁ、あっつい。")
        self.assertTrue(result.events[0].normalized)

    def test_short_plain_override_remains_backward_compatible(self) -> None:
        result = format_inline_dialogue(
            "Cut1\nA woman looks toward the camera.",
            "こんにちは。",
        )

        self.assertEqual(result.events[0].original_text, "こんにちは。")
        self.assertIn("<d>[Japanese] こんにちは。</d>", result.text)

    def test_instruction_like_or_multiline_plain_override_is_rejected(self) -> None:
        unsafe = (
            "キャラクターがカメラを見て一度だけ話す",
            "低い声でこんにちは。",
            "Say hello once.",
            "Cut4\n低い女性の声\nこんにちは。",
        )

        for override in unsafe:
            with self.subTest(override=override):
                with self.assertRaises(DialogueOverrideError):
                    format_inline_dialogue(
                        "Cut1\nA woman looks toward the camera.",
                        override,
                    )

    def test_quoted_multiline_override_remains_explicit_and_valid(self) -> None:
        result = format_inline_dialogue(
            "Cut1\nA woman looks toward the camera.",
            "Cut1\n低い女性の声\n「こんにちは。」",
        )

        self.assertEqual(result.events[0].original_text, "こんにちは。")
        self.assertIn("low-pitched female voice", result.text)

    def test_server_canonicalization_marks_only_generated_harness_as_trusted(self) -> None:
        result = format_inline_dialogue(
            "No narration. The woman (S1) says once: "
            "<d>[English] Hello.</d> The narrator explains continuously.",
            canonicalize_native_context=True,
        )

        self.assertEqual(len(result.trusted_fragments), 1)
        trusted = result.trusted_fragments[0]
        self.assertIn("<d>[English] Hello.</d>", trusted)
        self.assertIn("closes their lips after the line", trusted)
        self.assertNotIn("narrator", trusted.lower())
        self.assertIn("The narrator explains continuously", result.text)

    def test_parenthetical_voice_direction_enables_quote_only_next_line(self) -> None:
        prompt = "Cut1\n（明るい女性の声で）\n「こんにちは。」"

        result = format_inline_dialogue(prompt)

        self.assertIn("<d>[Japanese] こんにちは。</d>", result.text)
        self.assertNotIn("（明るい女性の声で）", result.text)
        self.assertIn("bright female voice", result.text)

    def test_english_dialogue_gets_english_language_tag(self) -> None:
        result = format_inline_dialogue('Cut1\nShe says "Hello there." and smiles.')

        self.assertIn("<d>[English] Hello there.</d>", result.text)

    def test_base_mode_can_preserve_decorative_punctuation(self) -> None:
        prompt = "Cut1\n彼女は「はぁ～～あっつい・・・。」と言う。"

        result = format_inline_dialogue(prompt, normalize_decorative=False)

        self.assertIn("<d>[Japanese] はぁ～～あっつい・・・。</d>", result.text)
        self.assertFalse(result.events[0].normalized)

    def test_visual_actions_and_sound_after_dialogue_are_preserved(self) -> None:
        prompt = (
            "Cut1\n彼女は椅子から立ち、窓へ歩いて低い声で"
            "「行こう。」と言い、ドアを開ける。足音が響く。"
        )

        result = format_inline_dialogue(prompt)

        self.assertIn("椅子から立ち", result.text)
        self.assertIn("窓へ歩いて", result.text)
        self.assertIn("ドアを開ける", result.text)
        self.assertIn("足音が響く", result.text)
        self.assertEqual(result.text.count("<d>[Japanese] 行こう。</d>"), 1)
        self.assertNotIn("低い声で", result.text)
        self.assertIn("low-pitched female voice", result.text)

    def test_speech_frequency_before_quote_is_not_left_as_visual_action(self) -> None:
        result = format_inline_dialogue(
            "Cut1\n<Picture 1>の女性が夏の海辺で一度だけ「こんにちは。」と言う。"
        )

        self.assertEqual(len(result.events), 1)
        self.assertIn("says once", result.text)
        self.assertNotIn("一度だけ", result.text)
        self.assertIn("夏の海辺", result.text)

    def test_dialogue_and_quoted_sound_effect_are_classified_per_quote(self) -> None:
        result = format_inline_dialogue(
            "Cut1\n彼女が「暑い。」と言い、グラスが「カラン」と鳴る。"
        )

        self.assertEqual(result.text.count("<d>"), 1)
        self.assertIn("<d>[Japanese] 暑い。</d>", result.text)
        self.assertIn("「カラン」と鳴る", result.text)

    def test_movie_title_is_not_spoken_but_following_review_is(self) -> None:
        result = format_inline_dialogue(
            "Cut1\n彼女は映画「夏」を見て「面白い。」と言う。"
        )

        self.assertIn("映画「夏」を見て", result.text)
        self.assertNotIn("<d>[Japanese] 夏</d>", result.text)
        self.assertIn("<d>[Japanese] 面白い。</d>", result.text)

    def test_display_objects_that_say_quoted_text_never_create_a_speaker(self) -> None:
        prompts = (
            'The sign says "OPEN".',
            'The title says "Summer".',
            'The subtitle says "Hello".',
            'The caption says "Day 1".',
            'The label says "100%".',
            'On-screen text says "OPEN".',
            "看板は「営業中」と言う文字を表示する。",
            "タイトルは「夏」と言う。",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                result = format_inline_dialogue(prompt)
                self.assertEqual(result.events, ())
                self.assertNotIn("<d>", result.text)
                self.assertNotIn("(S1)", result.text)

        spoken = format_inline_dialogue('The woman says "OPEN".')
        self.assertEqual(len(spoken.events), 1)
        self.assertIn("<d>[English] OPEN</d>", spoken.text)

    def test_utterance_script_wins_over_japanese_control_language(self) -> None:
        result = format_inline_dialogue(
            'Cut1\n彼女は英語で「Hello there.」と言う。'
        )

        self.assertIn("<d>[English] Hello there.</d>", result.text)

    def test_kana_corrects_an_explicit_english_language_request(self) -> None:
        result = format_inline_dialogue("Cut1\n彼女は英語で「こんにちは」と言う。")

        self.assertIn("<d>[Japanese] こんにちは</d>", result.text)
        self.assertEqual(result.events[0].original_text, "こんにちは")
        self.assertEqual(result.events[0].effective_text, "こんにちは")
        self.assertIn("DIALOGUE_LANGUAGE_SCRIPT_MISMATCH", result.diagnostics)
        self.assertIn("DIALOGUE_LANGUAGE_TAG_CORRECTED", result.adjustments)

    def test_kana_corrects_an_explicit_chinese_language_request(self) -> None:
        result = format_inline_dialogue("Cut1\n彼女は中国語で「こんにちは」と言う。")

        self.assertIn("<d>[Japanese] こんにちは</d>", result.text)
        self.assertEqual(result.events[0].effective_text, "こんにちは")
        self.assertIn("DIALOGUE_LANGUAGE_TAG_CORRECTED", result.adjustments)

    def test_latin_payload_does_not_override_explicit_japanese(self) -> None:
        result = format_inline_dialogue('Cut1\n彼女は日本語で「Hello」と言う。')

        self.assertIn("<d>[Japanese] Hello</d>", result.text)
        self.assertNotIn("DIALOGUE_LANGUAGE_SCRIPT_MISMATCH", result.diagnostics)
        self.assertNotIn("DIALOGUE_LANGUAGE_TAG_CORRECTED", result.adjustments)

    def test_hangul_arabic_and_cyrillic_correct_mismatched_requests(self) -> None:
        cases = (
            ("안녕하세요", "Korean"),
            ("مرحبا", "Arabic"),
            ("Здравствуйте", "Russian"),
        )

        for payload, expected_language in cases:
            with self.subTest(payload=payload):
                result = format_inline_dialogue(
                    f"Cut1\n彼女は英語で「{payload}」と言う。"
                )
                self.assertIn(
                    f"<d>[{expected_language}] {payload}</d>", result.text
                )
                self.assertEqual(result.events[0].effective_text, payload)
                self.assertIn(
                    "DIALOGUE_LANGUAGE_SCRIPT_MISMATCH", result.diagnostics
                )
                self.assertIn(
                    "DIALOGUE_LANGUAGE_TAG_CORRECTED", result.adjustments
                )

    def test_native_dialogue_tag_mismatch_is_corrected_idempotently(self) -> None:
        prompt = "Cut1\nThe woman says: <d>[English] こんにちは。</d>"

        first = format_inline_dialogue(prompt)
        second = format_inline_dialogue(first.text)

        self.assertIn("<d>[Japanese] こんにちは。</d>", first.text)
        self.assertNotIn("<d>[English]", first.text)
        self.assertEqual(first.events[0].original_text, "こんにちは。")
        self.assertEqual(first.events[0].effective_text, "こんにちは。")
        self.assertIn("DIALOGUE_LANGUAGE_SCRIPT_MISMATCH", first.diagnostics)
        self.assertIn("DIALOGUE_LANGUAGE_TAG_CORRECTED", first.adjustments)
        self.assertEqual(second.text, first.text)
        self.assertNotIn("DIALOGUE_LANGUAGE_SCRIPT_MISMATCH", second.diagnostics)

    def test_ambiguous_scripts_do_not_override_native_language_tags(self) -> None:
        cases = (
            ("<d>[Chinese] 了解</d>", "<d>[Chinese] 了解</d>"),
            ("<d>[Japanese] Hello</d>", "<d>[Japanese] Hello</d>"),
            ("<d>[English] こんにちは안녕</d>", "<d>[English] こんにちは안녕</d>"),
        )

        for native, expected in cases:
            with self.subTest(native=native):
                result = format_inline_dialogue(f"Cut1\nThe woman says: {native}")
                self.assertIn(expected, result.text)
                self.assertNotIn(
                    "DIALOGUE_LANGUAGE_TAG_CORRECTED", result.adjustments
                )

    def test_kanji_only_utterance_defaults_to_japanese(self) -> None:
        result = format_inline_dialogue("Cut1\n（低い声で）\n「了解」")

        self.assertIn("<d>[Japanese] 了解</d>", result.text)

    def test_two_picture_speakers_receive_distinct_stable_ids(self) -> None:
        result = format_inline_dialogue(
            "Cut1\n<Picture 1>の女性が「こんにちは。」と言い、"
            "<Picture 2>の男性が「やあ。」と答える。"
        )

        self.assertIn("<Picture 1> (S1)", result.text)
        self.assertIn("<Picture 2> (S2)", result.text)
        self.assertEqual([event.speaker_id for event in result.events], [1, 2])

    def test_speech_only_subject_fragment_is_not_left_behind(self) -> None:
        result = format_inline_dialogue(
            "Cut1\n<Picture 1>の女性は低く落ち着いた女性の声で"
            "「こんにちは。」と一度だけ言う。"
        )

        self.assertNotIn("<Picture 1>の女性は。", result.text)
        self.assertIn("<Picture 1> (S1)", result.text)

    def test_audio_reference_is_bound_to_timbre_without_leaving_a_fragment(self) -> None:
        result = format_inline_dialogue(
            "Cut1\n<Picture 1>の女性は「こんにちは。」（参照： <Audio 1>） "
            "と一度だけ言う。"
        )

        self.assertNotIn("女性は（参照", result.text)
        self.assertNotIn("<Picture 1>の女性は\n", result.text)
        self.assertNotIn("参照：", result.text)
        self.assertEqual(result.text.count("<d>[Japanese] こんにちは。</d>"), 1)
        self.assertIn("<Subject 1> (S1) is the visible character shown in <Picture 1>", result.text)
        self.assertIn(
            "<Audio 1> provides the voice timbre and measured delivery for <Subject 1> (S1)",
            result.text,
        )
        self.assertIn("<Subject 1> (S1), using <Audio 1>, says once", result.text)
        for forbidden in (
            "spoken content is limited",
            "non-speech",
            "without copying",
            "its words",
            "semantic content",
            "jaw ceases",
        ):
            self.assertNotIn(forbidden, result.text.lower())
        self.assertEqual(result.text.count("<d>"), 1)
        self.assertEqual(result.events[0].audio_reference_id, 1)
        self.assertEqual(result.events[0].to_dict()["audio_reference_label"], "<Audio 1>")

    def test_dialogue_priority_omits_audio_contract_but_keeps_requested_binding_for_audit(self) -> None:
        result = format_inline_dialogue(
            "Cut1\n<Picture 1>の女性は「こんにちは。」（参照： <Audio 1>）"
            "と一度だけ言う。",
            include_audio_references=False,
        )

        self.assertNotIn("<Audio 1>", result.text)
        self.assertNotIn("参照：", result.text)
        self.assertEqual(result.text.count("<d>[Japanese] こんにちは。</d>"), 1)
        self.assertEqual(result.events[0].audio_reference_id, 1)

    def test_audio_reference_can_be_suppressed_without_losing_the_spoken_line(self) -> None:
        result = format_inline_dialogue(
            'Cut 1\nThe woman shown in <Picture 1> says "Hello." using <Audio 1>.\n'
            "She closes the door while ocean surf continues.",
            include_audio_references=False,
        )

        self.assertEqual(result.text.count("<d>[English] Hello.</d>"), 1)
        self.assertNotIn("<Audio 1>", result.text)
        self.assertIn("closes the door", result.text)
        self.assertIn("ocean surf continues", result.text)
        # Keep the requested binding for audit while removing it from the H3
        # conditioning text.
        self.assertEqual(result.events[0].audio_reference_id, 1)

    def test_picture_and_audio_ordinals_are_not_conflated(self) -> None:
        result = format_inline_dialogue(
            "Cut1\n<Picture 2>の女性は（声質参照：<Audio 1>）「こんにちは。」と言う。"
        )

        self.assertIn("<Subject 2> (S2) is the visible character shown in <Picture 2>", result.text)
        self.assertIn(
            "<Audio 1> provides the voice timbre and measured delivery for <Subject 2> (S2)",
            result.text,
        )
        self.assertEqual(result.events[0].speaker_id, 2)
        self.assertEqual(result.events[0].audio_reference_id, 1)

    def test_ambiguous_audio_reference_is_reported_instead_of_guessed(self) -> None:
        result = format_inline_dialogue(
            "Cut1\n<Picture 1>の女性は<Audio 1>と<Audio 2>を参照して"
            "「こんにちは。」と言う。"
        )

        self.assertIsNone(result.events[0].audio_reference_id)
        self.assertIn("AMBIGUOUS_DIALOGUE_AUDIO_REFERENCE", result.diagnostics)
        self.assertNotIn("without copying the original signal", result.text)

    def test_inline_cut_label_is_retained_and_targets_dialogue(self) -> None:
        result = format_inline_dialogue(
            "Cut4 彼女は「暑い。」と言う。BGMなし。"
        )

        self.assertTrue(result.text.startswith("Cut4\n"))
        self.assertIn("BGMなし", result.text)
        self.assertEqual(result.events[0].target_cut, 4)

    def test_multiline_native_dialogue_tag_is_collapsed_and_recognized(self) -> None:
        result = format_inline_dialogue(
            "Cut1\nThe woman (S2) says:\n<d>[Japanese]\nこんにちは。\n</d>"
        )

        self.assertIn("<d>[Japanese] こんにちは。</d>", result.text)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].speaker_id, 2)
        self.assertNotIn("MALFORMED_DIALOGUE_TAG", result.diagnostics)

    def test_native_dialogue_is_replaced_instead_of_duplicated_by_override(self) -> None:
        result = format_inline_dialogue(
            "Cut2\nThe woman (S2) says: <d>[Japanese] 古い台詞。</d>",
            "Cut2\n（低い女性の声で）\n「新しい台詞。」",
        )

        self.assertNotIn("古い台詞", result.text)
        self.assertEqual(result.text.count("<d>"), 1)
        self.assertIn("<d>[Japanese] 新しい台詞。</d>", result.text)
        self.assertIn("low-pitched female voice", result.text)
        self.assertEqual(result.events[0].speaker_id, 2)

    def test_common_japanese_inflections_are_recognized(self) -> None:
        prompts = (
            "彼女は「帰ろう。」と言って立ち上がる。",
            "彼女は「帰ろう。」と言った。",
            "彼女は「帰ろう。」と話した。",
            "彼女は「帰ろう。」と答えた。",
            "彼女は「帰ろう。」と囁いた。",
        )

        for body in prompts:
            with self.subTest(body=body):
                result = format_inline_dialogue(f"Cut1\n{body}")
                self.assertEqual(result.text.count("<d>[Japanese] 帰ろう。</d>"), 1)

    def test_english_visual_actions_survive_japanese_dialogue(self) -> None:
        result = format_inline_dialogue(
            "Cut1\nThe woman stands up and waves, then says “こんにちは。” "
            "and walks to the door."
        )

        self.assertIn("stands up", result.text)
        self.assertIn("waves", result.text)
        self.assertIn("walks to the door", result.text)
        self.assertIn("<d>[Japanese] こんにちは。</d>", result.text)


if __name__ == "__main__":
    unittest.main()
