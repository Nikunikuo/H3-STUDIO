from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.server import _compose_effective_prompt  # noqa: E402
from webui.prompt_guard import sanitize_generation_text  # noqa: E402


class RawPromptGuardTests(unittest.TestCase):
    def test_empty_dialogue_removes_speech_cues_without_context_ir(self) -> None:
        prompt = (
            "日本の高品質なセルアニメ調。\n"
            "セリフ以外のカットではしゃべらないこと。\n"
            "Cut1\n海辺で青いグラスを持つ。\n"
            "Cut2\nストローでズズズーっとジュースを飲む。\n"
            "Cut3\nこちらを見てキャラクターのセリフ。\n"
            "最後にカメラ目線。"
        )

        effective, processing = _compose_effective_prompt(
            prompt,
            "anime",
            "auto",
            "",
            "",
            "none",
        )

        self.assertIn("日本の高品質なセルアニメ調", effective)
        self.assertIn("海辺で青いグラスを持つ", effective)
        self.assertIn("ストローで短く自然な吸引音を伴ってジュースを飲む", effective)
        self.assertIn("カメラを見る", effective)
        self.assertIn("最後にカメラ目線", effective)
        self.assertNotIn("セリフ以外", effective)
        self.assertNotIn("しゃべらない", effective)
        self.assertNotIn("キャラクターのセリフ", effective)
        self.assertNotIn("ズズズー", effective)
        self.assertNotIn("<d>", effective)
        self.assertNotIn("Audio:", effective)
        self.assertIn("Music: N/A", effective)
        lowered = effective.lower()
        for forbidden in (
            "subject_definitions:",
            "retention_analysis:",
            "detailed_description:",
            "overall_soundscape:",
            "non_diegetic_music:",
            "narration",
            "narrator",
            "voice-over",
            "dialogue",
            "human voice",
            "vocalization",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, lowered)
        self.assertEqual(processing["mode"], "raw_guarded")
        self.assertFalse(processing["context_ir"])
        self.assertEqual(processing["dialogue_policy"], "none")
        self.assertEqual(processing["rewritten_gaze_cues"], 1)
        self.assertEqual(len(processing["removed_speech_cues"]), 2)

    def test_explicit_dialogue_is_added_once_and_placeholder_is_removed(self) -> None:
        prompt = (
            "Cut1\n海辺で立つ。\n"
            "Cut2\nこちらを見てキャラクターのセリフ。"
        )
        dialogue = "Cut2で明るい声で「暑いね。」"

        effective, processing = _compose_effective_prompt(
            prompt,
            "natural",
            "dialogue",
            dialogue,
            "穏やかな波音。",
            "none",
        )

        self.assertNotIn(dialogue, effective)
        self.assertEqual(effective.count("暑いね。"), 1)
        self.assertEqual(effective.count("<d>[Japanese] 暑いね。</d>"), 1)
        tag = effective.split("<d>", 1)[1].split("</d>", 1)[0]
        self.assertNotIn("Cut2", tag)
        self.assertNotIn("明るい声", tag)
        self.assertLess(effective.index("Cut2"), effective.index("<d>"))
        self.assertNotIn("明るい声で", effective)
        self.assertIn("bright natural voice", effective)
        self.assertNotIn("キャラクターのセリフ", effective)
        self.assertNotIn("No narrator", effective)
        self.assertNotIn("voice-over", effective.lower())
        self.assertEqual(processing["dialogue_policy"], "inline_h3_native")
        self.assertEqual(processing["dialogue_source"], "override")
        self.assertEqual(processing["dialogue_count"], 1)
        self.assertEqual(processing["audio_preset_effective"], "dialogue")

    def test_positive_soundscape_survives_while_negative_voice_clause_is_removed(self) -> None:
        effective, processing = _compose_effective_prompt(
            "砂浜で髪が風に揺れる。",
            "natural",
            "ambience",
            "",
            "穏やかな波音と海風。ナレーションなし。",
            "auto",
        )

        self.assertIn("穏やかな波音と海風", effective)
        self.assertNotIn("ocean surf", effective)
        self.assertNotIn("A light natural breeze", effective)
        self.assertNotIn("ナレーション", effective)
        self.assertNotIn("narration", effective.lower())
        self.assertEqual(len(processing["removed_speech_cues"]), 1)

    def test_no_speech_connective_keeps_actor_wardrobe_camera_action_and_ambience(self) -> None:
        effective, processing = _compose_effective_prompt(
            "彼女は何もしゃべらずに参照画像どおりの白い夏服でカメラへ手を振る。",
            "natural",
            "ambience",
            "",
            "人の声は一切入れず、波と風の音だけにする。",
            "none",
        )

        self.assertIn("彼女は参照画像どおりの白い夏服でカメラへ手を振る", effective)
        self.assertIn("波と風の音だけにする", effective)
        self.assertNotIn("しゃべらず", effective)
        self.assertNotIn("人の声", effective)
        self.assertNotIn("に参照画像", effective)
        self.assertEqual(len(processing["removed_speech_cues"]), 2)

    def test_no_speech_guard_keeps_actions_without_dangling_connectives(self) -> None:
        cases = {
            "彼女はしゃべることなく歩く。": "彼女は歩く。",
            "彼女は手を振りながら話さない。": "彼女は手を振る。",
            "彼女は話すのをやめて、手を振る。": "彼女は手を振る。",
            "彼女は一言も話さず、笑顔で座る。": "彼女は笑顔で座る。",
            "彼女は手を振るが、話さない。": "彼女は手を振る。",
            "Audio: 波と風だけで、人の声はない。": "Audio: 波と風だけ。",
            "She does not speak and waves.": "She waves.",
            "Without narration, she waves.": "she waves.",
        }

        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(sanitize_generation_text(source).text, expected)

    def test_visible_quoted_no_speech_text_is_not_rewritten(self) -> None:
        for source in (
            "看板に「話さない」と書かれている。",
            "看板は「営業中」と言う文字を表示する。",
            "タイトルは「夏」と言う。",
            'The sign says "OPEN".',
            'The title says "Summer".',
            'On-screen text says "OPEN".',
        ):
            with self.subTest(source=source):
                guarded = sanitize_generation_text(source)
                self.assertEqual(guarded.text, source)
                self.assertEqual(guarded.removed_fragments, ())

    def test_lexical_speech_words_in_visual_or_figurative_context_are_kept(self) -> None:
        for source in (
            "A dialogue box appears on screen.",
            "古くからの言い伝えを描いた壁画がある。",
            "風が歌うように木々を揺らす。",
        ):
            with self.subTest(source=source):
                guarded = sanitize_generation_text(source)
                self.assertEqual(guarded.text, source)
                self.assertEqual(guarded.removed_fragments, ())

    def test_main_prompt_dialogue_stays_inside_cut_and_uses_native_h3_tags(self) -> None:
        prompt = (
            "Cut1\n海辺で寝そべる。波の音。\n"
            "Cut4\nキャラクターはカメラを見て、低くくぐもった女性の声で、"
            "疲れたように「はぁ～～あっつい・・・。」と一度だけ言う。\n"
            "穏やかな波の音が続く。グラスがカランと鳴る。"
        )

        effective, processing = _compose_effective_prompt(
            prompt,
            "anime",
            "ambience",
            "",
            "",
            "none",
            "omni",
        )

        expected = "<d>[Japanese] はぁ、あっつい。</d>"
        self.assertEqual(effective.count(expected), 1)
        self.assertLess(effective.index("Cut4"), effective.index(expected))
        self.assertNotIn("低くくぐもった女性の声", effective)
        self.assertIn("low-pitched, muffled female voice", effective)
        self.assertIn("穏やかな波の音", effective)
        self.assertIn("グラスがカランと鳴る", effective)
        self.assertNotIn("Audio: はぁ", effective)
        self.assertNotIn("visible speaking character", effective.lower())
        self.assertIn("speaker (s1) closes their lips after the line", effective.lower())
        for forbidden in (
            "spoken content is limited",
            "non-speech",
            "without copying",
            "semantic content",
            "its words",
            "jaw ceases",
        ):
            self.assertNotIn(forbidden, effective.lower())
        self.assertEqual(processing["dialogue_source"], "prompt")
        self.assertEqual(processing["dialogue_policy"], "inline_h3_native")
        self.assertEqual(processing["audio_preset_effective"], "dialogue+ambience")
        self.assertEqual(processing["dialogue_events"][0]["original_text"], "はぁ～～あっつい・・・。")
        self.assertEqual(processing["dialogue_events"][0]["effective_text"], "はぁ、あっつい。")

    def test_auto_audio_setting_keeps_main_prompt_as_the_only_audio_instruction(self) -> None:
        effective, processing = _compose_effective_prompt(
            'Cut1\n彼女は明るい声で「こんにちは。」と一度だけ言う。\n弱い風の音。',
            "natural",
            "auto",
            "",
            "",
            "auto",
            "omni",
        )

        self.assertIn("<d>[Japanese] こんにちは。</d>", effective)
        self.assertIn("弱い風の音", effective)
        self.assertNotIn("\n\nAudio:", effective)
        self.assertEqual(processing["audio_preset_effective"], "dialogue")

    def test_audio_timbre_contract_survives_the_final_prompt_guard(self) -> None:
        effective, processing = _compose_effective_prompt(
            "Cut1\n<Picture 1>の女性は「こんにちは。」（参照： <Audio 1>） "
            "と一度だけ言う。\n穏やかな波音が続く。",
            "anime",
            "auto",
            "",
            "",
            "none",
            "omni",
        )

        self.assertNotIn("女性は（参照", effective)
        self.assertNotIn("<Picture 1>の女性は\n", effective)
        self.assertEqual(effective.count("<d>[Japanese] こんにちは。</d>"), 1)
        self.assertEqual(
            effective.count(
                "<Audio 1> provides the voice timbre and measured delivery for "
                "<Subject 1> (S1)"
            ),
            1,
        )
        self.assertIn("<Subject 1> (S1), using <Audio 1>, says once", effective)
        for forbidden in (
            "spoken content is limited",
            "non-speech",
            "without copying",
            "semantic content",
            "its words",
            "jaw ceases",
        ):
            self.assertNotIn(forbidden, effective.lower())
        self.assertIn("穏やかな波音が続く", effective)
        self.assertEqual(processing["removed_speech_cues"], [])
        self.assertEqual(processing["dialogue_events"][0]["audio_reference_id"], 1)

    def test_native_dialogue_preserves_minimal_positive_context(self) -> None:
        prompt = (
            "summary:\nThe target video shows a woman at a seaside cafe.\n\n"
            "detailed_description:\n[Shot 1] The woman (S1) says once: "
            "<d>[Japanese] こんにちは。</d> She closes her lips after the line.\n\n"
            "overall_soundscape:\nGentle surf and a light breeze."
        )

        effective, processing = _compose_effective_prompt(
            prompt,
            "natural",
            "auto",
            "",
            "",
            "auto",
            "omni",
        )

        self.assertIn("woman at a seaside cafe", effective)
        self.assertIn("Speaker (S1) closes their lips after the line", effective)
        self.assertIn("Gentle surf and a light breeze", effective)
        for forbidden in ("dialogue", "narration", "voice-over", "human voice"):
            self.assertNotIn(forbidden, effective.lower())
        self.assertEqual(processing["removed_speech_cues"], [])

    def test_native_dialogue_never_bypasses_guard_for_surrounding_prose(self) -> None:
        prompt = (
            "No narration. The woman (S1) says once: "
            "<d>[English] Hello.</d> "
            "The narrator explains the workout continuously."
        )

        effective, processing = _compose_effective_prompt(
            prompt,
            "natural",
            "auto",
            "",
            "",
            "none",
            "omni",
        )

        self.assertEqual(effective.count("<d>[English] Hello.</d>"), 1)
        self.assertIn("says once", effective)
        self.assertIn("closes their lips after the line", effective)
        self.assertNotIn("narration", effective.lower())
        self.assertNotIn("narrator", effective.lower())
        self.assertNotIn("explains the workout", effective.lower())
        self.assertFalse(effective.startswith("."))
        self.assertGreaterEqual(len(processing["removed_speech_cues"]), 2)

        guarded = sanitize_generation_text(
            "No narration. <d>[English] Hello.</d> "
            "The narrator explains continuously.",
            preserve_speech_context=True,
        )
        self.assertEqual(guarded.text, "<d>[English] Hello.</d>")
        self.assertEqual(len(guarded.removed_fragments), 2)

    def test_dialogue_preset_without_dialogue_downgrades_to_ambience(self) -> None:
        effective, processing = _compose_effective_prompt(
            "静かな森で木漏れ日が揺れる。",
            "natural",
            "dialogue",
            "",
            "木々を抜ける風。",
            "auto",
        )

        self.assertNotIn("dialogue", effective.lower())
        self.assertNotIn("spoken", effective.lower())
        self.assertEqual(processing["audio_preset_requested"], "dialogue")
        self.assertEqual(processing["audio_preset_effective"], "ambience")


if __name__ == "__main__":
    unittest.main()
