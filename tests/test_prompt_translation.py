from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.prompt_translation import (  # noqa: E402
    OFFICIAL_BASE_SECTION_HEADERS,
    OFFICIAL_SECTION_HEADERS,
    PromptTranslationError,
    classify_visible_text_literals,
    contains_japanese_outside_dialogue,
    requires_translation,
    translate_and_compile_prompt,
    validate_authorized_visible_text_literals,
    validate_native_dialogue_blocks,
)
from webui.prompt_translation_worker import (  # noqa: E402
    LFM2EnglishTranslator,
    process_request,
)
from webui.h3_dialogue import format_inline_dialogue  # noqa: E402


_NAMES = {
    "アリス": "Alice",
    "ボブ": "Bob",
    "キャロル": "Carol",
    "デイビッド": "David",
    "エマ": "Emma",
    "フランク": "Frank",
}


class FakeLFM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> str:
        self.calls.append(text)
        aliases: list[str] = []
        for japanese_kind, english_kind in (
            ("参照画像", "reference image"),
            ("参照動画", "reference video"),
            ("参照音声", "reference audio"),
            ("被写体", "subject"),
        ):
            for japanese_name, english_name in _NAMES.items():
                phrase = japanese_kind + japanese_name
                if phrase in text:
                    replacement = f"{english_kind} {english_name}"
                    text = text.replace(phrase, replacement)
                    aliases.append(replacement)

        if "海辺のベンチ" in text:
            return " and ".join(aliases) + (
                " show two characters sitting side by side on a seaside bench, "
                "facing the locked-off camera."
            )
        if "穏やかな波音" in text:
            return "Gentle ocean waves and a light sea breeze."
        if "夏服" in text:
            prefix = " and ".join(aliases)
            return f"{prefix} preserve their summer outfits exactly.".strip()
        if "手を振る" in text and "小さく手を振る" not in text:
            prefix = " and ".join(aliases)
            return f"{prefix} waves one hand once.".strip()
        if "静かな海" in text:
            return "A quiet sea at sunset."
        if "看板" in text and "on-screen text" in text:
            alias = re.search(r"on-screen text [A-Za-z]+", text)
            return f"A sign clearly displays {alias.group(0)}." if alias else ""
        if "小さく手を振る" in text:
            return "The woman waves her hands slightly."
        if "青いコート" in text:
            prefix = " and ".join(aliases)
            return f"The character from {prefix} changes into a blue coat."
        if "青いワンピース" in text:
            prefix = " and ".join(aliases)
            return f"The character from {prefix} wears a blue dress."
        if "環境音" in text:
            prefix = " and ".join(aliases)
            return f"Use {prefix} only as an ambience reference."
        if "音楽参照" in text:
            prefix = " and ".join(aliases)
            return f"Use {prefix} only as a music reference."
        if "歩く" in text:
            prefix = " and ".join(aliases)
            return f"{prefix} walks slowly.".strip()
        raise AssertionError(f"Unexpected fake translation input: {text!r}")


def _headers(text: str) -> tuple[str, ...]:
    return tuple(
        match.group(1)
        for match in re.finditer(
            r"^(subject_definitions|summary|retention_analysis|detailed_description|"
            r"integrated_multimodal_description|overall_soundscape|non_diegetic_music):$",
            text,
            re.MULTILINE,
        )
    )


def _canonical_dialogue_line() -> str:
    return (
        "<Subject 1> (S1) is the visible character shown in <Picture 1>. "
        "<Audio 1> provides the voice timbre and measured delivery for <Subject 1> (S1). "
        "<Subject 1> (S1), using <Audio 1>, says once: "
        "<d>[Japanese] こんにちは。</d> Speaker (S1) closes their lips after the line."
    )


class PromptTranslationTests(unittest.TestCase):
    def test_lfm_chat_template_receives_a_list_conversation(self) -> None:
        import torch

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def __init__(self) -> None:
                self.chat: object = None

            def apply_chat_template(self, chat: object, **_: object) -> object:
                if not isinstance(chat, list):
                    raise TypeError("chat must be a list")
                self.chat = chat
                return {"input_ids": torch.tensor([[1, 2]])}

            @staticmethod
            def decode(_: object, **__: object) -> str:
                return "A quiet beach."

        class FakeModel:
            @staticmethod
            def generate(**_: object) -> object:
                return torch.tensor([[1, 2, 3]])

        translator = object.__new__(LFM2EnglishTranslator)
        translator._torch = torch  # type: ignore[attr-defined]
        translator._tokenizer = FakeTokenizer()  # type: ignore[attr-defined]
        translator._model = FakeModel()  # type: ignore[attr-defined]

        self.assertEqual(translator("静かな海。"), "A quiet beach.")
        self.assertIsInstance(translator._tokenizer.chat, list)

    def test_japanese_controls_become_official_six_sections_and_dialogue_is_immutable(self) -> None:
        canonical = _canonical_dialogue_line()
        source = (
            "Cut 1\n"
            "<Picture 1>と<Picture 2>のが海辺のベンチに並んで座り、正面の固定カメラを見る。\n"
            f"{canonical}\n"
            "Audio: 穏やかな波音。\n"
            "Music: N/A"
        )
        translator = FakeLFM()
        references = [
            {"kind": "image", "index": 1, "source_index": 0, "tag": "<Picture 1>"},
            {"kind": "image", "index": 2, "source_index": 1, "tag": "<Picture 2>"},
            {"kind": "audio", "index": 1, "source_index": 2, "tag": "<Audio 1>"},
        ]

        result = translate_and_compile_prompt(
            source,
            translator,
            dialogue_events=[
                {
                    "speaker_id": 1,
                    "speaker_label": "The visible character shown in <Picture 1>",
                    "audio_reference_id": 1,
                }
            ],
            music_policy="none",
            reference_inventory=references,
        )

        self.assertEqual(_headers(result.compiled_prompt), OFFICIAL_SECTION_HEADERS)
        self.assertFalse(contains_japanese_outside_dialogue(result.compiled_prompt))
        self.assertEqual(result.compiled_prompt.count("<d>[Japanese] こんにちは。</d>"), 1)
        self.assertIn(canonical, result.compiled_prompt)
        self.assertIn("<Subject 1> (S1) is the visible character", result.compiled_prompt)
        self.assertIn("<Subject 2> is the visible character", result.compiled_prompt)
        self.assertNotIn("<Subject 2> (S2)", result.compiled_prompt)
        self.assertIn("concept-sheet layout", result.compiled_prompt)
        self.assertIn(
            "[reference generation + audio reference]", result.compiled_prompt
        )
        self.assertIn("<Picture 1>: fully_preserved -", result.compiled_prompt)
        self.assertIn("<Picture 2>: fully_preserved -", result.compiled_prompt)
        self.assertIn("<Subject 1>: fully_preserved -", result.compiled_prompt)
        self.assertIn("<Subject 2>: fully_preserved -", result.compiled_prompt)
        self.assertIn(
            "<Audio 1>: reference - preserve its voice timbre and measured delivery for "
            "<Subject 1> (S1)",
            result.compiled_prompt,
        )
        self.assertIn("Gentle ocean waves", result.compiled_prompt)
        for forbidden in (
            "spoken content is limited",
            "non-speech",
            "without copying",
            "semantic content",
            "its words",
            "jaw ceases",
            "tagged dialogue",
            "human vocal content",
        ):
            self.assertNotIn(forbidden, result.compiled_prompt.lower())
        self.assertTrue(result.compiled_prompt.endswith("non_diegetic_music:\nN/A"))
        self.assertEqual(result.source_reference_tags, result.translated_reference_tags)
        self.assertEqual(result.translated_line_count, 2)
        self.assertNotIn("のが", translator.calls[0])

    def test_manifest_defines_unmentioned_omni_references_without_changing_detail_tags(self) -> None:
        result = translate_and_compile_prompt(
            "Cut 1\nA locked-off wide shot of an empty beach.",
            reference_inventory=[
                {"kind": "image", "index": 1, "source_index": 0, "tag": "<Picture 1>"},
                {"kind": "video", "index": 1, "source_index": 1, "tag": "<Video 1>"},
                {"kind": "audio", "index": 1, "source_index": 2, "tag": "<Audio 1>"},
            ],
        )

        definitions = result.compiled_prompt.split("summary:", 1)[0]
        self.assertIn("<Picture 1>", definitions)
        self.assertIn("<Video 1>", definitions)
        self.assertIn("<Audio 1>", definitions)
        self.assertEqual(result.source_reference_tags, ())
        self.assertEqual(result.translated_reference_tags, ())
        self.assertIn("<Picture 1>: weak_reference -", result.compiled_prompt)
        self.assertIn("<Video 1>: weak_reference -", result.compiled_prompt)
        self.assertIn("<Audio 1>: inactive -", result.compiled_prompt)

    def test_pure_english_compiles_without_calling_translator(self) -> None:
        class MustNotRun:
            def __call__(self, _: str) -> str:
                raise AssertionError("translator must not be called")

        result = translate_and_compile_prompt(
            "Cut 1\n<Picture 1> shows a woman who waves once.\nMusic: N/A",
            MustNotRun(),
        )

        self.assertEqual(result.translated_line_count, 0)
        self.assertEqual(_headers(result.compiled_prompt), OFFICIAL_SECTION_HEADERS)

    def test_already_valid_english_six_sections_passes_through_without_model(self) -> None:
        prompt = (
            "subject_definitions:\nA defined subject.\n\n"
            "summary:\nA short video.\n\n"
            "retention_analysis:\nSubject: fully_preserved - stable.\n\n"
            "detailed_description:\n[Shot 1] The subject waves.\n\n"
            "overall_soundscape:\nQuiet room tone.\n\n"
            "non_diegetic_music:\nN/A"
        )

        result = translate_and_compile_prompt(prompt)

        self.assertTrue(result.already_compiled)
        self.assertEqual(result.compiled_prompt, prompt)
        self.assertFalse(requires_translation(prompt))

    def test_japanese_requires_optional_translator_and_fails_closed_when_missing(self) -> None:
        with self.assertRaises(PromptTranslationError) as caught:
            translate_and_compile_prompt("Cut 1\n静かな海。")

        self.assertEqual(caught.exception.code, "TRANSLATOR_REQUIRED")

    def test_japanese_outside_dialogue_on_canonical_line_fails_closed(self) -> None:
        prompt = (
            "Cut 1\nThe woman says <d>[Japanese] こんにちは。</d> "
            "その後カメラを見る。"
        )

        with self.assertRaises(PromptTranslationError) as caught:
            translate_and_compile_prompt(prompt, FakeLFM())

        self.assertEqual(
            caught.exception.code, "JAPANESE_IN_IMMUTABLE_DIALOGUE_LINE"
        )

    def test_reference_alias_reordering_is_rejected(self) -> None:
        def reorder(_: str) -> str:
            return "reference image Bob and reference image Alice sit on a bench."

        with self.assertRaises(PromptTranslationError) as caught:
            translate_and_compile_prompt(
                "<Picture 1>と<Picture 2>が海辺のベンチに座る。", reorder
            )

        self.assertEqual(caught.exception.code, "REFERENCE_TAG_MISMATCH")

    def test_dropped_reference_alias_is_rejected(self) -> None:
        with self.assertRaises(PromptTranslationError) as caught:
            translate_and_compile_prompt(
                "<Picture 1>の人物が手を振る。",
                lambda _: "A character waves.",
            )

        self.assertEqual(caught.exception.code, "REFERENCE_ALIAS_MISMATCH")

    def test_reference_alias_translated_into_visible_text_is_rejected(self) -> None:
        with self.assertRaises(PromptTranslationError) as caught:
            translate_and_compile_prompt(
                "<Picture 1>の女性を映す。",
                lambda _: 'The words "reference image Alice" appear on screen.',
            )

        self.assertEqual(
            caught.exception.code, "REFERENCE_ALIAS_BECAME_VISIBLE_TEXT"
        )

    def test_leftover_japanese_and_multiline_output_are_rejected(self) -> None:
        for output, code in (
            ("A quiet 海。", "UNTRANSLATED_JAPANESE"),
            ("A quiet sea.\nSecond injected line.", "MULTILINE_TRANSLATOR_OUTPUT"),
        ):
            with self.subTest(output=output):
                with self.assertRaises(PromptTranslationError) as caught:
                    translate_and_compile_prompt("静かな海。", lambda _: output)
                self.assertEqual(caught.exception.code, code)

    def test_translator_cannot_invent_speech_in_visual_control_prose(self) -> None:
        for translated in (
            "The narrator says the woman runs.",
            "The narrator explains the workout continuously.",
            "A woman narrates continuously while running.",
            "Voice-over narration describes the workout.",
        ):
            with self.subTest(translated=translated):
                with self.assertRaises(PromptTranslationError) as caught:
                    translate_and_compile_prompt(
                        "女性が走る。",
                        lambda _: translated,
                        mode="omni",
                    )
                self.assertEqual(
                    caught.exception.code, "TRANSLATOR_INVENTED_SPEECH"
                )

    def test_translator_cannot_invent_vocals_in_music_control(self) -> None:
        with self.assertRaises(PromptTranslationError) as caught:
            translate_and_compile_prompt(
                "Cut 1\n女性が走る。\nMusic: 穏やかな音楽。",
                lambda text: (
                    "The woman runs."
                    if "走る" in text
                    else "A voice-over narration continues over soft music."
                ),
                mode="t2v",
            )

        self.assertEqual(caught.exception.code, "TRANSLATOR_INVENTED_SPEECH")

    def test_normal_visual_translation_without_vocal_role_still_passes(self) -> None:
        result = translate_and_compile_prompt(
            "女性が走る。",
            lambda _: "The woman runs.",
            mode="omni",
        )

        self.assertIn("The woman runs.", result.compiled_prompt)

    def test_worker_contract_with_injected_fake_translator(self) -> None:
        response = process_request(
            {
                "prompt": "Cut 1\n静かな海。",
                "music_policy": "none",
                "dialogue_events": [],
                "references": [],
                "model_id": "test/fake",
                "revision": "fixed",
            },
            translator=FakeLFM(),
        )

        self.assertTrue(response["ok"])
        self.assertIn("compiled_prompt", response)
        self.assertEqual(response["diagnostics"], [])
        metadata = response["compiler_metadata"]
        self.assertEqual(metadata["model_id"], "test/fake")
        self.assertEqual(metadata["revision"], "fixed")
        self.assertEqual(metadata["device"], "cpu")
        self.assertEqual(metadata["dtype"], "float32")
        self.assertEqual(metadata["system_prompt"], "Translate to English.")

    def test_worker_cli_emits_one_json_object_and_skips_missing_model_for_english(self) -> None:
        request = {
            "prompt": "Cut 1\nA woman waves once.\nMusic: N/A",
            "model_path": str(ROOT / "definitely-missing-model"),
            "dialogue_events": [],
            "references": [],
        }
        completed = subprocess.run(
            [sys.executable, "-m", "webui.prompt_translation_worker"],
            cwd=ROOT,
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        lines = completed.stdout.splitlines()
        self.assertEqual(len(lines), 1)
        response = json.loads(lines[0])
        self.assertTrue(response["ok"])
        self.assertIn("compiled_prompt", response)

    def test_invalid_manifest_and_partial_six_section_are_rejected(self) -> None:
        with self.assertRaises(PromptTranslationError) as inventory_error:
            translate_and_compile_prompt(
                "A quiet beach.",
                reference_inventory=[
                    {"kind": "image", "index": 1, "tag": "<Picture 2>"}
                ],
            )
        self.assertEqual(
            inventory_error.exception.code, "INVALID_REFERENCE_INVENTORY"
        )

        with self.assertRaises(PromptTranslationError) as schema_error:
            translate_and_compile_prompt("summary:\nOnly one section.")
        self.assertEqual(schema_error.exception.code, "INVALID_SIX_SECTION_SCHEMA")

    def test_all_modes_render_their_exact_official_schema_for_english(self) -> None:
        for mode in ("t2v", "i2v", "first_last", "omni"):
            with self.subTest(mode=mode):
                result = translate_and_compile_prompt(
                    "Cut 1\nA woman faces the camera.\nCut 3\nShe gives one small wave.",
                    mode=mode,
                    duration_seconds=10.125,
                )
                expected = (
                    OFFICIAL_SECTION_HEADERS
                    if mode == "omni"
                    else OFFICIAL_BASE_SECTION_HEADERS
                )
                self.assertEqual(_headers(result.compiled_prompt), expected)
                self.assertEqual(result.mode.value, mode)
                if mode == "i2v":
                    self.assertTrue(
                        result.compiled_prompt.startswith(
                            "For the target video, at 0.00 seconds into the target video, "
                            "<Picture 1> (from [Shot 1]) is fully referenced."
                        )
                    )
                elif mode == "first_last":
                    self.assertTrue(
                        result.compiled_prompt.startswith(
                            "How the reference pictures align with the target video — "
                            "Picture 1 (from Shot 1) aligns with the 0.00-second mark"
                        )
                    )
                    self.assertIn(
                        "Picture 2 (from Shot 3) aligns with the 10.12-second mark",
                        result.compiled_prompt,
                    )
                elif mode == "t2v":
                    self.assertTrue(
                        result.compiled_prompt.startswith(
                            "integrated_multimodal_description:"
                        )
                    )

    def test_all_modes_translate_japanese_into_mode_specific_schema(self) -> None:
        for mode in ("t2v", "i2v", "first_last", "omni"):
            with self.subTest(mode=mode):
                translator = FakeLFM()
                result = translate_and_compile_prompt(
                    "Cut 1\n静かな海。",
                    translator,
                    mode=mode,
                    duration_seconds=5.0,
                )
                expected = (
                    OFFICIAL_SECTION_HEADERS
                    if mode == "omni"
                    else OFFICIAL_BASE_SECTION_HEADERS
                )
                self.assertEqual(_headers(result.compiled_prompt), expected)
                self.assertIn("A quiet sea at sunset.", result.compiled_prompt)
                self.assertEqual(len(translator.calls), 1)

    def test_worker_consumes_mode_and_duration_for_base_schema(self) -> None:
        response = process_request(
            {
                "prompt": "Cut 1\nA woman waits.\nCut 2\nShe waves.",
                "mode": "first_last",
                "duration_seconds": 7.5,
                "dialogue_events": [],
                "references": [],
            }
        )

        self.assertTrue(response["ok"])
        self.assertEqual(
            _headers(response["compiled_prompt"]), OFFICIAL_BASE_SECTION_HEADERS
        )
        self.assertIn("Picture 2 (from Shot 2)", response["compiled_prompt"])
        self.assertIn("7.50-second mark", response["compiled_prompt"])
        self.assertEqual(response["compiler_metadata"]["mode"], "first_last")

    def test_invalid_mode_and_first_last_without_duration_fail_closed(self) -> None:
        with self.assertRaises(PromptTranslationError) as invalid:
            translate_and_compile_prompt("A quiet beach.", mode="video")
        self.assertEqual(invalid.exception.code, "INVALID_GENERATION_MODE")

        with self.assertRaises(PromptTranslationError) as missing_duration:
            translate_and_compile_prompt("A quiet beach.", mode="first_last")
        self.assertEqual(missing_duration.exception.code, "DURATION_REQUIRED")

    def test_existing_base_prompt_passes_only_for_a_base_mode(self) -> None:
        base = (
            "integrated_multimodal_description:\n[Shot 1] A woman waves.\n\n"
            "overall_soundscape:\nQuiet room tone.\n\n"
            "non_diegetic_music:\nN/A"
        )
        result = translate_and_compile_prompt(base, mode="t2v")
        self.assertTrue(result.already_compiled)
        self.assertEqual(result.compiled_prompt, base)

        with self.assertRaises(PromptTranslationError) as mismatch:
            translate_and_compile_prompt(base, mode="omni")
        self.assertEqual(mismatch.exception.code, "INVALID_SIX_SECTION_SCHEMA")

    def test_audio_and_music_reference_tags_survive_component_extraction_in_source_order(self) -> None:
        source = (
            "Audio: <Audio 1>を環境音として使う。\n"
            "Cut 1\n<Picture 1> shows a woman waiting.\n"
            "Music: <Audio 2>を音楽参照として使う。"
        )
        result = translate_and_compile_prompt(source, FakeLFM(), mode="omni")

        expected = ("<Audio 1>", "<Picture 1>", "<Audio 2>")
        self.assertEqual(result.source_reference_tags, expected)
        self.assertEqual(result.translated_reference_tags, expected)
        soundscape = result.compiled_prompt.split("overall_soundscape:\n", 1)[1].split(
            "\n\nnon_diegetic_music:", 1
        )[0]
        music = result.compiled_prompt.split("non_diegetic_music:\n", 1)[1]
        self.assertIn("<Audio 1>", soundscape)
        self.assertIn("<Audio 2>", music)
        self.assertGreaterEqual(result.compiled_prompt.count("<Picture 1>"), 1)

    def test_confirmed_small_wave_mistranslation_is_corrected_narrowly(self) -> None:
        corrected = translate_and_compile_prompt(
            "Cut 1\n女性が小さく手を振る。", FakeLFM(), mode="t2v"
        )
        self.assertIn("gives a small wave with one hand", corrected.compiled_prompt)
        self.assertNotIn("waves her hands slightly", corrected.compiled_prompt)

        untouched = translate_and_compile_prompt(
            "Cut 1\nA woman waves her hands slightly.", mode="t2v"
        )
        self.assertIn("waves her hands slightly", untouched.compiled_prompt)

    def test_visible_japanese_sign_is_protected_from_lfm_and_restored_verbatim(self) -> None:
        translator = FakeLFM()
        result = translate_and_compile_prompt(
            "Cut 1\n看板に「営業中！」と表示される。",
            translator,
            mode="t2v",
        )

        self.assertEqual(result.visible_text_literals, ("営業中！",))
        self.assertIn('A sign clearly displays "営業中！".', result.compiled_prompt)
        self.assertNotIn("営業中！", translator.calls[0])
        self.assertFalse(
            contains_japanese_outside_dialogue(
                result.compiled_prompt, result.visible_text_literals
            )
        )
        self.assertTrue(contains_japanese_outside_dialogue(result.compiled_prompt))
        self.assertEqual(
            validate_authorized_visible_text_literals(
                result.compiled_prompt, result.visible_text_literals
            ),
            ("営業中！",),
        )
        metadata = result.metadata()
        self.assertEqual(metadata["visible_text_literals"], ["営業中！"])
        self.assertEqual(len(metadata["visible_text_literal_sha256"]), 1)

    def test_visible_text_translated_into_human_speech_is_rejected(self) -> None:
        for output_template in (
            'The woman says "{alias}".',
            'The woman is speaking "{alias}" while a subtitle is visible.',
            'The narrator is reading "{alias}" from a sign.',
        ):
            with self.subTest(output_template=output_template):
                def mistranslate_as_speech(text: str) -> str:
                    alias = re.search(r"on-screen text ([A-Za-z]+)", text)
                    self.assertIsNotNone(alias)
                    return output_template.format(alias=alias.group(1))

                with self.assertRaises(PromptTranslationError) as caught:
                    translate_and_compile_prompt(
                        "Cut 1\n看板に「営業中」と表示される。",
                        mistranslate_as_speech,
                        mode="t2v",
                    )
                self.assertEqual(
                    caught.exception.code, "VISIBLE_TEXT_BECAME_SPEECH"
                )

    def test_public_visible_text_classifier_handles_compiled_prompt_locally(self) -> None:
        compiled = translate_and_compile_prompt(
            'Cut 1\nThe sign reads "営業中".', mode="t2v"
        ).compiled_prompt

        self.assertEqual(classify_visible_text_literals(compiled), ("営業中",))
        with self.assertRaises(PromptTranslationError) as caught:
            classify_visible_text_literals(123)  # type: ignore[arg-type]
        self.assertEqual(caught.exception.code, "INVALID_PROMPT")

    def test_english_subtitle_cue_with_japanese_literal_needs_no_model(self) -> None:
        result = translate_and_compile_prompt(
            "Cut 1\nThe subtitle displays 「つづく。」.", mode="t2v"
        )

        self.assertEqual(result.translated_line_count, 0)
        self.assertIn('The subtitle displays "つづく。".', result.compiled_prompt)
        self.assertFalse(requires_translation("The subtitle displays 「つづく。」."))

    def test_unclassified_cjk_quote_and_lyrics_fail_closed(self) -> None:
        with self.assertRaises(PromptTranslationError) as unclassified:
            translate_and_compile_prompt("Cut 1\nShe holds 「秘密」.", FakeLFM())
        self.assertEqual(
            unclassified.exception.code, "UNCLASSIFIED_CJK_QUOTED_TEXT"
        )

        with self.assertRaises(PromptTranslationError) as lyrics:
            translate_and_compile_prompt("Cut 1\n歌詞「ラララ」を歌う。", FakeLFM())
        self.assertEqual(lyrics.exception.code, "UNSUPPORTED_LYRICS")

    def test_visible_literal_mutation_is_rejected(self) -> None:
        result = translate_and_compile_prompt(
            "The sign reads 「営業中」.", mode="t2v"
        )
        mutated = result.compiled_prompt.replace('"営業中"', '"準備中"')
        with self.assertRaises(PromptTranslationError) as caught:
            validate_authorized_visible_text_literals(
                mutated, result.visible_text_literals
            )
        self.assertEqual(caught.exception.code, "VISIBLE_TEXT_LITERAL_MISMATCH")

    def test_native_dialogue_allowlist_case_spacing_and_reserved_tokens(self) -> None:
        valid = "<d>[Japanese] こんにちは。</d>"
        self.assertEqual(validate_native_dialogue_blocks(valid), (valid,))

        for value, code in (
            ("<d>[Klingon] Qapla.</d>", "UNSUPPORTED_DIALOGUE_LANGUAGE"),
            ("<d>[japanese] こんにちは。</d>", "INVALID_DIALOGUE_LANGUAGE_TAG"),
            ("<d>[ Japanese ] こんにちは。</d>", "MALFORMED_DIALOGUE_TAG"),
            ("<D>[Japanese] こんにちは。</D>", "INVALID_DIALOGUE_LANGUAGE_TAG"),
            ("<d>hello", "MALFORMED_DIALOGUE_TAG"),
            ("<d foo>hello</d>", "MALFORMED_DIALOGUE_TAG"),
            (
                "<d>[Japanese] <Picture 1>を見て。</d>",
                "RESERVED_TOKEN_IN_DIALOGUE",
            ),
        ):
            with self.subTest(value=value):
                with self.assertRaises(PromptTranslationError) as caught:
                    validate_native_dialogue_blocks(value)
                self.assertEqual(caught.exception.code, code)

    def test_reserved_token_from_natural_language_quote_is_rejected_after_formatter(self) -> None:
        formatted = format_inline_dialogue(
            'Cut 1\n彼女は「<Picture 1>を見て。」と言う。'
        ).text

        with self.assertRaises(PromptTranslationError) as caught:
            translate_and_compile_prompt(formatted, FakeLFM())
        self.assertEqual(caught.exception.code, "RESERVED_TOKEN_IN_DIALOGUE")

    def test_inventory_only_audio_is_explicitly_ignored_without_speech_activation(self) -> None:
        result = translate_and_compile_prompt(
            "Cut 1\nAn empty beach under a clear sky.",
            mode="omni",
            reference_inventory=[
                {"kind": "audio", "index": 1, "source_index": 0, "tag": "<Audio 1>"}
            ],
        )

        self.assertIn("[reference generation]", result.compiled_prompt)
        self.assertNotIn(
            "[reference generation + audio reference]", result.compiled_prompt
        )
        self.assertIn("<Audio 1> is inactive", result.compiled_prompt)
        self.assertIn("<Audio 1>: inactive -", result.compiled_prompt)
        self.assertNotIn("tagged dialogue", result.compiled_prompt)
        self.assertNotIn("voice-timbre and measured-delivery reference", result.compiled_prompt)

    def test_single_picture_actor_and_wardrobe_override_get_correct_relationships(self) -> None:
        actor = translate_and_compile_prompt(
            "Cut 1\n<Picture 1>が歩く。", FakeLFM(), mode="omni"
        )
        self.assertIn("<Subject 1> is the visible character", actor.compiled_prompt)
        self.assertIn("<Picture 1>: fully_preserved -", actor.compiled_prompt)

        wardrobe = translate_and_compile_prompt(
            "Cut 1\n<Picture 1>の人物を青いコートに着替えさせる。",
            FakeLFM(),
            mode="omni",
        )
        self.assertIn("target wardrobe", wardrobe.compiled_prompt)
        self.assertIn("<Picture 1>: partially_preserved -", wardrobe.compiled_prompt)
        self.assertIn("<Subject 1>: partially_preserved -", wardrobe.compiled_prompt)

    def test_wardrobe_mentions_preserve_reference_until_change_is_explicit(self) -> None:
        preserved = (
            "<Picture 1> shows a woman wearing a white dress.",
            "<Picture 1> shows a woman wearing the same white dress.",
            "<Picture 1> shows a woman wearing the reference outfit unchanged.",
        )
        for prompt in preserved:
            with self.subTest(prompt=prompt):
                result = translate_and_compile_prompt(
                    f"Cut 1\n{prompt}", mode="omni"
                )
                self.assertIn("<Picture 1>: fully_preserved -", result.compiled_prompt)
                self.assertNotIn("target wardrobe", result.compiled_prompt)

        changed = (
            "<Picture 1> shows a woman who changes into a new blue coat.",
            "<Picture 1> shows a woman replacing the source outfit with a new blue coat.",
            "<Picture 1> shows a woman who puts on a blue blouse.",
            "<Picture 1> shows a woman who has a new blue shirt.",
        )
        for prompt in changed:
            with self.subTest(prompt=prompt):
                result = translate_and_compile_prompt(
                    f"Cut 1\n{prompt}", mode="omni"
                )
                self.assertIn(
                    "<Picture 1>: partially_preserved -", result.compiled_prompt
                )
                self.assertIn("target wardrobe", result.compiled_prompt)

    def test_wardrobe_override_uses_source_and_translated_detail(self) -> None:
        for source in (
            "<Picture 1>の女性は、参照画像の衣装ではなく、新しい青いコートを着ている。",
            "<Picture 1>の女性は青いワンピースを着ている。",
        ):
            with self.subTest(source=source):
                result = translate_and_compile_prompt(
                    f"Cut 1\n{source}", FakeLFM(), mode="omni"
                )
                self.assertIn(
                    "<Picture 1>: partially_preserved -", result.compiled_prompt
                )

        translated_only = translate_and_compile_prompt(
            "Cut 1\n<Picture 1>の女性を新しい姿にする。",
            lambda _: (
                "The character from reference image Alice is wearing a new blue coat."
            ),
            mode="omni",
        )
        self.assertIn(
            "<Picture 1>: partially_preserved -", translated_only.compiled_prompt
        )

    def test_vocal_translation_in_audio_is_rebuilt_as_physical_ambience(self) -> None:
        def invent_narration(_: str) -> str:
            return "A woman narrates continuously in an unknown language."

        for detail in (
            "A quiet beach.",
            "The woman says exactly once: <d>[Japanese] こんにちは。</d>",
        ):
            with self.subTest(detail=detail):
                result = translate_and_compile_prompt(
                    f"Cut 1\n{detail}\nAudio: 穏やかな波音。",
                    invent_narration,
                    mode="t2v",
                )
                soundscape = result.compiled_prompt.split(
                    "overall_soundscape:\n", 1
                )[1].split("\n\nnon_diegetic_music:", 1)[0]
                self.assertNotIn("unknown language", soundscape)
                self.assertIn("environmental ambience", soundscape)
                if "<d>" not in detail:
                    self.assertEqual(
                        re.findall(
                            r"\b(?:speech|spoken|speak(?:s|ing)?|said|says?|"
                            r"talk(?:s|ing)?|dialogue|narrat(?:e|es|ed|ing|ion|or)|"
                            r"voice(?:[ -]?over)?|voices|vocal(?:s|ization)?|"
                            r"language|words?|utterance|greeting)\b",
                            soundscape,
                            re.IGNORECASE,
                        ),
                        [],
                    )


if __name__ == "__main__":
    unittest.main()
