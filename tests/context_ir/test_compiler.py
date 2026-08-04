from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.context_ir import (
    OMNI_SECTION_HEADERS,
    THREE_SECTION_HEADERS,
    CompilationStatus,
    compile_request,
    write_artifacts,
)


def request(mode: str = "t2v", **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "mode": mode,
        "prompt": "A paper kite rises in a summer breeze.",
        "num_frames": 243,
        "style": "natural",
        "dialogue": "",
        "soundscape": "",
        "music_policy": "none",
        "references": [],
    }
    values.update(overrides)
    return values


def positions(text: str, headers: tuple[str, ...]) -> list[int]:
    return [text.index(f"{header}:") for header in headers]


class BaseRendererTests(unittest.TestCase):
    def test_t2v_uses_official_three_sections_and_shot1_has_no_timestamp(self) -> None:
        result = compile_request(request())
        self.assertFalse(result.fatal)
        self.assertEqual(result.status, CompilationStatus.DEGRADED)
        assert result.ir_text is not None
        self.assertEqual(positions(result.ir_text, THREE_SECTION_HEADERS), sorted(positions(result.ir_text, THREE_SECTION_HEADERS)))
        self.assertIn("[Shot 1]\n", result.ir_text)
        self.assertNotIn("[Shot 1] At", result.ir_text)
        self.assertTrue(result.ir_text.endswith("non_diegetic_music:\nN/A"))

    def test_i2v_has_canonical_alignment_before_core(self) -> None:
        result = compile_request(request("i2v"))
        assert result.ir_text is not None
        self.assertTrue(
            result.ir_text.startswith(
                "For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced."
            )
        )
        self.assertFalse(result.fatal)

    def test_first_last_uses_actual_frame_duration_and_last_shot(self) -> None:
        result = compile_request(
            request(
                "first_last",
                num_frames=345,
                prompt="Cut1\nstart\nCut2\nfinish",
            )
        )
        assert result.ir_text is not None
        first_line = result.ir_text.splitlines()[0]
        self.assertIn("Picture 2 (from Shot 2)", first_line)
        self.assertIn("14.38-second mark", first_line)

    def test_cut_markers_get_even_later_timestamps_only(self) -> None:
        result = compile_request(
            request(
                prompt=(
                    "Global rule.\nCut1\nfirst\nCut2\nsecond\n"
                    "[Shot 3] third\nCut4\nfourth"
                ),
                duration_seconds=14.375,
                num_frames=None,
            )
        )
        assert result.ir_text is not None
        self.assertNotIn("[Shot 1] At", result.ir_text)
        self.assertIn("[Shot 2] At 00:03.594", result.ir_text)
        self.assertIn("[Shot 3] At 00:07.188", result.ir_text)
        self.assertIn("[Shot 4] At 00:10.781", result.ir_text)
        self.assertIn("Global rule.", result.ir_text)

    def test_japanese_text_is_not_rewritten_or_translated(self) -> None:
        prompt = "日本の高品質なセルアニメ調。\n空の色は薄い水色。"
        result = compile_request(request(prompt=prompt))
        assert result.ir_text is not None
        self.assertIn(prompt, result.ir_text)
        self.assertTrue(result.degraded)
        self.assertTrue(any(item.code == "IR_DEGRADED_LOCAL_SCAFFOLD" for item in result.diagnostics))

    def test_mode_specific_reference_labels_block_missing_inputs(self) -> None:
        cases = (
            ("t2v", "Use <Picture 1>.", []),
            ("t2v", "Use <Video 1>.", []),
            ("t2v", "Use <Audio 1>.", []),
            ("i2v", "Use <Picture 2>.", []),
            ("i2v", "Use <Audio 1>.", []),
            ("first_last", "Use <Picture 3>.", []),
            ("first_last", "Use <Video 1>.", []),
            ("omni", "Use <Picture 1>.", []),
        )
        for mode, prompt, references in cases:
            with self.subTest(mode=mode, prompt=prompt):
                result = compile_request(
                    request(mode, prompt=prompt, references=references)
                )
                self.assertTrue(result.fatal)
                self.assertIsNone(result.ir_text)
                self.assertTrue(
                    any(
                        item.code == "IR_UNKNOWN_REFERENCE_LABEL"
                        for item in result.diagnostics
                    )
                )

    def test_base_modes_accept_only_their_supplied_picture_labels(self) -> None:
        i2v = compile_request(request("i2v", prompt="Animate <Picture 1>."))
        first_last = compile_request(
            request(
                "first_last",
                prompt="Move from <Picture 1> to <Picture 2>.",
            )
        )
        self.assertFalse(i2v.fatal)
        self.assertFalse(first_last.fatal)


class DialogueTests(unittest.TestCase):
    def test_unknown_subject_speaker_aliases_are_fatal(self) -> None:
        for alias in ("<Subject 99>", "S3"):
            with self.subTest(alias=alias):
                result = compile_request(
                    request(dialogue=f"{alias} <d>[English] hello</d>")
                )
                self.assertTrue(result.fatal)
                self.assertIsNone(result.ir_text)
                diagnostic = next(
                    item
                    for item in result.diagnostics
                    if item.code == "IR_UNKNOWN_SPEAKER_SUBJECT"
                )
                self.assertEqual(
                    diagnostic.path,
                    "/shots/0/dialogue_events/0/speaker_id",
                )

    def test_defined_subject_speaker_alias_is_valid(self) -> None:
        result = compile_request(
            request(dialogue="S1 <d>[English] hello</d>")
        )
        self.assertFalse(result.fatal)
        self.assertFalse(
            any(
                item.code == "IR_UNKNOWN_SPEAKER_SUBJECT"
                for item in result.diagnostics
            )
        )

    def test_direct_visual_speaker_reference_does_not_require_subject_alias(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt="Use <Picture 1> as a character reference.",
                dialogue=(
                    "<Subject 99> <Picture 1> <d>[English] hello</d>"
                ),
                references=[{"kind": "image"}],
            )
        )
        self.assertFalse(result.fatal)
        assert result.ir_text is not None
        self.assertIn(
            "The visible character identified by <Picture 1>",
            result.ir_text,
        )
        self.assertFalse(
            any(
                item.code == "IR_UNKNOWN_SPEAKER_SUBJECT"
                for item in result.diagnostics
            )
        )

    def test_direct_picture_speaker_with_noncharacter_role_warns_but_is_retained(
        self,
    ) -> None:
        cases = (
            ("Use <Picture 1> as a clothing reference.", "clothing"),
            ("Use <Picture 1> as a style reference.", "style"),
            ("Use <Picture 1> as a background reference.", "background"),
            ("Use <Picture 1> as an object reference.", "object"),
        )
        for prompt, role in cases:
            with self.subTest(role=role):
                result = compile_request(
                    request(
                        "omni",
                        prompt=prompt,
                        dialogue="<Picture 1> <d>[English] hello</d>",
                        references=[{"kind": "image"}],
                    )
                )
                self.assertFalse(result.fatal)
                assert result.ir_text is not None
                self.assertIn(
                    "The visible character identified by <Picture 1>",
                    result.ir_text,
                )
                diagnostics = [
                    item
                    for item in result.diagnostics
                    if item.code == "IR_DIALOGUE_SPEAKER_ROLE_CONFLICT"
                ]
                self.assertEqual(len(diagnostics), 1)
                self.assertIn(f"resolved as a {role} reference", diagnostics[0].message)

    def test_direct_picture_speaker_with_negated_identity_warns_but_is_retained(
        self,
    ) -> None:
        result = compile_request(
            request(
                "omni",
                prompt=(
                    "Do not use <Picture 1> as a character identity reference."
                ),
                dialogue="<Picture 1> <d>[English] hello</d>",
                references=[{"kind": "image"}],
            )
        )
        self.assertFalse(result.fatal)
        assert result.ir_text is not None
        self.assertIn(
            "The visible character identified by <Picture 1>",
            result.ir_text,
        )
        diagnostic = next(
            item
            for item in result.diagnostics
            if item.code == "IR_DIALOGUE_SPEAKER_ROLE_CONFLICT"
        )
        self.assertIn("excluded from character identity use", diagnostic.message)

    def test_direct_picture_speaker_with_character_role_has_no_conflict_warning(
        self,
    ) -> None:
        result = compile_request(
            request(
                "omni",
                prompt="Use <Picture 1> as a character reference.",
                dialogue="<Picture 1> <d>[English] hello</d>",
                references=[{"kind": "image"}],
            )
        )
        self.assertFalse(result.fatal)
        self.assertFalse(
            any(
                item.code == "IR_DIALOGUE_SPEAKER_ROLE_CONFLICT"
                for item in result.diagnostics
            )
        )

    def test_dialogue_is_immutable_tagged_once_and_scoped_to_cut4(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt="Cut1\nwide\nCut2\nclose\nCut3\nsea\nCut4\nfinal",
                duration_seconds=12,
                num_frames=None,
                dialogue="Cut4\n（低い女性の声） <Audio 1> 「うっふ～ん・・・。」",
                references=[{"kind": "image"}, {"kind": "audio"}],
            )
        )
        self.assertFalse(result.fatal)
        assert result.ir_text is not None
        tag = "<d>[Japanese] うっふ～ん・・・。</d>"
        self.assertEqual(result.ir_text.count(tag), 1)
        shot4 = result.ir_text.index("[Shot 4] At 00:09.000")
        self.assertGreater(result.ir_text.index(tag), shot4)
        soundscape = result.ir_text.split("overall_soundscape:\n", 1)[1]
        self.assertNotIn("うっふ～ん", soundscape)
        self.assertIn("No narrator, voice-over, additional words", result.ir_text)

    def test_multiple_quoted_utterances_are_each_emitted_once(self) -> None:
        result = compile_request(
            request(dialogue="「はい。」そして「行きます。」")
        )
        assert result.ir_text is not None
        self.assertEqual(result.ir_text.count("<d>[Japanese] はい。</d>"), 1)
        self.assertEqual(result.ir_text.count("<d>[Japanese] 行きます。</d>"), 1)
        self.assertEqual(result.ir_text.count("<d>"), 2)

    def test_each_dialogue_event_keeps_its_own_audio_reference(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt="Two speakers alternate.",
                dialogue=(
                    "<Audio 1> <d>[English] first</d> "
                    "<Audio 2> <d>[English] second</d>"
                ),
                references=[{"kind": "audio"}, {"kind": "audio"}],
            )
        )
        self.assertFalse(result.fatal)
        assert result.document is not None
        events = tuple(
            event for shot in result.document.shots for event in shot.dialogue_events
        )
        self.assertEqual(
            [event.audio_reference.text for event in events if event.audio_reference],
            ["<Audio 1>", "<Audio 2>"],
        )
        assert result.ir_text is not None
        self.assertEqual(result.ir_text.count("The delivery uses <Audio 1>"), 1)
        self.assertEqual(result.ir_text.count("The delivery uses <Audio 2>"), 1)

    def test_identical_repeated_utterances_remain_valid_and_are_each_emitted(self) -> None:
        result = compile_request(request(dialogue="「はい。」「はい。」"))
        self.assertFalse(result.fatal)
        assert result.ir_text is not None
        self.assertEqual(result.ir_text.count("<d>[Japanese] はい。</d>"), 2)
        self.assertFalse(
            any(item.code == "IR_DIALOGUE_NOT_EXACTLY_ONCE" for item in result.diagnostics)
        )

    def test_every_reserved_h3_token_inside_dialogue_is_fatal(self) -> None:
        reserved_tokens = (
            "<d>",
            "</d>",
            "<|cutoff|>",
            "<|lyrics_start|>",
            "<|lyrics_end|>",
            "<|caption_start|>",
            "<|caption_end|>",
        )
        for token in reserved_tokens:
            with self.subTest(token=token):
                result = compile_request(request(dialogue=f"hello {token} injected"))
                self.assertTrue(result.fatal)
                self.assertIsNone(result.ir_text)
                self.assertTrue(
                    any(
                        item.code == "IR_DIALOGUE_TOKEN_INJECTION"
                        for item in result.diagnostics
                    )
                )

    def test_every_reserved_h3_token_inside_voice_direction_is_fatal(self) -> None:
        reserved_tokens = (
            "<d>",
            "</d>",
            "<|cutoff|>",
            "<|lyrics_start|>",
            "<|lyrics_end|>",
            "<|caption_start|>",
            "<|caption_end|>",
        )
        for token in reserved_tokens:
            with self.subTest(token=token):
                result = compile_request(request(dialogue=f"({token}) hello"))
                self.assertTrue(result.fatal)
                self.assertIsNone(result.ir_text)
                self.assertTrue(
                    any(
                        item.code == "IR_DIALOGUE_TOKEN_INJECTION"
                        and item.path.endswith("/voice_direction")
                        for item in result.diagnostics
                    )
                )

    def test_missing_dialogue_target_fails_open_to_last_shot_with_note(self) -> None:
        result = compile_request(
            request(
                prompt="Cut1\none\nCut2\ntwo",
                dialogue="Cut9\n「最後。」",
            )
        )
        self.assertFalse(result.fatal)
        assert result.ir_text is not None
        self.assertGreater(result.ir_text.index("<d>[Japanese] 最後。</d>"), result.ir_text.index("[Shot 2]"))
        self.assertTrue(any(item.code == "DIALOGUE_TARGET_RESOLVED" for item in result.auto_adjustments))


class ReferencePolicyTests(unittest.TestCase):
    def test_omni_uses_official_six_sections(self) -> None:
        result = compile_request(
            request("omni", references=[{"kind": "image"}])
        )
        assert result.ir_text is not None
        self.assertEqual(positions(result.ir_text, OMNI_SECTION_HEADERS), sorted(positions(result.ir_text, OMNI_SECTION_HEADERS)))
        self.assertNotIn("integrated_multimodal_description:", result.ir_text)

    def test_source_outfit_is_overridden_but_identity_and_body_are_preserved(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt="<Picture 1> wears the target red jacket.",
                references=[{"kind": "image"}],
            )
        )
        assert result.ir_text is not None
        self.assertIn("character identity and body-shape reference", result.ir_text)
        self.assertIn("replace the source outfit completely", result.ir_text)
        self.assertIn("source outfit must not override the target wardrobe", result.ir_text)
        self.assertTrue(any(item.code == "SOURCE_OUTFIT_OVERRIDE_POLICY" for item in result.auto_adjustments))

    def test_source_outfit_can_be_preserved_explicitly(self) -> None:
        result = compile_request(
            request(
                "omni",
                source_outfit_policy="preserve",
                references=[{"kind": "image"}],
            )
        )
        assert result.ir_text is not None
        self.assertIn("Preserve the source clothing", result.ir_text)
        self.assertIn("fully_preserved", result.ir_text)

    def test_clothing_only_picture_does_not_invent_character_identity(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt=(
                    "Use <Picture 1> only as a clothing design reference. "
                    "Do not copy the pictured person identity or body."
                ),
                references=[{"kind": "image"}],
            )
        )
        self.assertFalse(result.fatal)
        assert result.ir_text is not None
        self.assertIn("<Picture 1> is a clothing/costume design reference only", result.ir_text)
        self.assertNotIn(
            "<Subject 1> is the character identity and body-shape reference derived from <Picture 1>",
            result.ir_text,
        )
        self.assertFalse(
            any(
                item.code == "SOURCE_OUTFIT_OVERRIDE_POLICY"
                for item in result.auto_adjustments
            )
        )

    def test_ambiguous_picture_stays_neutral(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt="Use <Picture 1> as inspiration.",
                references=[{"kind": "image"}],
            )
        )
        self.assertFalse(result.fatal)
        assert result.ir_text is not None
        self.assertIn("<Picture 1> is a neutral visual reference", result.ir_text)
        self.assertFalse(
            any(
                item.code == "SOURCE_OUTFIT_OVERRIDE_POLICY"
                for item in result.auto_adjustments
            )
        )

    def test_adjacent_picture_roles_bind_to_the_correct_tag_in_both_orders(self) -> None:
        cases = (
            (
                "Use the clothing design from <Picture 1> and the character identity "
                "from <Picture 2>.",
                {"<Picture 1>": "clothing/costume", "<Picture 2>": "character identity"},
            ),
            (
                "Use character identity from <Picture 1> and clothing design "
                "from <Picture 2>.",
                {"<Picture 1>": "character identity", "<Picture 2>": "clothing/costume"},
            ),
            (
                "衣装デザインは<Picture 1>を参照し、キャラクター本人は<Picture 2>を参照。",
                {"<Picture 1>": "clothing/costume", "<Picture 2>": "character identity"},
            ),
            (
                "キャラクター本人は<Picture 1>を参照し、衣装は<Picture 2>を参照。",
                {"<Picture 1>": "character identity", "<Picture 2>": "clothing/costume"},
            ),
            (
                "衣装は<Picture 1>を使い、キャラクターは<Picture 2>を使う。",
                {"<Picture 1>": "clothing/costume", "<Picture 2>": "character identity"},
            ),
            (
                "The clothing should match <Picture 1>, while character identity should "
                "match <Picture 2>.",
                {"<Picture 1>": "clothing/costume", "<Picture 2>": "character identity"},
            ),
            (
                "The character identity should match <Picture 1>, while clothing should "
                "match <Picture 2>.",
                {"<Picture 1>": "character identity", "<Picture 2>": "clothing/costume"},
            ),
            (
                "For clothing, use <Picture 1>; for identity, use <Picture 2>.",
                {"<Picture 1>": "clothing/costume", "<Picture 2>": "character identity"},
            ),
            (
                "Use <Picture 1> with character identity from <Picture 2>.",
                {"<Picture 1>": "neutral visual", "<Picture 2>": "character identity"},
            ),
            (
                "Use <Picture 1> together with clothing design from <Picture 2>.",
                {"<Picture 1>": "neutral visual", "<Picture 2>": "clothing/costume"},
            ),
            (
                "Combine <Picture 1> and the character identity in <Picture 2>.",
                {"<Picture 1>": "neutral visual", "<Picture 2>": "character identity"},
            ),
        )
        for prompt, expected in cases:
            with self.subTest(prompt=prompt):
                result = compile_request(
                    request(
                        "omni",
                        prompt=prompt,
                        references=[{"kind": "image"}, {"kind": "image"}],
                    )
                )
                self.assertFalse(result.fatal)
                assert result.document is not None
                definitions = {
                    item.source_label.text: item.text
                    for item in result.document.subject_definitions
                    if item.source_label is not None
                }
                for label, phrase in expected.items():
                    self.assertIn(phrase, definitions[label])

    def test_standalone_audio_defaults_to_timbre_only(self) -> None:
        result = compile_request(
            request(
                "omni",
                references=[{"kind": "image"}, {"kind": "audio"}],
            )
        )
        assert result.ir_text is not None
        self.assertIn("<Audio 1> is a voice-timbre and delivery reference", result.ir_text)
        self.assertIn("Do not copy, continue, paraphrase, or reuse words", result.ir_text)
        self.assertIn("<Audio 1>: reference", result.ir_text)

    def test_explicit_standalone_audio_reuse_overrides_timbre_default(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt=(
                    "Reuse <Audio 1> exactly as the target soundtrack. "
                    "Do not use it as a voice timbre reference."
                ),
                references=[{"kind": "image"}, {"kind": "audio"}],
            )
        )
        self.assertFalse(result.fatal)
        assert result.ir_text is not None
        self.assertIn("<Audio 1>: partially_copy", result.ir_text)
        self.assertIn("audio source explicitly selected for partial signal reuse", result.ir_text)
        self.assertNotIn("<Audio 1> is a voice-timbre", result.ir_text)

    def test_standalone_audio_reuse_policy_is_scoped_to_its_own_tag(self) -> None:
        cases = (
            (
                "Do not reuse <Audio 1>; reuse <Audio 2> exactly.",
                {"<Audio 1>": "timbre", "<Audio 2>": "reuse"},
            ),
            (
                "<Audio 1> should be reused. <Audio 2> must not be reused.",
                {"<Audio 1>": "reuse", "<Audio 2>": "timbre"},
            ),
        )
        for prompt, expected in cases:
            with self.subTest(prompt=prompt):
                result = compile_request(
                    request(
                        "omni",
                        prompt=prompt,
                        references=[
                            {"kind": "image"},
                            {"kind": "audio"},
                            {"kind": "audio"},
                        ],
                    )
                )
                self.assertFalse(result.fatal)
                assert result.document is not None
                audio_policies = {
                    item.label.text: item.audio_policy.value
                    for item in result.document.references
                    if item.label.kind.value == "Audio"
                }
                self.assertEqual(audio_policies, expected)

    def test_reference_source_index_tracks_the_original_upload_order(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt="Use all supplied references.",
                references=[
                    {"kind": "image"},
                    {"kind": "video"},
                    {"kind": "audio"},
                ],
            )
        )
        self.assertFalse(result.fatal)
        assert result.document is not None
        standalone = next(
            item
            for item in result.document.references
            if item.origin.value == "standalone_audio"
        )
        self.assertEqual(standalone.source_index, 2)

    def test_audio_reuse_is_partial_copy_not_timbre(self) -> None:
        result = compile_request(
            request(
                "omni",
                audio_reference_policy="reuse",
                references=[{"kind": "image"}, {"kind": "audio"}],
            )
        )
        assert result.ir_text is not None
        self.assertIn("[reference generation + audio reuse]", result.ir_text)
        self.assertIn("<Audio 1>: partially_copy", result.ir_text)
        self.assertNotIn("Use only speaker timbre", result.ir_text)

    def test_video_embedded_audio_auto_defaults_to_ignore(self) -> None:
        result = compile_request(
            request("omni", references=[{"kind": "video"}])
        )
        self.assertEqual(result.embedded_video_audio_policy, "ignore")
        assert result.document is not None
        labels = [item.label.text for item in result.document.references]
        self.assertEqual(labels, ["<Video 1>"])
        self.assertTrue(any(item.code == "EMBEDDED_VIDEO_AUDIO_IGNORED" for item in result.auto_adjustments))

    def test_standalone_audio_tag_does_not_enable_embedded_video_audio(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt="<Video 1> movement, <Audio 1> voice timbre.",
                embedded_video_audio_policy="auto",
                references=[{"kind": "video"}, {"kind": "audio"}],
            )
        )
        self.assertFalse(result.fatal)
        self.assertEqual(result.embedded_video_audio_policy, "ignore")
        assert result.document is not None
        self.assertEqual(
            [item.label.text for item in result.document.references],
            ["<Video 1>", "<Audio 1>"],
        )
        standalone = result.document.references[1]
        self.assertEqual(standalone.origin.value, "standalone_audio")

    def test_video_embedded_audio_reference_is_numbered_before_standalone_audio(self) -> None:
        result = compile_request(
            request(
                "omni",
                embedded_video_audio_policy="reference",
                references=[{"kind": "video", "has_audio": True}, {"kind": "audio"}],
            )
        )
        self.assertEqual(result.embedded_video_audio_policy, "reference")
        assert result.document is not None
        audio = [
            item for item in result.document.references if item.label.kind.value == "Audio"
        ]
        self.assertEqual([item.label.text for item in audio], ["<Audio 1>", "<Audio 2>"])
        self.assertEqual(audio[0].origin.value, "embedded_video_audio")
        self.assertEqual(audio[1].origin.value, "standalone_audio")

    def test_video_audio_reuse_can_be_auto_detected(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt="Reuse the original video audio soundtrack.",
                embedded_video_audio_policy="auto",
                references=[{"kind": "video", "has_audio": True}],
            )
        )
        self.assertEqual(result.embedded_video_audio_policy, "reuse")
        assert result.ir_text is not None
        self.assertIn("<Audio 1>: partially_copy", result.ir_text)

    def test_negated_video_audio_requests_remain_ignored(self) -> None:
        prompts = (
            "Do not reuse the original video audio soundtrack.",
            "Never copy the reference video soundtrack.",
            "Preserve identity but ignore reference video audio.",
            "Ignore the video audio completely.",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                result = compile_request(
                    request(
                        "omni",
                        prompt=prompt,
                        embedded_video_audio_policy="auto",
                        references=[{"kind": "video"}],
                    )
                )
                self.assertFalse(result.fatal)
                self.assertEqual(result.embedded_video_audio_policy, "ignore")
                assert result.document is not None
                self.assertEqual(
                    [item.label.text for item in result.document.references],
                    ["<Video 1>"],
                )

    def test_video_audio_policy_is_resolved_per_video(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt=(
                    "<Video 2>の音声だけを声質参照にする。"
                    "<Video 1>の音声は使わない。"
                ),
                references=[
                    {"kind": "video", "has_audio": True},
                    {"kind": "video", "has_audio": True},
                ],
            )
        )
        self.assertFalse(result.fatal)
        self.assertEqual(result.embedded_video_audio_policy, "reference")
        self.assertEqual(result.embedded_video_audio_indices, (1,))
        assert result.document is not None
        self.assertEqual(
            [(item.label.text, item.source_index) for item in result.document.references],
            [("<Video 1>", 0), ("<Video 2>", 1), ("<Audio 1>", 1)],
        )

    def test_conjoined_video_audio_policies_bind_to_each_explicit_video(self) -> None:
        cases = (
            (
                "reference_then_reuse",
                "Use <Video 1> audio as a timbre reference and reuse <Video 2> audio.",
                "reuse",
                (0, 1),
                ((0, "as_prompted"), (1, "reuse")),
            ),
            (
                "reuse_then_reference_reverse_text_order",
                "Reuse <Video 2> audio and use <Video 1> audio as a timbre reference.",
                "reuse",
                (0, 1),
                ((0, "as_prompted"), (1, "reuse")),
            ),
            (
                "reuse_video1_reference_video2",
                "Reuse <Video 1> audio and use <Video 2> audio as a timbre reference.",
                "reuse",
                (0, 1),
                ((0, "reuse"), (1, "as_prompted")),
            ),
            (
                "reference_then_ignore",
                "Use <Video 1> audio as a timbre reference and ignore <Video 2> audio.",
                "reference",
                (0,),
                ((0, "as_prompted"),),
            ),
            (
                "ignore_then_reference_reverse_text_order",
                "Ignore <Video 2> audio and use <Video 1> audio as a timbre reference.",
                "reference",
                (0,),
                ((0, "as_prompted"),),
            ),
            (
                "ignore_video1_reference_video2",
                "Ignore <Video 1> audio and use <Video 2> audio as a timbre reference.",
                "reference",
                (1,),
                ((1, "as_prompted"),),
            ),
            (
                "reuse_then_negated_reuse",
                "Reuse <Video 1> audio and do not reuse <Video 2> audio.",
                "reuse",
                (0,),
                ((0, "reuse"),),
            ),
        )
        references = [
            {"kind": "video", "has_audio": True},
            {"kind": "video", "has_audio": True},
        ]
        for name, prompt, aggregate, indices, expected_audio in cases:
            with self.subTest(name=name):
                result = compile_request(
                    request("omni", prompt=prompt, references=references)
                )
                self.assertFalse(result.fatal)
                self.assertEqual(result.embedded_video_audio_policy, aggregate)
                self.assertEqual(result.embedded_video_audio_indices, indices)
                assert result.document is not None
                embedded_audio = tuple(
                    (item.source_index, item.audio_policy.value)
                    for item in result.document.references
                    if item.origin.value == "embedded_video_audio"
                )
                self.assertEqual(embedded_audio, expected_audio)

    def test_visual_video_tag_does_not_enable_its_unmentioned_soundtrack(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt=(
                    "Use <Video 1> for camera motion and "
                    "<Video 2> audio as a timbre reference."
                ),
                references=[
                    {"kind": "video", "has_audio": True},
                    {"kind": "video", "has_audio": True},
                ],
            )
        )
        self.assertFalse(result.fatal)
        self.assertEqual(result.embedded_video_audio_indices, (1,))

    def test_requesting_audio_from_a_silent_video_is_fatal(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt="Use <Video 1> audio as a voice timbre reference.",
                references=[{"kind": "video", "has_audio": False}],
            )
        )
        self.assertTrue(result.fatal)
        self.assertIsNone(result.ir_text)
        self.assertTrue(
            any(
                item.code == "IR_REFERENCE_VIDEO_HAS_NO_AUDIO"
                for item in result.diagnostics
            )
        )

    def test_requesting_unverified_embedded_audio_is_fatal(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt="Use <Video 1> audio as a voice timbre reference.",
                references=[{"kind": "video"}],
            )
        )
        self.assertTrue(result.fatal)
        self.assertIsNone(result.ir_text)
        self.assertTrue(
            any(
                item.code == "IR_REFERENCE_VIDEO_AUDIO_UNVERIFIED"
                for item in result.diagnostics
            )
        )

    def test_standalone_audio_reuse_does_not_promote_video_audio(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt=(
                    "Use the video audio as a timbre reference; "
                    "reuse <Audio 1> as the target soundtrack."
                ),
                references=[
                    {"kind": "video", "has_audio": True},
                    {"kind": "audio"},
                ],
            )
        )
        self.assertFalse(result.fatal)
        self.assertEqual(result.embedded_video_audio_policy, "reference")
        assert result.document is not None
        audios = [
            item
            for item in result.document.references
            if item.label.kind.value == "Audio"
        ]
        self.assertEqual(
            [item.audio_policy.value for item in audios],
            ["as_prompted", "reuse"],
        )

    def test_embedded_audio_opt_in_preserves_ui_standalone_audio_binding(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt=(
                    "Use <Video 1> audio as a timbre reference; "
                    "<Audio 1> is the standalone speaker voice."
                ),
                references=[{"kind": "video", "has_audio": True}, {"kind": "audio"}],
            )
        )
        self.assertFalse(result.fatal)
        self.assertEqual(result.embedded_video_audio_policy, "reference")
        assert result.document is not None
        audios = [
            item
            for item in result.document.references
            if item.label.kind.value == "Audio"
        ]
        self.assertEqual(
            [(item.label.text, item.origin.value) for item in audios],
            [
                ("<Audio 1>", "embedded_video_audio"),
                ("<Audio 2>", "standalone_audio"),
            ],
        )
        assert result.ir_text is not None
        self.assertIn("<Audio 2> is the standalone speaker voice", result.ir_text)
        self.assertTrue(
            any(
                item.code == "STANDALONE_AUDIO_LABELS_SHIFTED"
                for item in result.auto_adjustments
            )
        )

    def test_explicit_video_audio_reference_can_be_auto_detected(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt="<Video 1>の音声を声質だけの参照にする。",
                embedded_video_audio_policy="auto",
                references=[{"kind": "video", "has_audio": True}],
            )
        )
        self.assertFalse(result.fatal)
        self.assertEqual(result.embedded_video_audio_policy, "reference")
        assert result.document is not None
        self.assertEqual(
            [item.label.text for item in result.document.references],
            ["<Video 1>", "<Audio 1>"],
        )

    def test_unrelated_preserve_does_not_upgrade_video_audio_reference_to_reuse(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt=(
                    "Preserve character identity. "
                    "Use the video audio as a voice timbre reference."
                ),
                embedded_video_audio_policy="auto",
                references=[{"kind": "video", "has_audio": True}],
            )
        )
        self.assertFalse(result.fatal)
        self.assertEqual(result.embedded_video_audio_policy, "reference")
        assert result.ir_text is not None
        self.assertIn("<Audio 1>: reference", result.ir_text)
        self.assertNotIn("<Audio 1>: partially_copy", result.ir_text)

    def test_unknown_reference_against_upload_manifest_is_fatal(self) -> None:
        result = compile_request(
            request(
                "omni",
                prompt="Use <Picture 2>.",
                references=[{"kind": "image"}],
            )
        )
        self.assertTrue(result.fatal)
        self.assertIsNone(result.ir_text)
        self.assertTrue(any(item.code == "IR_UNKNOWN_REFERENCE_LABEL" for item in result.diagnostics))


class ResultAndArtifactTests(unittest.TestCase):
    def test_public_result_has_stable_diagnostic_contract(self) -> None:
        result = compile_request(request())
        public = result.to_public_dict()
        self.assertEqual(public["status"], "degraded")
        self.assertTrue(public["degraded"])
        self.assertFalse(public["fatal"])
        self.assertTrue(public["local_only"])
        self.assertFalse(public["model_inference"])
        self.assertIsInstance(public["auto_adjustments"], list)
        self.assertIsInstance(public["diagnostics"], list)
        for item in public["diagnostics"]:
            self.assertEqual(set(item), {"severity", "code", "message", "path", "fatal"})

    def test_provenance_is_local_deterministic_and_hashed(self) -> None:
        first = compile_request(request())
        second = compile_request(request())
        self.assertEqual(first.provenance.source_sha256, second.provenance.source_sha256)
        self.assertEqual(first.provenance.output_sha256, second.provenance.output_sha256)
        self.assertTrue(first.provenance.local_only)
        self.assertFalse(first.provenance.model_inference)

    def test_invalid_request_is_blocked_instead_of_raising(self) -> None:
        result = compile_request({"mode": "bad", "prompt": "x", "num_frames": 124})
        self.assertTrue(result.fatal)
        self.assertEqual(result.status, CompilationStatus.BLOCKED)
        self.assertIsNone(result.ir_text)

    def test_invalid_inline_dialogue_language_is_blocked_instead_of_raising(self) -> None:
        for dialogue in ("<d>[123] hi</d>", "<d>[日本語] hi</d>"):
            with self.subTest(dialogue=dialogue):
                result = compile_request(request(dialogue=dialogue))
                self.assertTrue(result.fatal)
                self.assertEqual(result.status, CompilationStatus.BLOCKED)
                self.assertIsNone(result.ir_text)
                self.assertTrue(
                    any(
                        item.code == "IR_COMPILE_INPUT_INVALID"
                        for item in result.diagnostics
                    )
                )

    def test_reserved_control_tokens_are_blocked_in_all_untrusted_text_fields(self) -> None:
        cases = (
            {"prompt": "A scene with <|cutoff|> injected."},
            {"soundscape": "Wind <|lyrics_start|>"},
            {"audio_direction": "Voice <|caption_start|>"},
            {"music_direction": "Score <|lyrics_end|>"},
            {"style_direction": "Look <|caption_end|>"},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                result = compile_request(request(**overrides))
                self.assertTrue(result.fatal)
                self.assertIsNone(result.ir_text)
                self.assertTrue(
                    any(
                        item.code == "IR_RESERVED_CONTROL_TOKEN"
                        for item in result.diagnostics
                    )
                )

    def test_valid_raw_ir_is_preserved_byte_for_byte(self) -> None:
        raw = (
            "integrated_multimodal_description:\r\n"
            "[Shot 1] A static view.\r\n\r\n"
            "overall_soundscape:\r\nQuiet room tone.\r\n\r\n"
            "non_diegetic_music:\r\nN/A\r\n"
        )
        result = compile_request(request(prompt=raw))
        self.assertEqual(result.status, CompilationStatus.RAW_VALIDATED)
        self.assertEqual(result.ir_text, raw)

    def test_raw_ir_reports_separate_controls_without_rewriting_or_blocking(self) -> None:
        raw = (
            "integrated_multimodal_description:\n"
            "[Shot 1] A static view.\n\n"
            "overall_soundscape:\nQuiet room tone.\n\n"
            "non_diegetic_music:\nN/A"
        )
        result = compile_request(
            request(
                prompt=raw,
                soundscape="A separately authored storm.",
                style="cinematic",
                music_policy="prominent",
            )
        )
        self.assertFalse(result.fatal)
        self.assertEqual(result.status, CompilationStatus.RAW_VALIDATED)
        self.assertEqual(result.ir_text, raw)
        warning = next(
            item
            for item in result.diagnostics
            if item.code == "IR_RAW_AUXILIARY_FIELDS_IGNORED"
        )
        self.assertEqual(warning.severity.value, "warning")
        self.assertIn("soundscape", warning.message)
        self.assertIn("style", warning.message)
        self.assertIn("music_policy", warning.message)

    def test_raw_ir_with_wrong_mode_schema_is_fatal(self) -> None:
        raw = "\n\n".join(f"{header}:\nvalue" for header in OMNI_SECTION_HEADERS)
        result = compile_request(request("t2v", prompt=raw))
        self.assertTrue(result.fatal)
        self.assertIsNone(result.ir_text)

    def test_raw_ir_with_unavailable_reference_is_fatal(self) -> None:
        raw = (
            "integrated_multimodal_description:\n"
            "[Shot 1] Use <Picture 1>.\n\n"
            "overall_soundscape:\nQuiet room tone.\n\n"
            "non_diegetic_music:\nN/A"
        )
        result = compile_request(request("t2v", prompt=raw))
        self.assertTrue(result.fatal)
        self.assertIsNone(result.ir_text)
        self.assertTrue(
            any(
                item.code == "IR_UNKNOWN_REFERENCE_LABEL"
                for item in result.diagnostics
            )
        )

    def test_raw_i2v_ir_accepts_its_implicit_first_picture(self) -> None:
        raw = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
            "integrated_multimodal_description:\n"
            "[Shot 1] Animate <Picture 1>.\n\n"
            "overall_soundscape:\nQuiet room tone.\n\n"
            "non_diegetic_music:\nN/A"
        )
        result = compile_request(request("i2v", prompt=raw))
        self.assertFalse(result.fatal)
        self.assertEqual(result.status, CompilationStatus.RAW_VALIDATED)
        self.assertEqual(result.ir_text, raw)

    def test_raw_ir_rejects_reserved_control_tokens_and_separate_dialogue(self) -> None:
        base = (
            "integrated_multimodal_description:\n"
            "[Shot 1] A static view.{suffix}\n\n"
            "overall_soundscape:\nQuiet room tone.\n\n"
            "non_diegetic_music:\nN/A"
        )
        injected = compile_request(
            request(prompt=base.format(suffix=" <|cutoff|>"))
        )
        separate = compile_request(
            request(
                prompt=base.format(suffix=""),
                dialogue="This must be spoken.",
            )
        )
        self.assertTrue(injected.fatal)
        self.assertTrue(separate.fatal)
        self.assertTrue(
            any(
                item.code == "IR_RESERVED_CONTROL_TOKEN"
                for item in injected.diagnostics
            )
        )
        self.assertTrue(
            any(
                item.code == "IR_RAW_WITH_SEPARATE_DIALOGUE"
                for item in separate.diagnostics
            )
        )

    def test_artifacts_are_atomic_json_safe_and_hashed(self) -> None:
        result = compile_request(request())
        with tempfile.TemporaryDirectory() as temporary:
            completed = write_artifacts(result, temporary)
            root = Path(temporary) / "context_ir"
            self.assertTrue((root / "final_ir.txt").is_file())
            self.assertTrue((root / "document.json").is_file())
            self.assertTrue((root / "diagnostics.json").is_file())
            self.assertTrue((root / "provenance.json").is_file())
            self.assertFalse(any(root.glob("*.tmp")))
            self.assertEqual(len(completed.artifacts), 4)
            json.loads((root / "document.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
