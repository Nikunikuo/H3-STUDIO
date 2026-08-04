from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.context_ir import compile_request


def omni_request(prompt: str, references: list[dict[str, object]]) -> dict[str, object]:
    return {
        "mode": "omni",
        "prompt": prompt,
        "num_frames": 124,
        "style": "natural",
        "dialogue": "",
        "soundscape": "",
        "music_policy": "none",
        "references": references,
    }


class GroupedReferencePolicyTests(unittest.TestCase):
    def test_grouped_reference_video_reuse_applies_to_every_soundtrack(self) -> None:
        cases = (
            ("Reuse <Video 1> audio and <Video 2> audio.", 2),
            (
                "Reuse <Video 1> audio, <Video 2> audio, and <Video 3> audio.",
                3,
            ),
            (
                "<Video 1> audio and <Video 2> audio should be reused as the soundtrack.",
                2,
            ),
            (
                "<Video 1> audio, <Video 2> audio, and <Video 3> audio should be "
                "reused as the soundtrack.",
                3,
            ),
        )
        for prompt, count in cases:
            with self.subTest(prompt=prompt):
                result = compile_request(
                    omni_request(
                        prompt,
                        [
                            {"kind": "video", "has_audio": True}
                            for _ in range(count)
                        ],
                    )
                )
                self.assertFalse(result.fatal)
                self.assertEqual(
                    result.embedded_video_audio_indices,
                    tuple(range(count)),
                )
                assert result.document is not None
                policies = [
                    item.audio_policy.value
                    for item in result.document.references
                    if item.origin.value == "embedded_video_audio"
                ]
                self.assertEqual(policies, ["reuse"] * count)

    def test_grouped_standalone_audio_reuse_applies_to_every_list_member(self) -> None:
        cases = (
            ("Reuse <Audio 1> and <Audio 2> as target soundtracks.", 2),
            ("Copy <Audio 1>, <Audio 2>, and <Audio 3> into the target soundtrack.", 3),
            ("<Audio 1> and <Audio 2> should be reused as target soundtracks.", 2),
            ("<Audio 1>, <Audio 2>, and <Audio 3> should be copied.", 3),
        )
        for prompt, count in cases:
            with self.subTest(prompt=prompt):
                result = compile_request(
                    omni_request(
                        prompt,
                        [{"kind": "image"}]
                        + [{"kind": "audio"} for _ in range(count)],
                    )
                )
                self.assertFalse(result.fatal)
                assert result.document is not None
                policies = [
                    item.audio_policy.value
                    for item in result.document.references
                    if item.label.kind.value == "Audio"
                ]
                self.assertEqual(policies, ["reuse"] * count)

    def test_grouped_audio_negation_is_not_upgraded_by_reuse_substring(self) -> None:
        result = compile_request(
            omni_request(
                "Do not reuse <Audio 1> and <Audio 2>.",
                [{"kind": "image"}, {"kind": "audio"}, {"kind": "audio"}],
            )
        )
        self.assertFalse(result.fatal)
        assert result.document is not None
        policies = [
            item.audio_policy.value
            for item in result.document.references
            if item.label.kind.value == "Audio"
        ]
        self.assertEqual(policies, ["timbre", "timbre"])

    def test_later_local_negation_overrides_a_grouped_reuse_for_that_tag(self) -> None:
        result = compile_request(
            omni_request(
                "Reuse <Audio 1> and <Audio 2> as soundtracks, but do not reuse <Audio 2>.",
                [{"kind": "image"}, {"kind": "audio"}, {"kind": "audio"}],
            )
        )
        self.assertFalse(result.fatal)
        assert result.document is not None
        policies = [
            item.audio_policy.value
            for item in result.document.references
            if item.label.kind.value == "Audio"
        ]
        self.assertEqual(policies, ["reuse", "timbre"])

    def test_later_local_reuse_overrides_an_earlier_grouped_negation(self) -> None:
        result = compile_request(
            omni_request(
                "Do not reuse <Audio 1> and <Audio 2>. Reuse <Audio 2> as the soundtrack.",
                [{"kind": "image"}, {"kind": "audio"}, {"kind": "audio"}],
            )
        )
        self.assertFalse(result.fatal)
        assert result.document is not None
        policies = [
            item.audio_policy.value
            for item in result.document.references
            if item.label.kind.value == "Audio"
        ]
        self.assertEqual(policies, ["timbre", "reuse"])

    def test_oxford_comma_character_list_shares_the_trailing_role(self) -> None:
        prompts = (
            "Use <Picture 1>, <Picture 2>, and <Picture 3> as character identity references.",
            "<Picture 1>、<Picture 2>、<Picture 3>のすべてをキャラクター参照にする。",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                result = compile_request(
                    omni_request(prompt, [{"kind": "image"} for _ in range(3)])
                )
                self.assertFalse(result.fatal)
                assert result.document is not None
                characters = [
                    item
                    for item in result.document.subject_definitions
                    if item.source_label is not None and item.subject_index is not None
                ]
                self.assertEqual(
                    [item.source_label.text for item in characters],
                    ["<Picture 1>", "<Picture 2>", "<Picture 3>"],
                )

    def test_oxford_comma_clothing_list_never_invents_character_identity(self) -> None:
        prompts = (
            "Use <Picture 1>, <Picture 2>, and <Picture 3> as clothing design references.",
            "<Picture 1>、<Picture 2>、<Picture 3>のすべてを衣装参照にする。",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                result = compile_request(
                    omni_request(prompt, [{"kind": "image"} for _ in range(3)])
                )
                self.assertFalse(result.fatal)
                assert result.document is not None
                picture_definitions = [
                    item
                    for item in result.document.subject_definitions
                    if item.source_label is not None
                ]
                self.assertEqual(len(picture_definitions), 3)
                self.assertTrue(
                    all(
                        "clothing/costume design reference only" in item.text
                        and item.subject_index is None
                        for item in picture_definitions
                    )
                )

    def test_mixed_picture_list_does_not_apply_the_last_role_to_every_tag(self) -> None:
        result = compile_request(
            omni_request(
                "Use <Picture 1>, <Picture 2> as character identity references, "
                "and <Picture 3> as a clothing reference.",
                [{"kind": "image"} for _ in range(3)],
            )
        )
        self.assertFalse(result.fatal)
        assert result.document is not None
        definitions = {
            item.source_label.text: item
            for item in result.document.subject_definitions
            if item.source_label is not None
        }
        self.assertIsNotNone(definitions["<Picture 1>"].subject_index)
        self.assertIsNotNone(definitions["<Picture 2>"].subject_index)
        self.assertIsNone(definitions["<Picture 3>"].subject_index)
        self.assertIn("clothing/costume", definitions["<Picture 3>"].text)

    def test_later_explicit_picture_role_overrides_an_earlier_group(self) -> None:
        result = compile_request(
            omni_request(
                "Use <Picture 1>, <Picture 2>, and <Picture 3> as character identity "
                "references. Use <Picture 3> only as a clothing reference.",
                [{"kind": "image"} for _ in range(3)],
            )
        )
        self.assertFalse(result.fatal)
        assert result.document is not None
        definitions = {
            item.source_label.text: item
            for item in result.document.subject_definitions
            if item.source_label is not None
        }
        self.assertIsNotNone(definitions["<Picture 1>"].subject_index)
        self.assertIsNotNone(definitions["<Picture 2>"].subject_index)
        self.assertIsNone(definitions["<Picture 3>"].subject_index)
        self.assertIn("clothing/costume", definitions["<Picture 3>"].text)

    def test_negative_identity_exception_removes_only_the_named_picture(self) -> None:
        prompts = (
            "Use <Picture 1>, <Picture 2>, and <Picture 3> as character identity "
            "references. Do not use <Picture 3> as a character identity reference.",
            "Use <Picture 1>, <Picture 2>, and <Picture 3> as character identity "
            "references, but do not preserve identity from <Picture 3>.",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                result = compile_request(
                    omni_request(prompt, [{"kind": "image"} for _ in range(3)])
                )
                self.assertFalse(result.fatal)
                assert result.document is not None
                definitions = {
                    item.source_label.text: item
                    for item in result.document.subject_definitions
                    if item.source_label is not None
                }
                self.assertIsNotNone(definitions["<Picture 1>"].subject_index)
                self.assertIsNotNone(definitions["<Picture 2>"].subject_index)
                self.assertIsNone(definitions["<Picture 3>"].subject_index)
                self.assertIn("neutral visual reference", definitions["<Picture 3>"].text)


if __name__ == "__main__":
    unittest.main()
