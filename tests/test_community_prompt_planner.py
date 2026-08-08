from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.community_prompt_planner import (  # noqa: E402
    MODEL_ID,
    MODEL_LOCK_FILENAME,
    MODEL_PROVENANCE_FILENAME,
    MODEL_RELATIVE_PATH,
    MODEL_REVISION,
    MODEL_RUNTIME_FILE_COUNT,
    MODEL_RUNTIME_TOTAL_BYTES,
    PLAN_SCHEMA_VERSION,
    SYSTEM_PROMPT,
    CommunityPromptPlannerError,
    build_model_messages,
    compile_model_result,
    has_explicit_source_dialogue,
    inspect_model_checkout,
    parse_plan_json,
    prepare_planner_input,
    require_verified_model_checkout,
)
from webui.community_prompt_worker import (  # noqa: E402
    _retry_instruction,
    process_request,
)


def _plan(
    *,
    action: str = "The athlete lifts the 120 kg barbell 3 times.",
    camera: str = "A locked low-angle camera looks upward without moving.",
    dialogue: bool = True,
    foley: list[str] | None = None,
    music: str = "A restrained instrumental pulse.",
) -> dict[str, object]:
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "style": "Polished cinematic realism with stable subject appearance.",
        "scene": "A focused athlete trains in a quiet modern gym.",
        "shots": [
            {
                "number": 1,
                "start_seconds": 0.0,
                "end_seconds": 5.0,
                "framing": "A medium full-body composition from below.",
                "camera": camera,
                "action": action,
            }
        ],
        "ambient": ["Quiet gym room tone and faint ventilation."],
        "foley": foley if foley is not None else ["Barbell plates clink on each lift."],
        "music": music,
        "dialogue_delivery": (
            [
                {
                    "dialogue_id": 1,
                    "shot": 1,
                    "start_seconds": 3.5,
                    "speaker": "the athlete",
                    "delivery": "confident, clear, and energetic",
                }
            ]
            if dialogue
            else []
        ),
    }


def _json(**kwargs: object) -> str:
    return json.dumps(_plan(**kwargs), ensure_ascii=False)


class CommunityPromptPlannerTests(unittest.TestCase):
    def _prepared(self, **kwargs: object):
        return prepare_planner_input(
            "Cut 1\n仰角で120kgのバーベルを3回持ち上げ、セリフ「頑張るぞ！」と言う。",
            duration_seconds=5,
            **kwargs,
        )

    def test_dialogue_is_redacted_before_qwen_and_never_uses_d_tags(self) -> None:
        prepared = self._prepared()
        self.assertEqual([item.text for item in prepared.dialogues], ["頑張るぞ！"])
        self.assertNotIn("頑張るぞ", prepared.redacted_prompt)
        messages = build_model_messages(prepared)
        self.assertNotIn("頑張るぞ", json.dumps(messages, ensure_ascii=False))
        self.assertIn("[DIALOGUE_1]", messages[1]["content"])
        self.assertNotIn("<Picture 1>", messages[1]["content"])

        compiled = compile_model_result(_json(), prepared)
        self.assertEqual(compiled.prompt.count('"頑張るぞ！"'), 1)
        self.assertNotIn("<d>", compiled.prompt.lower())
        without_dialogue = compiled.prompt.replace('"頑張るぞ！"', "")
        self.assertNotRegex(without_dialogue, r"[\u3040-\u30ff\u3400-\u9fff]")

    def test_explicit_unquoted_dialogue_field_is_also_hidden(self) -> None:
        literal = "今日は絶対に成功する！"
        prepared = prepare_planner_input(
            f"Cut 1\n主人公はカメラを見て、{literal}と力強く言う。",
            dialogue_texts=[literal],
            duration_seconds=5,
        )
        self.assertNotIn(literal, prepared.redacted_prompt)
        self.assertNotIn(literal, build_model_messages(prepared)[1]["content"])

    def test_reference_tag_variants_are_normalized_only_in_the_internal_copy(self) -> None:
        source = (
            "Cut 1\n<pICTURE1>の人物が<Ｖｉｄｅｏ　０１>の動きをまねる。"
        )
        prepared = prepare_planner_input(
            source,
            reference_inventory=[
                {"kind": "image", "index": 1},
                {"kind": "video", "index": 1},
            ],
            duration_seconds=5,
        )

        self.assertEqual(prepared.source_prompt, source)
        self.assertNotRegex(
            prepared.redacted_prompt,
            r"[<＜]\s*(?:picture|video|audio|subject|image|photo|sound)",
        )
        model_message = json.dumps(build_model_messages(prepared), ensure_ascii=False)
        self.assertNotRegex(
            model_message,
            r"[<＜]\s*(?:picture|video|audio|subject|image|photo|sound)",
        )
        self.assertIn(
            "SOURCE_REFERENCE_TAG_NORMALIZED",
            {warning.code for warning in prepared.source_warnings},
        )

    def test_unresolvable_reference_tag_variants_fail_before_model_invocation(self) -> None:
        cases = (
            ("<Picture 0>", "SOURCE_REFERENCE_TAG_UNSUPPORTED"),
            ("<Picture X>", "SOURCE_REFERENCE_TAG_UNSUPPORTED"),
            ("<Picture 2>", "SOURCE_REFERENCE_NOT_IN_INVENTORY"),
        )
        for tag, expected_code in cases:
            with self.subTest(tag=tag):
                with self.assertRaises(CommunityPromptPlannerError) as ctx:
                    prepare_planner_input(
                        f"Cut 1\n{tag}の人物が歩く。",
                        reference_inventory=[{"kind": "image", "index": 1}],
                        duration_seconds=5,
                    )
                self.assertEqual(ctx.exception.code, expected_code)

    def test_logo_quote_shape_is_control_prose_not_implicit_dialogue(self) -> None:
        source = """<Picture 1>のタイトルモーション。
最終フレームではタイトル文字「ファイナリーちゃん」の字形を維持する。
演出のテーマは「電子システムの起動」ではなく「古代紋章の覚醒」。
全体の流れは「暗闇」→「光」→「影」→「封印は解かれた」。
同じロゴ名「ファイナリーちゃん」を繰り返し表示する。
完成した「ファイナリーちゃん」のロゴを完全に見せる。"""
        prepared = prepare_planner_input(
            source,
            reference_inventory=[{"kind": "image", "index": 1}],
            duration_seconds=6,
            music_policy="none",
        )

        self.assertEqual(prepared.source_prompt, source)
        self.assertEqual(prepared.dialogues, ())
        compiled = compile_model_result(
            _json(
                dialogue=False,
                action="The title emblem emerges from darkness and settles into the reference composition.",
                camera="A fixed frontal camera holds the composition.",
            ),
            prepared,
        )
        self.assertTrue(compiled.prompt)
        self.assertEqual(compiled.metadata()["dialogue_count"], 0)

    def test_shared_source_dialogue_contract_matches_community_quote_classification(self) -> None:
        cases = (
            (
                "Cut 1\nタイトル「ファイナリーちゃん」を表示する。",
                "",
                False,
            ),
            (
                "Cut 1\nメイコが「行くよ！」と言う。",
                "",
                True,
            ),
            (
                "Cut 1\n<Picture 1> faces the camera.",
                "こんにちは。",
                True,
            ),
        )
        for prompt, dialogue, expected in cases:
            with self.subTest(prompt=prompt, dialogue=dialogue):
                prepared = prepare_planner_input(
                    prompt,
                    dialogue_texts=[dialogue] if dialogue else None,
                    reference_inventory=(
                        [{"kind": "image", "index": 1}]
                        if "<Picture 1>" in prompt
                        else None
                    ),
                    duration_seconds=5,
                )
                self.assertEqual(bool(prepared.dialogues), expected)
                self.assertEqual(
                    has_explicit_source_dialogue(
                        prompt,
                        prompt_processing_mode="community",
                        dialogue=dialogue,
                    ),
                    expected,
                )

        self.assertTrue(
            has_explicit_source_dialogue(
                'Cut 1\nThe title reads "Final Niku-chan".',
                prompt_processing_mode="raw_en",
            )
        )

    def test_real_logo_job_shape_keeps_repeated_title_quotes_out_of_dialogue(self) -> None:
        # Fixed representative copy of the production prompt shape; this test
        # intentionally does not read mutable webui_data job artifacts.
        source = """<Picture 1>のテキストモーション
参考画像を最終的な字形・ロゴデザイン・構図の絶対的な基準として使用し、6秒・16:9のダークファンタジー調タイトルロゴ出現アニメーションを制作する。
最終フレームでは、参考画像に存在するタイトル文字「ファイナリーちゃん」のスペル、字形、文字比率、文字間隔、大小関係、配置を正確に維持すること。
演出のテーマは「電子システムの起動」ではなく、
「長い年月、闇の中に封印されていた古代の紋章と聖剣が、不可視の力によってゆっくりと覚醒し、ひとつの荘厳な紋章として完成する瞬間」。
全体の流れは「暗闇」→「古代紋章の覚醒」→「文字が闇から削り出される」→「光」→「影」。
【2.0〜3.8秒｜「ファイナリーちゃん」の文字顕現】
タイトル文字「ファイナリーちゃん」が参考画像と同じ位置・字形・比率のまま出現する。
【5.2〜6.0秒｜荘厳な完成状態】
完成した「ファイナリーちゃん」のロゴを完全に見せる。
最後のフレームでは「封印は解かれた」「何かが目覚めた」「ここから物語が始まる」という余韻を残す。
絶対に避ける表現: 「ファイナリーちゃん」の誤字、文字の追加・削除、ロゴ配置の変更。"""
        prepared = prepare_planner_input(
            source,
            reference_inventory=[{"kind": "image", "index": 1}],
            duration_seconds=6,
            music_policy="none",
        )

        self.assertEqual(prepared.source_prompt, source)
        self.assertEqual(prepared.dialogues, ())
        self.assertEqual(prepared.visual_title_literals, ("ファイナリーちゃん",))
        model_input = json.dumps(build_model_messages(prepared), ensure_ascii=False)
        self.assertNotIn("ファイナリーちゃん", model_input)
        self.assertIn(
            "the exact visible title lettering from the supplied reference image",
            model_input,
        )
        compiled = compile_model_result(
            _json(
                dialogue=False,
                action="The dark fantasy title emblem emerges and settles into the supplied reference composition.",
                camera="A fixed frontal camera holds the logo composition.",
            ),
            prepared,
        )
        self.assertEqual(compiled.metadata()["dialogue_count"], 0)

    def test_reference_visual_title_echo_preserves_scene_and_action_semantics(self) -> None:
        source = (
            "Cut 1\n"
            "参考画像に存在するタイトル文字「ファイナリーちゃん」が暗闇から浮かび上がる。"
        )
        prepared = prepare_planner_input(
            source,
            reference_inventory=[{"kind": "image", "index": 1}],
            duration_seconds=5,
            music_policy="none",
        )
        raw = _plan(
            dialogue=False,
            action=(
                "The title 'ファイナリーちゃん' appears in full as engraved lettering; "
                "its edge emerges from darkness and settles into the reference composition."
            ),
        )
        raw["scene"] = (
            "A completely black background where the title 'ファイナリーちゃん' "
            "awakens as an ancient emblem."
        )

        compiled = compile_model_result(json.dumps(raw, ensure_ascii=False), prepared)
        action = compiled.plan.shots[0].action

        self.assertIn("emerges from darkness", action)
        self.assertIn("engraved lettering", action)
        self.assertNotEqual(
            action,
            "The visible action continues naturally with clear physical cause and effect.",
        )
        self.assertIn("completely black background", compiled.plan.scene)
        self.assertNotIn("ファイナリーちゃん", compiled.prompt)
        self.assertIn(
            "MODEL_VISUAL_TITLE_PRESERVED",
            {warning.code for warning in compiled.diagnostics()},
        )
        self.assertEqual(compiled.metadata()["dialogue_count"], 0)

    def test_unknown_japanese_subtitle_is_still_repaired_with_reference_title_present(self) -> None:
        source = (
            "Cut 1\n"
            "参考画像に存在するタイトル文字「ファイナリーちゃん」が暗闇から浮かび上がる。"
        )
        prepared = prepare_planner_input(
            source,
            reference_inventory=[{"kind": "image", "index": 1}],
            duration_seconds=5,
        )
        raw = _plan(
            dialogue=False,
            action="A readable subtitle appears under the title: 秘密の言葉.",
        )

        compiled = compile_model_result(json.dumps(raw, ensure_ascii=False), prepared)
        warning_codes = {warning.code for warning in compiled.diagnostics()}

        self.assertIn("MODEL_ONSCREEN_TEXT_REPAIRED", warning_codes)
        self.assertNotIn("秘密の言葉", compiled.prompt)
        self.assertNotIn("subtitle", compiled.plan.shots[0].action.casefold())

    def test_reference_title_exception_does_not_capture_dialogue_or_unrelated_quotes(self) -> None:
        source = (
            "Cut 1\n"
            "参考画像に存在するタイトル文字「ファイナリーちゃん」が暗闇から浮かび上がる。\n"
            "テーマ「暗闇」として演出し、画面には非参照の引用「秘密の言葉」を出さない。\n"
            "少女が「行くよ！」と言う。"
        )
        prepared = prepare_planner_input(
            source,
            reference_inventory=[{"kind": "image", "index": 1}],
            duration_seconds=5,
        )

        self.assertEqual(prepared.visual_title_literals, ("ファイナリーちゃん",))
        self.assertEqual([item.text for item in prepared.dialogues], ["行くよ！"])
        model_input = json.dumps(build_model_messages(prepared), ensure_ascii=False)
        self.assertNotIn("ファイナリーちゃん", model_input)
        self.assertNotIn("行くよ！", model_input)
        self.assertIn("the exact visible title lettering from the supplied reference image", model_input)

        compiled = compile_model_result(json.dumps(_plan(), ensure_ascii=False), prepared)
        self.assertEqual(compiled.prompt.count('"行くよ！"'), 1)
        self.assertNotIn("ファイナリーちゃん", compiled.prompt)

    def test_explicitly_attributed_quoted_dialogue_is_rendered_exactly_once(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n少女が「行くよ！」と言う。",
            duration_seconds=5,
        )
        compiled = compile_model_result(_json(), prepared)

        self.assertEqual([item.text for item in prepared.dialogues], ["行くよ！"])
        self.assertEqual(compiled.prompt.count('"行くよ！"'), 1)

    def test_named_character_and_latin_speaker_attribution_are_supported(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\nニケ「行くぞ！」\nLuna7:「開始！」\n"
            "タイトル「ファイナリーちゃん」\nテーマ:「暗闇」",
            duration_seconds=5,
        )

        self.assertEqual(
            [item.text for item in prepared.dialogues],
            ["行くぞ！", "開始！"],
        )

    def test_production_labels_with_quote_marks_remain_control_prose(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\nタイトル「ファイナリーちゃん」\n"
            "演出「古代紋章の覚醒」\n色「黒鉄と銀」\n"
            "カメラ「完全固定」\nロゴ「完成形」",
            duration_seconds=5,
        )

        self.assertEqual(prepared.dialogues, ())

    def test_dialogue_cue_does_not_cross_into_the_next_production_label_line(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\nニケが「行くぜ」と言う。\nテーマ「暗闇」で演出する。",
            duration_seconds=5,
        )

        self.assertEqual([item.text for item in prepared.dialogues], ["行くぜ"])

    def test_explicit_dialogue_texts_override_visual_quote_context(self) -> None:
        literal = "行くよ！"
        prepared = prepare_planner_input(
            f"Cut 1\n画面に「{literal}」を表示する。",
            dialogue_texts=[literal],
            duration_seconds=5,
        )
        compiled = compile_model_result(_json(), prepared)

        self.assertEqual([item.text for item in prepared.dialogues], [literal])
        self.assertEqual(compiled.prompt.count(f'"{literal}"'), 1)

    def test_duplicate_dialogue_words_outside_the_quote_are_redacted_by_position(self) -> None:
        literal = "行くよ！"
        prepared = prepare_planner_input(
            f"Cut 1\n少女が「{literal}」と言う。ポスターには{literal}という文字も残る。",
            duration_seconds=5,
        )

        self.assertNotIn(literal, prepared.redacted_prompt)
        compiled = compile_model_result(_json(), prepared)
        self.assertEqual(compiled.prompt.count(f'"{literal}"'), 1)

    def test_public_example_contract_uses_only_ordinary_quote_for_japanese(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n少女が楽しそうに一度だけ『ミニマックス、H3で遊んでみない？』と言う。",
            duration_seconds=5,
        )
        plan = _plan(
            action="The girl smiles toward the camera and gives a small inviting gesture.",
            camera="A locked eye-level medium shot holds steady.",
        )
        plan["dialogue_delivery"] = [
            {
                "dialogue_id": 1,
                "shot": 1,
                "start_seconds": 2.0,
                "speaker": "the girl",
                "delivery": "bright, friendly, and naturally paced",
            }
        ]
        compiled = compile_model_result(json.dumps(plan, ensure_ascii=False), prepared)
        self.assertIn('"ミニマックス、H3で遊んでみない？"', compiled.prompt)
        self.assertEqual(compiled.prompt.count("ミニマックス"), 1)
        self.assertNotIn("<d>", compiled.prompt.casefold())

    def test_reference_tags_are_python_owned_and_match_inventory_exactly(self) -> None:
        prepared = self._prepared(
            reference_inventory=[
                {"kind": "image", "index": 1, "tag": "<Picture 1>"},
                {"kind": "image", "index": 2},
                {"kind": "video", "index": 1},
                {"kind": "audio", "index": 1, "role": "voice"},
            ]
        )
        compiled = compile_model_result(_json(), prepared)
        for tag in ("<Picture 1>", "<Picture 2>", "<Video 1>", "<Audio 1>"):
            self.assertEqual(compiled.prompt.count(tag), 1)
        messages = json.dumps(build_model_messages(prepared), ensure_ascii=False)
        self.assertNotIn("<Picture", messages)
        self.assertNotIn("<Video", messages)
        self.assertNotIn("<Audio", messages)

    def test_source_reference_tags_are_described_to_qwen_then_restored_by_python(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n<Picture 1>の女性が、<Video 1>と同じ動きで手を振る。"
            "音の雰囲気は<Audio 1>を参考にする。",
            reference_inventory=[
                {"kind": "image", "index": 1},
                {"kind": "video", "index": 1},
                {"kind": "audio", "index": 1, "role": "ambience"},
            ],
            duration_seconds=5,
        )
        self.assertNotRegex(prepared.redacted_prompt, r"<(?:Picture|Video|Audio)\s")
        self.assertIn("supplied image reference number 1", prepared.redacted_prompt)
        self.assertIn("supplied video reference number 1", prepared.redacted_prompt)
        self.assertIn("supplied audio reference number 1", prepared.redacted_prompt)
        model_input = json.dumps(build_model_messages(prepared), ensure_ascii=False)
        self.assertNotIn("<Picture 1>", model_input)
        self.assertNotIn("<Video 1>", model_input)
        self.assertNotIn("<Audio 1>", model_input)

        compiled = compile_model_result(
            _json(
                dialogue=False,
                action="The woman follows the supplied motion reference and waves once.",
                camera="A locked eye-level medium shot holds steady.",
            ),
            prepared,
        )
        for tag in ("<Picture 1>", "<Video 1>", "<Audio 1>"):
            self.assertEqual(compiled.prompt.count(tag), 1)

    def test_source_reference_tag_must_exist_in_actual_inventory(self) -> None:
        with self.assertRaises(CommunityPromptPlannerError) as ctx:
            prepare_planner_input(
                "Cut 1\n<Picture 2>の人物が歩く。",
                reference_inventory=[{"kind": "image", "index": 1}],
                duration_seconds=5,
            )
        self.assertEqual(ctx.exception.code, "SOURCE_REFERENCE_NOT_IN_INVENTORY")

    def test_latest_workout_job_reference_never_reaches_model_messages(self) -> None:
        source = (
            "日本の高品質なセルアニメ調の塗り、演出。\n"
            "BGMなし。セリフなし。SEのみ。\n---\n"
            "<Picture 1>が、ジムで、トレーニングウェア"
            "（ショートパンツ、ランニング）を着て運動している。\n"
            "Cut1\nウォーキングマシン。斜め構図で仰角煽り視点。鼻息を荒く。\n"
            "Cut2\n両手ダンベルスクワット。ハァハァ息が上がっている。\n"
            "Cut4\nバーベルデッドリフト。重りは120kg。\n"
            "Cut5\nバーベル上げ。重量は200kg。\n"
            "女性キャラクターのセリフ（高いかわいいロリ声）"
            "「うおおおおおおおお！！！」\n"
            "Cut6\n暗転し、甘いプロテインドリンクを飲んで終了。"
        )
        prepared = prepare_planner_input(
            source,
            reference_inventory=[{"kind": "image", "index": 1}],
            duration_seconds=14.375,
        )
        model_input = json.dumps(build_model_messages(prepared), ensure_ascii=False)
        self.assertNotIn("<Picture 1>", model_input)
        self.assertIn("supplied image reference number 1", model_input)
        self.assertEqual(prepared.source_shot_labels, (1, 2, 4, 5, 6))
        self.assertEqual(prepared.source_shot_numbers, (1, 2, 3, 4, 5))
        self.assertIn(
            "SOURCE_SHOT_NUMBERING_NORMALIZED",
            {warning.code for warning in prepared.source_warnings},
        )
        self.assertEqual(
            {(fact.value, fact.unit) for fact in prepared.numeric_facts},
            {("120", "kg"), ("200", "kg")},
        )
        self.assertEqual(
            [dialogue.text for dialogue in prepared.dialogues],
            ["うおおおおおおおお！！！"],
        )
        self.assertEqual(
            prepared.dialogues[0].voice_direction,
            "a high-pitched, cute, youthful anime voice, delivered as a forceful "
            "exertion shout",
        )
        self.assertTrue(prepared.wardrobe_override)
        self.assertEqual(
            prepared.wardrobe_direction,
            "workout wear consisting of a tank top and shorts",
        )
        self.assertEqual(prepared.wardrobe_required_terms, ("tank top", "shorts"))
        self.assertEqual(
            {cue.english_sound for cue in prepared.nonverbal_cues},
            {"strained breathing and panting"},
        )
        self.assertIn("ウォーキングマシン", prepared.redacted_prompt)

    def test_explicit_speech_context_overrides_nonverbal_shape(self) -> None:
        literal = "うおおおおおおおお！！！"
        prepared = prepare_planner_input(
            f"Cut 1\n女性キャラクターのセリフ（高いかわいいロリ声）「{literal}」",
            duration_seconds=5,
        )
        self.assertEqual([item.text for item in prepared.dialogues], [literal])
        self.assertEqual(prepared.nonverbal_cues, ())
        self.assertNotIn(literal, build_model_messages(prepared)[1]["content"])
        self.assertEqual(
            prepared.dialogues[0].voice_direction,
            "a high-pitched, cute, youthful anime voice, delivered as a forceful "
            "exertion shout",
        )
        compiled = compile_model_result(_json(), prepared)
        self.assertIn(
            "with a high-pitched, cute, youthful anime voice, delivered as a "
            "forceful exertion shout",
            compiled.prompt,
        )
        self.assertNotIn("warm, clear, and conversational", compiled.prompt)
        self.assertTrue(
            compiled.metadata()["dialogue_voice_directions"][0][
                "deterministic_override"
            ]
        )

    def test_dialogue_texts_always_wins_for_an_interjection(self) -> None:
        literal = "ハァ、ハァ！"
        prepared = prepare_planner_input(
            "Cut 1\n人物が息を切らしている。",
            dialogue_texts=[literal],
            duration_seconds=5,
        )
        self.assertEqual([item.text for item in prepared.dialogues], [literal])
        self.assertEqual(prepared.nonverbal_cues, ())

    def test_unquoted_sound_match_never_corrupts_walking_machine_word(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\nウォーキングマシンで歩き、ハァハァゼェゼェ息を切らす。",
            duration_seconds=5,
        )
        self.assertIn("ウォーキングマシン", prepared.redacted_prompt)
        self.assertNotIn("[NONVERBAL", prepared.redacted_prompt.split("で歩き", 1)[0])

    def test_authored_wardrobe_replaces_only_reference_clothing(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n<Picture 1>の女性がトレーニングウェア"
            "（ショートパンツ、ランニング）を着て運動する。",
            reference_inventory=[{"kind": "image", "index": 1}],
            duration_seconds=5,
        )
        self.assertTrue(prepared.wardrobe_override)
        missing = _plan(dialogue=False, action="The woman exercises in the gym.")
        repaired = compile_model_result(json.dumps(missing), prepared)
        self.assertIn(
            "SOURCE_WARDROBE_DETAIL_REPAIRED",
            {warning.code for warning in repaired.diagnostics()},
        )

        valid = _plan(
            dialogue=False,
            action="The woman wears a tank top and shorts while exercising in the gym.",
        )
        compiled = compile_model_result(json.dumps(valid), prepared)
        self.assertIn(
            "defines the primary visible subject's identity, face, body shape, and hair",
            compiled.prompt,
        )
        self.assertIn(
            "the wardrobe must follow the Scene and Shot instructions and replaces the "
            "reference outfit",
            compiled.prompt,
        )
        self.assertIn(
            "Wardrobe: The primary visible subject wears workout wear consisting of a "
            "tank top and shorts, replacing the reference outfit.",
            compiled.prompt,
        )
        self.assertNotIn(
            "<Picture 1> defines the primary visible subject identity and appearance",
            compiled.prompt,
        )

    def test_reference_outfit_is_preserved_when_no_new_clothing_is_authored(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n<Picture 1>の女性が手を振る。",
            reference_inventory=[{"kind": "image", "index": 1}],
            duration_seconds=5,
        )
        compiled = compile_model_result(
            _json(
                dialogue=False,
                action="The woman waves one hand.",
                camera="A locked eye-level camera holds steady.",
            ),
            prepared,
        )
        self.assertFalse(prepared.wardrobe_override)
        self.assertIn(
            "<Picture 1> defines the primary visible subject identity and appearance.",
            compiled.prompt,
        )

    def test_model_cannot_invent_a_reference_tag(self) -> None:
        prepared = self._prepared(reference_inventory=[])
        plan = _plan(action="The athlete from <Picture 9> lifts the 120 kg barbell 3 times.")
        compiled = compile_model_result(json.dumps(plan), prepared)
        self.assertNotIn("<Picture 9>", compiled.prompt)
        self.assertIn(
            "MODEL_CONTROL_TAGS_REMOVED",
            {warning.code for warning in compiled.diagnostics()},
        )

    def test_model_result_wrappers_and_extra_keys_are_recoverable(self) -> None:
        prepared = self._prepared()
        fenced = compile_model_result(
            f"preamble\n```json\n{_json()}\n```\ntrailing", prepared
        )
        self.assertIn(
            "MODEL_MARKDOWN_WRAPPER_REMOVED",
            {warning.code for warning in fenced.diagnostics()},
        )
        plan = _plan()
        plan["commentary"] = "Looks good"
        extra = compile_model_result(json.dumps(plan), prepared)
        self.assertIn(
            "MODEL_EXTRA_FIELDS_IGNORED",
            {warning.code for warning in extra.diagnostics()},
        )

    def test_non_english_control_from_model_is_repaired(self) -> None:
        prepared = self._prepared()
        plan = _plan(action="選手が120 kgのバーベルを3 times持ち上げる。")
        compiled = compile_model_result(json.dumps(plan, ensure_ascii=False), prepared)
        self.assertIn(
            "MODEL_CONTROL_TEXT_REPAIRED",
            {warning.code for warning in compiled.diagnostics()},
        )
        without_exact_dialogue = compiled.prompt.replace('"頑張るぞ！"', "")
        self.assertNotRegex(without_exact_dialogue, r"[\u3040-\u30ff\u3400-\u9fff]")

    def test_model_quoted_control_text_is_repaired(self) -> None:
        prepared = self._prepared()
        plan = _plan(action='The athlete mouths "頑張るぞ" while lifting 120 kg 3 times.')
        compiled = compile_model_result(json.dumps(plan, ensure_ascii=False), prepared)
        self.assertIn(
            "MODEL_CONTROL_TEXT_REPAIRED",
            {warning.code for warning in compiled.diagnostics()},
        )
        self.assertEqual(compiled.prompt.count('"頑張るぞ！"'), 1)

    def test_model_invented_readable_subtitle_is_repaired_with_a_warning(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\nA heroine looks toward the dark emblem.",
            duration_seconds=5,
        )
        plan = _plan(
            dialogue=False,
            action="A readable rune subtitle appears under her face.",
        )

        compiled = compile_model_result(json.dumps(plan), prepared)

        warning_codes = {warning.code for warning in compiled.diagnostics()}
        self.assertIn("MODEL_ONSCREEN_TEXT_REPAIRED", warning_codes)
        self.assertNotIn("subtitle", compiled.prompt.casefold())
        self.assertNotIn("rune", compiled.prompt.casefold())

    def test_numeric_values_and_units_are_deterministically_repaired(self) -> None:
        prepared = self._prepared()
        plan = _plan(action="The athlete repeatedly lifts the heavy barbell.")
        compiled = compile_model_result(json.dumps(plan), prepared)
        self.assertIn("120 kg", compiled.prompt)
        self.assertIn("3 times", compiled.prompt)
        self.assertIn(
            "SOURCE_NUMERIC_DETAIL_REPAIRED",
            {warning.code for warning in compiled.diagnostics()},
        )

    def test_source_cut_order_is_mandatory_and_model_overlap_is_repaired(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n静かな部屋。\nCut 2\n人物が歩く。", duration_seconds=5
        )
        plan = _plan(dialogue=False, action="The person waits quietly.")
        missing_cut = compile_model_result(json.dumps(plan), prepared)
        self.assertIn(
            "SOURCE_MODEL_SHOT_COUNT_DIFFERENCE",
            {warning.code for warning in missing_cut.diagnostics()},
        )

        plan["shots"] = [
            {
                "number": 1,
                "start_seconds": 0,
                "end_seconds": 3,
                "framing": "A medium shot.",
                "camera": "A locked eye-level camera.",
                "action": "The person waits quietly.",
            },
            {
                "number": 2,
                "start_seconds": 2.5,
                "end_seconds": 5,
                "framing": "A wide shot.",
                "camera": "A slow lateral tracking move.",
                "action": "The person walks across the room.",
            },
        ]
        compiled = compile_model_result(json.dumps(plan), prepared)
        self.assertEqual(compiled.plan.shots[0].start_seconds, 0)
        self.assertEqual(compiled.plan.shots[-1].end_seconds, 5)
        self.assertEqual(
            compiled.plan.shots[0].end_seconds,
            compiled.plan.shots[1].start_seconds,
        )
        self.assertTrue(compiled.metadata()["timeline_structure_repaired"])
        self.assertIn(
            "PLANNER_TIMELINE_REPAIRED",
            {warning.code for warning in compiled.diagnostics()},
        )

    def test_bracketed_shot_headers_are_preserved(self) -> None:
        prepared = prepare_planner_input(
            "[Shot 1]\n静かな部屋。\n[Shot 2]\n人物が歩く。", duration_seconds=5
        )
        self.assertEqual(prepared.source_shot_numbers, (1, 2))

    def test_fullwidth_parenthesized_cut_headers_are_preserved(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\uFF080.0-2.6\u79D2\uFF09\nChurch exterior.\n"
            "cut 2\nA hand reaches for the door.\n"
            "Cut3\nThe boss breaks through the ceiling.\n"
            "Cut4\nThe boss turns in profile.\n"
            "Cut5\nThe heroine raises her sword.",
            duration_seconds=14.4,
        )
        self.assertEqual(prepared.source_shot_labels, (1, 2, 3, 4, 5))
        self.assertEqual(prepared.source_shot_numbers, (1, 2, 3, 4, 5))

    def test_current_dark_fantasy_prompt_normalizes_duplicate_cut_without_rejection(self) -> None:
        source = """日本の高品質なセルアニメ調。
スタイル: HD 16:9、14.4秒のダークファンタジーゲームPV、
強いパース、深い青黒の影、冷たい月光、輪郭は明瞭。
参照素材と役割: <Picture 1> 主人公ニケ
<Picture 2> は暗い教会。
<Picture 3>敵のボス
暗い教会。断崖、崩れた橋、青い霧の洞窟、枯れ森を場所変更の基準にする。
登場人物と継続: 全編を通して人間はニケ一人だけ。敵は指定されたボスのみ。
タイムラインとカット割り: 0.0秒開始。

Cut 1
<Picture 2>の教会の外観を見上げる <Picture 1> のニケ。
cut 2
教会の扉へ手をかけるシーンのアップ。
Cut3
教会へ踏み込むと、上から天井を破り<Picture 3>が登場。
Cut4
ボスは謎の言語で一言しゃべる。字幕もルーン文字のような象形文字で表示。
Cut5
ニケが剣を構える。
Cut6
ボスが剣を振り回し、剣先をニケに向けてポーズ。
Cut6
ニケの顔から瞳へ寄り、瞳の中にボスの姿が映る。

音声: BGMなし、音楽なし、歌なし、台詞なし。
禁止: 画面内字幕、ロゴ、透かし、読めない文字。"""
        prepared = prepare_planner_input(
            source,
            reference_inventory=[
                {"kind": "image", "index": 1},
                {"kind": "image", "index": 2},
                {"kind": "image", "index": 3},
            ],
            duration_seconds=14.375,
            music_policy="none",
        )

        self.assertEqual(prepared.source_prompt, source)
        self.assertEqual(prepared.source_shot_labels, (1, 2, 3, 4, 5, 6, 6))
        self.assertEqual(prepared.source_shot_numbers, (1, 2, 3, 4, 5, 6, 7))
        self.assertIn("Cut 7\nニケの顔から瞳へ寄り", prepared.redacted_prompt)
        warnings = {warning.code for warning in prepared.source_warnings}
        self.assertEqual(
            warnings,
            {
                "SOURCE_SHOT_NUMBERING_NORMALIZED",
                "SOURCE_DURATION_BOUND_TO_FRAME_COUNT",
                "SOURCE_SPEECH_CONFLICT",
                "SOURCE_ONSCREEN_TEXT_CONFLICT",
            },
        )
        model_input = build_model_messages(prepared)[1]["content"]
        self.assertIn("Source Cut/Shot order: 1, 2, 3, 4, 5, 6, 7", model_input)

        shot_ends = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.375)
        plan = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "style": "A polished HD 16:9 cel-animated dark fantasy game trailer lasting 14.4 seconds.",
            "scene": "At 0.0 seconds, a ruined moonlit church establishes the setting.",
            "shots": [
                {
                    "number": number,
                    "start_seconds": 0.0 if number == 1 else shot_ends[number - 2],
                    "end_seconds": end,
                    "framing": "A distinct cinematic composition advances the confrontation.",
                    "camera": "The camera changes angle with a purposeful fast cinematic move.",
                    "action": "The heroine and boss advance through the next visual story beat without speech or text.",
                }
                for number, end in enumerate(shot_ends, start=1)
            ],
            "ambient": ["Cold wind reverberates through the ruined church."],
            "foley": ["Stone breaks, armor shifts, and swords cut through the air."],
            "music": "N/A",
            "dialogue_delivery": [],
        }
        compiled = compile_model_result(json.dumps(plan), prepared)
        self.assertEqual(tuple(shot.number for shot in compiled.plan.shots), tuple(range(1, 8)))
        self.assertEqual(compiled.metadata()["source_shot_labels"], [1, 2, 3, 4, 5, 6, 6])
        self.assertEqual(
            compiled.metadata()["source_shot_numbers"], [1, 2, 3, 4, 5, 6, 7]
        )

    def test_zero_length_final_shot_is_repaired_for_exact_dark_fantasy_shape(self) -> None:
        prepared = prepare_planner_input(
            "\n".join(f"Cut {number}\nVisual beat {number}." for number in range(1, 8)),
            duration_seconds=14.375,
        )
        shot_ends = (2.0, 4.0, 6.0, 8.0, 10.0, 14.375, 14.375)
        plan = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "style": "A polished HD 16:9 cel-animated dark fantasy trailer.",
            "scene": "A ruined moonlit church contains one heroine and one boss.",
            "shots": [
                {
                    "number": number,
                    "start_seconds": 0.0 if number == 1 else shot_ends[number - 2],
                    "end_seconds": end,
                    "framing": "A distinct cinematic composition advances the scene.",
                    "camera": "A purposeful cinematic camera move follows the action.",
                    "action": "The next visual story beat unfolds without speech or text.",
                }
                for number, end in enumerate(shot_ends, start=1)
            ],
            "ambient": ["Cold wind reverberates through the ruined church."],
            "foley": ["Stone and armor produce synchronized physical impacts."],
            "music": "N/A",
            "dialogue_delivery": [],
        }

        compiled = compile_model_result(json.dumps(plan), prepared)

        self.assertEqual(tuple(shot.number for shot in compiled.plan.shots), tuple(range(1, 8)))
        self.assertEqual(compiled.plan.shots[0].start_seconds, 0)
        self.assertEqual(compiled.plan.shots[-1].end_seconds, 14.375)
        self.assertTrue(
            all(shot.end_seconds > shot.start_seconds for shot in compiled.plan.shots)
        )
        self.assertTrue(
            all(
                left.end_seconds == right.start_seconds
                for left, right in zip(compiled.plan.shots, compiled.plan.shots[1:])
            )
        )
        self.assertTrue(compiled.timeline_repaired)

    def test_duration_none_does_not_error_on_a_valid_model_timeline(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\nA traveler crosses a ruined hall."
        )
        diagnostics: list[object] = []

        parsed = parse_plan_json(
            json.dumps(
                _plan(
                    dialogue=False,
                    action="The traveler crosses the ruined hall.",
                    camera="A neutral eye-level camera tracks the traveler.",
                )
            ),
            prepared,
            diagnostics=diagnostics,
        )

        self.assertIsNone(prepared.duration_seconds)
        self.assertEqual(parsed.shots[-1].end_seconds, 5.0)
        self.assertNotIn(
            "PLANNER_DURATION_NORMALIZED",
            {getattr(warning, "code", None) for warning in diagnostics},
        )

    def test_normalize_duration_false_preserves_a_differing_valid_timeline(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\nThe traveler enters the ruined hall.\nCut 2\nThe traveler finds a sealed door.",
            duration_seconds=10,
        )
        plan = _plan(
            dialogue=False,
            action="The traveler enters the ruined hall.",
            camera="A neutral eye-level camera tracks the traveler.",
        )
        plan["shots"] = [
            {
                "number": 1,
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "framing": "A medium composition follows the traveler.",
                "camera": "A lateral tracking camera follows the entrance.",
                "action": "The traveler enters the ruined hall.",
            },
            {
                "number": 2,
                "start_seconds": 2.0,
                "end_seconds": 5.0,
                "framing": "A wide composition reveals the sealed door.",
                "camera": "A gentle push-in approaches the door.",
                "action": "The traveler finds the sealed door.",
            },
        ]
        diagnostics: list[object] = []

        parsed = parse_plan_json(
            json.dumps(plan),
            prepared,
            normalize_duration=False,
            diagnostics=diagnostics,
        )

        self.assertEqual(
            [(shot.start_seconds, shot.end_seconds) for shot in parsed.shots],
            [(0.0, 2.0), (2.0, 5.0)],
        )
        self.assertNotIn(
            "PLANNER_DURATION_NORMALIZED",
            {getattr(warning, "code", None) for warning in diagnostics},
        )

    def test_normalize_duration_true_fits_a_differing_valid_timeline(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\nThe traveler enters the ruined hall.\nCut 2\nThe traveler finds a sealed door.",
            duration_seconds=10,
        )
        plan = _plan(
            dialogue=False,
            action="The traveler enters the ruined hall.",
            camera="A neutral eye-level camera tracks the traveler.",
        )
        plan["shots"] = [
            {
                "number": 1,
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "framing": "A medium composition follows the traveler.",
                "camera": "A lateral tracking camera follows the entrance.",
                "action": "The traveler enters the ruined hall.",
            },
            {
                "number": 2,
                "start_seconds": 2.0,
                "end_seconds": 5.0,
                "framing": "A wide composition reveals the sealed door.",
                "camera": "A gentle push-in approaches the door.",
                "action": "The traveler finds the sealed door.",
            },
        ]
        diagnostics: list[object] = []

        parsed = parse_plan_json(
            json.dumps(plan),
            prepared,
            normalize_duration=True,
            diagnostics=diagnostics,
        )

        self.assertEqual(
            [(shot.start_seconds, shot.end_seconds) for shot in parsed.shots],
            [(0.0, 4.0), (4.0, 10.0)],
        )
        self.assertIn(
            "PLANNER_DURATION_NORMALIZED",
            {getattr(warning, "code", None) for warning in diagnostics},
        )

    def test_contentless_model_shot_is_fatal_before_field_defaults(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\nA traveler crosses the ruined hall.",
            duration_seconds=5,
        )
        plan = _plan(dialogue=False)
        plan["shots"][0]["framing"] = ""
        plan["shots"][0]["camera"] = ""
        plan["shots"][0]["action"] = ""

        with self.assertRaises(CommunityPromptPlannerError) as ctx:
            parse_plan_json(json.dumps(plan), prepared)
        self.assertEqual(ctx.exception.code, "EMPTY_MODEL_SHOT_CONTENT")

    def test_isolated_empty_shot_fields_and_substantive_shot_strings_are_repaired(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\nA traveler crosses the ruined hall.",
            duration_seconds=5,
        )
        plan = _plan(
            dialogue=False,
            action="The traveler crosses the ruined hall.",
        )
        plan["shots"][0]["framing"] = ""
        plan["shots"][0]["camera"] = ""
        parsed = parse_plan_json(json.dumps(plan), prepared)
        self.assertIn("traveler", parsed.shots[0].action)
        self.assertTrue(parsed.shots[0].framing)
        self.assertTrue(parsed.shots[0].camera)

        string_shot = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "style": "Cinematic style.",
            "scene": "A ruined hall.",
            "shots": ["The traveler crosses the ruined hall."],
            "ambient": [],
            "foley": [],
            "music": "N/A",
            "dialogue_delivery": [],
        }
        parsed_string_shot = parse_plan_json(json.dumps(string_shot), prepared)
        self.assertIn("traveler", parsed_string_shot.shots[0].action)

    def test_irregular_cut_labels_are_generically_normalized_by_occurrence(self) -> None:
        cases = (
            ((1, 1), "Cut1\nfirst\nCut1\nsecond"),
            ((4, 7), "Cut4\nfirst\nCut7\nsecond"),
            ((9, 3, 8), "Shot 9\nfirst\nShot 3\nsecond\nShot 8\nthird"),
        )
        for labels, source in cases:
            with self.subTest(labels=labels):
                prepared = prepare_planner_input(source, duration_seconds=5)
                self.assertEqual(prepared.source_shot_labels, labels)
                self.assertEqual(
                    prepared.source_shot_numbers,
                    tuple(range(1, len(labels) + 1)),
                )
                self.assertIn(
                    "SOURCE_SHOT_NUMBERING_NORMALIZED",
                    {warning.code for warning in prepared.source_warnings},
                )

    def test_fatal_boundaries_remain_small_and_explicit(self) -> None:
        with self.assertRaises(CommunityPromptPlannerError) as empty:
            prepare_planner_input("", duration_seconds=5)
        self.assertEqual(empty.exception.code, "EMPTY_SOURCE_PROMPT")

        with self.assertRaises(CommunityPromptPlannerError) as missing_reference:
            prepare_planner_input(
                "Cut 1\n<Picture 2> is visible.",
                reference_inventory=[{"kind": "image", "index": 1}],
                duration_seconds=5,
            )
        self.assertEqual(missing_reference.exception.code, "SOURCE_REFERENCE_NOT_IN_INVENTORY")

        with self.assertRaises(CommunityPromptPlannerError) as unsafe_dialogue:
            prepare_planner_input(
                "Cut 1\nThe subject speaks.",
                dialogue_texts=['unsafe " literal'],
                duration_seconds=5,
            )
        self.assertEqual(unsafe_dialogue.exception.code, "UNSAFE_DIALOGUE_LITERAL")

        prepared = prepare_planner_input("Cut 1\nThe subject moves.", duration_seconds=5)
        with self.assertRaises(CommunityPromptPlannerError) as no_shots:
            compile_model_result(json.dumps({"style": "cinematic", "scene": "room"}), prepared)
        self.assertEqual(no_shots.exception.code, "NO_SHOTS")

        with self.assertRaises(CommunityPromptPlannerError) as empty_shot:
            prepare_planner_input("Cut 1\n\nCut 2\n", duration_seconds=5)
        self.assertEqual(empty_shot.exception.code, "EMPTY_SOURCE_SHOT_CONTENT")

    def test_independent_japanese_corpus_acceptance_oracle(self) -> None:
        """20+ authored scenarios × independent model faults use the real compiler."""

        cases = (
            ("lighthouse", "labels", "Cut 0\n港の灯台が点滅する。\nCut 999\n霧の中を船が進む。", [], "A lighthouse sweeps its beacon over a foggy harbor.", "A boat glides through the fog.", "A wide tracking camera follows the boat.", ["fog horn"], "auto", [], 5.0),
            ("desert_beast", "labels", "Shot 9\n砂漠の機械獣が砂を巻き上げる。\nShot 2\n旅人が槍を抜く。\nShot 2\n機械獣が突進する。", [], "A mechanical beast bursts from the dunes.", "The traveler draws a spear and braces.", "A low tracking camera follows the charge.", ["metal scrape"], "none", [], 5.0),
            ("deep_submersible", "labels", "【カット９】\n深海探査艇の窓に発光する魚群。\n（ショット ４）\n探査艇が海溝へ沈む。", [{"kind": "image", "index": 1}], "Bioluminescent fish circle a submersible.", "The submersible descends toward the trench.", "A vertical camera follows the descent.", ["hull creak"], "auto", [], 5.0),
            ("snow_cabin", "labels", "第３カット\n雪山の山小屋で火が揺れる。\n第４シーン\n登山者が扉を開ける。", [], "Firelight flickers inside the cabin.", "The mountaineer opens the frozen door.", "A handheld push-in approaches the doorway.", ["door groan"], "none", [], 5.0),
            ("masked_theater", "labels", "ｃｕｔ　６\n古い劇場の幕が上がる。\nSHOT #７\n仮面役者が舞台を横切る。", [], "The curtain rises over an empty stage.", "A masked actor crosses the stage sharply.", "A lateral dolly keeps the actor centered.", ["curtain rustle"], "auto", [], 5.0),
            ("greenhouse_prose", "prose", "夕暮れの温室で、botanistが枯れた花に水を与える。Cutという単語は説明にだけ登場する。", [], "A botanist waters a wilted flower.", "The botanist trims one stem and sees a new bud.", "A macro push-in reveals the bud.", ["water droplets"], "none", [], 5.0),
            ("night_tram", "labels", "音声条件: BGMなし、雨音のみ。\nScene 12\n路面電車が夜の坂道を登る。\nShot 1\n少年が窓を拭く。", [{"kind": "image", "index": 1}, {"kind": "image", "index": 2}], "A tram climbs a wet hillside.", "A boy wipes fog from the window.", "A reflective tracking shot moves beside the tram.", ["tram bell"], "none", [], 5.0),
            ("detective_conflict", "contradiction", "Cut 1\n探偵が『犯人は時計塔にいる』と叫ぶ。台詞なし。字幕を表示するが文字は禁止。", [], "A detective points toward a clock tower.", "The detective gestures urgently as the bell rings.", "A rapid push-in ends on the tower face.", ["bell strike"], "none", ["犯人は時計塔にいる"], 5.0),
            ("volcano_conflict", "contradiction", "Cut 1\nBGMは壮大に鳴らす。BGMなし。静止画のように立つが激しく走る。", [], "A runner stands at a volcanic crater.", "The runner sprints and leaps over a fissure.", "A fast orbit keeps the horizon readable.", ["ash gust"], "none", [], 5.0),
            ("market_conflict", "contradiction", "カット１\n俯瞰で見下ろす。煽りで見上げる。人物は一人、背後には群衆。", [{"kind": "image", "index": 1}], "A courier stands in a crowded night market.", "The courier pushes through the crowd with a sealed letter.", "A high view descends into a low reveal.", ["crowd murmur"], "auto", [], 5.0),
            ("temple_station_conflict", "contradiction", "Cut 1\n寺院から動かない。Cut 3\n一瞬で宇宙港へ場所変更。Cut 3\n人物は一人で群衆もいる。", [{"kind": "image", "index": 1}, {"kind": "image", "index": 2}], "A monk kneels inside a candlelit temple.", "The temple dissolves into an orbital port.", "A whip pan bridges the location change.", ["alarm chirp"], "auto", [], 5.0),
            ("witch_rune_conflict", "contradiction", "Scene 2\n魔女が謎の言語で話しルーン字幕を出す。台詞も字幕も歌も禁止。", [{"kind": "image", "index": 1}], "A witch raises a lantern beneath ancient trees.", "The witch mouths a warning as the lantern flares.", "A slow arc circles the lantern.", ["lantern crackle"], "none", [], 5.0),
            ("warehouse_motion_conflict", "contradiction", "Cut 1\n絶対に動かない静止演技。次の瞬間、壁を蹴って三回転する。", [], "A dancer holds a rigid silhouette in a warehouse.", "The dancer kicks the wall and completes three rotations.", "A locked frame snaps into an upward tilt.", ["shoe impact"], "none", [], 5.0),
            ("rounded_duration", "near_miss", "14.4秒のHD PV。Cut 1（0.0-2.6秒）\n海賊船が波を越える。Cut 2\n帆が裂ける。", [{"kind": "image", "index": 1}], "A pirate ship climbs a black wave.", "The sail tears and snaps in the wind.", "A crane rises from deck to mast.", ["canvas rip"], "none", [], 14.375),
            ("crlf_clockmaker", "near_miss", "Cut 1  \r\n\r\n  時計職人が歯車を組み立てる。  \r\nCut 2\r\n工房の窓から朝日が差す。", [], "A clockmaker assembles a brass mechanism.", "Sunlight enters as the mechanism starts turning.", "A macro rack focus shifts to the window.", ["gear clicks"], "auto", [], 5.0),
            ("missing_numbers", "near_miss", "Cut 7\n重量120kgの石を3回持ち上げる。数字は重要。", [], "An athlete prepares beside a heavy stone.", "The athlete completes a controlled lift.", "A low-angle push-in emphasizes effort.", ["stone scrape"], "none", [], 5.0),
            ("missing_foley", "near_miss", "Cut 1\nSEは鎖の擦れる音、靴音、雷鳴。BGMなし。看守が地下牢を歩く。", [], "A jailer walks through a damp underground cell block.", "The jailer unlocks an iron gate.", "A narrow tracking shot follows the jailer.", [], "none", [], 5.0),
            ("missing_wardrobe", "near_miss", "Cut 1\n<Picture 1>の王女が軍服から赤い雨合羽へ着替える。", [{"kind": "image", "index": 1}], "A princess waits on a station platform.", "The princess raises an umbrella in rain.", "A medium shot tracks the umbrella.", ["rain patter"], "auto", [], 5.0),
            ("missing_camera", "near_miss", "Cut 1\n明示: 俯瞰で城壁全体を見せる。", [], "A fortress wall surrounds a silent valley.", "The gate remains closed as banners pull in wind.", "A neutral eye-level camera observes the gate.", ["banner flap"], "auto", [], 5.0),
            ("robot_motion_refs", "materials", "Cut 1\n<Picture 1>のロボットが<Video 1>の動きをまねる。", [{"kind": "image", "index": 1}, {"kind": "video", "index": 1}], "A small robot waits in a machine room.", "The robot imitates the supplied motion.", "A compact tracking shot follows its arm.", ["servo whir"], "auto", [], 5.0),
            ("paper_airplane_zero_refs", "materials", "Cut 1\n白い紙飛行機が図書館の吹き抜けを滑空する。", [], "A paper airplane glides through a library atrium.", "The airplane lands on an open book.", "A floating camera follows the airplane.", ["paper flutter"], "none", [], 5.0),
            ("cat_repeated_ref", "materials", "Cut 1\n<Picture 1>の猫が<Picture 1>の窓辺で眠る。", [{"kind": "image", "index": 1}], "A cat sleeps beside a rainy window.", "The cat opens one eye as raindrops crawl down glass.", "A close-up pushes toward the eye.", ["soft purr"], "auto", [], 5.0),
            ("singer_audio_ref", "materials", "Shot 2\n<Picture 1>の歌手を<Picture 2>の舞台に置き、<Audio 1>の声質を参照する。", [{"kind": "image", "index": 1}, {"kind": "image", "index": 2}, {"kind": "audio", "index": 1, "role": "voice"}], "A singer stands on an empty theater stage.", "The singer turns toward the balcony.", "A slow dolly moves from the stage edge.", ["microphone handling"], "auto", [], 5.0),
            ("worker_se_only", "audio", "Cut 1\n工事現場で作業員がヘルメットを拾う。音楽なし、BGMなし、SEのみ。", [], "A worker searches a construction site.", "The worker picks up a helmet as a crane swings.", "A handheld follow shot stays close.", ["crane cable rattle"], "none", [], 5.0),
            ("snow_auto_music", "audio", "Cut 1\n雪原を走る少女。音楽指定は自動、環境音は風。", [{"kind": "image", "index": 1}], "A girl runs across a snowfield.", "She turns toward a distant cabin and keeps running.", "A wide lateral tracking shot crosses the snow.", ["footsteps in snow"], "auto", [], 5.0),
            ("soldier_dialogue", "dialogue", "Cut 1\n兵士が『ここは俺に任せろ！』と叫び、剣を構える。BGMなし。", [{"kind": "image", "index": 1}], "A soldier raises a sword before smoke.", "The soldier plants his feet and faces the threat.", "A low-angle push-in stops on the sword guard.", ["sword ring"], "none", ["ここは俺に任せろ！"], 5.0),
            ("mother_child_dialogue", "dialogue", "Cut 1\n母が『早く逃げて！』と叫ぶ。Cut 2\n子どもが『いやだ！』と答える。SEのみ。", [{"kind": "image", "index": 1}], "A mother reaches toward a child in a shaking house.", "The child answers and runs to the door.", "A whip pan connects the rooms.", ["window rattle"], "none", ["早く逃げて！", "いやだ！"], 5.0),
            ("fighter_breath", "dialogue", "Cut 1\n格闘家が『ハァ、ハァ』と息を切らしながら砂袋を殴る。台詞なし。", [], "A fighter strikes a hanging sandbag.", "The fighter exhales and lands three precise punches.", "A tight handheld camera tracks every impact.", ["strained breathing and panting", "sandbag thump"], "none", [], 5.0),
            ("librarian_audio_conflict", "dialogue", "Cut 1\n『静かにして』と囁く。BGMあり、BGMなし、音声なし。", [], "A librarian raises one finger in a silent hallway.", "The librarian gestures for silence and closes a door.", "A rack focus moves from finger to latch.", ["door latch"], "none", ["静かにして"], 5.0),
        )

        def model_json(case: tuple[object, ...], fault: str) -> str:
            name, category, source, references, scene, action, camera, foley, music_policy, dialogue_texts, duration = case
            has_dialogue = bool(dialogue_texts)
            plan = _plan(dialogue=has_dialogue, action=str(action), camera=str(camera), foley=list(foley), music="N/A")
            plan["scene"] = str(scene)
            plan["style"] = "Cinematic animation with clear physical motion and continuity."
            if fault == "wrapper":
                return "説明:\n" + "\x60\x60\x60json\n" + json.dumps({**plan, "commentary": "retry"}, ensure_ascii=False) + "\n\x60\x60\x60\n以上"
            if fault == "drift":
                return json.dumps({"cuts": [{"shot_number": "99", "start": "bad", "end": "0.0 sec", "composition": "medium shot", "camera_move": "eye-level camera", "description": "The visible event continues."}], "scene": str(scene)}, ensure_ascii=False)
            if fault == "unsafe":
                plan["shots"][0]["action"] = 'The subject repeats "不要な日本語" while moving.'
                plan["shots"][0]["camera"] = "A camera shows <Picture 99> from above."
            if fault == "drop":
                plan["shots"][0]["action"] = "The visible event continues."
                plan["shots"][0]["foley"] = []
            if fault == "timing":
                plan["shots"][0]["number"] = 77
                plan["shots"][0]["start_seconds"] = 3.5
                plan["shots"][0]["end_seconds"] = 3.5
            return json.dumps(plan, ensure_ascii=False)

        faults = {
            "labels": ("clean", "wrapper", "drift"),
            "prose": ("clean", "unsafe", "timing"),
            "contradiction": ("clean", "wrapper", "unsafe"),
            "near_miss": ("clean", "drop", "timing"),
            "materials": ("clean", "wrapper", "unsafe"),
            "audio": ("clean", "drop", "wrapper"),
            "dialogue": ("clean", "timing", "unsafe"),
        }
        counts = {category: 0 for category in faults}
        accepted = {category: 0 for category in faults}
        rejected: list[tuple[str, str, str]] = []
        for case in cases:
            name, category, source, references, _scene, _action, _camera, _foley, music_policy, dialogue_texts, duration = case
            for fault in faults[str(category)]:
                counts[str(category)] += 1
                try:
                    prepared = prepare_planner_input(str(source), reference_inventory=list(references), dialogue_texts=list(dialogue_texts) or None, duration_seconds=float(duration), music_policy=str(music_policy))
                    self.assertEqual(prepared.source_prompt, source)
                    compiled = compile_model_result(model_json(case, fault), prepared)
                    accepted[str(category)] += 1
                    self.assertTrue(compiled.prompt)
                    if fault in {"wrapper", "drift", "unsafe", "timing"}:
                        self.assertTrue(compiled.diagnostics(), f"{name}:{fault}")
                    for literal in dialogue_texts:
                        self.assertEqual(compiled.prompt.count(f'"{literal}"'), 1)
                except CommunityPromptPlannerError as exc:
                    rejected.append((str(name), fault, exc.code))

        self.assertGreaterEqual(len(cases), 20)
        self.assertGreaterEqual(sum(counts.values()), 50)
        self.assertEqual(rejected, [])
        self.assertEqual(counts, accepted)
        self.assertEqual(
            counts,
            {"labels": 18, "prose": 3, "contradiction": 18, "near_miss": 18, "materials": 12, "audio": 6, "dialogue": 12},
        )
        self.assertEqual(set(counts), {"labels", "prose", "contradiction", "near_miss", "materials", "audio", "dialogue"})

    def test_rounded_clip_duration_is_bound_to_frames_without_dropping_cut_times(self) -> None:
        prepared = prepare_planner_input(
            "14.4秒のPV。\nCut 1（0.0-2.6秒）\n導入。\nCut 2\n結末。",
            duration_seconds=14.375,
        )
        facts = {(fact.value, fact.unit) for fact in prepared.numeric_facts}
        self.assertNotIn(("14.4", "seconds"), facts)
        self.assertIn(("0", "seconds"), facts)
        self.assertIn(("2.6", "seconds"), facts)
        self.assertIn(
            "SOURCE_DURATION_BOUND_TO_FRAME_COUNT",
            {warning.code for warning in prepared.source_warnings},
        )

    def test_source_instruction_conflicts_are_advisory_warnings(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n"
            "The boss speaks one line in an unknown language.\n"
            "字幕を表示する。\n"
            "\u53f0\u8a5e\u306a\u3057\u3002\n"
            "\u7981\u6b62: \u753b\u9762\u5185\u5b57\u5e55\u3002",
            duration_seconds=5,
        )
        warnings = {item.code: item.to_dict() for item in prepared.source_warnings}
        self.assertEqual(
            set(warnings),
            {"SOURCE_SPEECH_CONFLICT", "SOURCE_ONSCREEN_TEXT_CONFLICT"},
        )
        self.assertTrue(all(item["severity"] == "warning" for item in warnings.values()))
        self.assertTrue(all(item["fatal"] is False for item in warnings.values()))

    def test_no_dialogue_clause_alone_is_not_treated_as_positive_speech(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n台詞なし。BGMなし。", duration_seconds=5
        )
        self.assertNotIn(
            "SOURCE_SPEECH_CONFLICT",
            {item.code for item in prepared.source_warnings},
        )

    def test_each_source_warning_axis_requires_both_polarities(self) -> None:
        cases = {
            "speech": (
                "少女が話す。",
                "台詞なし。",
                "少女が話す。台詞なし。",
                "SOURCE_SPEECH_CONFLICT",
            ),
            "onscreen": (
                "字幕を表示する。",
                "字幕は禁止。",
                "字幕を表示する。しかし字幕は禁止。",
                "SOURCE_ONSCREEN_TEXT_CONFLICT",
            ),
            "music": (
                "BGMを鳴らす。",
                "BGMなし。",
                "BGMを鳴らす。BGMなし。",
                "SOURCE_MUSIC_CONFLICT",
            ),
            "location": (
                "場所を変更して洞窟へ移動する。",
                "場所は変えない。",
                "場所は変えない。しかし場所を変更する。",
                "SOURCE_LOCATION_CONFLICT",
            ),
            "camera": (
                "カメラが一周してトラッキングする。",
                "カメラは完全固定。",
                "カメラは完全固定。しかし一周してトラッキングする。",
                "SOURCE_CAMERA_CONFLICT",
            ),
            "motion": (
                "静止した画面で始める。",
                "静止画のまま引き延ばさない。次に走る。",
                "絶対に動かない。次に走る。",
                "SOURCE_MOTION_CONFLICT",
            ),
        }
        for name, (positive, negative, contradiction, code) in cases.items():
            with self.subTest(axis=name, polarity="positive"):
                self.assertNotIn(
                    code,
                    {item.code for item in prepare_planner_input(positive, duration_seconds=5).source_warnings},
                )
            with self.subTest(axis=name, polarity="negative"):
                self.assertNotIn(
                    code,
                    {item.code for item in prepare_planner_input(negative, duration_seconds=5).source_warnings},
                )
            with self.subTest(axis=name, polarity="contradiction"):
                self.assertIn(
                    code,
                    {item.code for item in prepare_planner_input(contradiction, duration_seconds=5).source_warnings},
                )

    def test_natural_language_warning_gaps_use_local_context(self) -> None:
        """Keep unrelated nouns, sound effects, and subject actions out of warnings."""

        cases = {
            "music": {
                "positive": ("BGMを静かに流す。",),
                "negative": (
                    "BGMなし。噴水の水を流す効果音だけを残す。",
                    "音楽は禁止。風の音を鳴らし、足音を入れる。",
                ),
                "contradiction": (
                    "BGMを静かに流す。だがBGMなし。",
                    "音楽を小さく再生する。しかし音楽は不要。",
                ),
                "near_miss": (
                    "BGMの文字を画面に表示する。BGMなし。",
                    "BGMという語を説明するだけで、効果音は水を流す音にする。",
                ),
                "code": "SOURCE_MUSIC_CONFLICT",
            },
            "camera": {
                "positive": ("カメラが人物の周囲を一周して移動する。",),
                "negative": (
                    "カメラは完全固定。人物は廊下を走って移動する。",
                    "カメラは固定。主人公が階段を駆け上がり、敵の周囲を一周する。",
                ),
                "contradiction": (
                    "カメラは完全固定。しかしカメラが人物の周囲を一周して移動する。",
                    "カメラは固定。だがカメラがゆっくり周回する。",
                ),
                "near_miss": (
                    "人物が敵の周囲を一周して移動する。カメラは完全固定。",
                    "カメラは固定したまま、人物を追うように見せる。",
                ),
                "code": "SOURCE_CAMERA_CONFLICT",
            },
            "motion": {
                "positive": ("主人公は一瞬だけ動かない。",),
                "negative": (
                    "静止した標識が背景にある。主人公は激しく走る。",
                    "静止画の参考素材を使い、主人公は回転して敵を倒す。",
                ),
                "contradiction": (
                    "主人公は動かない。しかし主人公は激しく走る。",
                    "画面を完全に静止させる。次の瞬間、主人公が高速で走り出す。",
                ),
                "near_miss": (
                    "静止した背景の看板を映す。霧だけが揺れ、主人公は走る。",
                    "静止画のまま引き延ばさず、次のカットで主人公が走る。",
                ),
                "code": "SOURCE_MOTION_CONFLICT",
            },
            "speech": {
                "positive": (
                    "主人公が短く宣言する。",
                    "兵士が声を発声し、伝令が命令を言い渡す。",
                ),
                "negative": (
                    "画面に宣言文を表示する。台詞は禁止。",
                    "画面の文字を読みやすく表示し、発話は入れない。",
                ),
                "contradiction": (
                    "主人公が短く宣言する。しかし台詞は禁止する。",
                    "ナレーターが古い詩を読み上げる。発話は不要。",
                ),
                "near_miss": (
                    "画面に宣言文を表示する。台詞は禁止。",
                    "読み上げ用の文章を画面に表示するだけで、声は出さない。",
                ),
                "code": "SOURCE_SPEECH_CONFLICT",
            },
            "location": {
                "positive": (
                    "一つの場所に留まり、周囲を見渡す。",
                    "舞台から離れず、人物だけが歩く。",
                ),
                "negative": ("場所は変更しない。", "場所から移動せず、背景を維持する。"),
                "contradiction": (
                    "一つの場所から移動しない。しかし後半で場所を変更する。",
                    "舞台から離れない。だが後半で別の場所へ移る。",
                    "場所は変えないが、後半で場所を変更する。",
                ),
                "near_miss": (
                    "主人公はその場から動かない。しかしカメラだけが別の方向へパンする。",
                    "主人公は場所を変えずに歩き、背景は固定する。",
                ),
                "code": "SOURCE_LOCATION_CONFLICT",
            },
        }
        for axis, data in cases.items():
            code = data["code"]
            for polarity in ("positive", "negative", "near_miss"):
                for prompt in data[polarity]:
                    with self.subTest(axis=axis, polarity=polarity, prompt=prompt):
                        self.assertNotIn(
                            code,
                            {
                                item.code
                                for item in prepare_planner_input(prompt, duration_seconds=5).source_warnings
                            },
                        )
            for prompt in data["contradiction"]:
                with self.subTest(axis=axis, polarity="contradiction", prompt=prompt):
                    self.assertIn(
                        code,
                        {
                            item.code
                            for item in prepare_planner_input(prompt, duration_seconds=5).source_warnings
                        },
                    )

    def test_natural_language_contradictions_remain_nonfatal_in_worker(self) -> None:
        prompt = """Cut 1
BGMを静かに流す。しかしBGMなし。
カメラは完全固定。しかしカメラが周回する。
主人公は動かない。しかし主人公は激しく走る。
主人公が短く宣言する。しかし台詞は禁止。
一つの場所から移動しない。しかし後半で場所を変更する。"""
        response = process_request(
            {
                "prompt": prompt,
                "duration_seconds": 5,
                "music_policy": "none",
            },
            planner=lambda _: _json(dialogue=False),
        )
        self.assertTrue(response["ok"])
        diagnostics = {
            item["code"]: item
            for item in response["diagnostics"]
            if item["code"].startswith("SOURCE_")
        }
        self.assertTrue(
            {
                "SOURCE_MUSIC_CONFLICT",
                "SOURCE_CAMERA_CONFLICT",
                "SOURCE_MOTION_CONFLICT",
                "SOURCE_SPEECH_CONFLICT",
                "SOURCE_LOCATION_CONFLICT",
            }.issubset(diagnostics)
        )
        self.assertTrue(all(item["fatal"] is False for item in diagnostics.values()))

    def test_prohibited_stillness_phrases_are_not_motion_conflicts(self) -> None:
        for prompt in (
            "静止画のまま引き延ばさない。次に走る。",
            "静止画は禁止。次に走る。",
            "静止を避ける。次に走る。",
            "静止画のまま作らない。次に走る。",
        ):
            with self.subTest(prompt=prompt):
                self.assertNotIn(
                    "SOURCE_MOTION_CONFLICT",
                    {item.code for item in prepare_planner_input(prompt, duration_seconds=5).source_warnings},
                )

    def test_negative_onscreen_list_clause_stays_negative_without_comma_split(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n字幕、テロップ、読める文字は禁止。", duration_seconds=5
        )
        self.assertNotIn(
            "SOURCE_ONSCREEN_TEXT_CONFLICT",
            {item.code for item in prepared.source_warnings},
        )

    def test_final_hardening_keeps_semantic_warnings_local_and_advisory(self) -> None:
        cases = {
            "onscreen": {
                "near_miss": (
                    "字幕用に「封印解除」を表示する。台詞なし。",
                    "字幕を表示する。音声なし。",
                ),
                "conflict": (
                    "字幕を表示する。字幕は禁止。",
                    "画面内字幕を出す。しかし画面内字幕は不要。",
                ),
                "code": "SOURCE_ONSCREEN_TEXT_CONFLICT",
            },
            "camera": {
                "near_miss": (
                    "カメラは固定。パンのように髪が風で揺れる。",
                    "カメラは固定。パンを食べる音がする。",
                ),
                "conflict": (
                    "カメラは固定。しかしカメラを右へパンする。",
                    "カメラは固定。カメラがゆっくりパンしながら人物を追う。",
                ),
                "code": "SOURCE_CAMERA_CONFLICT",
            },
            "motion": {
                "near_miss": (
                    "静止した標識が背景にある。主人公は激しく走る。",
                    "静止画の参考素材を使い、主人公は飛び込んで敵を倒す。",
                ),
                "conflict": (
                    "主人公は絶対に動かない。しかし敵が走って接近する。",
                    "主人公は動かない。敵が駆けて飛び込んでくる。",
                ),
                "code": "SOURCE_MOTION_CONFLICT",
            },
            "location": {
                "near_miss": (
                    "主人公はその場から動かない。しかし次のカットで別の場所へ移る。",
                    "人物は同じ場所に留まる間、カット後は別の場所へ移る。",
                ),
                "conflict": (
                    "場面転換をせずに、次のカットで別の場所へ移る。",
                    "シーン転換しない。次のカットで別の場所に切り替える。",
                ),
                "code": "SOURCE_LOCATION_CONFLICT",
            },
        }
        for axis, data in cases.items():
            code = data["code"]
            for prompt in data["near_miss"]:
                with self.subTest(axis=axis, polarity="near_miss", prompt=prompt):
                    warnings = prepare_planner_input(prompt, duration_seconds=5).source_warnings
                    self.assertNotIn(code, {warning.code for warning in warnings})
            for prompt in data["conflict"]:
                with self.subTest(axis=axis, polarity="conflict", prompt=prompt):
                    warnings = prepare_planner_input(prompt, duration_seconds=5).source_warnings
                    matching = [warning for warning in warnings if warning.code == code]
                    self.assertEqual(len(matching), 1)
                    self.assertTrue(all(warning.to_dict()["fatal"] is False for warning in matching))

    def test_natural_japanese_source_warning_conflicts_are_detected(self) -> None:
        prepared = prepare_planner_input(
            "\n".join(
                (
                    "Cut 1",
                    "場所は絶対に変えない。次のカットで場所を変更する。",
                    "字幕、テロップ、読める文字は禁止。ただし字幕を表示する。",
                    "壮大なBGMを鳴らす。BGMなし。",
                    "カメラは完全固定。しかし次の瞬間に一周してトラッキングする。",
                )
            ),
            duration_seconds=5,
        )
        self.assertTrue(
            {
                "SOURCE_LOCATION_CONFLICT",
                "SOURCE_ONSCREEN_TEXT_CONFLICT",
                "SOURCE_MUSIC_CONFLICT",
                "SOURCE_CAMERA_CONFLICT",
            }.issubset({item.code for item in prepared.source_warnings})
        )

    def test_exact_dark_fantasy_job_prompt_reports_only_speech_and_onscreen_conflicts(self) -> None:
        prompt = """日本の高品質なセルアニメ調。
スタイル: HD 16:9、14.4秒のダークファンタジーゲームPV、
強いパース、深い青黒の影、冷たい月光、輪郭は明瞭。
参照素材と役割: <Picture 1> 主人公ニケ
<Picture 2> は暗い教会。
<Picture 3>敵のボス
暗い教会。断崖、崩れた橋、青い霧の洞窟、枯れ森を場所変更の基準にする。
登場人物と継続: 全編を通して人間はニケ一人だけ。敵は指定されたボスのみ。ニケの衣装、髪型、顔、体格、剣の形をカット間で維持する。画像参照を別々の人物へ取り違えない。
タイムラインとカット割り: 0.0秒開始。

Cut 1
<Picture 2>の教会の外観を見上げる <Picture 1> のニケ。
cut 2
教会の扉へ手をかけるシーンのアップ。
Cut3
教会で歩いて踏み込むと、急激な速度で上から天井を破り<Picture 3>が登場。
登場時の土煙、大きな着地音、重厚な動き、剣の構えなど、大物のボスが現れた状態を表す。
カメラの角度や追従、速度なども状況にあわせて荘厳に表現する。
Cut4
ボスは謎の言語で一言しゃべる。ボスの横顔、少しカメラはパンしながら。
字幕もルーン文字のような象形文字で謎の言葉の字幕。
Cut5
ニケが剣を構える。
Cut6
ボスも、大げさに剣を振り回し力を誇示し、剣先をニケに向けてポーズ。
Cut6
ニケの顔のアップから一気に瞳に寄るクローズアップ。瞳の中に、ボスの姿が映る。

各カットは前のカットより新しい場所、視点、人物状態、攻撃情報を加え、意味のない静止や間延びを作らない。
カメラ: Push In、Tracking Shot、Arc Shot、Tilt、Rack Focus、Pull Out、Whip Pan、Crash Zoomを自然な動作として連続させ、速いが被写体と接触点が読める速度にする。急な角度変更、前景越しの遮蔽、画面端からの侵入を積極的に使う。
音声: 風、砂利、洞窟の反響、枯れ枝、骨の組み上がる音、盾の擦れ、長柄武器の風切り、剣戟、短い衝撃音。BGMなし、音楽なし、歌なし、台詞なし。
禁止: 余計な人物、余計な敵、敵の増殖、別の髪色や衣装、武器の形状変化、画面内字幕、ロゴ、透かし、読めない文字、血の過剰表現、参照画像のコラージュ表示、静止画のままのカット、場所の無説明な飛び、二重露光、崩れた手足。"""
        prepared = prepare_planner_input(
            prompt,
            reference_inventory=[
                {"kind": "image", "index": 1},
                {"kind": "image", "index": 2},
                {"kind": "image", "index": 3},
            ],
            duration_seconds=14.375,
            music_policy="none",
        )
        warning_codes = {item.code for item in prepared.source_warnings}
        self.assertTrue(
            {
                "SOURCE_SPEECH_CONFLICT",
                "SOURCE_ONSCREEN_TEXT_CONFLICT",
            }.issubset(warning_codes)
        )
        for code in (
            "SOURCE_LOCATION_CONFLICT",
            "SOURCE_CAMERA_CONFLICT",
            "SOURCE_MUSIC_CONFLICT",
            "SOURCE_MOTION_CONFLICT",
        ):
            self.assertNotIn(code, warning_codes)
        self.assertTrue(all(item.to_dict()["fatal"] is False for item in prepared.source_warnings))

    def test_worker_exposes_source_warnings_without_rejecting_the_request(self) -> None:
        response = process_request(
            {
                "prompt": (
                    "Cut 1\nThe boss speaks one line.\n"
                    "字幕を表示する。\n"
                    "\u53f0\u8a5e\u306a\u3057\u3002\n"
                    "\u7981\u6b62: \u753b\u9762\u5185\u5b57\u5e55\u3002"
                ),
                "duration_seconds": 5,
                "music_policy": "none",
            },
            planner=lambda _: _json(dialogue=False),
        )
        self.assertTrue(response["ok"])
        self.assertEqual(
            {item["code"] for item in response["diagnostics"]},
            {"SOURCE_SPEECH_CONFLICT", "SOURCE_ONSCREEN_TEXT_CONFLICT"},
        )

    def test_camera_glossary_corrects_low_and_high_angle_meaning(self) -> None:
        low = self._prepared()
        bad_low = _plan(camera="A static high-angle camera looks downward.")
        repaired_low = compile_model_result(json.dumps(bad_low), low)
        self.assertIn("low-angle", repaired_low.plan.shots[0].camera)
        self.assertIn("looks upward", repaired_low.plan.shots[0].camera)

        high = prepare_planner_input("Cut 1\n俯瞰で部屋全体を映す。", duration_seconds=5)
        high_plan = _plan(
            dialogue=False,
            action="The person stands in the center of the room.",
            camera="A static high-angle camera looks downward over the full room.",
        )
        high_plan["shots"][0]["framing"] = "A wide overhead composition from above."
        parsed = parse_plan_json(json.dumps(high_plan), high)
        self.assertEqual(parsed.shots[0].number, 1)
        self.assertIn("low-angle upward view", SYSTEM_PROMPT)
        self.assertIn("high-angle downward view", SYSTEM_PROMPT)

    def test_unrequested_camera_ambiguity_is_warning_instead_of_render_blocker(self) -> None:
        neutral = prepare_planner_input("Cut 1\n人物が立っている。", duration_seconds=5)
        conflicts = (
            (
                "A low-angle upward composition from below.",
                "The camera is positioned slightly above the subject.",
            ),
            (
                "A high-angle downward composition from above.",
                "The camera is positioned slightly below the subject and looks upward.",
            ),
        )
        for framing, camera in conflicts:
            with self.subTest(framing=framing, camera=camera):
                plan = _plan(dialogue=False, camera=camera)
                plan["shots"][0]["framing"] = framing
                compiled = compile_model_result(json.dumps(plan), neutral)
                self.assertIn(
                    "PLANNER_CAMERA_GEOMETRY_AMBIGUOUS",
                    {warning.code for warning in compiled.diagnostics()},
                )

    def test_subject_entry_direction_stays_in_action_not_camera_geometry(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\nThe boss enters from above through the ceiling.",
            duration_seconds=5,
        )
        plan = _plan(
            dialogue=False,
            action="The boss breaks through the ceiling from above and descends into the church.",
            camera="A low-angle camera below the ceiling looks upward at the descending subject.",
        )
        plan["shots"][0]["framing"] = "A wide low-angle composition from below the ceiling."
        parsed = parse_plan_json(json.dumps(plan), prepared)
        self.assertEqual(parsed.shots[0].number, 1)

    def test_camera_retry_instruction_separates_subject_trajectory(self) -> None:
        error = CommunityPromptPlannerError(
            "Shot 3 contains contradictory camera geometry: high, low.",
            code="CAMERA_DIRECTION_CONFLICT",
        )
        instruction = _retry_instruction(error)
        self.assertIn("action field", instruction)
        self.assertIn("not the camera viewpoint", instruction)
        self.assertIn("Choose one coherent framing/camera pair", instruction)
        self.assertIn("from above", instruction)

    def test_long_source_message_requires_a_compact_complete_json_budget(self) -> None:
        source = "Cut 1\n" + (
            "崩れた聖堂で主人公が封印紋章を調べ、冷たい風と石片の動きを追う。"
            * 100
        )
        prepared = prepare_planner_input(source, duration_seconds=14.4)
        message = build_model_messages(prepared)[1]["content"]

        self.assertGreaterEqual(len(prepared.redacted_prompt), 2400)
        self.assertIn("OUTPUT BUDGET:", message)
        self.assertIn("under 1050 generated tokens", message)
        self.assertIn("one concise sentence per string field", message)
        self.assertIn("ambient and foley to at most 4 short items each", message)
        self.assertIn("no repetition or commentary", message)

    def test_explicit_three_shots_require_exact_compact_count_and_order(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n少女が扉へ近づく。\n"
            "Cut 2\n扉を押し開ける。\n"
            "Cut 3\n奥の騎士を見つける。",
            duration_seconds=6,
        )
        message = build_model_messages(prepared)[1]["content"]

        self.assertIn(
            "return exactly 3 compact shot objects numbered 1..3, one per "
            "canonical source block; never merge/drop blocks.",
            message,
        )

    def test_long_unnumbered_source_uses_duration_bounded_merged_shots(self) -> None:
        source = (
            "少女が暗い回廊を進み、壁画、階段、祭壇、崩れた天井を順に発見する。"
            * 100
        )
        for duration, expected_max in ((4, 6), (30, 8)):
            with self.subTest(duration=duration):
                prepared = prepare_planner_input(source, duration_seconds=duration)
                message = build_model_messages(prepared)[1]["content"]

                self.assertEqual(prepared.source_shot_numbers, ())
                self.assertIn(
                    f"merge adjacent descriptive beats into at most {expected_max} shot objects",
                    message,
                )
                self.assertIn("hard cap 8", message)
                self.assertIn("preserve essential chronological transitions", message)

    def test_short_unnumbered_source_has_no_long_output_budget(self) -> None:
        prepared = prepare_planner_input(
            "少女が扉を開け、暗い回廊を慎重に進む。",
            duration_seconds=5,
        )
        message = build_model_messages(prepared)[1]["content"]

        self.assertEqual(prepared.source_shot_numbers, ())
        self.assertNotIn("OUTPUT BUDGET:", message)
        self.assertNotIn("merge adjacent descriptive beats", message)

    def test_json_shape_retry_instruction_demands_a_short_complete_plan(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n少女が120kgの扉を押す。\n"
            "Cut 2\n少女が『行くぞ！』と言う。\n"
            "Cut 3\n扉が開く。",
            duration_seconds=6,
        )
        for code in ("MODEL_JSON_INVALID", "EMPTY_MODEL_RESULT", "NO_SHOTS"):
            with self.subTest(code=code):
                error = CommunityPromptPlannerError(
                    "The planner output was truncated before a complete shot object.",
                    code=code,
                )
                instruction = _retry_instruction(error, prepared)

                self.assertIn("under 750 generated tokens", instruction)
                self.assertIn("one short sentence per string field", instruction)
                self.assertIn("at most 4 short ambient items", instruction)
                self.assertIn("at most 4 short foley items", instruction)
                self.assertIn("Close every JSON array", instruction)
                self.assertIn(
                    "Preserve exactly 3 shot objects numbered 1..3 in source order",
                    instruction,
                )
                self.assertIn("120 kg", instruction)
                self.assertIn("dialogue IDs exactly (1)", instruction)
                self.assertIn("No Markdown, preamble, trailing text, or commentary", instruction)

    def test_numbered_source_camera_cue_cannot_be_satisfied_by_another_shot(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n仰角煽り視点で人物が歩く。\nCut 2\n人物が立ち止まる。",
            duration_seconds=5,
        )
        self.assertIn(
            "Shot 1 = low-angle upward view",
            build_model_messages(prepared)[1]["content"],
        )
        plan = _plan(dialogue=False, action="The person waits quietly.")
        plan["shots"] = [
            {
                "number": 1,
                "start_seconds": 0,
                "end_seconds": 2.5,
                "framing": "A neutral eye-level medium composition.",
                "camera": "A locked eye-level camera holds steady.",
                "action": "The person walks forward.",
            },
            {
                "number": 2,
                "start_seconds": 2.5,
                "end_seconds": 5,
                "framing": "A low-angle full-body composition from below.",
                "camera": "A low-angle camera looks upward at the person.",
                "action": "The person stops and waits quietly.",
            },
        ]
        compiled = compile_model_result(json.dumps(plan), prepared)
        self.assertIn("low-angle", compiled.plan.shots[0].camera)
        self.assertIn("looks upward", compiled.plan.shots[0].camera)
        self.assertIn(
            "SOURCE_CAMERA_DIRECTION_REPAIRED",
            {warning.code for warning in compiled.diagnostics()},
        )

    def test_latest_generated_camera_controls_are_repaired(self) -> None:
        latest_shot_1 = prepare_planner_input(
            "Cut 1\nウォーキングマシン。斜め構図で仰角煽り視点。",
            duration_seconds=5,
        )
        plan = _plan(
            dialogue=False,
            action=(
                "The visible subject walks on a treadmill with intense focus, "
                "breathing heavily"
            ),
            camera=(
                "Camera positioned slightly above and to the side, tracking the subject "
                "as she walks on a treadmill with a dynamic upward tilt"
            ),
        )
        plan["shots"][0]["framing"] = (
            "Diagonal composition with a low-angle upward view, emphasizing the "
            "character's motion and physical effort"
        )
        repaired = compile_model_result(json.dumps(plan), latest_shot_1)
        self.assertIn("looks upward", repaired.plan.shots[0].camera)
        self.assertIn(
            "SOURCE_CAMERA_DIRECTION_REPAIRED",
            {warning.code for warning in repaired.diagnostics()},
        )

        latest_shot_6 = prepare_planner_input(
            "Cut 6\n暗転し、甘いプロテインドリンクを飲んで終了。",
            duration_seconds=5,
        )
        plan["shots"][0].update(
            {
                "number": 1,
                "framing": "Low-angle shot from above, soft focus on the character's hands and drink",
                "camera": (
                    "Slow pan down and in on the character as she drinks from a bottle, "
                    "ending in a still frame"
                ),
                "action": (
                    "The scene darkens gradually, and the character drinks a sweet protein "
                    "drink slowly and deliberately"
                ),
            }
        )
        compiled = compile_model_result(json.dumps(plan), latest_shot_6)
        self.assertIn(
            "PLANNER_CAMERA_GEOMETRY_AMBIGUOUS",
            {warning.code for warning in compiled.diagnostics()},
        )

    def test_panting_and_exertion_roar_are_foley_not_dialogue(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n選手が『ハァ、ハァ』と息を切らし、うおおお！と力を込める。",
            duration_seconds=5,
        )
        self.assertEqual(prepared.dialogues, ())
        self.assertEqual(len(prepared.nonverbal_cues), 2)
        plan = _plan(
            dialogue=False,
            action="The athlete strains through one final controlled lift.",
            foley=[
                "strained breathing and panting",
                "a nonverbal exertion roar",
                "Metal plates clink under load.",
            ],
        )
        compiled = compile_model_result(json.dumps(plan), prepared)
        self.assertNotIn('"', compiled.prompt)
        self.assertIn("Foley: strained breathing and panting", compiled.prompt)
        self.assertNotIn("ハァ", compiled.prompt)
        self.assertNotIn("うお", compiled.prompt)

    def test_missing_required_nonverbal_foley_is_repaired(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n人物が『うおおお！』と力を込める。", duration_seconds=5
        )
        compiled = compile_model_result(json.dumps(_plan(dialogue=False)), prepared)
        self.assertIn("a nonverbal exertion roar", compiled.prompt)
        self.assertIn(
            "SOURCE_NONVERBAL_FOLEY_REPAIRED",
            {warning.code for warning in compiled.diagnostics()},
        )

    def test_model_invented_speech_is_removed_when_no_dialogue_was_supplied(self) -> None:
        prepared = prepare_planner_input("Cut 1\n人物が静かに歩く。", duration_seconds=5)
        plan = _plan(
            dialogue=False,
            action="The person walks and narrates the full scene.",
        )
        compiled = compile_model_result(json.dumps(plan), prepared)
        self.assertIn("The person walks", compiled.prompt)
        self.assertNotIn("narrates", compiled.prompt)
        self.assertIn(
            "MODEL_SPEECH_CONTROL_REMOVED",
            {warning.code for warning in compiled.diagnostics()},
        )

    def test_model_speech_control_removes_only_speech_clauses(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n整備士の少女が「開けて」と言う。", duration_seconds=5
        )
        plan = _plan(
            dialogue=False,
            action=(
                "The maintenance robot pushes open the storage cabinet door; "
                "the pilot speaks clearly"
            ),
        )
        plan["scene"] = (
            "A maintenance robot pushes open a storage cabinet door; "
            "the pilot speaks clearly in response. After a pause, "
            "a hydraulic arm extends and slides the door horizontally."
        )
        compiled = compile_model_result(json.dumps(plan), prepared)
        self.assertIn("robot pushes open a storage cabinet door", compiled.prompt)
        self.assertIn("hydraulic arm extends and slides the door horizontally", compiled.prompt)
        self.assertNotIn("speaks clearly", compiled.prompt)
        self.assertEqual(compiled.prompt.count('"開けて"'), 1)
        speech_warnings = [
            warning
            for warning in compiled.diagnostics()
            if warning.code == "MODEL_SPEECH_CONTROL_REMOVED"
        ]
        self.assertGreaterEqual(len(speech_warnings), 2)
        self.assertTrue(any("scene" in warning.message for warning in speech_warnings))
        self.assertTrue(any("action" in warning.message for warning in speech_warnings))

    def test_model_invented_mixed_speech_preserves_physical_clauses(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\nロボットが扉を押し開ける。", duration_seconds=5
        )
        plan = _plan(
            dialogue=False,
            action=(
                'The robot pushes the door; the pilot says "go"; '
                "the hydraulic arm extends and locks."
            ),
        )
        compiled = compile_model_result(json.dumps(plan), prepared)
        self.assertIn("The robot pushes the door", compiled.prompt)
        self.assertIn("the hydraulic arm extends and locks", compiled.prompt)
        self.assertNotIn("says", compiled.prompt)
        self.assertIn(
            "MODEL_SPEECH_CONTROL_REMOVED",
            {warning.code for warning in compiled.diagnostics()},
        )

    def test_model_speech_only_still_uses_existing_default(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n人物が静かに歩く。", duration_seconds=5
        )
        plan = _plan(
            dialogue=False,
            action="The pilot speaks clearly and narrates the scene.",
        )
        compiled = compile_model_result(json.dumps(plan), prepared)
        self.assertIn(
            "The visible action continues naturally with clear physical cause and effect.",
            compiled.prompt,
        )
        self.assertNotIn("speaks", compiled.prompt)
        self.assertNotIn("narrates", compiled.prompt)
        self.assertIn(
            "MODEL_SPEECH_CONTROL_REMOVED",
            {warning.code for warning in compiled.diagnostics()},
        )

    def test_lexical_assent_is_dialogue_but_nonverbal_cries_are_foley(self) -> None:
        for assent in ("ええ", "ああ"):
            prepared = prepare_planner_input(
                f"Cut 1\n少女が「{assent}」とうなずく。", duration_seconds=5
            )
            self.assertEqual(tuple(item.text for item in prepared.dialogues), (assent,))
            self.assertEqual(prepared.nonverbal_cues, ())

        nonverbal = prepare_planner_input(
            "Cut 1\n少女が「うわーっ」と息を漏らす。", duration_seconds=5
        )
        self.assertEqual(nonverbal.dialogues, ())
        self.assertEqual(len(nonverbal.nonverbal_cues), 1)

        explicit = prepare_planner_input(
            "Cut 1\n少女が「うわーっ」と叫ぶ。", duration_seconds=5
        )
        self.assertEqual(tuple(item.text for item in explicit.dialogues), ("うわーっ",))
        self.assertEqual(explicit.nonverbal_cues, ())

    def test_explicit_dialogue_text_keeps_even_a_nonverbal_literal_as_dialogue(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n少女が驚いて振り返る。",
            dialogue_texts=["ああ"],
            duration_seconds=5,
        )
        self.assertEqual(tuple(item.text for item in prepared.dialogues), ("ああ",))
        self.assertEqual(prepared.nonverbal_cues, ())

    def test_style_sound_audio_and_music_controls_are_separate_and_deterministic(self) -> None:
        prepared = self._prepared(
            style_direction="暖かな夏の2Dアニメ調。",
            soundscape="遠くの蝉と穏やかな風。",
            audio_preset="effects",
            music_policy="none",
        )
        model_input = build_model_messages(prepared)[1]["content"]
        self.assertIn("暖かな夏の2Dアニメ調", model_input)
        self.assertIn("遠くの蝉と穏やかな風", model_input)
        self.assertNotIn("頑張るぞ", model_input)
        compiled = compile_model_result(_json(), prepared)
        self.assertIn("Mix: Prioritize precisely synchronized physical foley", compiled.prompt)
        self.assertIn("Music: N/A", compiled.prompt)
        self.assertNotIn("A restrained instrumental pulse", compiled.prompt)

    def test_auxiliary_control_cannot_smuggle_dialogue_or_tags(self) -> None:
        with self.assertRaises(CommunityPromptPlannerError) as quote:
            self._prepared(soundscape="少女が『秘密だよ』と言う。")
        self.assertEqual(quote.exception.code, "DIALOGUE_IN_AUXILIARY_CONTROL")
        with self.assertRaises(CommunityPromptPlannerError) as tag:
            self._prepared(style_direction="Use <Picture 9>.")
        self.assertEqual(tag.exception.code, "AUXILIARY_CONTROL_TAG_FORBIDDEN")

    def test_visible_text_is_not_automatically_treated_as_dialogue(self) -> None:
        prepared = prepare_planner_input(
            "Cut 1\n店の看板に「営業中」と表示される。", duration_seconds=5
        )
        self.assertEqual(prepared.dialogues, ())

    def test_model_checkout_revision_metadata_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            root = repo / MODEL_RELATIVE_PATH
            root.mkdir(parents=True)
            for name in (
                "config.json",
                "tokenizer_config.json",
                "model.safetensors.index.json",
            ):
                (root / name).write_text("{}", encoding="utf-8")
            files = [
                {
                    "path": (MODEL_RELATIVE_PATH / f"runtime-{index}.bin").as_posix(),
                    "size": 0,
                    "sha256": "0" * 64,
                }
                for index in range(MODEL_RUNTIME_FILE_COUNT - 1)
            ]
            files.append(
                {
                    "path": (MODEL_RELATIVE_PATH / "runtime-final.bin").as_posix(),
                    "size": MODEL_RUNTIME_TOTAL_BYTES,
                    "sha256": "0" * 64,
                }
            )
            lock = {
                "schema_version": 1,
                "source": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
                "verification": {"total_bytes": MODEL_RUNTIME_TOTAL_BYTES},
                "files": files,
            }
            lock_path = repo / MODEL_LOCK_FILENAME
            lock_path.write_text(
                json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
            (root / MODEL_PROVENANCE_FILENAME).write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_id": MODEL_ID,
                        "revision": MODEL_REVISION,
                        "lock_sha256": lock_sha,
                        "file_count": MODEL_RUNTIME_FILE_COUNT,
                        "total_bytes": MODEL_RUNTIME_TOTAL_BYTES,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            inspected = inspect_model_checkout(root)
            self.assertTrue(inspected.verified)
            self.assertEqual(require_verified_model_checkout(root).detected_revision, MODEL_REVISION)
            self.assertEqual(inspected.lock_sha256, lock_sha)
            self.assertFalse((root / ".cache").exists())

            marker = json.loads(
                (root / MODEL_PROVENANCE_FILENAME).read_text(encoding="utf-8")
            )
            marker["lock_sha256"] = "f" * 64
            (root / MODEL_PROVENANCE_FILENAME).write_text(
                json.dumps(marker), encoding="utf-8"
            )
            self.assertFalse(inspect_model_checkout(root).verified)

    def test_unverified_checkout_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(CommunityPromptPlannerError) as ctx:
                require_verified_model_checkout(temporary)
        self.assertEqual(ctx.exception.code, "MODEL_REVISION_UNVERIFIED")

    def test_worker_contract_with_injected_model_and_retry(self) -> None:
        class FakePlanner:
            def __init__(self) -> None:
                self.calls = 0
                self.load_metadata = {"total_load_ms": 12.5}
                self.last_generation_metadata = None

            def generate(self, messages):
                self.calls += 1
                self.last_generation_metadata = {
                    "output_tokens_including_eos": 101 + self.calls,
                    "content_tokens_before_eos": 100 + self.calls,
                    "eos_reached": True,
                    "generation_ms": 250.0,
                    "max_new_tokens": 1280,
                }
                if self.calls == 1:
                    return "not JSON"
                self.last_messages = messages
                return _json()

        fake = FakePlanner()
        response = process_request(
            {
                "prompt": "Cut 1\n仰角で120kgを3回持ち上げ、セリフ「頑張るぞ！」と言う。",
                "references": [{"kind": "image", "index": 1}],
                "duration_seconds": 5,
                "audio_preset": "effects",
                "music_policy": "none",
                "max_attempts": 2,
            },
            planner=fake,
        )
        self.assertTrue(response["ok"])
        self.assertEqual(response["planner_metadata"]["attempts"], 2)
        self.assertEqual(response["planner_metadata"]["max_new_tokens"], 1280)
        self.assertEqual(
            response["planner_metadata"]["runtime_timing"][
                "measured_generation_ms_total"
            ],
            500.0,
        )
        self.assertEqual(
            response["planner_metadata"]["runtime_timing"]["generation_attempts"][-1][
                "content_tokens_before_eos"
            ],
            102,
        )
        self.assertEqual(response["compiled_prompt"].count('"頑張るぞ！"'), 1)
        self.assertIn("<Picture 1>", response["compiled_prompt"])
        self.assertEqual(response["plan"]["schema_version"], PLAN_SCHEMA_VERSION)
        self.assertIn("REJECTED", fake.last_messages[-1]["content"])

    def test_worker_retries_truncated_json_with_compact_guidance(self) -> None:
        class TruncatingPlanner:
            def __init__(self) -> None:
                self.calls = 0
                self.retry_message = ""
                self.last_generation_metadata = None

            def generate(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return (
                        '{"schema_version":"h3-community-plan-v1","style":"dark",'
                        '"scene":"ruins","shots":[{"number":1'
                    )
                self.retry_message = messages[-1]["content"]
                return _json(dialogue=False)

        fake = TruncatingPlanner()
        response = process_request(
            {
                "prompt": "Cut 1\n人物が120kgの扉を3回押す。",
                "duration_seconds": 5,
                "music_policy": "none",
                "max_attempts": 2,
            },
            planner=fake,
        )

        self.assertTrue(response["ok"])
        self.assertEqual(response["planner_metadata"]["attempts"], 2)
        self.assertEqual(fake.calls, 2)
        self.assertIn("REJECTED (MODEL_JSON_INVALID)", fake.retry_message)
        self.assertIn("under 750 generated tokens", fake.retry_message)
        self.assertIn("Close every JSON array", fake.retry_message)

    def test_worker_rejects_an_unsafe_generation_cap(self) -> None:
        with self.assertRaises(CommunityPromptPlannerError) as ctx:
            process_request(
                {
                    "prompt": "Cut 1\n人物が歩く。",
                    "duration_seconds": 5,
                    "max_new_tokens": 256,
                },
                planner=lambda _: _json(dialogue=False),
            )
        self.assertEqual(ctx.exception.code, "INVALID_WORKER_REQUEST")


if __name__ == "__main__":
    unittest.main()
