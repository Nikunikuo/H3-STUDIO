"""Independent adversarial E2E corpus for the Japanese community prompt path.

This file is separate from test_community_prompt_resilience.py.  The prompts
are original authoring requests with different subjects, locations, and
production constraints.  The tests exercise FastAPI /api/jobs preflight and
community_prompt_worker.process_request with an injected model so they remain
deterministic and cheap.

Every case records a recoverable/fatal verdict and expected diagnostic codes.
This file is a test artifact only; production code is not edited here.
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

import webui.job_manager as job_manager_module  # noqa: E402
from webui.community_prompt_planner import (  # noqa: E402
    PLAN_SCHEMA_VERSION,
    CommunityPromptPlannerError,
    extract_shot_numbers,
)
from webui.community_prompt_worker import process_request  # noqa: E402


@dataclass(frozen=True)
class AdversarialCase:
    name: str
    family: str
    prompt: str
    verdict: str
    expected_diagnostics: tuple[str, ...]
    model_fault: str = "normal"
    references: tuple[tuple[str, int, str], ...] = ()
    dialogue_texts: tuple[str, ...] = ()
    duration_seconds: float = 6.0
    music_policy: str = "auto"
    audio_preset: str = "effects"
    api_mode: str | None = None
    api_expected_status: int | None = None

    @property
    def shot_count(self) -> int:
        return max(1, len(extract_shot_numbers(self.prompt)))

    @property
    def inventory(self) -> list[dict[str, Any]]:
        return [
            {"kind": kind, "index": index, "role": role}
            for kind, index, role in self.references
        ]


def case(
    name: str,
    family: str,
    prompt: str,
    *,
    expected: tuple[str, ...] = (),
    model_fault: str = "normal",
    references: tuple[tuple[str, int, str], ...] = (),
    dialogue_texts: tuple[str, ...] = (),
    duration: float = 6.0,
    music: str = "auto",
    audio: str = "effects",
    verdict: str = "recoverable",
    api_mode: str | None = None,
    api_status: int | None = None,
) -> AdversarialCase:
    return AdversarialCase(
        name=name,
        family=family,
        prompt=prompt,
        verdict=verdict,
        expected_diagnostics=expected,
        model_fault=model_fault,
        references=references,
        dialogue_texts=dialogue_texts,
        duration_seconds=duration,
        music_policy=music,
        audio_preset=audio,
        api_mode=api_mode,
        api_expected_status=api_status,
    )


IMAGE_1 = (("image", 1, "subject"),)
IMAGE_1_2 = (("image", 1, "subject"), ("image", 2, "environment"))
MIXED_MEDIA = (
    ("image", 1, "subject"),
    ("video", 1, "motion"),
    ("audio", 1, "ambience"),
)


# The stories are intentionally independent.  The categories are repeated to
# get coverage, but no case is a string variant of the Niku/church prompt.
CASES: tuple[AdversarialCase, ...] = (
    # Cut/Shot/Scene/第Nカット; duplicate, skipped, reverse, full-width,
    # bracketed, and very large labels.
    case(
        "cut_duplicate_mixed_railway",
        "cut_label_chaos",
        """Cut ００４
凍った駅で信号係が赤い旗を線路へ差し出す。
Shot 004
遠くの列車灯が雪煙を割り、係は足場を踏み直す。
第００３カット
警報灯の反射が濡れたレールを走る。
Scene 900000
列車が安全に停止し、係は旗を下ろす。""",
        expected=("SOURCE_SHOT_NUMBERING_NORMALIZED",),
        model_fault="aliases",
        duration=8.0,
        api_mode="t2v",
        api_status=200,
    ),
    case(
        "cut_reverse_gap_observatory",
        "cut_label_chaos",
        """第１２カット
砂漠の観測所で天文学者が望遠鏡の蓋を外す。
Cut 2
二つの月が雲間から現れ、彼女は重い歯車を回す。
Shot 0002
流星の光が真鍮の鏡筒を横切る。""",
        expected=("SOURCE_SHOT_NUMBERING_NORMALIZED",),
        model_fault="shot_number_drift",
        duration=7.0,
    ),
    case(
        "cut_fullwidth_large_coral",
        "cut_label_chaos",
        """Scene ０１
珊瑚の郵便門を水中配達員がくぐり、封蝋の手紙を胸に抱える。
カット ０９
マンタの群れを回り込み、沈没船の投函口へ近づく。
Shot ９９９９９９
泡の渦が手紙を守り、錆びた蓋が音を立てて開く。""",
        expected=("SOURCE_SHOT_NUMBERING_NORMALIZED",),
        model_fault="timing_overlap",
        references=(("video", 1, "motion"),),
        duration=7.5,
    ),
    case(
        "cut_bracketed_clockwork",
        "cut_label_chaos",
        """[Shot 0000007]
劇場の機械仕掛けの幕が開き、代役の女優が古い剣を拾う。
【Cut 0002】
歯車の腕が赤い衣装の留め具を一つずつ閉じる。
（Scene 0002）
女優が回転舞台を横切り、スポットライトの中心で止まる。""",
        expected=("SOURCE_SHOT_NUMBERING_NORMALIZED",),
        model_fault="missing_optional",
        references=IMAGE_1,
        duration=6.5,
    ),
    case(
        "cut_hash_colon_lighthouse",
        "cut_label_chaos",
        """Cut #8:
灯台の修理工が暴風雨の屋上で受信機の蓋を開ける。
Shot: 8
アンテナ線が風に振られ、彼女は片手でボルトを押さえる。
第8カット
海面の稲妻と同時に、受信機が短い信号を返す。""",
        expected=("SOURCE_SHOT_NUMBERING_NORMALIZED",),
        model_fault="wrapper",
        duration=6.75,
        music="none",
    ),
    case(
        "cut_huge_single_volcano",
        "cut_label_chaos",
        """Scene ４２４２４２４２
火山海の潜水士が熱水噴出口の横に計測器を立てる。
崩れた玄武岩の天井から砂が落ち、潜水士は逆噴射でトンネルを抜ける。""",
        expected=("SOURCE_SHOT_NUMBERING_NORMALIZED",),
        model_fault="duration_mismatch",
        duration=5.5,
        music="none",
    ),
    # Advisory contradictions: people, location, motion, timing, audio, and
    # visible text all receive their own independently authored case.
    case(
        "contradiction_fixed_location_then_port",
        "contradictory_authoring",
        """Cut 1
古い港の倉庫を舞台にし、場所は変えない。
Cut 2
合図のあと、場所変更で灯台の階段へ移動する。
錆びた扉を押し開け、海霧の向こうを確認する。""",
        expected=("SOURCE_LOCATION_CONFLICT",),
    ),
    case(
        "contradiction_still_then_sprint",
        "contradictory_authoring",
        """Cut 1
陶芸家の手元を静止させ、動かない構図で粘土の表面を見せる。
Cut 2
警報が鳴った瞬間、陶芸家は工房を激しく走り、窓を閉じる。""",
        expected=("SOURCE_MOTION_CONFLICT",),
        model_fault="camera_mixed",
        duration=5.0,
    ),
    case(
        "contradiction_one_person_and_crowd",
        "contradictory_authoring",
        """Cut 1
街灯の下に郵便配達員が一人だけ立つ。
Cut 2
一人のまま、群衆が橋を埋め尽くす中を自転車で横切る。""",
        expected=("SOURCE_SUBJECT_COUNT_CONFLICT",),
        model_fault="missing_action",
    ),
    case(
        "contradiction_bgm_on_off",
        "contradictory_authoring",
        """Cut 1
雪原の救難ビーコンを探す。BGMはなし、音楽なしで風音だけにする。
Cut 2
救難信号が見えた瞬間だけBGMあり、短い音楽ありとして高揚させる。""",
        expected=("SOURCE_MUSIC_CONFLICT",),
        model_fault="extra_fields",
        music="none",
    ),
    case(
        "contradiction_caption_on_off",
        "contradictory_authoring",
        """Cut 1
地下温室の水滴を映し、字幕は表示しない。
Cut 2
植物の名前を字幕として画面中央に表示するが、他のテロップは禁止する。""",
        expected=("SOURCE_ONSCREEN_TEXT_CONFLICT",),
        model_fault="invented_caption",
        music="none",
    ),
    case(
        "contradiction_speech_and_no_speech",
        "contradictory_authoring",
        """Cut 1
夜の遊園地で音声なし、セリフなしのまま観覧車を見上げる。
Cut 2
整備員が「安全確認、完了」と叫ぶが、台詞は禁止したままにする。""",
        expected=("SOURCE_SPEECH_CONFLICT",),
        model_fault="dialogue_delivery_drift",
        dialogue_texts=("安全確認、完了",),
        music="none",
        audio="dialogue",
    ),
    case(
        "contradiction_opposing_camera",
        "contradictory_authoring",
        """Cut 1
廃駅の屋根を俯瞰で見下ろし、同じカットの後半は仰角で空を煽って捉える。
Cut 2
落ちた時計を拾う手元へ視線を切り替える。""",
        expected=("SOURCE_CAMERA_CONFLICT",),
        model_fault="camera_mixed",
        duration=5.0,
    ),
    case(
        "contradiction_timecodes_and_budget",
        "contradictory_authoring",
        """Cut 1（0.0-8.0秒）
峡谷の測量士が赤い杭を打つ。
Cut 2（7.5-3.0秒）
崩れた橋を渡り、風で地図を押さえる。
映像全体は3秒しかない。""",
        expected=("SOURCE_DURATION_BOUND_TO_FRAME_COUNT",),
        model_fault="timing_overlap",
        duration=3.0,
        music="none",
    ),
    case(
        "contradiction_all_axes",
        "contradictory_authoring",
        """Cut 1
同じ場所から動かず、人物は一人だけ、静止した画面で始める。
Cut 2
場所変更で群衆の市場へ移り、主人公は激しく走る。BGMありで盛り上げる。
Cut 3
BGMなし、字幕を表示し、しかし字幕は禁止。""",
        expected=(
            "SOURCE_LOCATION_CONFLICT",
            "SOURCE_SUBJECT_COUNT_CONFLICT",
            "SOURCE_MOTION_CONFLICT",
            "SOURCE_MUSIC_CONFLICT",
            "SOURCE_ONSCREEN_TEXT_CONFLICT",
        ),
        model_fault="aliases",
        music="none",
        duration=7.0,
    ),
    case(
        "contradiction_live_prompt_warning_shape",
        "contradictory_authoring",
        """日本の高品質なセルアニメ調。
スタイル: HD 16:9、14.4秒のダークファンタジーゲームPV。
暗い教会。断崖、崩れた橋、青い霧の洞窟、枯れ森を場所変更の基準にする。
場所は絶対に変えない。しかし次の瞬間、別の場所へ場所を変更する。
Cut 1
教会の外観を見上げる主人公。
Cut 2
教会で歩いて踏み込むと、上からボスが登場。
字幕もルーン文字のような象形文字で謎の言葉の字幕。
各カットは意味のない静止や間延びを作らない。
カメラは完全固定。
カメラ: Push In、Tracking Shot、Arc Shot、Whip Panを連続させる。
音声: 壮大なBGMを鳴らす。BGMなし。台詞なし。
禁止: 画面内字幕、読めない文字。""",
        expected=(
            "SOURCE_LOCATION_CONFLICT",
            "SOURCE_CAMERA_CONFLICT",
            "SOURCE_ONSCREEN_TEXT_CONFLICT",
            "SOURCE_MUSIC_CONFLICT",
        ),
        model_fault="wrapper",
        music="none",
        duration=14.375,
    ),
    # Quoted visual labels versus actual speech and nonverbal sound.
    case(
        "visual_title_quote_not_dialogue",
        "audio_dialogue_policy",
        """Cut 1
展示会の黒い壁にタイトル「月面の庭」を大きく描く。これは画面文字であり台詞ではない。
Cut 2
カメラが文字から模型の月面へパンし、音声なしで機械音だけを聞かせる。""",
        model_fault="invented_caption",
        music="none",
        audio="ambience",
        duration=5.5,
    ),
    case(
        "named_speaker_quote_is_dialogue",
        "audio_dialogue_policy",
        """Cut 1
整備ロボットが格納庫の扉を押し、操縦士が「開けて」と明確に言う。
Cut 2
一拍後に油圧アームが伸び、扉が横へ滑る。""",
        model_fault="dialogue_delivery_drift",
        dialogue_texts=("開けて",),
        music="none",
        audio="dialogue",
        duration=5.0,
    ),
    case(
        "production_label_quote_control_text",
        "audio_dialogue_policy",
        """Cut 1
映像制作メモとして「予告編・第2稿」というラベルを画面外の設計情報にする。
Cut 2
森の記録員がランタンを掲げ、実際の発話は入れない。""",
        model_fault="wrapper",
        music="none",
        duration=5.0,
    ),
    case(
        "nonverbal_roar_becomes_foley",
        "audio_dialogue_policy",
        """Cut 1
洞窟の奥で巨大な蒸気弁が開き、作業員が「うわーっ」と息を漏らす。
Cut 2
蒸気の噴出音が反響し、作業員は壁際へ身を寄せる。""",
        model_fault="missing_optional",
        music="none",
        audio="effects",
        duration=5.5,
    ),
    case(
        "audio_reference_without_dialogue",
        "audio_dialogue_policy",
        """Cut 1
<Picture 1>の料理人が夜明け前の厨房で包丁を研ぐ。
<Audio 1>は換気扇と食器の音だけを環境参照として使い、発話は作らない。
Cut 2
窓が明るくなり、料理人は鍋の蓋を開ける。""",
        references=(("image", 1, "subject"), ("audio", 1, "ambience")),
        music="none",
        audio="ambience",
        api_mode="omni",
        api_status=200,
        duration=6.0,
    ),
    case(
        "dialogue_and_audio_reference_policy",
        "audio_dialogue_policy",
        """Cut 1
<Picture 1>の船長がデッキを見回し、「面舵いっぱい」と指示する。
<Audio 1>は波の音の参照だが、指定台詞を優先する。
Cut 2
帆が張り、船体が大きく傾く。""",
        dialogue_texts=("面舵いっぱい",),
        references=(("image", 1, "subject"), ("audio", 1, "ambience")),
        music="none",
        audio="dialogue",
        duration=6.0,
    ),
    # Reference spelling, leading zeros, full-width forms, and intentional
    # repeated use of the same item.
    case(
        "reference_case_folded_picture",
        "reference_binding",
        """Cut 1
<picture1>の気球操縦士が雲の裂け目へ降下する。
Cut 2
操縦士はロープを二度引き、青い旗を風上へ向ける。""",
        expected=("SOURCE_REFERENCE_TAG_NORMALIZED",),
        references=IMAGE_1,
        api_mode="omni",
        api_status=200,
        duration=5.0,
    ),
    case(
        "reference_leading_zero_picture",
        "reference_binding",
        """Cut 1
<Picture 0001>の庭師が凍った温室の鍵を回す。
Cut 2
蔓がガラスを押し、庭師は非常レバーへ駆け寄る。""",
        expected=("SOURCE_REFERENCE_TAG_NORMALIZED",),
        model_fault="aliases",
        references=IMAGE_1,
        duration=6.0,
    ),
    case(
        "reference_fullwidth_picture_video",
        "reference_binding",
        """Cut 1
＜Ｐｉｃｔｕｒｅ　０１＞の踊り子が廃劇場の床を踏む。
＜Ｖｉｄｅｏ　０１＞の動きのリズムに合わせる。
Cut 2
回転しながら幕を引き、舞台裏へ消える。""",
        expected=("SOURCE_REFERENCE_TAG_NORMALIZED",),
        model_fault="shot_number_drift",
        references=(("image", 1, "subject"), ("video", 1, "motion")),
        duration=6.0,
    ),
    case(
        "reference_audio_no_space_and_repeat",
        "reference_binding",
        """Cut 1
<Audio01>の雨音を使い、<Picture 01>の写真家が濡れた路地を走る。
同じ<Picture 01>を次の瞬間の手元の寄りでも使う。
Cut 2
写真家がシャッターを切り、雨粒がレンズを横切る。""",
        expected=("SOURCE_REFERENCE_TAG_NORMALIZED",),
        model_fault="missing_optional",
        references=(("image", 1, "subject"), ("audio", 1, "ambience")),
        music="none",
        audio="ambience",
        duration=5.5,
    ),
    case(
        "reference_mixed_kind_leading_zero",
        "reference_binding",
        """Cut 1
<VIDEO０１>を動きの基準に、＜ＰＩＣＴＵＲＥ　０００１＞の消防士が煙の中を進む。
＜Ａｕｄｉｏ０１＞の警報音が遠くで続く。
Cut 2
消防士がホースを開き、白い水幕が画面を横切る。""",
        expected=("SOURCE_REFERENCE_TAG_NORMALIZED",),
        model_fault="extra_fields",
        references=MIXED_MEDIA,
        music="none",
        duration=6.0,
    ),
    case(
        "reference_two_images_same_prop",
        "reference_binding",
        """Cut 1
<Picture 02>の古い鍵を<Picture 01>の錬金術師が拾う。
Cut 2
<Picture 02>を回転させ、錬金術師の顔に青い反射を返す。
鍵穴が開き、壁の奥から光が漏れる。""",
        expected=("SOURCE_REFERENCE_TAG_NORMALIZED",),
        model_fault="wrapper",
        references=IMAGE_1_2,
        duration=6.0,
    ),
    # Near-valid planner JSON and explicit fatal controls.
    case(
        "planner_aliases_and_extra_fields",
        "near_valid_planner_json",
        """Cut 1
移動図書館の司書が閉館前に最後の本を棚へ戻す。
Cut 2
窓から夕日が差し、司書が静かに照明を落とす。""",
        expected=("MODEL_SCHEMA_NORMALIZED", "MODEL_EXTRA_FIELDS_IGNORED"),
        model_fault="aliases",
        duration=6.0,
    ),
    case(
        "planner_fence_preamble_trailing",
        "near_valid_planner_json",
        """Cut 1
蒸気機関車の運転士が圧力計を確認する。
Cut 2
列車が橋を渡り、煙が谷へ流れる。""",
        expected=("MODEL_MARKDOWN_WRAPPER_REMOVED", "MODEL_WRAPPER_TEXT_IGNORED"),
        model_fault="wrapper",
        duration=5.0,
    ),
    case(
        "planner_missing_optional_arrays",
        "near_valid_planner_json",
        """Cut 1
雨上がりの屋上で気象観測員が風向計を調整する。
Cut 2
雲が割れ、観測員は濡れたノートを閉じる。""",
        model_fault="missing_optional",
        music="none",
        duration=5.5,
    ),
    case(
        "planner_duplicate_numbers_and_clocks",
        "near_valid_planner_json",
        """第０８カット
地下鉄の清掃員が止まったエスカレーターを点検する。
Shot 0008
非常灯が点滅し、清掃員は工具箱を閉じる。
Cut 2
駅員室への扉が開く。""",
        expected=("SOURCE_SHOT_NUMBERING_NORMALIZED", "MODEL_SHOT_NUMBER_NORMALIZED"),
        model_fault="shot_number_drift",
        duration=6.0,
    ),
    case(
        "planner_overlapping_timeline_repaired",
        "near_valid_planner_json",
        """Cut 1
雪山の救助隊員がロープを岩へ固定する。
Cut 2
足場が崩れ、隊員は滑りながら仲間を引き上げる。
Cut 3
雪煙が晴れ、安全な棚へ二人で移動する。""",
        expected=("PLANNER_TIMELINE_REPAIRED",),
        model_fault="timing_overlap",
        duration=7.0,
    ),
    case(
        "planner_duration_rescaled",
        "near_valid_planner_json",
        """Cut 1
水上都市の配電士が浮橋の端子を外す。
Cut 2
閃光が走り、配電士は濡れた足場を飛び越える。""",
        expected=("PLANNER_DURATION_NORMALIZED",),
        model_fault="duration_mismatch",
        duration=4.0,
    ),
    case(
        "planner_invented_reference_removed",
        "near_valid_planner_json",
        """Cut 1
<Picture 1>の地質学者が赤い断層面を調べる。
Cut 2
岩盤が鳴り、地質学者は計測器を抱えて後退する。""",
        expected=("MODEL_CONTROL_TAGS_REMOVED",),
        model_fault="invented_reference",
        references=IMAGE_1,
        duration=5.0,
    ),
    case(
        "planner_caption_repaired",
        "near_valid_planner_json",
        """Cut 1
砂時計職人が工房の中央で歯車を組み直す。
Cut 2
砂が落ち切る前に、職人が最後の輪を押し込む。""",
        expected=("MODEL_ONSCREEN_TEXT_REPAIRED",),
        model_fault="invented_caption",
        music="none",
        duration=5.0,
    ),
    case(
        "planner_dialogue_delivery_drift",
        "near_valid_planner_json",
        """Cut 1
<Picture 1>の船員が霧の中で「右舷へ」と叫ぶ。
Cut 2
灯りが右へ振れ、船が暗礁をかわす。""",
        expected=("MODEL_DIALOGUE_DELIVERY_NORMALIZED",),
        model_fault="dialogue_delivery_drift",
        references=IMAGE_1,
        dialogue_texts=("右舷へ",),
        music="none",
        audio="dialogue",
        duration=5.5,
    ),
    case(
        "fatal_source_picture_missing_inventory",
        "fatal_unresolved_reference",
        """Cut 1
<Picture 9>の探検家が塩湖の縁を歩く。
Cut 2
白い地面に足跡が続き、探検家は旗を立てる。""",
        expected=("SOURCE_REFERENCE_NOT_IN_INVENTORY",),
        references=IMAGE_1,
        verdict="fatal",
        api_mode="omni",
        api_status=409,
        duration=5.0,
    ),
    case(
        "fatal_source_picture_non_numeric",
        "fatal_unresolved_reference",
        """Cut 1
<Picture X>の潜水艇が深海の窓を照らす。
潜水士は海溝の壁を指差す。""",
        expected=("SOURCE_REFERENCE_TAG_UNSUPPORTED",),
        references=IMAGE_1,
        verdict="fatal",
        api_mode="omni",
        api_status=400,
        duration=4.5,
    ),
    case(
        "fatal_source_picture_zero",
        "fatal_unresolved_reference",
        """Cut 1
<Picture ０>の路面電車が霧の駅へ入る。
駅員が手旗を上げ、車輪が止まる。""",
        expected=("SOURCE_REFERENCE_TAG_UNSUPPORTED",),
        references=IMAGE_1,
        verdict="fatal",
        api_mode="omni",
        api_status=400,
        duration=4.0,
    ),
    case(
        "fatal_model_empty_explicit_shots",
        "fatal_planner_output",
        """Cut 1
海底研究所の研究員が培養槽の栓を外す。
Cut 2
赤い警告灯が点く。""",
        expected=("EMPTY_MODEL_SHOT_CONTENT",),
        model_fault="empty_shot",
        verdict="fatal",
        duration=5.0,
    ),
    case(
        "fatal_model_no_shots",
        "fatal_planner_output",
        """Cut 1
風車守が丘の上で羽根の軸を点検する。
雲が低く流れる。""",
        expected=("NO_SHOTS",),
        model_fault="no_shots",
        verdict="fatal",
        duration=4.0,
    ),
)


def references_for_model_prompt(case_item: AdversarialCase) -> list[str]:
    labels = {"image": "Picture", "video": "Video", "audio": "Audio"}
    return [
        f"<{labels[kind]} {index}>"
        for kind, index, _role in case_item.references
    ]


def base_plan(case_item: AdversarialCase) -> dict[str, Any]:
    count = case_item.shot_count
    step = case_item.duration_seconds / count
    shots: list[dict[str, Any]] = []
    for index in range(count):
        start = round(index * step, 4)
        end = round(
            case_item.duration_seconds if index == count - 1 else (index + 1) * step,
            4,
        )
        shots.append(
            {
                "number": index + 1,
                "start_seconds": start,
                "end_seconds": end,
                "framing": "A clear medium-wide composition keeps the subject and environment readable.",
                "camera": "The camera tracks the subject with a smooth lateral move and readable reframing.",
                "action": f"The primary subject completes physical beat {index + 1} with clear cause and effect.",
            }
        )
    deliveries = [
        {
            "dialogue_id": index,
            "shot": index,
            "start_seconds": round(min(case_item.duration_seconds, index * step * 0.75), 4),
            "speaker": "the primary visible subject",
            "delivery": "clear, urgent, and natural",
        }
        for index, _literal in enumerate(case_item.dialogue_texts, start=1)
    ]
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "style": "Cinematic animation with crisp silhouettes and coherent physical motion.",
        "scene": "The authored environment continues with the requested subject and readable spatial continuity.",
        "shots": shots,
        "ambient": ["Air movement and natural room tone continue through the scene."],
        "foley": ["Footsteps, material contact, and environmental movement remain audible."],
        "music": "N/A" if case_item.music_policy == "none" else "A restrained instrumental texture.",
        "dialogue_delivery": deliveries,
    }


def model_result(case_item: AdversarialCase) -> str:
    plan = base_plan(case_item)
    fault = case_item.model_fault
    if fault == "aliases":
        shots = [
            {
                "shot": f"{shot['number']:02d}",
                "start": f"{shot['start_seconds']} sec",
                "end": f"{shot['end_seconds']} seconds",
                "composition": shot["framing"],
                "movement": shot["camera"],
                "event": shot["action"],
            }
            for shot in plan["shots"]
        ]
        return json.dumps(
            {
                "schema": "h3-community-plan-old",
                "visual_style": plan["style"],
                "setting": plan["scene"],
                "timeline": shots,
                "ambience": plan["ambient"],
                "sfx": plan["foley"],
                "bgm": plan["music"],
                "dialogue": plan["dialogue_delivery"],
                "irrelevant_debug_note": "discard this field",
            },
            ensure_ascii=False,
        )
    if fault == "wrapper":
        fence = chr(96) * 3
        return (
            "Here is a compact plan.\n"
            + fence
            + "json\n"
            + json.dumps(plan, ensure_ascii=False)
            + "\n"
            + fence
            + "\nTrailing author note."
        )
    if fault == "missing_optional":
        plan.pop("ambient", None)
        plan.pop("foley", None)
        plan["dialogue_delivery"] = []
        return json.dumps(plan, ensure_ascii=False)
    if fault == "shot_number_drift":
        for index, shot in enumerate(plan["shots"], start=1):
            shot["number"] = 9000 - index
            shot["start_seconds"] = f"00:{index - 1:02d}.0"
            shot["end_seconds"] = f"00:{index:02d}.0"
        return json.dumps(plan, ensure_ascii=False)
    if fault == "timing_overlap":
        for index, shot in enumerate(plan["shots"]):
            shot["start_seconds"] = -0.5 if index == 0 else max(0.0, index * 0.4)
            shot["end_seconds"] = 2.0 if index == 0 else 1.5 + index * 0.4
        return json.dumps(plan, ensure_ascii=False)
    if fault == "duration_mismatch":
        cursor = 0.0
        for shot in plan["shots"]:
            shot["start_seconds"] = cursor
            cursor += 3.0
            shot["end_seconds"] = cursor
        return json.dumps(plan, ensure_ascii=False)
    if fault == "camera_mixed":
        plan["shots"][0]["framing"] = (
            "A low-angle composition from below looks upward while also looking down from above."
        )
        plan["shots"][0]["camera"] = (
            "The camera pushes from a high-angle downward view, then tilts to a low-angle upward view."
        )
        return json.dumps(plan, ensure_ascii=False)
    if fault == "missing_action":
        plan["shots"][0]["action"] = None
        return json.dumps(plan, ensure_ascii=False)
    if fault == "extra_fields":
        plan["camera_notes"] = ["Unknown field must not reach the effective prompt."]
        plan["shots"][0]["producer_comment"] = "Ignore this comment."
        return json.dumps(plan, ensure_ascii=False)
    if fault == "dialogue_delivery_drift":
        plan["dialogue_delivery"] = (
            [
                {
                    "id": 99,
                    "cut": 999,
                    "timestamp": "not-a-time",
                    "character": "the primary visible subject",
                    "tone": "urgent but controlled",
                }
            ]
            if case_item.dialogue_texts
            else [
                {
                    "dialogue_id": 77,
                    "shot": 77,
                    "start_seconds": 999,
                    "speaker": "the primary visible subject",
                    "delivery": "invented speech",
                }
            ]
        )
        return json.dumps(plan, ensure_ascii=False)
    if fault == "invented_reference":
        plan["shots"][0]["action"] = (
            "The subject moves with <Picture 9> as an invented model-owned tag."
        )
        return json.dumps(plan, ensure_ascii=False)
    if fault == "invented_caption":
        plan["shots"][0]["action"] = (
            "A readable subtitle appears as on-screen text during the movement."
        )
        return json.dumps(plan, ensure_ascii=False)
    if fault == "empty_shot":
        for shot in plan["shots"]:
            shot["framing"] = ""
            shot["camera"] = ""
            shot["action"] = ""
        return json.dumps(plan, ensure_ascii=False)
    if fault == "no_shots":
        plan["shots"] = []
        return json.dumps(plan, ensure_ascii=False)
    return json.dumps(plan, ensure_ascii=False)


def worker_payload(case_item: AdversarialCase) -> dict[str, Any]:
    return {
        "prompt": case_item.prompt,
        "references": case_item.inventory,
        "dialogue_texts": list(case_item.dialogue_texts),
        "duration_seconds": case_item.duration_seconds,
        "audio_preset": case_item.audio_preset,
        "music_policy": case_item.music_policy,
        "max_attempts": 1,
    }


class InjectedPlanner:
    load_metadata = {"injected": True, "total_load_ms": 0.0}
    last_generation_metadata = {
        "content_tokens_before_eos": 0,
        "output_tokens_including_eos": 0,
        "eos_reached": True,
        "generation_ms": 0.0,
        "max_new_tokens": 1280,
    }

    def __init__(self, raw: str) -> None:
        self.raw = raw

    def generate(self, _messages: list[Mapping[str, str]]) -> str:
        return self.raw


class IsolatedManager:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.jobs_dir = self.root / "webui_data" / "jobs"
        self.outputs_dir = self.root / "outputs" / "webui"
        self.jobs_dir.mkdir(parents=True)
        self.outputs_dir.mkdir(parents=True)
        self.submitted: list[dict[str, Any]] = []

    def submit(self, job: dict[str, Any]) -> dict[str, Any]:
        saved = copy.deepcopy(job)
        self.submitted.append(saved)
        job_dir = self.jobs_dir / saved["id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(
            json.dumps(saved, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return copy.deepcopy(saved)

    def list_jobs(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.submitted)

    def current_job_id(self) -> None:
        return None


class CommunityPromptAdversarialE2ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.test_root = Path(cls.temporary.name).resolve()
        cls.manager = IsolatedManager(cls.test_root)
        sys.modules.pop("webui.server", None)
        with patch.object(job_manager_module, "JobManager", return_value=cls.manager):
            cls.server = importlib.import_module("webui.server")

        cls.original_model_root = cls.server.COMFY_MODEL_ROOT
        cls.original_model_files = cls.server.COMFY_MODEL_FILES
        cls.original_planner_status = cls.server.community_planner_status
        model_root = cls.test_root / "models" / "comfy"
        model_files = {
            "fl2va": model_root / "diffusion_models" / "fl2va.safetensors",
            "ref2va": model_root / "diffusion_models" / "ref2va.safetensors",
            "text_encoder": model_root / "text_encoders" / "qwen.safetensors",
            "video_vae": model_root / "vae" / "video_vae.safetensors",
            "audio_vae": model_root / "vae" / "audio_vae.safetensors",
        }
        for model_path in model_files.values():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"adversarial-test-model-marker")
        cls.server.COMFY_MODEL_ROOT = model_root
        cls.server.COMFY_MODEL_FILES = model_files
        cls.server.community_planner_status = lambda _root: {
            "ready": True,
            "status": "ready",
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "repo_id": "Qwen/Qwen3-4B-Instruct-2507",
            "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
            "local_only": True,
            "model_inference": True,
            "total_bytes": 8_056_459_158,
            "missing_files": [],
            "invalid_files": [],
        }
        cls.client = TestClient(cls.server.app, base_url="http://127.0.0.1")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        cls.server.COMFY_MODEL_ROOT = cls.original_model_root
        cls.server.COMFY_MODEL_FILES = cls.original_model_files
        cls.server.community_planner_status = cls.original_planner_status
        sys.modules.pop("webui.server", None)
        cls.temporary.cleanup()

    def test_corpus_is_independent_and_complete(self) -> None:
        self.assertGreaterEqual(len(CASES), 30)
        prompts = [item.prompt for item in CASES]
        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertGreaterEqual(len({item.family for item in CASES}), 6)
        for family in {
            "cut_label_chaos",
            "contradictory_authoring",
            "audio_dialogue_policy",
            "reference_binding",
            "near_valid_planner_json",
            "fatal_unresolved_reference",
            "fatal_planner_output",
        }:
            self.assertTrue(any(item.family == family for item in CASES), family)
        self.assertTrue(any(item.verdict == "recoverable" for item in CASES))
        self.assertTrue(any(item.verdict == "fatal" for item in CASES))

    def test_worker_preparation_and_compilation(self) -> None:
        failures: list[str] = []
        recoverable_success = 0
        fatal_success = 0
        for item in CASES:
            with self.subTest(case=item.name, family=item.family):
                try:
                    response = process_request(
                        worker_payload(item),
                        planner=InjectedPlanner(model_result(item)),
                    )
                except CommunityPromptPlannerError as exc:
                    if item.verdict == "fatal" and exc.code in item.expected_diagnostics:
                        fatal_success += 1
                        continue
                    failures.append(
                        f"{item.name}: expected {item.verdict}, raised {exc.code}: {exc}"
                    )
                    continue
                except Exception as exc:
                    failures.append(f"{item.name}: unexpected {type(exc).__name__}: {exc}")
                    continue

                if item.verdict == "fatal":
                    failures.append(
                        f"{item.name}: expected fatal {item.expected_diagnostics}, but accepted"
                    )
                    continue

                recoverable_success += 1
                diagnostics = response.get("diagnostics", [])
                actual_codes = {str(value.get("code")) for value in diagnostics}
                missing = set(item.expected_diagnostics) - actual_codes
                if missing:
                    failures.append(
                        f"{item.name}: missing diagnostics {sorted(missing)}, "
                        f"actual={sorted(actual_codes)}"
                    )
                if any(value.get("severity") == "error" for value in diagnostics):
                    failures.append(f"{item.name}: fatal diagnostic leaked into success")
                compiled = str(response.get("compiled_prompt") or "")
                if not all(section in compiled for section in ("Style:", "Scene:", "Audio:")):
                    failures.append(f"{item.name}: compiled prompt lost a required section")
                metadata = response.get("planner_metadata") or {}
                expected_tags = references_for_model_prompt(item)
                if metadata.get("reference_tags", []) != expected_tags:
                    failures.append(
                        f"{item.name}: tags {metadata.get('reference_tags')!r} != {expected_tags!r}"
                    )
                for literal in item.dialogue_texts:
                    if compiled.count(f'"{literal}"') != 1:
                        failures.append(
                            f"{item.name}: exact dialogue {literal!r} was not rendered once"
                        )
                if not item.dialogue_texts and metadata.get("dialogue_count", 0) != 0:
                    failures.append(f"{item.name}: unexpected dialogue metadata")

        if failures:
            self.fail(
                f"{len(failures)} worker cases failed; recoverable={recoverable_success}, "
                f"fatal_controls={fatal_success}\n" + "\n".join(failures)
            )
        self.assertEqual(
            recoverable_success,
            sum(item.verdict == "recoverable" for item in CASES),
        )
        self.assertEqual(
            fatal_success,
            sum(item.verdict == "fatal" for item in CASES),
        )

    def api_form(self, item: AdversarialCase) -> dict[str, str]:
        return {
            "mode": item.api_mode or "t2v",
            "style": "natural",
            "prompt": item.prompt,
            "width": "640",
            "height": "384",
            "num_frames": "124",
            "steps": "8",
            "seed": "130813",
            "acceleration": "off",
            "ref_image_size": "match",
            "prompt_processing_mode": "community",
            "audio_preset": item.audio_preset,
            "dialogue": "",
            "soundscape": "",
            "music_policy": item.music_policy,
            "audio_gain_db": "0",
        }

    def api_files(self, item: AdversarialCase) -> list[tuple[str, tuple[str, bytes, str]]]:
        extensions = {
            "image": ("png", "image/png"),
            "video": ("mp4", "video/mp4"),
            "audio": ("wav", "audio/wav"),
        }
        return [
            (
                "references",
                (
                    f"adversarial-{ordinal}.{extensions[kind][0]}",
                    b"fixture",
                    extensions[kind][1],
                ),
            )
            for ordinal, (kind, _index, _role) in enumerate(item.references, start=1)
        ]

    def test_fastapi_create_job_entry(self) -> None:
        api_cases = [item for item in CASES if item.api_expected_status is not None]
        self.assertGreaterEqual(len(api_cases), 6)
        failures: list[str] = []
        for item in api_cases:
            with self.subTest(case=item.name):
                self.manager.submitted.clear()
                response = self.client.post(
                    "/api/jobs",
                    data=self.api_form(item),
                    files=self.api_files(item),
                    headers={
                        self.server.LOCAL_MUTATION_HEADER: self.server.LOCAL_MUTATION_VALUE
                    },
                )
                if response.status_code != item.api_expected_status:
                    failures.append(
                        f"{item.name}: verdict={item.verdict}, expected HTTP "
                        f"{item.api_expected_status}, got {response.status_code}; "
                        f"body={response.text[:500]!r}; "
                        f"expected_diagnostics={item.expected_diagnostics!r}"
                    )
                    continue
                if item.verdict == "recoverable":
                    if len(self.manager.submitted) != 1:
                        failures.append(f"{item.name}: HTTP accepted but no job persisted")
                        continue
                    job = self.manager.submitted[0]
                    request_path = self.manager.jobs_dir / job["id"] / "request.json"
                    if not request_path.exists():
                        failures.append(f"{item.name}: request.json was not persisted")
                        continue
                    request = json.loads(request_path.read_text(encoding="utf-8"))
                    if request.get("prompt") != item.prompt:
                        failures.append(f"{item.name}: authoring prompt changed before persistence")
                elif self.manager.submitted:
                    failures.append(f"{item.name}: fatal API input created a job")
        if failures:
            self.fail("FastAPI adversarial entry failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
