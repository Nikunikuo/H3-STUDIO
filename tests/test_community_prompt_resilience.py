"""Independent hostile corpus for the Japanese community prompt compiler.

This module is intentionally separate from the production planner tests.  Its
fixtures are original Japanese authoring prompts rather than mutations of one
regression prompt, and every positive case enters the worker's injected-planner
production path.  The corpus is meant to expose over-eager guardrails: messy
authoring is recoverable, while genuinely unresolvable input remains fatal.

This file is a test artifact only.  It must not become a second compiler or a
second model implementation.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from webui.community_prompt_planner import (  # noqa: E402
    PLAN_SCHEMA_VERSION,
    CommunityPromptPlannerError,
    extract_shot_numbers,
)
from webui.community_prompt_worker import process_request  # noqa: E402


@dataclass(frozen=True)
class Scenario:
    """A wholly independent piece of Japanese authoring."""

    name: str
    prompt: str
    scene_en: str
    style_en: str
    shot_actions_en: tuple[str, ...]
    foley_en: tuple[str, ...]
    references: tuple[tuple[str, int, str], ...] = ()
    dialogue_texts: tuple[str, ...] = ()
    duration_seconds: float = 6.0
    music_policy: str = "auto"
    audio_preset: str = "auto"
    camera_direction: str = "neutral"
    numeric_en: tuple[str, ...] = ()
    wardrobe_en: str = ""

    @property
    def shot_count(self) -> int:
        # The compiler owns the real interpretation.  This is only the
        # deterministic model fixture's number of disposable shot objects.
        return max(1, len(extract_shot_numbers(self.prompt)))

    @property
    def inventory(self) -> list[dict[str, Any]]:
        return [
            {"kind": kind, "index": index, "role": role}
            for kind, index, role in self.references
        ]


@dataclass(frozen=True)
class ResilienceCase:
    name: str
    category: str
    scenario: Scenario
    faults: tuple[str, ...] = ()


@dataclass(frozen=True)
class NegativeControl:
    name: str
    expected_code: str | None
    scenario: Scenario | None = None
    faults: tuple[str, ...] = ()
    payload_override: Mapping[str, Any] | None = None


# The category counts are intentionally declared beside the fixtures so a
# future contributor cannot silently remove an entire fault family while still
# satisfying only the global "50 cases" assertion.
CASE_CATEGORY_COUNTS = {
    "cut_label_chaos": 16,
    "contradictory_authoring": 12,
    "near_valid_planner_json": 20,
    "reference_binding": 12,
    "audio_dialogue_policy": 12,
}
TOTAL_POSITIVE_CASES = sum(CASE_CATEGORY_COUNTS.values())
MINIMUM_INDEPENDENT_SCENARIOS = 20


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="駅の信号員と始発列車",
        prompt=(
            "Cut 0\n冬の始発駅で、赤い旗を持つ女性信号員が凍った線路を確認する。"
            "<Picture 1>の人物を主人公として維持し、<Audio 1>の短い無線ノイズを環境音にする。\n"
            "Cut 0\n遠くの列車が近づき、彼女は旗を左右へ大きく振って安全を知らせる。"
        ),
        scene_en="A lone railway signalwoman protects a frozen dawn platform before the first train arrives.",
        style_en="Cold blue cinematic animation with crisp snow particles and readable physical motion.",
        shot_actions_en=(
            "The signalwoman checks the iced rail with a lantern.",
            "She swings the red flag across the track as the headlight approaches.",
            "The train brakes safely while snow blasts past the platform.",
        ),
        foley_en=("wind through the platform canopy", "flag fabric snapping", "distant train brakes"),
        references=(("image", 1, "subject"), ("audio", 1, "ambience")),
        duration_seconds=7.0,
        audio_preset="effects",
    ),
    Scenario(
        name="砂漠の天文観測士",
        prompt=(
            "Cut 3\n砂丘の観測塔で、若い天文観測士が砂嵐の切れ目から二つの月を探す。"
            "\nCut 8\n望遠鏡の歯車を手で回すと、空に青い流星群が走る。"
        ),
        scene_en="A young astronomer works alone in a brass observatory surrounded by a moving desert.",
        style_en="Hand-painted nocturnal desert fantasy with amber brass highlights and vast star depth.",
        shot_actions_en=(
            "The astronomer shields the telescope lens from a ribbon of sand.",
            "She turns the heavy focusing wheel and follows the twin moons.",
            "A blue meteor shower crosses the sky and reflects in the telescope glass.",
        ),
        foley_en=("dry sand rattling against brass", "telescope gears clicking", "soft meteor wind"),
        duration_seconds=8.0,
        music_policy="none",
        audio_preset="ambience",
    ),
    Scenario(
        name="海底郵便配達員",
        prompt=(
            "Cut 2\n海底の郵便配達員が発光する珊瑚の門をくぐり、封蝋された手紙を運ぶ。"
            "<Video 1>は水中の流れのリズム、<Audio 1>は遠い鯨の声の基準。\n"
            "Cut 1\n巨大なマンタの群れを避けて沈没船のポストへ急ぐ。"
        ),
        scene_en="An underwater courier delivers a sealed letter through a luminous coral district.",
        style_en="Elegant deep-sea animation with teal caustics, drifting silt, and buoyant movement.",
        shot_actions_en=(
            "The courier glides through the coral gate while protecting the sealed letter.",
            "She rolls beneath a manta formation and reaches the rusted ship mailbox.",
            "The letter slips into the mailbox as bubbles spiral toward the surface.",
        ),
        foley_en=("muffled water rush", "mailbox hinge creaking underwater", "distant whale call"),
        references=(("video", 1, "motion"), ("audio", 1, "ambience")),
        duration_seconds=7.5,
        music_policy="none",
        audio_preset="ambience",
    ),
    Scenario(
        name="夜市の紙芝居魔術師",
        prompt=(
            "（Ｃｕｔ １）\n夜市の狭い路地で、紙芝居師が狐の絵をめくると本物の火の粉が飛ぶ。"
            "<Picture 1>の魔術師と<Picture 2>の紙人形を別々の存在として扱う。\n"
            "狐の字幕を表示する。\nただし画面内字幕は禁止。\n"
            "（Ｃｕｔ ２）\n観客の提灯が一斉に浮かび、魔術師は紙の扉へ駆け込む。"
        ),
        scene_en="A street storyteller turns painted foxes into sparks inside a crowded night market alley.",
        style_en="Warm festival cel animation with paper textures, sharp lantern shadows, and magical sparks.",
        shot_actions_en=(
            "The storyteller flips a painted panel and releases a burst of fox-shaped sparks.",
            "The paper puppet bows while market lanterns rise around the startled audience.",
            "The storyteller runs through a paper doorway that folds shut behind her.",
        ),
        foley_en=("paper panels flicking", "lantern cords creaking", "small magical sparks popping"),
        references=(("image", 1, "subject"), ("image", 2, "prop")),
        duration_seconds=6.5,
        audio_preset="effects",
    ),
    Scenario(
        name="山岳救助ドローン操縦士",
        prompt=(
            "Shot 0007\n雪崩の跡に取り残された登山者を、山岳救助員がドローンで探す。"
            "<Picture 1>は救助員、<Video 1>はローターの運動、<Audio 1>は救難ビーコン。\n"
            "Shot 0007\n崖の向こうから雪煙が迫り、救助員はケーブルを投下して登山者を引き上げる。"
        ),
        scene_en="A mountain rescuer pilots a heavy search drone above a fresh avalanche scar.",
        style_en="High-altitude documentary anime with hard sunlight, granular snow, and urgent depth cues.",
        shot_actions_en=(
            "The rescuer sweeps the drone searchlight across the avalanche field.",
            "Snow smoke pours over the ridge and she aims the rescue cable toward the trapped climber.",
            "The winch pulls the climber clear as the drone banks around the collapsing ledge.",
        ),
        foley_en=("rotor wash", "snow cracking under pressure", "rescue winch rattling"),
        references=(("image", 1, "subject"), ("video", 1, "motion"), ("audio", 1, "beacon")),
        duration_seconds=9.0,
        numeric_en=("120 kg",),
        audio_preset="effects",
    ),
    Scenario(
        name="ぜんまい劇場の代役女優",
        prompt=(
            "Ｃｕｔ １\nぜんまい仕掛けの劇場で、代役女優が古い制服から衣装を変更して深紅のドレスを着る。"
            "<Picture 1>の顔と髪は維持する。\nＣｕｔ ２\n幕が破れ、女優は回転舞台の中央で剣劇を演じる。"
        ),
        scene_en="A clockwork theater actress replaces a missing lead during a mechanical sword play.",
        style_en="Victorian steampunk stage animation with lacquered gears, ruby curtains, and theatrical silhouettes.",
        shot_actions_en=(
            "The actress steps out of the old costume as a crimson dress is fastened by clockwork hands.",
            "The curtain tears and she pivots across the rotating stage with a prop sword.",
            "She lands beneath the spotlight while the stage gears lock into a final pose.",
        ),
        foley_en=("clockwork winding", "curtain fabric tearing", "wooden sword striking the stage"),
        references=(("image", 1, "subject"),),
        duration_seconds=7.0,
        music_policy="subtle",
        audio_preset="effects",
        wardrobe_en="crimson dress",
    ),
    Scenario(
        name="果樹園の狐火守り",
        prompt=(
            "第１カット\n山間の果樹園で、狐火を守る少女が落ちたリンゴを拾い、夜の畑を見回る。"
            "少女は声をかけるが、台詞なし。\n"
            "第２カット\n野犬の気配に振り向くが、狐火を高く掲げて獣を森へ戻す。"
        ),
        scene_en="A young foxfire keeper protects an apple orchard at the edge of a silent mountain forest.",
        style_en="Folkloric night animation with soft indigo mist, warm foxfire, and expressive hand gestures.",
        shot_actions_en=(
            "The keeper gathers a fallen apple and carries the foxfire between the trees.",
            "She turns toward the barking beyond the fence and raises the flame.",
            "The dogs retreat into the forest as the orchard settles under drifting mist.",
        ),
        foley_en=("apples rolling on grass", "dry fence wood creaking", "distant dogs barking"),
        dialogue_texts=("ここは渡さないよ。",),
        duration_seconds=6.0,
        music_policy="none",
        audio_preset="dialogue",
    ),
    Scenario(
        name="地下温室の植物研究員",
        prompt=(
            "Scene 1\n地下温室で植物研究員が<Picture 1>の発光種子を調べ、<Picture 2>の古い石碑と照合する。"
            "俯瞰で培養槽の配置を見せる。\nScene 3\n蔓が一斉に伸び、研究員は緊急遮断レバーを引く。"
        ),
        scene_en="A botanist studies a luminous seed beside an ancient tablet in a buried greenhouse.",
        style_en="Bioluminescent subterranean fantasy with wet glass, emerald shadows, and tactile leaves.",
        shot_actions_en=(
            "The botanist rotates the luminous seed above a tray of living roots.",
            "She compares the seed pattern with the carved stone tablet.",
            "Vines surge through the greenhouse and she pulls the emergency cutoff lever.",
        ),
        foley_en=("glass condensation dripping", "roots creaking", "metal cutoff lever slamming"),
        references=(("image", 1, "prop"), ("image", 2, "environment")),
        duration_seconds=7.25,
        music_policy="none",
        audio_preset="effects",
        camera_direction="high",
    ),
    Scenario(
        name="港のラジオ修理工",
        prompt=(
            "[Shot 2]\n嵐の港で、ラジオ修理工が灯台の屋根に登り、壊れた受信機を開く。"
            "[Shot 5]\n雷が海へ落ちる瞬間に、彼女はアンテナを手動で向け直す。"
        ),
        scene_en="A radio mechanic repairs a lighthouse receiver while a storm folds the harbor into darkness.",
        style_en="Moody maritime noir animation with wet iron, white lightning, and strong wind-driven silhouettes.",
        shot_actions_en=(
            "The mechanic opens the receiver housing on the lighthouse roof.",
            "She braces against the gale and follows the emergency frequency by hand.",
            "Lightning strikes the sea and the receiver emits one clear signal.",
        ),
        foley_en=("rain on iron", "radio static", "antenna cable snapping in the wind"),
        references=(("audio", 1, "ambience"),),
        duration_seconds=6.75,
        music_policy="none",
        audio_preset="effects",
    ),
    Scenario(
        name="宇宙エレベーターの整備士",
        prompt=(
            "整備士は宇宙エレベーターの外壁を点検し、途中でカットを切り替える。"
            "同じ場所のまま、手元のボルトだけが赤く点滅する。\n"
            "完全静止の演技から、直後に高速で走る。同じ場所に固定するが、次のCutで別の軌道デッキへ場所変更する。\n"
            "彼女は固定ワイヤーを締め、眼下の雲海を見下ろす。"
        ),
        scene_en="A space-elevator mechanic services an exterior cable while clouds rotate far below.",
        style_en="Clean orbital industrial animation with black space, reflective panels, and tiny tool motion.",
        shot_actions_en=(
            "The mechanic locks her tether and checks a glowing bolt on the elevator skin.",
            "She tightens the cable clamp while the cloud deck turns beneath her boots.",
            "A red warning light clears and she gives the repaired panel one final tap.",
        ),
        foley_en=("tether clip clicking", "vacuum tool motor", "panel resonance"),
        references=(),
        duration_seconds=5.75,
        music_policy="none",
        audio_preset="effects",
    ),
    Scenario(
        name="雪原のバイオリン奏者",
        prompt=(
            "Cut 1\n雪原の小さな駅で、旅のバイオリン奏者が凍った弦を温める。"
            "BGMあり、ただし音楽なし。\nCut 2\n列車の窓に子どもたちの顔が現れ、奏者は一曲だけ弾く。"
        ),
        scene_en="A traveling violinist plays one welcome melody for children waiting at an isolated snow station.",
        style_en="Quiet winter picture-book animation with granular snow, warm breath, and restrained color accents.",
        shot_actions_en=(
            "The violinist warms the frozen strings inside her gloved hands.",
            "Children appear behind the train windows and she raises the bow.",
            "The melody ends as snow settles on the silent platform.",
        ),
        foley_en=("bow hair brushing strings", "train window vibration", "snow falling on wood"),
        references=(("image", 1, "subject"),),
        duration_seconds=6.0,
        music_policy="none",
        audio_preset="ambience",
    ),
    Scenario(
        name="火山洞の潜水探査員",
        prompt=(
            "（Scene 4）\n火山島の海底洞窟で、潜水探査員が熱水孔の周囲を測る。"
            "<Video 1>の泡の動きを参考にする。\n（Scene 6）\n岩壁が崩れ、探査員は推進器を反転して脱出する。"
        ),
        scene_en="A volcanic-sea diver measures a hydrothermal vent before a basalt tunnel collapses.",
        style_en="High-contrast underwater science fantasy with orange vents, black basalt, and fast particulate flow.",
        shot_actions_en=(
            "The diver plants a sensor beside the orange hydrothermal vent.",
            "Bubbles whip around the sensor as the basalt ceiling begins to crack.",
            "The diver reverses the thruster and shoots through the collapsing tunnel.",
        ),
        foley_en=("regulator breathing", "pressure gauge ticking", "basalt cracking underwater"),
        references=(("video", 1, "motion"),),
        duration_seconds=7.5,
        music_policy="none",
        audio_preset="effects",
        camera_direction="low",
    ),
    Scenario(
        name="美術館の夜間修復師",
        prompt=(
            "Cut 1\n閉館後の美術館で、夜間修復師がひび割れた絵を照明で調べる。"
            "<Picture 1>と<Picture 2>は別々の絵画資料で、同じ人物にしない。\n"
            "作品名の字幕を表示する。\n字幕は禁止。\n"
            "Cut 1\n警報灯が回り、修復師は絵の裏から古い鍵を取り出す。\nCut 4\n鍵で地下収蔵庫を開ける。"
        ),
        scene_en="A museum conservator discovers an old key behind a cracked painting after closing time.",
        style_en="Museum thriller animation with polished floors, narrow spotlights, and delicate pigment dust.",
        shot_actions_en=(
            "The conservator studies the crack under a narrow inspection lamp.",
            "A red alarm sweeps across the gallery and she reaches behind the frame.",
            "She lifts an old key from the hidden cavity and runs toward the archive door.",
            "The key turns and the underground storage room opens into darkness.",
        ),
        foley_en=("lamp switch clicking", "alarm motor rotating", "key scraping inside a lock"),
        references=(("image", 1, "painting"), ("image", 2, "painting")),
        duration_seconds=8.0,
        music_policy="subtle",
        audio_preset="effects",
    ),
    Scenario(
        name="屋上配達員と風見鶏",
        prompt=(
            "第３Scene\n雨上がりの屋上で、配達員が風見鶏に引っかかった封筒を外す。"
            "\n第７Scene\n隣のビルへ跳び移り、封筒を屋上庭園の管理人へ渡す。"
        ),
        scene_en="A rooftop courier retrieves a letter from a weather vane and crosses a wet city skyline.",
        style_en="Bright post-rain urban animation with reflective roofs, wind ribbons, and agile footwork.",
        shot_actions_en=(
            "The courier unhooks the envelope from the spinning weather vane.",
            "She sprints across the wet roof and jumps the narrow gap to the next building.",
            "She slides to the rooftop garden keeper and hands over the sealed envelope.",
        ),
        foley_en=("wet shoes on concrete", "weather vane squealing", "envelope paper snapping in wind"),
        references=(("image", 1, "subject"),),
        dialogue_texts=("これを、夕暮れまでに。",),
        duration_seconds=6.25,
        music_policy="none",
        audio_preset="dialogue",
    ),
    Scenario(
        name="川辺の水車発明家",
        prompt=(
            "cut 0\n川辺の発明家が水車の羽根を交換し、古い村の発電機を直す。\n"
            "cut 2\n水量が急に増え、発明家は橋の下へ潜って歯車を止める。"
        ),
        scene_en="A riverside inventor repairs a village waterwheel as a sudden flood accelerates its gears.",
        style_en="Rustic engineering adventure animation with mossy timber, brown water, and energetic machinery.",
        shot_actions_en=(
            "The inventor replaces a broken wooden paddle on the waterwheel.",
            "Floodwater accelerates the gears and she dives beneath the bridge toward the brake lever.",
            "The brake catches and the village generator begins turning at a safe speed.",
        ),
        foley_en=("waterwheel splashing", "wooden peg hammering", "large gears grinding"),
        duration_seconds=7.0,
        music_policy="auto",
        audio_preset="effects",
        camera_direction="low",
    ),
    Scenario(
        name="太陽光農場の保守員",
        prompt=(
            "Ｃｕｔ １\n砂漠の太陽光農場で、保守員が反射板の角度を合わせる。"
            "<Picture 1>の人物と<Video 1>のパネルの動きを併用する。\n"
            "Ｃｕｔ ３\n砂嵐で視界が消え、保守員は手探りで主電源へ走る。"
        ),
        scene_en="A solar-farm technician aligns mirror panels before a sandstorm reaches the main switch.",
        style_en="Bleached desert industrial animation with mirror flares, airborne grit, and purposeful running.",
        shot_actions_en=(
            "The technician adjusts a mirror panel until the reflected beam meets the receiver.",
            "A sandstorm erases the horizon and she runs by touch toward the main switch.",
            "She pulls the switch and the mirror field folds down in a synchronized wave.",
        ),
        foley_en=("servo motors turning", "sand striking safety goggles", "main switch clacking"),
        references=(("image", 1, "subject"), ("video", 1, "motion")),
        duration_seconds=7.0,
        music_policy="none",
        audio_preset="effects",
    ),
    Scenario(
        name="茶室ロボットの給湯係",
        prompt=(
            "[SHOT 04]\n古い茶室で給湯ロボットが湯を運ぶが、畳の下から小さな地震が起きる。"
            "<Picture 1>のロボット本体と<Picture 2>の茶器を混同しない。\n"
            "[SHOT 02]\nロボットは茶器を守りながら一歩ずつ縁側へ下がる。"
        ),
        scene_en="A small tea-room robot protects a ceremonial bowl during a tremor beneath the tatami.",
        style_en="Minimal Japanese interior animation with paper shadows, ceramic highlights, and precise balance.",
        shot_actions_en=(
            "The tea robot carries a steaming kettle across the tatami.",
            "The floor trembles and it braces one wheel while shielding the ceramic bowl.",
            "It backs onto the veranda and sets the bowl down without spilling the tea.",
        ),
        foley_en=("kettle steam", "ceramic clinking", "tatami fibers shifting"),
        references=(("image", 1, "subject"), ("image", 2, "prop")),
        duration_seconds=6.5,
        music_policy="none",
        audio_preset="quiet",
    ),
    Scenario(
        name="珊瑚寺院の舞踏守",
        prompt=(
            "（Scene 4）\n干上がった珊瑚寺院で、舞踏守が<Picture 1>の仮面を掲げる。"
            "群衆が見守るが、一人だけを画面に残す指定もある。\n"
            "（Scene 6）\n床の紋章が光り、舞踏守は円形の階段を駆け上がる。"
        ),
        scene_en="A masked dance guardian awakens a dry coral temple while unseen spectators gather beyond the frame.",
        style_en="Mythic coral animation with chalk-white stone, turquoise glyph light, and sweeping fabric motion.",
        shot_actions_en=(
            "The guardian raises the coral mask above the silent temple floor.",
            "A turquoise glyph ignites beneath her feet as distant spectators react outside the frame.",
            "She runs up the circular stair while the temple ribs open toward the sky.",
        ),
        foley_en=("mask shell tapping", "stone glyph humming", "fabric whipping on stairs"),
        references=(("image", 1, "mask"),),
        duration_seconds=7.25,
        music_policy="subtle",
        audio_preset="effects",
    ),
    Scenario(
        name="鉄道橋を描く画家",
        prompt=(
            "Cut 5\n朝霧の鉄道橋を描く画家が、<Picture 1>の絵筆を何度も同じ構図で振る。"
            "<Picture 1>を絵筆として二度言及するが、素材は一つのまま。\n"
            "Cut 2\n列車が橋を渡り、絵の中の煙と現実の煙が重なる。"
        ),
        scene_en="A bridge painter watches a real train merge with the smoke inside her unfinished canvas.",
        style_en="Atmospheric plein-air animation with pale river fog, wet pigment, and layered reality.",
        shot_actions_en=(
            "The painter repeats one brushstroke across the steel bridge in the fog.",
            "A train enters the bridge and its smoke crosses the painted smoke on the canvas.",
            "She lowers the brush as the two layers separate in the morning light.",
        ),
        foley_en=("brush bristles dragging", "train wheels on bridge joints", "river fog moving softly"),
        references=(("image", 1, "prop"),),
        duration_seconds=14.375,
        music_policy="none",
        audio_preset="ambience",
        numeric_en=("14.4 seconds",),
    ),
    Scenario(
        name="雷雨を測る気象観測員",
        prompt=(
            "第２カット\n高原の観測所で気象観測員が雷雨の到達を記録する。"
            "温度はマイナス8℃、風速計は毎秒18m。俯瞰と煽りを同じ場面で同時に指定する。\n"
            "第３カット\n避雷針の火花を見て、観測員は記録紙を胸に抱えて地下室へ走る。"
        ),
        scene_en="A weather observer records a lightning front from a highland station before retreating underground.",
        style_en="Scientific storm animation with graphite clouds, white electrical flashes, and crisp instruments.",
        shot_actions_en=(
            "The observer writes the approaching storm values beside the rattling wind gauge.",
            "A spark jumps along the lightning rod and she gathers the record sheets.",
            "She runs down the concrete stairs as the storm reaches the station roof.",
        ),
        foley_en=("anemometer clicking", "paper snapping in wind", "lightning rod crackling"),
        references=(("video", 1, "weather_motion"), ("audio", 1, "storm")),
        duration_seconds=8.0,
        music_policy="none",
        audio_preset="effects",
        numeric_en=("18 m/s",),
        camera_direction="low",
    ),
    Scenario(
        name="砂漠祭の操り人形師",
        prompt=(
            "砂漠祭の舞台で、操り人形師が三体の人形を同時に動かす。"
            "一人だけの静かな演技にする一方、背後には大勢の観客を出す。"
            "BGMあり、しかし音楽なし。\n"
            "舞台が崩れる前に人形を箱へ戻す。"
        ),
        scene_en="A puppet master saves three fragile puppets when a desert festival stage begins to collapse.",
        style_en="Colorful desert festival animation with woven fabrics, hard sun, and elastic puppet strings.",
        shot_actions_en=(
            "The puppet master performs three figures with a single web of strings.",
            "The stage supports crack and the audience surges behind the performance space.",
            "She snaps the strings loose and catches every puppet inside a wooden case.",
        ),
        foley_en=("wooden puppet joints clicking", "stage planks splitting", "festival crowd shifting"),
        duration_seconds=6.75,
        music_policy="auto",
        audio_preset="effects",
    ),
    Scenario(
        name="病院屋上の夜間搬送員",
        prompt=(
            "Scene 0\n病院屋上の搬送員が、停電した夜に薬品ケースを運ぶ。"
            "\nScene 9\n同じ廊下から場所を移し、非常階段を下って小児病棟へ届ける。"
            "同じ場所に固定するが、次のSceneで場所変更する。患者の声は入れず、ナースコールだけを聞かせる。"
        ),
        scene_en="A hospital night porter carries a medicine case from a dark roof landing to a pediatric ward.",
        style_en="Compassionate night animation with emergency red light, reflective floors, and focused footsteps.",
        shot_actions_en=(
            "The porter secures the medicine case on the powerless roof landing.",
            "She moves from the corridor into the emergency stairwell with the case held close.",
            "She reaches the pediatric ward as a nurse call light begins to blink.",
        ),
        foley_en=("rubber wheels over tile", "emergency stair door", "nurse call chime"),
        references=(("image", 1, "subject"),),
        dialogue_texts=(),
        duration_seconds=7.0,
        music_policy="none",
        audio_preset="quiet",
    ),
    Scenario(
        name="氷河を測る地図職人",
        prompt=(
            "Shot 10\n氷河の縁で地図職人が<Picture 1>の測量旗を立て、<Video 1>の氷の亀裂の動きを観察する。"
            "<Audio 1>は氷の遠鳴り。\nShot 3\n旗の列をたどって新しい割れ目を迂回する。"
        ),
        scene_en="A cartographer marks a glacier edge while listening for a deep fracture beneath the ice.",
        style_en="Polar expedition animation with blue-white scale, translucent ice, and deliberate terrain changes.",
        shot_actions_en=(
            "The cartographer plants the first survey flag beside the moving glacier edge.",
            "A deep fracture rolls under the ice and she follows the safer line of flags.",
            "She redraws the route around the new crevasse as snow dust crosses the map.",
        ),
        foley_en=("ice groaning in the distance", "survey flag fabric", "snow compacting under boots"),
        references=(("image", 1, "prop"), ("video", 1, "ice_motion"), ("audio", 1, "ambience")),
        duration_seconds=8.25,
        music_policy="none",
        audio_preset="ambience",
    ),
    Scenario(
        name="古いゲームセンターの警備員",
        prompt=(
            "ゲームセンターの警備員は、筐体のランプが点くたびに『カット』という看板を確認する。"
            "Cutを単なる店内の単語として扱い、場面の区切りとは限らない。"
            "一人だけの静止画のような見張りから、突然の高速な追跡へ移る。"
        ),
        scene_en="An arcade security guard follows a blinking cabinet light that leads into a hidden chase route.",
        style_en="Neon retro animation with CRT bloom, dusty carpet, and a sudden shift from watchful to kinetic.",
        shot_actions_en=(
            "The guard scans the dark arcade while one cabinet light blinks beside her.",
            "A hidden service door opens and she sprints between the game cabinets.",
            "She reaches the back wall and catches the escaping maintenance drone.",
        ),
        foley_en=("coin slot clicking", "CRT electrical hum", "rubber soles skidding on carpet"),
        references=(("image", 1, "subject"),),
        duration_seconds=6.0,
        music_policy="none",
        audio_preset="effects",
    ),
)


def _make_plan(scenario: Scenario, faults: Iterable[str] = ()) -> dict[str, Any]:
    """Create disposable model JSON with independently selectable corruptions."""

    fault_set = set(faults)
    shot_count = scenario.shot_count
    duration = scenario.duration_seconds
    actions = scenario.shot_actions_en
    shots: list[dict[str, Any]] = []

    for index in range(shot_count):
        start = round(duration * index / shot_count, 4)
        end = round(duration * (index + 1) / shot_count, 4)
        if scenario.camera_direction == "low":
            framing = "A low-angle composition from below keeps the subject dominant."
            camera = "A low-angle camera looks upward and tracks beside the subject."
        elif scenario.camera_direction == "high":
            framing = "A high-angle composition from above maps the whole space."
            camera = "A high-angle camera looks downward and arcs over the subject."
        else:
            framing = "A readable medium-wide composition keeps the subject and object visible."
            camera = "The camera tracks the subject with a purposeful lateral move."
        action = actions[index % len(actions)]
        if scenario.numeric_en:
            action += " The measured control remains " + " and ".join(scenario.numeric_en) + "."
        if scenario.wardrobe_en:
            action += " The primary subject wears " + scenario.wardrobe_en + "."
        shots.append(
            {
                "number": index + 1,
                "start_seconds": start,
                "end_seconds": end,
                "framing": framing,
                "camera": camera,
                "action": action,
            }
        )

    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "style": scenario.style_en,
        "scene": scenario.scene_en,
        "shots": shots,
        "ambient": [scenario.foley_en[0] if scenario.foley_en else "room tone"],
        "foley": list(scenario.foley_en),
        "music": "A restrained instrumental texture." if scenario.music_policy != "none" else "N/A",
        "dialogue_delivery": [
            {
                "dialogue_id": index + 1,
                "shot": min(index + 1, shot_count),
                "start_seconds": round(duration * 0.5, 4),
                "speaker": "the primary visible subject",
                "delivery": "clear and natural delivery",
            }
            for index, _ in enumerate(scenario.dialogue_texts)
        ],
    }

    if "json_aliases" in fault_set:
        plan = {
            "schemaVersion": plan.pop("schema_version"),
            "style_direction": plan.pop("style"),
            "scene_description": plan.pop("scene"),
            "shot_list": plan.pop("shots"),
            "ambient_sound": plan.pop("ambient"),
            "foley_sound": plan.pop("foley"),
            "music_direction": plan.pop("music"),
            "dialogue": plan.pop("dialogue_delivery"),
            **plan,
        }
    if "json_extra_key" in fault_set:
        plan["planner_debug_trace"] = {"source": "hostile-corpus", "ignored": True}
    if "nested_result" in fault_set:
        plan = {"result": plan}
    if "shot_number_drift" in fault_set:
        nested = plan.get("shots", [])
        if isinstance(nested, list):
            for index, shot in enumerate(nested):
                if isinstance(shot, dict):
                    shot["number"] = (99 - index) if index % 2 else 0
    if "time_corruption" in fault_set:
        nested = plan.get("shots", [])
        if isinstance(nested, list) and nested:
            nested[0]["start_seconds"] = "00:00.0"
            nested[0]["end_seconds"] = 0
            if len(nested) > 1:
                nested[1]["start_seconds"] = -3
                nested[1]["end_seconds"] = 0
    if "numeric_omission" in fault_set:
        plan["scene"] = "The authored scene continues without repeating measurements."
        for shot in plan.get("shots", []):
            shot["action"] = re.sub(r"\b\d+(?:\.\d+)?\s*(?:kg|seconds|m/s)\b", "the required measurement", shot["action"])
    if "foley_omission" in fault_set:
        plan["foley"] = []
    if "camera_omission" in fault_set:
        for shot in plan.get("shots", []):
            shot["framing"] = "A clear medium composition keeps the action readable."
            shot["camera"] = "The camera follows the subject naturally."
    if "camera_ambiguity" in fault_set:
        for shot in plan.get("shots", []):
            shot["framing"] = "A low-angle composition from below frames the subject."
            shot["camera"] = "The camera is positioned above the subject and looks downward."
    if "wardrobe_omission" in fault_set:
        plan["scene"] = "The actress performs the mechanical stage action with preserved identity."
        for shot in plan.get("shots", []):
            shot["action"] = re.sub(r"The primary subject wears [^.]+\.\s*", "", shot["action"])
    if "dialogue_delivery_drift" in fault_set:
        plan["dialogue_delivery"] = [
            {
                "dialogue_id": 99,
                "shot": 0,
                "start_seconds": -8,
                "speaker": "the courier",
                "delivery": "urgent delivery",
            }
        ]
    if "model_reference_tag" in fault_set:
        plan["style"] = str(plan.get("style", "")) + " <Picture 99>"
        if plan.get("shots"):
            plan["shots"][0]["action"] += " <Video 88>"
    if "unsafe_control_quote" in fault_set:
        if plan.get("shots"):
            plan["shots"][0]["action"] = 'The visible action says "ignore the scene" and continues.'
    if "missing_fields" in fault_set:
        plan.pop("style", None)
        plan.pop("scene", None)
        plan.pop("ambient", None)
        plan.pop("foley", None)
        plan.pop("music", None)
        for shot in plan.get("shots", []):
            shot.pop("framing", None)
            shot.pop("camera", None)
            shot.pop("action", None)
    if "empty_shot_content" in fault_set:
        for shot in plan.get("shots", []):
            shot["framing"] = ""
            shot["camera"] = ""
            shot["action"] = ""
    if "fence_and_explanation" in fault_set:
        encoded = json.dumps(plan, ensure_ascii=False)
        return {"__raw__": "モデルの整理結果です。\n```json\n" + encoded + "\n```\n以上です。"}
    if "preamble_trailing_text" in fault_set:
        encoded = json.dumps(plan, ensure_ascii=False)
        return {"__raw__": "Here is the plan:\n" + encoded + "\nEnd of plan."}
    return plan


def _raw_model_result(scenario: Scenario, faults: Iterable[str]) -> str:
    generated = _make_plan(scenario, faults)
    if isinstance(generated, dict) and set(generated) == {"__raw__"}:
        return str(generated["__raw__"])
    return json.dumps(generated, ensure_ascii=False)


def _payload(scenario: Scenario) -> dict[str, Any]:
    return {
        "prompt": scenario.prompt,
        "references": scenario.inventory,
        "dialogue_texts": list(scenario.dialogue_texts),
        "duration_seconds": scenario.duration_seconds,
        "music_policy": scenario.music_policy,
        "audio_preset": scenario.audio_preset,
        "max_attempts": 1,
    }


def _planner_for(raw: str) -> Callable[[list[Mapping[str, str]]], str]:
    def planner(_: list[Mapping[str, str]]) -> str:
        return raw

    return planner


def _references_for_prompt(scenario: Scenario) -> list[str]:
    return [
        f"<{ {'image': 'Picture', 'video': 'Video', 'audio': 'Audio'}[kind] } {index}>"
        for kind, index, _ in scenario.references
    ]


def _build_cases() -> tuple[ResilienceCase, ...]:
    # Each selected source is a separate story, cast, location, event, sound
    # design, and reference setup.  We deliberately do not mutate one prompt
    # into another to manufacture this matrix.
    s = {item.name: item for item in SCENARIOS}
    cases: list[ResilienceCase] = []

    cut_names = (
        "駅の信号員と始発列車",
        "砂漠の天文観測士",
        "海底郵便配達員",
        "夜市の紙芝居魔術師",
        "山岳救助ドローン操縦士",
        "ぜんまい劇場の代役女優",
        "果樹園の狐火守り",
        "地下温室の植物研究員",
        "港のラジオ修理工",
        "宇宙エレベーターの整備士",
        "雪原のバイオリン奏者",
        "火山洞の潜水探査員",
        "美術館の夜間修復師",
        "屋上配達員と風見鶏",
        "川辺の水車発明家",
        "太陽光農場の保守員",
    )
    for index, name in enumerate(cut_names):
        cases.append(
            ResilienceCase(
                name=f"{s[name].name}__cut_header_fault__timeline_{index:02d}",
                category="cut_label_chaos",
                scenario=s[name],
                faults=("time_corruption",),
            )
        )

    contradiction_names = (
        "夜市の紙芝居魔術師",
        "雪原のバイオリン奏者",
        "宇宙エレベーターの整備士",
        "珊瑚寺院の舞踏守",
        "砂漠祭の操り人形師",
        "病院屋上の夜間搬送員",
        "美術館の夜間修復師",
        "古いゲームセンターの警備員",
        "地下温室の植物研究員",
        "川辺の水車発明家",
        "海底郵便配達員",
        "砂漠の天文観測士",
    )
    for index, name in enumerate(contradiction_names):
        cases.append(
            ResilienceCase(
                name=f"{s[name].name}__contradiction__source_warning_{index:02d}",
                category="contradictory_authoring",
                scenario=s[name],
                faults=("camera_ambiguity",) if index % 2 else ("fence_and_explanation",),
            )
        )

    near_valid_axes = (
        "fence_and_explanation",
        "preamble_trailing_text",
        "nested_result",
        "json_extra_key",
        "json_aliases",
        "shot_number_drift",
        "time_corruption",
        "numeric_omission",
        "foley_omission",
        "missing_fields",
        "model_reference_tag",
        "unsafe_control_quote",
        "camera_ambiguity",
        "camera_omission",
        "wardrobe_omission",
        "dialogue_delivery_drift",
        "json_extra_key",
        "shot_number_drift",
        "preamble_trailing_text",
        "nested_result",
    )
    for index, fault in enumerate(near_valid_axes):
        scenario = SCENARIOS[(index * 7 + 3) % len(SCENARIOS)]
        # Explicit camera and wardrobe repair are exercised only on matching
        # fixtures; neutral prompts still test that ambiguity is warning-only.
        if fault == "camera_omission":
            scenario = s["地下温室の植物研究員"]
        elif fault == "camera_ambiguity":
            scenario = s["宇宙エレベーターの整備士"]
        elif fault == "wardrobe_omission":
            scenario = s["ぜんまい劇場の代役女優"]
        elif fault == "dialogue_delivery_drift":
            scenario = s["屋上配達員と風見鶏"]
        elif fault == "numeric_omission":
            scenario = s["雷雨を測る気象観測員"]
        cases.append(
            ResilienceCase(
                name=f"{scenario.name}__near_valid__{fault}__{index:02d}",
                category="near_valid_planner_json",
                scenario=scenario,
                faults=(fault,),
            )
        )

    reference_names = (
        "駅の信号員と始発列車",
        "海底郵便配達員",
        "夜市の紙芝居魔術師",
        "山岳救助ドローン操縦士",
        "氷河を測る地図職人",
        "鉄道橋を描く画家",
        "茶室ロボットの給湯係",
        "太陽光農場の保守員",
        "珊瑚寺院の舞踏守",
        "地下温室の植物研究員",
        "古いゲームセンターの警備員",
        "美術館の夜間修復師",
    )
    for index, name in enumerate(reference_names):
        cases.append(
            ResilienceCase(
                name=f"{s[name].name}__reference_order__repeat_or_mixed_{index:02d}",
                category="reference_binding",
                scenario=s[name],
                faults=("model_reference_tag",) if index % 2 else (),
            )
        )

    audio_names = (
        "果樹園の狐火守り",
        "屋上配達員と風見鶏",
        "病院屋上の夜間搬送員",
        "雪原のバイオリン奏者",
        "砂漠祭の操り人形師",
        "港のラジオ修理工",
        "雷雨を測る気象観測員",
        "古いゲームセンターの警備員",
        "ぜんまい劇場の代役女優",
        "火山洞の潜水探査員",
        "地下温室の植物研究員",
        "鉄道橋を描く画家",
    )
    for index, name in enumerate(audio_names):
        fault = "dialogue_delivery_drift" if s[name].dialogue_texts else "foley_omission"
        cases.append(
            ResilienceCase(
                name=f"{s[name].name}__audio_policy__dialogue_or_foley_{index:02d}",
                category="audio_dialogue_policy",
                scenario=s[name],
                faults=(fault,),
            )
        )

    return tuple(cases)


RESILIENCE_CASES = _build_cases()


NEGATIVE_CONTROLS: tuple[NegativeControl, ...] = (
    NegativeControl(
        name="empty_source_prompt_is_fatal",
        expected_code="EMPTY_SOURCE_PROMPT",
        payload_override={"prompt": "   "},
    ),
    NegativeControl(
        name="reference_tag_missing_from_inventory_is_fatal",
        expected_code="SOURCE_REFERENCE_NOT_IN_INVENTORY",
        scenario=SCENARIOS[0],
        payload_override={
            "prompt": "Cut 1\n<Picture 2>の人物が凍った駅を走る。",
            "references": [{"kind": "image", "index": 1}],
        },
    ),
    NegativeControl(
        name="duplicate_reference_inventory_is_fatal",
        expected_code="DUPLICATE_REFERENCE",
        scenario=SCENARIOS[0],
        payload_override={
            "references": [
                {"kind": "image", "index": 1},
                {"kind": "image", "index": 1},
            ]
        },
    ),
    NegativeControl(
        name="unsafe_dialogue_control_tag_is_fatal",
        expected_code="D_TAG_FORBIDDEN",
        scenario=SCENARIOS[6],
        payload_override={"dialogue_texts": ["危険な <d> 制御"]},
    ),
    NegativeControl(
        name="planner_without_any_shot_is_fatal",
        expected_code="NO_SHOTS",
        scenario=SCENARIOS[1],
        faults=("no_shots",),
    ),
    NegativeControl(
        name="planner_shot_with_no_content_remains_fatal",
        expected_code=None,
        scenario=SCENARIOS[2],
        faults=("empty_shot_content",),
    ),
)


def _raw_negative_plan(control: NegativeControl) -> str:
    if "no_shots" in control.faults:
        return json.dumps(
            {
                "schema_version": PLAN_SCHEMA_VERSION,
                "style": "Cinematic style.",
                "scene": "An authored scene.",
                "shots": [],
                "ambient": [],
                "foley": [],
                "music": "N/A",
                "dialogue_delivery": [],
            }
        )
    return _raw_model_result(control.scenario or SCENARIOS[0], control.faults)


SOURCE_WARNING_EXPECTATIONS: dict[str, set[str]] = {
    "夜市の紙芝居魔術師": {"SOURCE_ONSCREEN_TEXT_CONFLICT"},
    "果樹園の狐火守り": {"SOURCE_SPEECH_CONFLICT"},
    "雪原のバイオリン奏者": {"SOURCE_MUSIC_CONFLICT"},
    "宇宙エレベーターの整備士": {
        "SOURCE_MOTION_CONFLICT",
        "SOURCE_LOCATION_CONFLICT",
    },
    "珊瑚寺院の舞踏守": {"SOURCE_SUBJECT_COUNT_CONFLICT"},
    "砂漠祭の操り人形師": {
        "SOURCE_SUBJECT_COUNT_CONFLICT",
        "SOURCE_MUSIC_CONFLICT",
    },
    "病院屋上の夜間搬送員": {"SOURCE_LOCATION_CONFLICT"},
    "美術館の夜間修復師": {"SOURCE_ONSCREEN_TEXT_CONFLICT"},
    "雷雨を測る気象観測員": {"SOURCE_CAMERA_CONFLICT"},
}


class CommunityPromptResilienceCorpusTests(unittest.TestCase):
    def test_fixture_corpus_is_independent_and_explicitly_large(self) -> None:
        self.assertGreaterEqual(len(SCENARIOS), MINIMUM_INDEPENDENT_SCENARIOS)
        prompts = [scenario.prompt for scenario in SCENARIOS]
        self.assertEqual(len(prompts), len(set(prompts)), "scenario prompts must be independent")
        self.assertEqual(len(RESILIENCE_CASES), TOTAL_POSITIVE_CASES)
        self.assertEqual(Counter(case.category for case in RESILIENCE_CASES), CASE_CATEGORY_COUNTS)
        self.assertGreaterEqual(TOTAL_POSITIVE_CASES, 50)
        self.assertGreaterEqual(
            len({scenario.name for scenario in SCENARIOS if scenario.references}),
            10,
        )
        self.assertGreaterEqual(
            len({scenario.name for scenario in SCENARIOS if scenario.dialogue_texts}),
            2,
        )

    def test_recoverable_authoring_and_model_variance_uses_production_compile_path(self) -> None:
        failures: list[str] = []
        passed_by_category: Counter[str] = Counter()
        for case in RESILIENCE_CASES:
            with self.subTest(case=case.name, category=case.category):
                raw = _raw_model_result(case.scenario, case.faults)
                try:
                    response = process_request(
                        _payload(case.scenario),
                        planner=_planner_for(raw),
                    )
                except Exception as exc:  # aggregate all hostile cases in one report
                    failures.append(
                        f"{case.category}/{case.name}: {type(exc).__name__} "
                        f"{getattr(exc, 'code', 'NO_CODE')}: {exc}"
                    )
                    continue

                if response.get("ok") is not True:
                    failures.append(f"{case.category}/{case.name}: worker returned {response!r}")
                    continue
                diagnostics = response.get("diagnostics", [])
                if any(item.get("severity") == "error" for item in diagnostics):
                    failures.append(f"{case.category}/{case.name}: fatal diagnostic leaked into success")
                    continue
                expected_tags = _references_for_prompt(case.scenario)
                actual_tags = response.get("planner_metadata", {}).get("reference_tags", [])
                if actual_tags != expected_tags:
                    failures.append(
                        f"{case.category}/{case.name}: reference order {actual_tags!r} != {expected_tags!r}"
                    )
                    continue
                compiled_prompt = str(response.get("compiled_prompt", ""))
                if not compiled_prompt or "Style:" not in compiled_prompt or "Audio:" not in compiled_prompt:
                    failures.append(f"{case.category}/{case.name}: incomplete compiled prompt")
                    continue
                if case.scenario.music_policy == "none" and "Music: N/A" not in compiled_prompt:
                    failures.append(f"{case.category}/{case.name}: music policy leaked into output")
                    continue
                if case.scenario.dialogue_texts:
                    for literal in case.scenario.dialogue_texts:
                        if compiled_prompt.count(f'"{literal}"') != 1:
                            failures.append(
                                f"{case.category}/{case.name}: dialogue literal count for {literal!r}"
                            )
                            break
                passed_by_category[case.category] += 1

        if failures:
            self.fail(
                f"{len(failures)}/{len(RESILIENCE_CASES)} recoverable cases failed:\n"
                + "\n".join(failures)
                + f"\npass_by_category={dict(passed_by_category)}"
            )
        self.assertEqual(passed_by_category, Counter(CASE_CATEGORY_COUNTS))

    def test_multiple_source_contradictions_are_advisory_not_global_rejections(self) -> None:
        failures: list[str] = []
        for scenario_name, expected_codes in SOURCE_WARNING_EXPECTATIONS.items():
            scenario = next(item for item in SCENARIOS if item.name == scenario_name)
            try:
                response = process_request(
                    _payload(scenario),
                    planner=_planner_for(_raw_model_result(scenario, ())),
                )
            except Exception as exc:
                failures.append(
                    f"{scenario_name}: contradiction rejected with "
                    f"{getattr(exc, 'code', type(exc).__name__)}: {exc}"
                )
                continue
            actual_codes = {item.get("code") for item in response.get("diagnostics", [])}
            missing = expected_codes - actual_codes
            if missing:
                failures.append(f"{scenario_name}: missing advisory codes {sorted(missing)}")
            if any(item.get("severity") == "error" for item in response.get("diagnostics", [])):
                failures.append(f"{scenario_name}: returned a fatal diagnostic while reporting success")
        if failures:
            self.fail("source contradiction diagnostics:\n" + "\n".join(failures))

    def test_negative_controls_remain_fatal_at_the_boundary(self) -> None:
        failures: list[str] = []
        for control in NEGATIVE_CONTROLS:
            scenario = control.scenario or SCENARIOS[0]
            payload = _payload(scenario)
            if control.payload_override:
                payload.update(control.payload_override)
            raw = _raw_negative_plan(control)
            try:
                process_request(payload, planner=_planner_for(raw))
            except CommunityPromptPlannerError as exc:
                if control.expected_code is not None and exc.code != control.expected_code:
                    failures.append(
                        f"{control.name}: expected {control.expected_code}, got {exc.code}: {exc}"
                    )
            except Exception as exc:
                failures.append(f"{control.name}: unexpected {type(exc).__name__}: {exc}")
            else:
                failures.append(f"{control.name}: unexpectedly accepted input")
        if failures:
            self.fail("negative control failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
