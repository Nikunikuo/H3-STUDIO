"""Strict Japanese-to-English prompt planning for the community H3 workflow.

The public MiniMax H3 examples that reliably speak Japanese use English for
all scene control and keep the literal Japanese utterance inside one ordinary
pair of double quotes.  This module enforces that contract at a trust boundary:

* literal dialogue is removed before the language model is called;
* the language model may return only a small JSON document;
* reference tags and literal dialogue are rendered by Python, never by the
  model; and
* the finished prompt is rejected if control prose is non-English, references
  were invented, a number/unit disappeared, or dialogue occurs more than once.

Model loading deliberately lives in :mod:`webui.community_prompt_worker` so
the parser, validator, and renderer remain deterministic and unit-testable.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_RELATIVE_PATH = Path("models/prompt_planner/Qwen3-4B-Instruct-2507")
MODEL_PROVENANCE_FILENAME = "h3-studio-provenance.json"
MODEL_LOCK_FILENAME = "prompt_planner.lock.json"
MODEL_RUNTIME_FILE_COUNT = 9
MODEL_RUNTIME_TOTAL_BYTES = 8_056_459_158
PLAN_SCHEMA_VERSION = "h3-community-plan-v1"
LONG_SOURCE_PROMPT_CHAR_THRESHOLD = 2400
LONG_SOURCE_NUMERIC_FACT_THRESHOLD = 5
LONG_SOURCE_OUTPUT_TOKEN_BUDGET = 1050
LONG_SOURCE_DEFAULT_MAX_SHOTS = 6
LONG_SOURCE_MAX_SHOTS = 8
LONG_SOURCE_SECONDS_PER_SHOT = 2.0


_NON_ENGLISH_RE = re.compile(
    r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff\uf900-\ufaff"
    r"\uac00-\ud7af\u0600-\u06ff\u0400-\u04ff]"
)
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff]")
_REFERENCE_TAG_RE = re.compile(
    r"<(Picture|Video|Audio)\s+([1-9][0-9]*)>", re.IGNORECASE
)
_REFERENCE_TAG_CANDIDATE_RE = re.compile(
    r"[<＜][^<>＜＞\r\n]*?[>＞]"
)
_ANY_REFERENCE_LIKE_RE = re.compile(
    r"[<＜]\s*(?:Picture|Video|Audio|Subject|Image|Photo|Sound)"
    r"[^<>＜＞\r\n]*[>＞]",
    re.IGNORECASE,
)
_D_TAG_RE = re.compile(r"</?d(?:\s+[^>]*)?>", re.IGNORECASE)
_QUOTE_RE = re.compile(
    r"「(?P<corner>[^」\r\n]+)」|『(?P<double_corner>[^』\r\n]+)』|"
    r"“(?P<curly>[^”\r\n]+)”|\"(?P<ascii>[^\"\r\n]+)\""
)
_SHOT_HEADER_RE = re.compile(
    r"(?im)^[ \t]*(?:"
    r"(?:[\[\(【（「『]?\s*(?:Cut|Shot|Scene|カット|ショット|シーン)\s*#?\s*"
    r"[\[\(【（「『:]?\s*"
    r"(?P<label_number>[0-9]+)\s*[\]\)】）」』]?)"
    r"|(?:第\s*(?P<ordinal_number>[0-9]+)\s*(?:カット|ショット|シーン))"
    r")"
    # Japanese authors commonly append an inline time range with full-width
    # punctuation, for example ``Cut 1（0.0-2.6秒）``.  The source copy is
    # NFKC-normalized before matching, so full-width digits, spaces, brackets,
    # and Latin labels use the same path as their ASCII forms.
    r"(?=\s|$|[:\-\u2010-\u2015\uFF08\u3010\u300C\u300E(\[])"
)
_VISUAL_TEXT_CUE_RE = re.compile(
    r"(?:字幕|テロップ|看板|標識|画面|文字|題名|タイトル|表示|書か|"
    r"on[- ]screen|caption|subtitle|sign|label|written|displayed)",
    re.IGNORECASE,
)
# A quoted title is not automatically a dialogue event, but only a much
# narrower subset is safe to carry through the model boundary as visual text.
# In particular, generic captions, runes, signs, and subtitles must not inherit
# the reference-title exception merely because an image is present somewhere in
# the same prompt.
_REFERENCE_IMAGE_CUE_RE = re.compile(
    r"(?:参考(?:画像|イメージ|写真|素材)|参照(?:画像|イメージ|素材)|"
    r"reference\s+(?:image|picture|logo)|supplied\s+(?:image|picture)\s+reference|"
    r"provided\s+(?:image|picture)\s+reference|image\s+reference)",
    re.IGNORECASE,
)
_VISUAL_TITLE_CUE_RE = re.compile(
    r"(?:タイトル(?:文字|名|ロゴ)?|ロゴ(?:文字|名)?|題名(?:文字)?|"
    r"文字ロゴ|ワードマーク|字形|文字比率|文字間隔|スペル|綴り|"
    r"title(?:\s+(?:text|lettering|logo|wordmark))?|"
    r"logo(?:\s+(?:text|lettering|name|wordmark))?|"
    r"title\s+lettering|logo\s+lettering|wordmark|letterform)",
    re.IGNORECASE,
)
_VISUAL_TITLE_PLACEHOLDER = "the exact visible title lettering from the supplied reference image"
_MODEL_ONSCREEN_TEXT_RE = re.compile(
    r"\b(?:subtitles?|captions?|on[- ]screen(?:\s+(?:text|writing|words?|letters?))?|"
    r"readable\s+(?:text|writing|words?|letters?|runes?|glyphs?|captions?|subtitles?)|"
    r"legible\s+(?:text|writing|words?|letters?|runes?|glyphs?)|"
    r"(?:runes?|glyphs?|letters?|words?|text)\s+"
    r"(?:are|is|appear(?:s)?|displayed|visible|readable)|"
    r"(?:written|displayed)\s+(?:text|words?|letters?|runes?|glyphs?)|"
    r"(?:subtitle|caption|label|title)\s+(?:appears?|is|reads?))\b",
    re.IGNORECASE,
)
_EXPLICIT_SPEECH_CUE_RE = re.compile(
    r"(?:セリフ|台詞|発話|言(?:う|って|い)|話(?:す|して)|喋|しゃべ|"
    r"つぶや|囁|ささや|叫(?:ぶ|ん)|口にする|"
    r"宣言(?:する|し|した|して)|発声(?:する|し|した|して)|"
    r"言い渡(?:す|し|した|して)|告げ(?:る|た|て)|読み上げ(?:る|た|て)|"
    r"\b(?:dialogue|spoken line|says?|speaks?|whispers?|shouts?)\b)",
    re.IGNORECASE,
)
_SPOKEN_QUOTE_CUE_RE = re.compile(
    r"(?:セリフ|台詞|発話|会話|"
    r"言(?:う|って|った|い)|話(?:す|して|した)|叫(?:ぶ|んで|んだ)|"
    r"囁(?:く|いて|いた)|ささや|つぶや|答(?:える|えて|えた)|"
    r"尋(?:ねる|ねて|ねた)|呼びかけ|"
    r"\b(?:dialogue|spoken\s+line|says?|speaks?|whispers?|"
    r"shouts?|replies?|asks?)\b)",
    re.IGNORECASE,
)
_ATTRIBUTED_SPEAKER_PREFIX_RE = re.compile(
    r"(?:少女|少年|女の子|男の子|女性|男性|彼女|彼|主人公|ヒロイン|"
    r"ヒーロー|兵士|騎士|魔女|王女|王子|王|女王|子ども|子供|母|父|"
    r"教師|医師|作業員|キャラクター|人物|声)\s*(?:が|は|の)?\s*[:：]?\s*$"
)
_NAMED_SPEAKER_TOKEN_RE = re.compile(
    r"[A-Za-z0-9_\-\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff\uf900-\ufaff]"
    r"[A-Za-z0-9 _\-\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff\uf900-\ufaff]{0,23}$"
)
_PRODUCTION_CONTROL_LABELS = frozenset(
    {
        "テーマ",
        "演出",
        "色",
        "構図",
        "カメラ",
        "タイトル",
        "文字",
        "ロゴ",
        "注記",
        "禁止",
        "音声",
        "bgm",
        "se",
        "字幕",
        "テロップ",
        "画面",
        "説明",
        "項目",
        "見出し",
        "表示",
        "演出テーマ",
        "ロゴ名",
        "スタイル",
        "シーン",
        "場面",
        "時間",
        "秒",
        "動き",
        "音楽",
        "環境音",
        "効果音",
        "素材",
        "参考",
        "参照",
        "条件",
        "役割",
        "注意",
        "設定",
        "台詞",
        "セリフ",
        "title",
        "camera",
        "logo",
        "note",
        "forbidden",
        "audio",
        "music",
        "sound",
        "style",
        "scene",
    }
)
_NON_SPEAKER_PREDICATE_SUFFIX_RE = re.compile(
    r"(?:しました|します|している|していた|していく|してくる|"
    r"されました|されます|された|される|されて|した|する|して|"
    r"だった|である|になる|となる|にする|見せる|示す|表示する|"
    r"描かれた|置かれた|現れた|現れる|存在する|続く|始まる|終わる)$"
)
_NON_SPEAKER_CONTROL_FRAGMENT_RE = re.compile(
    r"(?:絶対に|避ける|禁止事項|禁則|事項|表現|文言|内容|説明|方法|一覧|"
    r"基準|条件|指示|要求|注意事項)"
)
_SPEECH_CONTROL_RE = re.compile(
    r"\b(?:"
    r"say(?:s|ing)?|said|"
    r"speak(?:s|ing)?|spoke|spoken|"
    r"talk(?:s|ed|ing)?|"
    r"dialogue(?:s)?|speech(?:es)?|"
    r"narrat(?:e|es|ed|ing|ion)|"
    r"voice[- ]?over|"
    r"whisper(?:s|ed|ing)?|"
    r"shout(?:s|ed|ing)?|"
    r"yell(?:s|ed|ing)?|"
    r"utter(?:s|ed|ing)?|"
    r"announce(?:s|d|ing)?|"
    r"recite(?:s|d|ing)?"
    r")\b",
    re.IGNORECASE,
)
_MODEL_SPEECH_SENTENCE_SPLIT_RE = re.compile(
    r"\s*;\s*|(?<=[.!?])\s+",
    re.IGNORECASE,
)
_MODEL_SPEECH_FINE_SPLIT_RE = re.compile(
    r"\s*(?:,|:|\bthen\b|\band\b|\bwhile\b|\bas\b|\bbut\b|"
    r"\byet\b|\bbecause\b|\balthough\b|\bbefore\b|\bafter\b)\s*",
    re.IGNORECASE,
)
_MODEL_ONSCREEN_SENTENCE_SPLIT_RE = re.compile(
    r"\s*;\s*|(?<=[.!?])\s+",
    re.IGNORECASE,
)
_MODEL_ONSCREEN_FINE_SPLIT_RE = re.compile(
    r"\s*(?:,|:|\bthen\b|\band\b|\bwhile\b|\bas\b|\bbut\b|"
    r"\byet\b|\bbecause\b|\balthough\b|\bbefore\b|\bafter\b)\s*",
    re.IGNORECASE,
)
_FORBIDDEN_AUDIO_SPEECH_RE = re.compile(
    r"\b(?:speech|spoken|dialogue|narration|voice[- ]?over|says?|speaks?)\b",
    re.IGNORECASE,
)

# Camera direction is validated as geometry, not merely as a bag of glossary
# words.  Keep viewpoint height (low/high) separate from viewing direction
# (up/down), then reject opposing evidence inside one shot.
_CAMERA_LOW_VIEWPOINT_RE = re.compile(
    r"\b(?:low[- ]angle(?:d)?|worm(?:'s|s)?[- ]eye|from below|"
    r"positioned(?:\s+slightly)?\s+below|below\s+(?:the\s+)?(?:subject|character))\b",
    re.IGNORECASE,
)
_CAMERA_HIGH_VIEWPOINT_RE = re.compile(
    r"\b(?:high[- ]angle(?:d)?|bird(?:'s|s)?[- ]eye|top[- ]down|overhead|"
    r"from above|positioned(?:\s+slightly)?\s+above|"
    r"above\s+(?:the\s+)?(?:subject|character))\b",
    re.IGNORECASE,
)
_CAMERA_UPWARD_VIEW_RE = re.compile(
    r"\b(?:upward(?:[- ]looking|\s+view|\s+angle|\s+tilt)?|"
    r"look(?:s|ing)?\s+up(?:ward)?|points?\s+up(?:ward)?|viewed\s+upward)\b",
    re.IGNORECASE,
)
_CAMERA_DOWNWARD_VIEW_RE = re.compile(
    r"\b(?:downward(?:[- ]looking|\s+view|\s+angle|\s+tilt)?|"
    r"look(?:s|ing)?\s+down(?:ward)?|points?\s+down(?:ward)?|viewed\s+downward)\b",
    re.IGNORECASE,
)


class CommunityPromptPlannerError(ValueError):
    """A planner input, model result, or rendered prompt violated the contract."""

    def __init__(self, message: str, *, code: str = "INVALID_COMMUNITY_PLAN") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class DialogueLiteral:
    dialogue_id: int
    text: str
    voice_direction: str = ""

    @property
    def placeholder(self) -> str:
        return f"[DIALOGUE_{self.dialogue_id}]"


@dataclass(frozen=True, slots=True)
class NonverbalCue:
    cue_id: int
    source_text: str
    english_sound: str

    @property
    def placeholder(self) -> str:
        return f"[NONVERBAL_SOUND_{self.cue_id}: {self.english_sound}]"


@dataclass(frozen=True, slots=True)
class ReferenceItem:
    kind: str
    index: int
    role: str = "auto"

    @property
    def label(self) -> str:
        label = {"image": "Picture", "video": "Video", "audio": "Audio"}[self.kind]
        return f"<{label} {self.index}>"


@dataclass(frozen=True, slots=True)
class SourceReferencePreflight:
    """Deterministic source-reference binding shared by planner and API.

    ``source_prompt`` is never rewritten by this helper.  The occurrence
    offsets point into the original string so the planner can build its
    private model-facing copy while the API can use the same canonical tag
    view for entry checks (including standalone-audio orphan checks).
    """

    references: tuple[ReferenceItem, ...]
    occurrences: tuple[tuple[int, int, ReferenceItem], ...]
    warnings: tuple[SourceWarning, ...]

    @property
    def canonical_tags(self) -> tuple[str, ...]:
        """Return canonical tags in source occurrence order, with duplicates."""

        return tuple(reference.label for _, _, reference in self.occurrences)


@dataclass(frozen=True, slots=True)
class NumericFact:
    value: str
    unit: str
    source_text: str

    @property
    def key(self) -> tuple[str, str]:
        return self.value, self.unit


@dataclass(frozen=True, slots=True)
class SourceWarning:
    """Non-fatal authoring ambiguity surfaced beside a compiled prompt."""

    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": "warning",
            "code": self.code,
            "message": self.message,
            "fatal": False,
        }


@dataclass(frozen=True, slots=True)
class PreparedPlannerInput:
    source_prompt: str
    redacted_prompt: str
    dialogues: tuple[DialogueLiteral, ...]
    nonverbal_cues: tuple[NonverbalCue, ...]
    references: tuple[ReferenceItem, ...]
    source_shot_labels: tuple[int, ...]
    source_shot_numbers: tuple[int, ...]
    numeric_facts: tuple[NumericFact, ...]
    visual_title_literals: tuple[str, ...] = ()
    duration_seconds: float | None = None
    style_direction: str = ""
    soundscape: str = ""
    audio_preset: str = "auto"
    music_policy: str = "auto"
    wardrobe_override: bool = False
    wardrobe_direction: str = ""
    wardrobe_required_terms: tuple[str, ...] = ()
    source_warnings: tuple[SourceWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class ShotPlan:
    number: int
    start_seconds: float
    end_seconds: float
    framing: str
    camera: str
    action: str


@dataclass(frozen=True, slots=True)
class DialogueDelivery:
    dialogue_id: int
    shot: int
    start_seconds: float
    speaker: str
    delivery: str


@dataclass(frozen=True, slots=True)
class CommunityPromptPlan:
    schema_version: str
    style: str
    scene: str
    shots: tuple[ShotPlan, ...]
    ambient: tuple[str, ...]
    foley: tuple[str, ...]
    music: str
    dialogue_delivery: tuple[DialogueDelivery, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CompiledCommunityPrompt:
    prompt: str
    plan: CommunityPromptPlan
    prepared: PreparedPlannerInput
    duration_normalized: bool = False
    timeline_repaired: bool = False
    plan_warnings: tuple[SourceWarning, ...] = ()

    def _plan_warnings(self) -> tuple[SourceWarning, ...]:
        warnings = list(self.plan_warnings)
        warnings.extend(_plan_camera_warnings(self.plan))
        if self.timeline_repaired and not any(
            warning.code == "PLANNER_TIMELINE_REPAIRED" for warning in warnings
        ):
            warnings.append(
                SourceWarning(
                    "PLANNER_TIMELINE_REPAIRED",
                    (
                        "The planner returned unusable or overlapping shot times. "
                        "H3 Studio deterministically rebuilt a contiguous timeline "
                        "inside the exact frame budget while preserving Cut order and "
                        "the authored source prompt."
                    ),
                )
            )
        return tuple(warnings)

    def diagnostics(self) -> tuple[SourceWarning, ...]:
        return self.prepared.source_warnings + self._plan_warnings()

    def metadata(self) -> dict[str, Any]:
        plan_warnings = self._plan_warnings()
        return {
            "contract": PLAN_SCHEMA_VERSION,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dialogue_count": len(self.prepared.dialogues),
            "dialogue_voice_directions": [
                {
                    "dialogue_id": item.dialogue_id,
                    "effective": item.voice_direction or None,
                    "deterministic_override": bool(item.voice_direction),
                }
                for item in self.prepared.dialogues
            ],
            "reference_tags": [item.label for item in self.prepared.references],
            "source_shot_labels": list(self.prepared.source_shot_labels),
            "source_shot_numbers": list(self.prepared.source_shot_numbers),
            "numeric_facts": [
                {"value": fact.value, "unit": fact.unit}
                for fact in self.prepared.numeric_facts
            ],
            "audio_preset": self.prepared.audio_preset,
            "music_policy": self.prepared.music_policy,
            "wardrobe_override": self.prepared.wardrobe_override,
            "wardrobe_direction": self.prepared.wardrobe_direction or None,
            "source_warnings": [
                warning.to_dict() for warning in self.prepared.source_warnings
            ],
            "plan_warnings": [warning.to_dict() for warning in plan_warnings],
            "timeline_duration_normalized": self.duration_normalized,
            "timeline_structure_repaired": self.timeline_repaired,
        }


@dataclass(frozen=True, slots=True)
class ModelCheckoutMetadata:
    model_id: str
    expected_revision: str
    detected_revision: str | None
    verified: bool
    path: str
    provenance_path: str | None = None
    lock_path: str | None = None
    lock_sha256: str | None = None
    file_count: int | None = None
    total_bytes: int | None = None
    verification_method: str = "h3-studio-provenance-v1"


_TOP_LEVEL_KEYS = {
    "schema_version",
    "style",
    "scene",
    "shots",
    "ambient",
    "foley",
    "music",
    "dialogue_delivery",
}
_SHOT_KEYS = {
    "number",
    "start_seconds",
    "end_seconds",
    "framing",
    "camera",
    "action",
}
_DELIVERY_KEYS = {
    "dialogue_id",
    "shot",
    "start_seconds",
    "speaker",
    "delivery",
}

_AUDIO_PRESET_DIRECTIONS = {
    "auto": "Use a balanced, physically coherent natural mix.",
    "dialogue": "Keep the exact dialogue clear and naturally integrated with the visible space.",
    "ambience": "Prioritize detailed spatial ambience and convincing room tone.",
    "effects": "Prioritize precisely synchronized physical foley and clear natural transients.",
    "music": "Use a music-led mix while retaining physically matched diegetic sound.",
    "quiet": "Use a restrained, quiet mix with subtle room tone and only necessary diegetic sounds.",
}
_MUSIC_POLICY_DIRECTIONS = {
    "none": "N/A",
    "subtle": "A subtle instrumental score supports the ambience and important physical sounds.",
    "prominent": "A prominent instrumental score follows the rhythm and emotional arc of the edit.",
}


SYSTEM_PROMPT = r"""You are a compiler for a local MiniMax H3 video workflow.
Return exactly one raw JSON object. Do not use Markdown or commentary.

Translate the Japanese production request into concise, concrete English H3
control prose. Dialogue words have already been hidden from you. Never invent,
quote, paraphrase, or repeat any spoken words. Never create narration,
voice-over, subtitles, captions, labels, reference tags, XML tags, or <d> tags.
Source reference tags have been rewritten as ordinary descriptive phrases.
Never turn those phrases back into tags. Python will insert all reference tags
and exact dialogue later.

The phrase "the exact visible title lettering from the supplied reference image"
is a protected semantic visual cue. It represents title/logo lettering already
present in a supplied reference image, not a subtitle, caption, rune, label, or
spoken text. Preserve its physical emergence, engraving, placement, and
continuity actions in English without inventing or repeating the hidden literal.

If the source mentions a character speaking but there are no supplied DIALOGUE
IDs, keep the shot visual-only: show a reaction or mouth movement without
speech, narration, subtitles, captions, runes, or other readable on-screen
text. If the source contains both a request and a prohibition for speech or
on-screen text, the prohibition wins so the compiled plan stays safe.

Use this exact JSON shape and no additional keys:
{
  "schema_version": "h3-community-plan-v1",
  "style": "English visual style and continuity instructions",
  "scene": "English scene overview",
  "shots": [
    {
      "number": 1,
      "start_seconds": 0.0,
      "end_seconds": 5.0,
      "framing": "English framing and composition",
      "camera": "English camera placement and movement",
      "action": "English visible action"
    }
  ],
  "ambient": ["English environmental sound"],
  "foley": ["English physical sound effect"],
  "music": "English music direction or N/A",
  "dialogue_delivery": [
    {
      "dialogue_id": 1,
      "shot": 1,
      "start_seconds": 1.5,
      "speaker": "the woman",
      "delivery": "warm, clear, and conversational"
    }
  ]
}

Every string value must be English control prose with no quotation marks.
Preserve every supplied number and unit exactly. Preserve the supplied Cut or
Shot numbers and order. Use one simple dominant action per shot. Make camera
instructions explicit. A Japanese 仰角 or 煽り instruction means a low-angle upward view;
俯瞰 means a high-angle downward view. Keep the full camera meaning in the same
numbered Shot where the source requested it. Within one Shot, never combine a
low-angle or upward view with an above, high-angle, or downward view, and never
combine a high-angle or downward view with a below, low-angle, or upward view.
The direction of a subject entering from above is an action, not camera
geometry. For a boss breaking through a ceiling, put that entry direction in
action and use one neutral or low-angle camera view; do not write "from above"
or "high-angle" in framing or camera unless the camera itself is above the
subject and looks down.
Treat breathing, panting, grunts, and
exertion cries such as はぁはぁ or うおお as nonverbal physical foley unless
the source explicitly labels the quoted literal as dialogue. Include each
supplied NONVERBAL_SOUND phrase verbatim
in foley. dialogue_delivery must contain exactly one entry for every supplied
DIALOGUE id, but must never contain its words. Preserve every supplied English
DIALOGUE_VOICE_CONSTRAINT exactly in the corresponding delivery; never replace
an explicit voice with a generic conversational voice. Preserve every supplied
WARDROBE_OVERRIDE phrase exactly in scene or visible shot action; it replaces
the clothing from an image reference while identity, face, body shape, and hair
stay referenced. If no dialogue IDs are supplied, dialogue_delivery must be an
empty list."""


def _error(message: str, code: str) -> CommunityPromptPlannerError:
    return CommunityPromptPlannerError(message, code=code)


def _is_long_or_complex_source(prepared: PreparedPlannerInput) -> bool:
    return (
        len(prepared.redacted_prompt) >= LONG_SOURCE_PROMPT_CHAR_THRESHOLD
        or len(prepared.numeric_facts) >= LONG_SOURCE_NUMERIC_FACT_THRESHOLD
    )


def _long_source_max_shots(prepared: PreparedPlannerInput) -> int:
    if prepared.duration_seconds is None:
        return LONG_SOURCE_DEFAULT_MAX_SHOTS
    duration = max(0.0, float(prepared.duration_seconds))
    derived = math.ceil(duration / LONG_SOURCE_SECONDS_PER_SHOT)
    return max(LONG_SOURCE_DEFAULT_MAX_SHOTS, min(LONG_SOURCE_MAX_SHOTS, derived))


def _quote_text(match: re.Match[str]) -> str:
    return next(value for value in match.groupdict().values() if value is not None).strip()


def _contains_japanese(text: str) -> bool:
    return bool(_JAPANESE_RE.search(text))


def _is_named_speaker_prefix(line_prefix: str) -> bool:
    """Accept bounded ``Name「...」``/``Name:「...」`` screenplay syntax."""

    candidate = unicodedata.normalize("NFKC", line_prefix).strip()
    colon_match = re.fullmatch(r"(?P<name>[^:：]{1,24})\s*[:：]\s*", candidate)
    has_colon = colon_match is not None
    name = colon_match.group("name").strip() if colon_match else candidate
    if not _NAMED_SPEAKER_TOKEN_RE.fullmatch(name):
        return False
    if _NON_SPEAKER_PREDICATE_SUFFIX_RE.search(name):
        return False
    if _NON_SPEAKER_CONTROL_FRAGMENT_RE.search(name):
        return False
    normalized_name = re.sub(r"\s+", "", name).casefold()
    if any(label in normalized_name for label in _PRODUCTION_CONTROL_LABELS):
        return False
    if not has_colon and re.search(r"(?:の|は|が|を|に|へ|と|で|より|から|な)$", name):
        return False
    return True


def _has_explicit_speech_cue(prompt: str, match: re.Match[str]) -> bool:
    before, after = _quote_local_context(prompt, match)
    return bool(_SPOKEN_QUOTE_CUE_RE.search(before + after))


def _quote_local_context(prompt: str, match: re.Match[str]) -> tuple[str, str]:
    """Return the quote's own line/clause context, never a neighboring line."""

    line_start = prompt.rfind("\n", 0, match.start()) + 1
    line_end = prompt.find("\n", match.end())
    if line_end < 0:
        line_end = len(prompt)
    before = prompt[line_start:match.start()]
    after = prompt[match.end() : line_end]
    before = re.split(r"[。！？!?；;]", before)[-1]
    after = re.split(r"[。！？!?；;]", after, maxsplit=1)[0]
    return before[-72:], after[:72]


def _has_explicit_spoken_quote_context(prompt: str, match: re.Match[str]) -> bool:
    """Return whether one quote has clear speech attribution nearby.

    Japanese quotation marks are also routinely used for titles, labels,
    slogans, mood words, and visual text.  Only an explicit speech cue or a
    recognizable speaker attribution makes an otherwise implicit quote a
    dialogue event; ``dialogue_texts`` remains the stronger explicit contract.
    """

    before, _after = _quote_local_context(prompt, match)
    if _has_explicit_speech_cue(prompt, match):
        return True
    line_prefix = before.rsplit("\n", 1)[-1]
    return bool(
        _ATTRIBUTED_SPEAKER_PREFIX_RE.search(line_prefix)
        or _is_named_speaker_prefix(line_prefix)
    )


def _is_community_dialogue_quote(
    prompt: str,
    match: re.Match[str],
    *,
    requested_literals: Iterable[str] = (),
) -> bool:
    """Return the planner's exact classification for one quoted literal.

    Japanese quotation marks are used for far more than speech in an
    authoring prompt.  Keep this small predicate next to the planner's quote
    extraction so the API preflight can share the same decision boundary.
    ``requested_literals`` represents the explicit ``dialogue_texts``
    contract; an explicitly supplied literal remains dialogue even when it
    looks like a title or a non-verbal exertion sound.
    """

    literal = re.sub(r"\s+", " ", _quote_text(match)).strip()
    if not _contains_japanese(literal):
        return False
    requested = literal in set(requested_literals)
    speech_cue = _has_explicit_speech_cue(prompt, match)
    explicitly_spoken = requested or speech_cue or _has_explicit_spoken_quote_context(
        prompt, match
    )
    # The planner routes breath/grunt/roar literals to deterministic foley
    # unless the author explicitly marks them as speech.  Mirror that detail
    # here so server preflight cannot demand an audio policy for a non-verbal
    # cue that the community compiler will not count as dialogue.
    if _nonverbal_meaning(literal) is not None and not (requested or speech_cue):
        return False
    return explicitly_spoken


def _visual_title_scope(prompt: str, match: re.Match[str]) -> str:
    """Return the authoring paragraph containing one quoted literal.

    A whole-prompt lookup is deliberately avoided for the title cue: a prompt
    may contain a reference-image note in one section and an unrelated quoted
    production label elsewhere.  Paragraph scope keeps the exception tied to
    the authored visual-title block while still allowing a short reference note
    on the preceding line.
    """

    paragraph_start = prompt.rfind("\n\n", 0, match.start()) + 2
    paragraph_end = prompt.find("\n\n", match.end())
    if paragraph_end < 0:
        paragraph_end = len(prompt)
    return prompt[paragraph_start:paragraph_end]


def _is_reference_visual_title_quote(prompt: str, match: re.Match[str]) -> bool:
    """Return whether a quoted literal is an image-backed title/logo cue.

    The reference-title exception is intentionally narrower than the general
    visible-text classifier.  It requires both an explicit reference-image
    cue and a title/logo/letterform cue in the same paragraph, and it never
    overrides an explicit speech attribution or ``dialogue_texts`` contract.
    """

    literal = _quote_text(match)
    if not literal or _is_community_dialogue_quote(prompt, match):
        return False
    scope = _visual_title_scope(prompt, match)
    return bool(
        _REFERENCE_IMAGE_CUE_RE.search(scope)
        and _VISUAL_TITLE_CUE_RE.search(scope)
        and _VISUAL_TITLE_CUE_RE.search(
            prompt[prompt.rfind("\n", 0, match.start()) + 1 : prompt.find("\n", match.end()) if prompt.find("\n", match.end()) >= 0 else len(prompt)]
        )
    )


def _extract_visual_title_literals(prompt: str) -> tuple[str, ...]:
    """Collect source-authorized visual title literals in source order."""

    literals: list[str] = []
    for match in _QUOTE_RE.finditer(prompt):
        if not _is_reference_visual_title_quote(prompt, match):
            continue
        literal = _normalize_dialogue_text(_quote_text(match))
        if literal not in literals:
            literals.append(literal)
    return tuple(literals)


def has_explicit_source_dialogue(
    prompt: str,
    *,
    prompt_processing_mode: str = "community",
    dialogue: str = "",
    dialogue_texts: Iterable[str] | None = None,
) -> bool:
    """Apply the shared source-dialogue contract used by planner and server.

    The separate ``dialogue`` form field and explicit ``dialogue_texts`` list
    are authoritative.  In ``community`` mode, only Japanese quoted text
    with an actual speech cue/speaker attribution is considered dialogue;
    title, emphasis, production-label, and visual-text quotes remain control
    prose.  ``raw_en`` intentionally keeps the legacy conservative behavior:
    any non-empty supported quotation is treated as dialogue because the
    native prompt is passed through without community interpretation.

    This function is deliberately a boolean preflight contract.  The full
    planner still owns exact literal extraction, redaction, and rendering.
    """

    if isinstance(dialogue, str) and dialogue.strip():
        return True
    requested = tuple(
        value.strip()
        for value in (dialogue_texts or ())
        if isinstance(value, str) and value.strip()
    )
    if requested:
        return True
    if prompt_processing_mode == "raw_en":
        return bool(_QUOTE_RE.search(prompt))
    return any(
        _is_community_dialogue_quote(prompt, match)
        for match in _QUOTE_RE.finditer(prompt)
    )


def _normal_decimal(value: str | int | float | Decimal) -> str:
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise _error(f"Invalid numeric value: {value!r}", "INVALID_NUMBER") from exc
    if not number.is_finite():
        raise _error(f"Non-finite numeric value: {value!r}", "INVALID_NUMBER")
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", ""} else normalized


def _nonverbal_meaning(text: str) -> str | None:
    # Normalize small kana, katakana, and long-vowel punctuation before
    # matching.  Short surprise/pain cries are authored in many equivalent
    # forms (for example, うわーっ / うわぁ / ウワッ), so an exact literal
    # table would leave the same class of sound inconsistently classified.
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(
        chr(ord(character) - 0x60)
        if 0x30A1 <= ord(character) <= 0x30F6
        else character
        for character in normalized
    ).translate(
        str.maketrans(
            {
                "ぁ": "あ",
                "ぃ": "い",
                "ぅ": "う",
                "ぇ": "え",
                "ぉ": "お",
                "ゃ": "や",
                "ゅ": "ゆ",
                "ょ": "よ",
                "ゎ": "わ",
            }
        )
    )
    has_vocalization_marker = bool(re.search(r"[っ!！…ー〜～~]", normalized))
    compact = re.sub(r"[\s、。,.!！?？…・ー〜～~\-]", "", normalized)
    if not compact:
        return None
    # ``ええ`` and ``ああ`` are ordinary lexical assent when they are not
    # marked as a cry.  Do not turn an attributed line such as
    # ``少女が「ええ」とうなずく`` into foley merely because it is short.
    if compact in {"ええ", "ああ"} and not has_vocalization_marker:
        return None
    if re.fullmatch(r"(?:はあ|ぜえ|ふう|haa?|hah?)+", compact):
        return "strained breathing and panting"
    if re.fullmatch(r"(?:うお|おお)(?:[あいうえおっ]*)", compact):
        return "a nonverbal exertion roar"
    if re.fullmatch(r"(?:ぐっ|んっ|ぬっ|くっ|ふん|grunt)+", compact):
        return "a brief nonverbal exertion grunt"
    # A short vocalization with an exclamation/prolonged/small-tsu marker is
    # normally a physical reaction rather than lexical dialogue.  Requiring
    # a marker for one-syllable roots prevents ordinary prose fragments from
    # being converted, while repeated two-syllable cries (うわ, ああ, etc.)
    # remain recoverable without requiring a particular punctuation mark.
    if re.fullmatch(
        r"(?:うわ|うあ|わあ|ああ|ええ|ひゃ|ぎゃ|きゃ|"
        r"あ|え|お|う|ひ|ふ)(?:[あいうえおっ]{0,4})",
        compact,
    ) and (has_vocalization_marker or len(compact) >= 2):
        return "a brief nonverbal surprise or pain cry"
    return None


def _voice_direction_for(context: str, literal: str) -> str:
    """Compile explicit Japanese voice controls without model interpretation."""

    traits: list[str] = []
    if re.search(r"(?:高い|高め|ハイトーン|high[- ]pitched)", context, re.IGNORECASE):
        traits.append("high-pitched")
    if re.search(r"(?:低い|低め|low[- ]pitched|deep voice)", context, re.IGNORECASE):
        traits.append("low-pitched")
    if re.search(r"(?:かわいい|可愛い|キュート|cute|sweet voice)", context, re.IGNORECASE):
        traits.append("cute")
    if re.search(r"(?:ロリ声|幼い声|幼め|youthful anime voice)", context, re.IGNORECASE):
        traits.append("youthful anime voice")
    if re.search(r"(?:くぐもった|こもった|muffled)", context, re.IGNORECASE):
        traits.append("muffled")
    if re.search(r"(?:落ち着いた|穏やか|calm)", context, re.IGNORECASE):
        traits.append("calm")
    if re.search(r"(?:明るい声|元気な声|bright voice|cheerful)", context, re.IGNORECASE):
        traits.append("bright and cheerful")
    if re.search(r"(?:かすれた|ハスキー|hoarse|husky)", context, re.IGNORECASE):
        traits.append("husky")
    if re.search(r"(?:囁|ささや|whisper)", context, re.IGNORECASE):
        traits.append("softly whispered")
    if re.search(r"(?:力強|強い口調|forceful)", context, re.IGNORECASE):
        traits.append("forceful")

    # An exertion cry is normally foley, but once the user explicitly marks it
    # as dialogue its performance must still match its literal shape.
    performance = ""
    meaning = _nonverbal_meaning(literal)
    if meaning == "a nonverbal exertion roar":
        performance = "delivered as a forceful exertion shout"
    elif meaning == "strained breathing and panting":
        performance = "delivered as breathless panting"

    traits = list(dict.fromkeys(traits))
    if traits:
        direction = "a " + ", ".join(traits) + " voice"
        if "youthful anime voice" in traits:
            # Avoid the awkward duplicated phrase "voice voice".
            direction = "a " + ", ".join(traits)
        return f"{direction}, {performance}" if performance else direction
    return performance


_NO_DIALOGUE_RE = re.compile(
    r"(?:セリフ|台詞|発話|会話|音声|声)\s*(?:は|を)?\s*"
    r"(?:なし|無し|無|禁止|不要|入れない|入れず)",
    re.IGNORECASE,
)
_SOURCE_NO_DIALOGUE_LITERAL_RE = re.compile(
    r"(?:台詞|セリフ|せりふ|dialogue|spoken\s+line|speech)\s*"
    r"(?:なし|無し|禁止|不要|off|none|disabled)",
    re.IGNORECASE,
)
_ONSCREEN_TEXT_TERM_RE = re.compile(
    r"(?:字幕|テロップ|キャプション|subtitles?|captions?|"
    r"on[- ]screen(?:\s+(?:text|captions?|subtitles?))?)",
    re.IGNORECASE,
)
_ONSCREEN_TEXT_ACTION_RE = re.compile(
    r"(?:表示(?:する|し|して|され(?:る|た|て)?|させ(?:る|た|て)?)?|"
    r"出(?:す|し|して|現)|載せ(?:る|て)?|映(?:す|し|して)|"
    r"入れ(?:る|て)?|付け(?:る|て)?|見せ(?:る|て)?|"
    r"掲示(?:する|し|して)?|挿入(?:する|し|して)?|"
    r"display(?:ed|s|ing)?|show(?:s|ing)?|add(?:ed|s|ing)?)",
    re.IGNORECASE,
)
_ONSCREEN_TEXT_NEGATION_RE = re.compile(
    r"(?:禁止|なし|無し|無|不要|非表示|表示不可|"
    r"(?:表示|出|載せ|映|入れ|付け|見せ|掲示|挿入)(?:し|させ)?(?:ない|ません|ず|ずに|ぬ)|"
    r"(?:出さ|載せ|映さ|入れ|付け|見せ|消し|削除し|避け|許可し)(?:ない|ません|ず|ずに)|"
    r"(?:no|without|off|none|disabled))",
    re.IGNORECASE,
)
_ONSCREEN_TEXT_CLAUSE_SPLIT_RE = re.compile(
    r"(?:\r?\n|[。！？!?；;]+|しかし|だが|ただし|一方で|反面)",
    re.IGNORECASE,
)
_SOURCE_SEMANTIC_CLAUSE_SPLIT_RE = re.compile(
    r"(?:\r?\n|[。！？!?；;]+|しかし|だが|ただし|一方で|反面|"
    r"が、|けれども|けれど|けど|のに)",
    re.IGNORECASE,
)
_MUSIC_POSITIVE_RE = re.compile(
    r"(?:BGM\s*(?:あり|有り)|音楽\s*(?:あり|有り)|music\s*(?:on|enabled)|"
    r"(?:BGM|音楽)(?:を|は|も)?\s*"
    r"(?:静かに|小さく|大きく|短く|控えめに|力強く|背景に|低く)?\s*"
    r"(?:鳴らす|流す|再生する|入れる|かける|再生)|"
    r"(?:play|enable|turn\s+on|add)\s+(?:the\s+)?music)",
    re.I,
)
_MUSIC_NEGATIVE_RE = re.compile(r"(?:BGM\s*(?:なし|無し)|音楽\s*(?:なし|無し)|(?:no|without)\s+music|music\s*(?:off|none))", re.I)
_MOTION_STILL_RE = re.compile(r"(?:静止|動かない|動きなし|still|motionless)", re.I)
_MOTION_ACTIVE_RE = re.compile(
    r"(?:急激な速度|高速(?:で|に)?\s*(?:走|駆け|飛び|動)|"
    r"走(?:る|って|った|り(?:出す|込む|抜ける|続ける|[、,。！？!?]))|"
    r"駆け(?:る|て|った|上がる|上がって|下りる|下りて|込む|込んで|寄る|寄って)|"
    r"飛び(?:込む|込んで|込んだ|越える|越えて|越えた|跳ぶ|跳んで)|"
    r"回転(?:する|し|して|した|しながら)|"
    r"動(?:く|いて|いた|き出す|き始める)|"
    r"追従(?:する|し|して|した)|踏み込(?:む|んで|んだ)|"
    r"スプリント|sprint(?:s|ed|ing)?|spin(?:s|ning)?|rapid|violent)",
    re.I,
)
_LOCATION_FIXED_RE = re.compile(
    r"(?:同じ場所|同一(?:の)?場所|一つの場所|一か所|一箇所|"
    r"場所|舞台|ロケーション|環境|シーン)"
    r"(?:は|を|が|に|の|から)?\s*"
    r"(?:変え(?:ない|ず|ません)|変わら(?:ない|ず|ません)|"
    r"変更(?:しない|せず|しません)|移さ(?:ない|ず|ません)|"
    r"移動(?:しない|せず|しません)|離れ(?:ない|ず|ません)|"
    r"留ま(?:る|り|って|らない|らず)|固定(?:する|して|のまま)|そのまま|"
    r"動か(?:ない|ず|しません))"
    r"|(?:その場|現地|場所)から\s*(?:動か|移動|離れ)(?:ない|ず|ません)"
    r"|(?:同じ場所|同一(?:の)?場所|one\s+place|same\s+(?:place|location)|"
    r"fixed\s+(?:place|location)|stay\s+in\s+(?:the\s+)?(?:same|one)\s+(?:place|location)|"
    r"no\s+location\s+change)",
    re.I,
)
_SOURCE_LOCATION_FIXED_LITERAL_RE = re.compile(
    r"(?:場面|シーン)\s*転換\s*(?:を\s*)?(?:しない|せず(?:に)?|なし|無し|不要)|"
    r"(?:場所|ロケーション|scene|location)\s*"
    r"(?:は|を|も|が)?\s*(?:絶対に|決して|一切)?\s*"
    r"(?:変えない|変更しない|移さない|固定(?:する|して|のまま)?)",
    re.IGNORECASE,
)
_LOCATION_CHANGE_RE = re.compile(
    r"(?:場所|舞台|ロケーション|シーン)"
    r"(?:を|へ|に|が)?\s*"
    r"(?:変更(?:する|し|して|された|される)?|"
    r"変え(?:る|て|た)|移(?:る|り|動(?:する|し|して|した)?|す|し)|"
    r"切り替え(?:る|て|た)|転換(?:する|し|して)?)"
    r"|(?:場所変更|別の場所|別ロケーション|場所移動|ロケーションチェンジ|"
    r"location\s+change|change\s+(?:the\s+)?(?:place|location)|teleport)",
    re.I,
)
_SUBJECT_SINGLE_RE = re.compile(r"(?:一人だけ|一人|単独|alone|one\s+(?:person|character))", re.I)
_SUBJECT_GROUP_RE = re.compile(r"(?:群衆|大勢|複数|crowd|multiple|many\s+(?:people|characters))", re.I)
_SOURCE_CAMERA_FIXED_RE = re.compile(
    r"(?:カメラ|camera).{0,24}(?:完全固定|固定|動かさない|locked|static|fixed)",
    re.IGNORECASE,
)
_SOURCE_CAMERA_MOVING_RE = re.compile(
    r"(?:パン|パンニング|トラッキング|tracking\s*shot|tracking|whip\s*pan|"
    r"arc\s*shot|orbit\s*shot|dolly|truck|push\s*in|pull\s*out|"
    r"crash\s+zoom|zoom|ズーム|ティルト|チルト|カメラワーク|"
    r"(?:一周|周回|移動|動く|動かす))",
    re.IGNORECASE,
)
_SOURCE_CAMERA_TERM_RE = re.compile(r"(?:カメラ|camera)", re.IGNORECASE)
_SOURCE_CAMERA_EXPLICIT_MOVE_RE = re.compile(
    r"(?:パンニング|パン(?:する|し(?:て|ながら|つつ)|した|している)|"
    r"トラッキング|tracking\s*shot|tracking|whip\s*pan|"
    r"arc\s*shot|orbit\s*shot|dolly|truck|push\s*in|pull\s*out|"
    r"crash\s+zoom|zoom|ズーム|ティルト|チルト|カメラワーク)",
    re.IGNORECASE,
)
_SOURCE_CAMERA_GENERIC_MOVE_RE = re.compile(
    r"(?:一周|周回|移動|動く|動かす|回り込む|追従する|追いかける)",
    re.IGNORECASE,
)
_SOURCE_SPEECH_POSITIVE_LITERAL_RE = re.compile(
    r"(?:しゃべる|喋る|話す|叫ぶ|囁く|ささやく|一言|台詞|セリフ|"
    r"宣言(?:する|し|した|して)|発声(?:する|し|した|して)|"
    r"言い渡(?:す|し|した|して)|告げ(?:る|た|て)|読み上げ(?:る|た|て))",
    re.IGNORECASE,
)
_SOURCE_ONSCREEN_POSITIVE_LITERAL_RE = re.compile(
    r"(?:字幕|テロップ|キャプション|ルーン文字|読める文字).{0,24}"
    r"(?:表示(?:する|し(?:て|た)?|され(?:る|た|て)?|させ(?:る|た|て)?)|"
    r"出(?:す|し(?:て|た)?|現(?:する|し|して|した)?)|"
    r"載せ(?:る|て|た)?|映(?:す|し(?:て|た)?)|"
    r"入れ(?:る|て|た)?|付け(?:る|て|た)?|見せ(?:る|て|た)?|"
    r"掲示(?:する|し|して|した)?|挿入(?:する|し|して|した)?|"
    r"display(?:ed|s|ing)?|show(?:s|ing)?|add(?:ed|s|ing)?)"
    r"(?!\s*(?:ない|ません|ず|ずに|ぬ|不可|禁止))",
    re.IGNORECASE,
)
_SOURCE_ONSCREEN_DESCRIPTIVE_LITERAL_RE = re.compile(
    r"(?:字幕|テロップ|キャプション|ルーン文字|読める文字)\s*"
    r"(?:も|として|という|のような|で)\s*.{0,36}?"
    r"(?:字幕|テロップ|キャプション|ルーン文字|読める文字|文字)",
    re.IGNORECASE,
)
_SOURCE_ONSCREEN_NEGATIVE_LITERAL_RE = re.compile(
    r"(?:画面内字幕|字幕|テロップ|キャプション|読める文字|文字)"
    r"(?:[、,]\s*(?:画面内字幕|字幕|テロップ|キャプション|読める文字|文字))*\s*"
    r"(?:は|を|も|が|など)?\s*"
    r"(?:禁止|なし|無し|不要|入れない|出さない|表示しない|"
    r"(?:表示|出|載せ|映|入れ|付け|見せ|掲示|挿入)(?:し|させ)?"
    r"(?:ない|ません|ず|ずに|ぬ|不可))"
    r"|(?:禁止[:：]\s*)[^。！？!?\r\n]{0,80}"
    r"(?:画面内字幕|字幕|テロップ|キャプション|読める文字|文字)"
    r"|(?:no|without|off|none|disabled)\s*"
    r"(?:on[- ]screen\s*)?(?:text|subtitles?|captions?)",
    re.IGNORECASE,
)
_SOURCE_MUSIC_NEGATIVE_LITERAL_RE = re.compile(
    r"(?:BGM|音楽).{0,8}(?:なし|無し|禁止|不要|off|none)|音楽なし|BGMなし",
    re.IGNORECASE,
)
_SOURCE_LOCATION_CHANGE_LITERAL_RE = re.compile(
    r"(?:場所変更|場所を変更|ロケーションチェンジ|別の場所|移動する|切り替える)",
    re.IGNORECASE,
)
_SOURCE_LOCATION_CHANGE_NEGATIVE_RE = re.compile(
    r"(?:場所|舞台|ロケーション|シーン|環境).{0,20}"
    r"(?:変更|変え|移動|移さ|切り替え|転換)(?:し|せ|さ)?"
    r"(?:ない|ません|ず|ずに|ぬ|ないこと)",
    re.IGNORECASE,
)
_SOURCE_MOTION_STILL_LITERAL_RE = re.compile(
    r"(?:静止|間延び|動かない|止まったまま)",
    re.IGNORECASE,
)
_SOURCE_MOTION_ACTIVE_LITERAL_RE = re.compile(
    r"(?:急激な速度|高速(?:で|に)?\s*(?:走|駆け|飛び|動)|"
    r"走(?:る|って|った|り(?:出す|込む|抜ける|続ける|[、,。！？!?]))|"
    r"駆け(?:る|て|った|上がる|上がって|下りる|下りて|込む|込んで|寄る|寄って)|"
    r"飛び(?:込む|込んで|込んだ|越える|越えて|越えた|跳ぶ|跳んで)|"
    r"回転(?:する|し|して|した|しながら)|踏み込(?:む|んで|んだ)|"
    r"追従(?:する|し|して|した)|連続させる|"
    r"スプリント|sprint(?:s|ed|ing)?|spin(?:s|ning)?|rapid|violent)",
    re.IGNORECASE,
)
_MOTION_STILL_NEGATION_RE = re.compile(
    r"(?:禁止|なし|無し|不要|避け(?:る|て|た)?|"
    r"引き延ばさ(?:ない|ず|ずに)|作ら(?:ない|ず|ずに)|"
    r"し(?:ない|ません|ず|ずに)|させ(?:ない|ません|ず|ずに)|"
    r"維持し(?:ない|ません|ず)|保た(?:ない|ず)|"
    r"\b(?:no|without|avoid|prohibit(?:ed|s)?|do\s+not|don't)\b)",
    re.IGNORECASE,
)
_MOTION_OBJECT_STILLNESS_RE = re.compile(
    r"静止(?:した|している|する)?\s*"
    r"(?:標識|看板|標|背景|背景物|建物|塔|木|枝|雲|霧|"
    r"参考素材|参考画像|画像|イメージ|素材|資料|オブジェクト|物体)",
    re.IGNORECASE,
)
_MOTION_REFERENCE_STILLNESS_RE = re.compile(
    r"静止画(?:の|として)?\s*(?:参考素材|参考画像|画像|イメージ|素材|資料)",
    re.IGNORECASE,
)
_LOCATION_ACTOR_STAY_RE = re.compile(
    r"(?:主人公|人物|少女|少年|女性|男性|敵|騎士|兵士|彼女|彼|"
    r"actor|character|hero|girl|boy)\s*(?:は|が|だけが)?[^。！？!?\r\n]{0,24}"
    r"(?:その場|同じ場所|一つの場所|その位置|場所)[^。！？!?\r\n]{0,10}"
    r"(?:動かない|移動しない|留まる|留まり|離れない|変えない|変わらない)",
    re.IGNORECASE,
)
_LOCATION_ENVIRONMENT_CUE_RE = re.compile(
    r"(?:背景|舞台|ロケーション|シーン|場面|場面転換|背景を維持|"
    r"background|stage|location|scene)",
    re.IGNORECASE,
)


def _strip_source_no_dialogue_clauses(text: str) -> str:
    """Remove local no-dialogue clauses before positive speech detection."""

    stripped = _NO_DIALOGUE_RE.sub(" ", text)
    stripped = _SOURCE_NO_DIALOGUE_LITERAL_RE.sub(" ", stripped)
    return stripped


def _has_unnegated_onscreen_text_request(prompt: str) -> bool:
    """Detect positive caption clauses independently from nearby prohibitions."""

    normalized_prompt = unicodedata.normalize("NFKC", prompt)
    for clause in _ONSCREEN_TEXT_CLAUSE_SPLIT_RE.split(
        normalized_prompt
    ):
        terms = list(_ONSCREEN_TEXT_TERM_RE.finditer(clause))
        if not terms:
            continue
        for descriptive in _SOURCE_ONSCREEN_DESCRIPTIVE_LITERAL_RE.finditer(clause):
            # A descriptive subtitle request such as "字幕もルーン文字の
            # ような…字幕" has no action verb, but is still a positive request.
            # Only inspect the matched phrase itself so a later prohibition in
            # the same clause remains a genuine contradiction.
            if not _ONSCREEN_TEXT_NEGATION_RE.search(descriptive.group()):
                return True
        for term in terms:
            search_end = min(len(clause), term.end() + 48)
            for action in _ONSCREEN_TEXT_ACTION_RE.finditer(clause, term.end(), search_end):
                suffix = clause[action.end() : search_end]
                if not re.match(
                    r"\s*(?:ない|ません|ず|ずに|ぬ|不可|禁止)", suffix, re.I
                ):
                    return True
    return False


def _has_negated_onscreen_text_request(prompt: str) -> bool:
    """Detect natural Japanese/English caption prohibition clauses."""

    normalized_prompt = unicodedata.normalize("NFKC", prompt)
    for clause in _ONSCREEN_TEXT_CLAUSE_SPLIT_RE.split(
        normalized_prompt
    ):
        terms = list(_ONSCREEN_TEXT_TERM_RE.finditer(clause))
        if not terms:
            continue
        if _SOURCE_ONSCREEN_NEGATIVE_LITERAL_RE.search(clause):
            return True
        for term in terms:
            local = clause[term.start() : min(len(clause), term.end() + 48)]
            if re.match(
                r"(?:字幕|テロップ|キャプション|画面内字幕|読める文字|文字)\s*"
                r"(?:は|を|も|が|など)?\s*"
                r"(?:禁止|なし|無し|不要|入れない|出さない|表示しない|"
                r"(?:表示|出|載せ|映|入れ|付け|見せ|掲示|挿入)(?:し|させ)?"
                r"(?:ない|ません|ず|ずに|ぬ|不可))",
                local,
                re.IGNORECASE,
            ):
                return True
    return False


def _source_semantic_clauses(prompt: str) -> tuple[str, ...]:
    """Split source prose at boundaries where a new instruction can begin.

    A comma is intentionally not a boundary here: Japanese prohibition lists
    such as ``字幕、テロップ、読める文字は禁止`` must remain one negative
    clause.  Full stops, line breaks, and contrastive conjunctions are enough
    to keep unrelated sound, camera, and subject actions from borrowing one
    another's keywords.
    """

    normalized = unicodedata.normalize("NFKC", prompt)
    return tuple(
        clause.strip()
        for clause in _SOURCE_SEMANTIC_CLAUSE_SPLIT_RE.split(normalized)
        if clause.strip()
    )


def _has_unnegated_music_request(prompt: str) -> bool:
    """Detect an enabled music request in the same local clause as music."""

    for clause in _source_semantic_clauses(prompt):
        match = _MUSIC_POSITIVE_RE.search(clause)
        if not match:
            continue
        # Keep a negative suffix from being accepted by a shorter verb branch
        # such as ``再生``.  The normal Japanese forms are also covered by the
        # negative music regex, but this local guard protects future variants.
        if re.search(
            r"(?:なし|無し|禁止|不要|しない|せず|ずに|off|none|disabled)",
            clause[match.start() : match.end() + 12],
            re.IGNORECASE,
        ):
            continue
        return True
    return False


def _has_negated_music_request(prompt: str) -> bool:
    """Detect an explicit music prohibition without inferring one."""

    normalized = unicodedata.normalize("NFKC", prompt)
    return bool(_MUSIC_NEGATIVE_RE.search(normalized) or _SOURCE_MUSIC_NEGATIVE_LITERAL_RE.search(normalized))


def _has_unnegated_speech_request(prompt: str) -> bool:
    """Detect vocal performance cues while excluding visual declaration text."""

    stripped = _strip_source_no_dialogue_clauses(unicodedata.normalize("NFKC", prompt))
    for clause in _source_semantic_clauses(stripped):
        # ``宣言文を画面に表示`` is a visual-text instruction.  It contains
        # neither an actual speech verb nor a vocal performance request.
        if re.search(r"(?:画面|文字|字幕|テロップ).{0,20}(?:表示|出す|載せる|映す)", clause, re.I):
            continue
        if _EXPLICIT_SPEECH_CUE_RE.search(clause) or _SOURCE_SPEECH_POSITIVE_LITERAL_RE.search(clause):
            return True
    return False


def _has_camera_movement_request(prompt: str) -> bool:
    """Detect camera movement without treating subject movement as camera motion."""

    for clause in _source_semantic_clauses(prompt):
        if _SOURCE_CAMERA_EXPLICIT_MOVE_RE.search(clause):
            return True
        camera_terms = list(_SOURCE_CAMERA_TERM_RE.finditer(clause))
        if not camera_terms:
            continue
        for term in camera_terms:
            local = clause[max(0, term.start() - 12) : min(len(clause), term.end() + 64)]
            if _SOURCE_CAMERA_GENERIC_MOVE_RE.search(local):
                return True
    return False


def _has_location_change_request(prompt: str) -> bool:
    """Detect a positive location change while excluding local prohibitions."""

    for clause in _source_semantic_clauses(prompt):
        if _SOURCE_LOCATION_CHANGE_NEGATIVE_RE.search(clause):
            continue
        if _LOCATION_CHANGE_RE.search(clause) or _SOURCE_LOCATION_CHANGE_LITERAL_RE.search(clause):
            return True
    return False


def _has_fixed_location_request(prompt: str) -> bool:
    """Detect a fixed setting while excluding an actor staying in place."""

    for clause in _source_semantic_clauses(prompt):
        if not (_LOCATION_FIXED_RE.search(clause) or _SOURCE_LOCATION_FIXED_LITERAL_RE.search(clause)):
            continue
        if _LOCATION_ACTOR_STAY_RE.search(clause) and not _LOCATION_ENVIRONMENT_CUE_RE.search(clause):
            continue
        return True
    return False


def _has_active_motion_request(prompt: str) -> bool:
    """Detect active verb forms without counting static/action nouns as motion."""

    return any(
        _MOTION_ACTIVE_RE.search(clause) or _SOURCE_MOTION_ACTIVE_LITERAL_RE.search(clause)
        for clause in _source_semantic_clauses(prompt)
    )


def _stillness_match_is_negated(prompt: str, match: re.Match[str]) -> bool:
    """Return whether one static/stillness term is explicitly prohibited.

    ``動かない`` is an authored stillness instruction even though it ends in
    the Japanese negative suffix ``ない``.  For that reason the helper checks
    only the surrounding text and treats prohibition verbs attached to static
    nouns (``静止画は禁止``, ``静止を避ける``, ``静止画を作らない``) as
    negation.  This keeps the warning semantic instead of doing a raw keyword
    search.
    """

    token = match.group(0).casefold()
    head = prompt[max(0, match.start() - 32) : match.start()]
    tail = prompt[match.end() : min(len(prompt), match.end() + 48)]
    if token == "動かない":
        return bool(
            re.search(r"\b(?:no|without|avoid|do\s+not|don't)\b", head, re.I)
        )
    if _MOTION_STILL_NEGATION_RE.search(tail):
        return True
    if re.search(r"(?:禁止|なし|無し|不要)\s*[:：]?\s*$", head):
        return True
    clause_start = max(
        prompt.rfind(separator, 0, match.start())
        for separator in ("\n", "。", "！", "？", "!", "?", ";", "；")
    )
    clause_prefix = prompt[clause_start + 1 : match.start()]
    if re.search(r"(?:禁止|禁止事項)\s*[:：]", clause_prefix):
        return True
    if re.search(r"\b(?:no|without|avoid|prohibit(?:ed|s)?|do\s+not|don't)\s*$", head, re.I):
        return True
    return False


def _has_unnegated_stillness_request(prompt: str) -> bool:
    """Detect an authored static beat while ignoring prohibited stillness."""

    normalized_prompt = unicodedata.normalize("NFKC", prompt)
    for pattern in (_MOTION_STILL_RE, _SOURCE_MOTION_STILL_LITERAL_RE):
        for match in pattern.finditer(normalized_prompt):
            local_suffix = normalized_prompt[match.start() : min(len(normalized_prompt), match.start() + 64)]
            if _MOTION_OBJECT_STILLNESS_RE.match(local_suffix) or _MOTION_REFERENCE_STILLNESS_RE.match(
                local_suffix
            ):
                # ``静止した標識`` and ``静止画の参考素材`` describe a
                # reference/background object, not the requested performance
                # or the camera/frame as a whole.
                continue
            if not _stillness_match_is_negated(normalized_prompt, match):
                return True
    return False


def _source_warnings(
    prompt: str,
    dialogues: Sequence[DialogueLiteral],
) -> tuple[SourceWarning, ...]:
    """Return advisory diagnostics without rejecting ordinary authoring prose."""

    warnings: list[SourceWarning] = []
    normalized_prompt = unicodedata.normalize("NFKC", prompt)
    no_dialogue = bool(
        _NO_DIALOGUE_RE.search(normalized_prompt)
        or _SOURCE_NO_DIALOGUE_LITERAL_RE.search(normalized_prompt)
    )
    stripped_prompt = _strip_source_no_dialogue_clauses(normalized_prompt)
    if no_dialogue and (bool(dialogues) or _has_unnegated_speech_request(normalized_prompt)):
        warnings.append(
            SourceWarning(
                "SOURCE_SPEECH_CONFLICT",
                "The source asks for speech while also forbidding dialogue; exact supplied "
                "dialogue remains Python-owned, otherwise the prohibition wins.",
            )
        )
    if _has_unnegated_onscreen_text_request(normalized_prompt) and _has_negated_onscreen_text_request(
        normalized_prompt
    ):
        warnings.append(
            SourceWarning(
                "SOURCE_ONSCREEN_TEXT_CONFLICT",
                "The source both requests and prohibits on-screen captions; "
                "the H3 planner contract does not invent subtitles or captions.",
            )
        )
    if _has_unnegated_music_request(normalized_prompt) and _has_negated_music_request(normalized_prompt):
        warnings.append(
            SourceWarning(
                "SOURCE_MUSIC_CONFLICT",
                "The source both enables and disables music; the Python music_policy/UI setting is authoritative.",
            )
        )
    if (
        _has_unnegated_stillness_request(normalized_prompt)
        and _has_active_motion_request(normalized_prompt)
    ):
        warnings.append(
            SourceWarning(
                "SOURCE_MOTION_CONFLICT",
                "The source mixes stillness and active motion; explicit action beats and cut-local movement take priority over blanket stillness.",
            )
        )
    if _has_location_change_request(normalized_prompt) and _has_fixed_location_request(normalized_prompt):
        warnings.append(
            SourceWarning(
                "SOURCE_LOCATION_CONFLICT",
                "The source mixes fixed-location and location-change instructions; explicit cut-local location changes are retained in sequence order.",
            )
        )
    if _SUBJECT_SINGLE_RE.search(normalized_prompt) and _SUBJECT_GROUP_RE.search(normalized_prompt):
        warnings.append(
            SourceWarning(
                "SOURCE_SUBJECT_COUNT_CONFLICT",
                "The source mixes single-subject and group presence; both are retained as scoped visual context and do not block compilation.",
            )
        )
    source_camera_by_shot, global_camera = _source_camera_requirements(normalized_prompt)
    camera_conflict = any(
        {"low_up", "high_down"}.issubset(requirements)
        for requirements in (*source_camera_by_shot.values(), global_camera)
    )
    camera_conflict = camera_conflict or (
        _SOURCE_CAMERA_FIXED_RE.search(normalized_prompt)
        and _has_camera_movement_request(normalized_prompt)
    )
    if camera_conflict:
        warnings.append(
            SourceWarning(
                "SOURCE_CAMERA_CONFLICT",
                "The source mixes opposing camera instructions in one scope; this remains advisory and does not block compilation.",
            )
        )
    return tuple(warnings)


_WARDROBE_TERM_RE = re.compile(
    r"(?:衣装|服装|ウェア|ショートパンツ|タンクトップ|ランニング|ビキニ|水着|"
    r"制服|ジャージ|スカート|シャツ|ブラウス|ジーンズ|ズボン|ドレス|ワンピース)",
    re.IGNORECASE,
)
_WARDROBE_CHANGE_RE = re.compile(
    r"(?:着(?:て|る|用)|変更|替え|換え)", re.IGNORECASE
)
_WARDROBE_ENGLISH_OVERRIDE_RE = re.compile(
    r"(?:wears?|wearing|dressed in|change(?:s|d)? (?:the )?(?:outfit|clothes)|"
    r"wardrobe must)",
    re.IGNORECASE,
)


def _wardrobe_override_for(prompt: str) -> tuple[bool, str, tuple[str, ...]]:
    """Detect an authored replacement outfit and compile common garments."""

    lines = [
        line.strip()
        for line in prompt.splitlines()
        if (
            (_WARDROBE_TERM_RE.search(line) and _WARDROBE_CHANGE_RE.search(line))
            or _WARDROBE_ENGLISH_OVERRIDE_RE.search(line)
        )
    ]
    if not lines:
        return False, "", ()
    text = " ".join(lines)
    # Explicit preservation is not an override; the default image-reference
    # contract already preserves the original appearance and clothing.
    if re.search(r"(?:維持|保つ|そのまま|変えない|preserve|keep unchanged)", text, re.I):
        return False, "", ()

    terms: list[str] = []
    if re.search(r"(?:タンクトップ|ランニング)", text, re.I):
        terms.append("tank top")
    if re.search(r"ショートパンツ", text, re.I):
        terms.append("shorts")
    for pattern, english in (
        (r"マイクロビキニ", "micro bikini"),
        (r"(?<!マイクロ)ビキニ", "bikini"),
        (r"水着", "swimsuit"),
        (r"制服", "uniform"),
        (r"ジャージ", "tracksuit"),
        (r"スカート", "skirt"),
        (r"(?:Tシャツ|Ｔシャツ)", "T-shirt"),
        (r"ブラウス", "blouse"),
        (r"ジーンズ", "jeans"),
        (r"ズボン", "trousers"),
        (r"ドレス", "dress"),
        (r"ワンピース", "one-piece dress"),
    ):
        if re.search(pattern, text, re.I):
            terms.append(english)
    terms = list(dict.fromkeys(terms))

    if "トレーニングウェア" in text and {"tank top", "shorts"}.issubset(terms):
        direction = "workout wear consisting of a tank top and shorts"
    elif "トレーニングウェア" in text and terms:
        direction = "workout wear consisting of " + " and ".join(terms)
    elif "トレーニングウェア" in text:
        direction = "workout wear"
    elif terms:
        direction = " and ".join(terms)
    else:
        # Unknown clothing remains model-translated, but the reference contract
        # still switches from outfit preservation to authored wardrobe.
        direction = ""
    return True, direction, tuple(terms)


_UNQUOTED_NONVERBAL_RE = re.compile(
    r"(?<![\w\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff])"
    r"(?P<sound>(?:(?:(?:は[ぁあ]|ハ[ァア]|ぜ[ぇえ]|ゼ[ェエ])"
    r"(?:[、, ]*)){2,}[!！…]*|"
    r"う[ぉおォオー〜～]{3,}[!！…]*|ウ[ォオぉおー〜～]{3,}[!！…]*|"
    r"(?:ぐっ|んっ|ぬっ|くっ)(?:[、, ]*(?:ぐっ|んっ|ぬっ|くっ))*)[!！…]*)"
    r"(?=$|[\s、。,.!！?？…]|息|声|と|し)"
)


def _normalize_dialogue_text(text: str) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    if not value:
        raise _error("Dialogue cannot be empty.", "EMPTY_DIALOGUE")
    if "\"" in value or "\r" in value or "\n" in value:
        raise _error(
            "Dialogue may not contain an ASCII double quote or a line break.",
            "UNSAFE_DIALOGUE_LITERAL",
        )
    if _D_TAG_RE.search(value):
        raise _error("Dialogue may not contain <d> tags.", "D_TAG_FORBIDDEN")
    return value


_REFERENCE_KIND_ALIASES = {
    "picture": "image",
    "image": "image",
    "photo": "image",
    "video": "video",
    "audio": "audio",
    "sound": "audio",
    "subject": "subject",
}


def _parse_reference_tag_variant(raw_tag: str) -> tuple[str, int | None] | None:
    """Parse a source/inventory reference tag on an internal normalized copy.

    The wire format stays canonical (``<Picture 1>``), but authoring often
    contains harmless case, spacing, compatibility-width, or leading-zero
    variation.  Return ``index=None`` for a recognized reference kind whose
    index cannot safely resolve; callers can then report the boundary-specific
    diagnostic without ever forwarding the raw tag to Qwen.
    """

    normalized = unicodedata.normalize("NFKC", str(raw_tag))
    if len(normalized) < 2 or normalized[0] != "<" or normalized[-1] != ">":
        return None
    body = re.sub(r"\s+", " ", normalized[1:-1]).strip()
    match = re.fullmatch(r"(?P<kind>[A-Za-z]+)(?P<index>.*)", body)
    if not match:
        return None
    kind = _REFERENCE_KIND_ALIASES.get(match.group("kind").casefold())
    if kind is None:
        return None
    index_text = match.group("index").strip()
    if not re.fullmatch(r"[0-9]+", index_text):
        return kind, None
    index = int(index_text)
    return kind, index if index >= 1 else None


def _parse_reference_inventory(
    inventory: Sequence[Mapping[str, Any]] | None,
) -> tuple[ReferenceItem, ...]:
    if not inventory:
        return ()
    aliases = {
        key: value
        for key, value in _REFERENCE_KIND_ALIASES.items()
        if value != "subject"
    }
    counters = {"image": 0, "video": 0, "audio": 0}
    result: list[ReferenceItem] = []
    seen: set[tuple[str, int]] = set()
    for position, raw in enumerate(inventory, start=1):
        if not isinstance(raw, Mapping):
            raise _error(
                f"Reference #{position} must be an object.", "INVALID_REFERENCE_INVENTORY"
            )
        raw_kind = str(raw.get("kind") or raw.get("type") or "").strip().casefold()
        kind = aliases.get(raw_kind)
        if kind is None:
            raise _error(
                f"Reference #{position} has unsupported kind {raw_kind!r}.",
                "INVALID_REFERENCE_KIND",
            )
        tag_value = raw.get("tag")
        tag_index: int | None = None
        if tag_value is not None:
            parsed_tag = _parse_reference_tag_variant(str(tag_value).strip())
            if parsed_tag is None or parsed_tag[0] == "subject" or parsed_tag[1] is None:
                raise _error(
                    f"Reference #{position} has invalid tag {tag_value!r}.",
                    "INVALID_REFERENCE_TAG",
                )
            tag_kind, tag_index = parsed_tag
            if tag_kind != kind:
                raise _error(
                    f"Reference #{position} kind and tag disagree.",
                    "REFERENCE_KIND_TAG_MISMATCH",
                )
        raw_index = raw.get("index")
        if raw_index is None:
            index = tag_index if tag_index is not None else counters[kind] + 1
        else:
            if isinstance(raw_index, bool):
                raise _error("Reference index must be an integer.", "INVALID_REFERENCE_INDEX")
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise _error(
                    f"Reference #{position} index is invalid.", "INVALID_REFERENCE_INDEX"
                ) from exc
            normalized_raw_index = unicodedata.normalize("NFKC", str(raw_index)).strip()
            if (
                not re.fullmatch(r"[0-9]+", normalized_raw_index)
                or int(normalized_raw_index) != index
            ) and not isinstance(raw_index, int):
                raise _error(
                    f"Reference #{position} index must be a whole integer.",
                    "INVALID_REFERENCE_INDEX",
                )
            if tag_index is not None and tag_index != index:
                raise _error(
                    f"Reference #{position} index and tag disagree.",
                    "REFERENCE_INDEX_TAG_MISMATCH",
                )
        if index < 1 or (kind, index) in seen:
            raise _error(
                f"Reference {kind} index {index} is invalid or duplicated.",
                "DUPLICATE_REFERENCE",
            )
        counters[kind] = max(counters[kind], index)
        seen.add((kind, index))
        role = str(raw.get("role") or raw.get("purpose") or "auto").strip().casefold()
        result.append(ReferenceItem(kind=kind, index=index, role=role))
    return tuple(result)


def preflight_source_references(
    prompt: str,
    *,
    reference_inventory: Sequence[Mapping[str, Any]] | None = None,
    mode: str | None = None,
) -> SourceReferencePreflight:
    """Validate and canonically bind source reference tags without rewriting.

    This is the single reference-tag rule used at the community API boundary
    and by :func:`prepare_planner_input`.  Harmless authoring variants such as
    ``<Picture01>``, case/spacing differences, compatibility-width characters,
    and leading zeroes resolve to the same Python-owned ``ReferenceItem``.
    Invalid indices, unsupported kinds, missing inventory entries, and tags
    that are not legal for the H3 mode raise before any model call.  The
    returned occurrence offsets always refer to ``prompt`` itself; callers
    must keep that source string byte-for-byte intact.

    ``mode`` is the H3 continuity mode (``i2v``, ``first_last``, ``omni``, or
    ``t2v``).  The planner calls this helper without a mode because its
    payload inventory is the authoritative reference set; the server passes
    the mode to retain the native mode-specific allowlist at the API edge.
    """

    if not isinstance(prompt, str):
        raise _error("The source prompt must be a string.", "INVALID_SOURCE_PROMPT")

    references = _parse_reference_inventory(reference_inventory)
    allowed_by_mode: set[tuple[str, int]] | None = None
    normalized_mode = str(mode or "").strip().casefold()
    if normalized_mode:
        if normalized_mode == "i2v":
            allowed_by_mode = {("image", 1)}
        elif normalized_mode == "first_last":
            allowed_by_mode = {("image", 1), ("image", 2)}
        elif normalized_mode == "omni":
            allowed_by_mode = {(item.kind, item.index) for item in references}
        elif normalized_mode == "t2v":
            allowed_by_mode = set()
        else:
            allowed_by_mode = set()

    allowed_by_inventory = {(item.kind, item.index): item for item in references}
    occurrences: list[tuple[int, int, ReferenceItem]] = []
    warnings: list[SourceWarning] = []
    for match in _REFERENCE_TAG_CANDIDATE_RE.finditer(prompt):
        raw_tag = match.group(0)
        parsed_tag = _parse_reference_tag_variant(raw_tag)
        if parsed_tag is None:
            # Preserve the existing planner boundary for reference-like but
            # malformed tokens (for example an unsupported ``<Subject 1>``).
            # Unrelated angle-bracket prose is left for the normal prompt
            # safety checks, exactly as before.
            if _ANY_REFERENCE_LIKE_RE.search(raw_tag):
                raise _error(
                    f"Source prompt contains an unsupported reference tag {raw_tag!r}.",
                    "SOURCE_REFERENCE_TAG_UNSUPPORTED",
                )
            continue
        kind, index = parsed_tag
        if kind == "subject" or index is None:
            raise _error(
                f"Source prompt contains an unsupported reference tag {raw_tag!r}.",
                "SOURCE_REFERENCE_TAG_UNSUPPORTED",
            )
        reference = allowed_by_inventory.get((kind, index))
        if reference is None and allowed_by_mode is not None and (
            kind,
            index,
        ) in allowed_by_mode and normalized_mode in {"i2v", "first_last"}:
            # i2v/first_last frames are uploaded in dedicated slots rather
            # than the Omni reference inventory.  Keep their established H3
            # allowlist while still returning a planner-compatible binding.
            reference = ReferenceItem(kind=kind, index=index)
        if reference is None:
            canonical = {
                "image": "Picture",
                "video": "Video",
                "audio": "Audio",
            }[kind]
            raise _error(
                f"Source prompt uses <{canonical} {index}>, but it is absent from "
                "the actual reference inventory.",
                "SOURCE_REFERENCE_NOT_IN_INVENTORY",
            )
        if allowed_by_mode is not None and (kind, index) not in allowed_by_mode:
            raise _error(
                f"Source reference {reference.label} is not allowed for H3 mode "
                f"{normalized_mode!r}.",
                "SOURCE_REFERENCE_NOT_ALLOWED_FOR_MODE",
            )
        occurrences.append((match.start(), match.end(), reference))
        if raw_tag != reference.label:
            warnings.append(
                SourceWarning(
                    "SOURCE_REFERENCE_TAG_NORMALIZED",
                    f"Source reference {raw_tag!r} was normalized internally to "
                    f"{reference.label}; source_prompt was preserved.",
                )
            )
    return SourceReferencePreflight(
        references=references,
        occurrences=tuple(occurrences),
        warnings=tuple(warnings),
    )


def _source_shot_numbering(
    text: str,
) -> tuple[str, tuple[int, ...], tuple[int, ...], SourceWarning | None]:
    """Normalize authoring labels by occurrence without changing source text.

    Cut labels are editing aids, not H3 semantic IDs. Duplicate, skipped,
    out-of-order, or non-one-based labels still delimit an unambiguous sequence
    of blocks, so the planner copy is renumbered to 1..N while the original
    Japanese prompt and its raw labels remain available for provenance.
    """

    # This is deliberately an internal copy.  ``source_prompt`` remains the
    # exact authoring string, including line endings, full-width punctuation,
    # and the user's original Cut/Shot spelling.
    normalized_source = unicodedata.normalize("NFKC", text)
    normalized_source = normalized_source.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_SHOT_HEADER_RE.finditer(normalized_source))

    def label_number(match: re.Match[str]) -> int:
        raw = match.group("label_number") or match.group("ordinal_number")
        assert raw is not None
        return int(raw)

    labels = tuple(label_number(match) for match in matches)
    numbers = tuple(range(1, len(labels) + 1))
    normalized = normalized_source
    for number, match in reversed(list(zip(numbers, matches))):
        start, end = match.span()
        normalized = normalized[:start] + f"Cut {number}" + normalized[end:]

    warnings: list[str] = []
    if normalized_source != text:
        warnings.append(
            "SOURCE_AUTHORING_COPY_NORMALIZED: line endings and compatibility-width "
            "characters were normalized only in the internal planner copy."
        )
    if labels != numbers or any(
        match.group("ordinal_number") is not None
        or match.group("label_number") is not None
        and match.group(0).strip() != f"Cut {index}"
        for index, match in enumerate(matches, start=1)
    ):
        warnings.append(
            f"Cut/Shot labels {labels!r} were normalized by appearance order to "
            f"{numbers!r}; the original authoring prompt was preserved."
        )
    warning = None
    if warnings:
        warning = SourceWarning(
            "SOURCE_SHOT_NUMBERING_NORMALIZED"
            if any("Cut/Shot labels" in message for message in warnings)
            else "SOURCE_AUTHORING_COPY_NORMALIZED",
            " ".join(warnings),
        )
    return normalized, labels, numbers, warning


def _source_has_shot_content(text: str) -> bool:
    """Reject only an explicitly segmented prompt whose every block is empty."""

    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_SHOT_HEADER_RE.finditer(normalized))
    if not matches:
        return True
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        body = normalized[match.end() : end]
        body = re.sub(r"\d+(?:\.\d+)?\s*(?:秒|s|sec|second|seconds|分|min|minute|minutes)", " ", body, flags=re.I)
        body = re.sub(r"[\s\W_]+", "", body, flags=re.UNICODE)
        if body:
            return True
    return False


def extract_shot_numbers(text: str) -> tuple[int, ...]:
    """Return canonical 1..N shot numbers for explicit authoring headers."""

    return _source_shot_numbering(text)[2]


def _camera_direction_flags(text: str) -> frozenset[str]:
    """Return semantic camera-direction evidence found in English controls."""

    flags: set[str] = set()
    if _CAMERA_LOW_VIEWPOINT_RE.search(text):
        flags.add("low")
    if _CAMERA_HIGH_VIEWPOINT_RE.search(text):
        flags.add("high")
    if _CAMERA_UPWARD_VIEW_RE.search(text):
        flags.add("up")
    if _CAMERA_DOWNWARD_VIEW_RE.search(text):
        flags.add("down")
    return frozenset(flags)


def _plan_camera_warnings(plan: CommunityPromptPlan) -> tuple[SourceWarning, ...]:
    """Surface model-authored camera ambiguity without blocking a render.

    When the author did not request a semantic camera direction, mixed camera
    vocabulary is a quality warning rather than a reason to discard the whole
    generation. Explicit Japanese 仰角/煽り/俯瞰 requirements are still enforced
    separately and reject an opposing geometry.
    """

    warnings: list[SourceWarning] = []
    for shot in plan.shots:
        flags = _camera_direction_flags(f"{shot.framing} {shot.camera}")
        if flags.intersection({"low", "up"}) and flags.intersection({"high", "down"}):
            warnings.append(
                SourceWarning(
                    "PLANNER_CAMERA_GEOMETRY_AMBIGUOUS",
                    (
                        f"Shot {shot.number} contains mixed camera geometry "
                        f"({', '.join(sorted(flags))}); no explicit authored camera "
                        "direction was overridden, so generation may continue."
                    ),
                )
            )
    return tuple(warnings)


def _source_camera_requirements(
    text: str,
) -> tuple[dict[int, frozenset[str]], frozenset[str]]:
    """Map authored Japanese camera cues to their numbered source shot.

    Text before the first explicit Cut/Shot header has no unambiguous target and
    is returned as a global requirement. Text within a numbered block is never
    allowed to be satisfied by a different generated shot.
    """

    normalized_text = _source_shot_numbering(text)[0]
    matches = list(_SHOT_HEADER_RE.finditer(normalized_text))

    def requirements(fragment: str) -> frozenset[str]:
        result: set[str] = set()
        if re.search(r"(?:仰角|煽り)", fragment):
            result.add("low_up")
        if "俯瞰" in fragment:
            result.add("high_down")
        return frozenset(result)

    if not matches:
        return {}, requirements(text)

    per_shot: dict[int, frozenset[str]] = {}
    for index, match in enumerate(matches):
        number = index + 1
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_text)
        shot_requirements = requirements(normalized_text[match.end() : end])
        if shot_requirements:
            per_shot[number] = shot_requirements
    return per_shot, requirements(normalized_text[: matches[0].start()])


def _prioritized_camera_requirement(requirements: Iterable[str]) -> str | None:
    """Resolve an authored camera contradiction without making it fatal."""

    values = set(requirements)
    if "low_up" in values:
        return "low_up"
    if "high_down" in values:
        return "high_down"
    return None


_UNIT_ALIASES = {
    "秒": "seconds",
    "s": "seconds",
    "sec": "seconds",
    "second": "seconds",
    "seconds": "seconds",
    "分": "minutes",
    "min": "minutes",
    "minute": "minutes",
    "minutes": "minutes",
    "時間": "hours",
    "h": "hours",
    "hour": "hours",
    "hours": "hours",
    "kg": "kg",
    "キログラム": "kg",
    "g": "g",
    "グラム": "g",
    "km": "km",
    "キロメートル": "km",
    "m": "m",
    "メートル": "m",
    "cm": "cm",
    "センチ": "cm",
    "センチメートル": "cm",
    "mm": "mm",
    "%": "%",
    "％": "%",
    "fps": "fps",
    "フレーム": "frames",
    "frame": "frames",
    "frames": "frames",
    "回": "times",
    "times": "times",
}
_UNIT_PATTERN = "|".join(
    sorted((re.escape(unit) for unit in _UNIT_ALIASES), key=len, reverse=True)
)
_VALUE_UNIT_RE = re.compile(
    rf"(?<![A-Za-z0-9.])(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>{_UNIT_PATTERN})(?![A-Za-z])",
    re.IGNORECASE,
)
_RANGE_UNIT_RE = re.compile(
    rf"(?<![A-Za-z0-9.])(?P<left>\d+(?:\.\d+)?)\s*"
    rf"(?:-|–|—|~|〜|～)\s*(?P<right>\d+(?:\.\d+)?)\s*"
    rf"(?P<unit>{_UNIT_PATTERN})(?![A-Za-z])",
    re.IGNORECASE,
)
_RESOLUTION_RE = re.compile(r"(?<!\d)(?P<w>\d{2,5})\s*[x×]\s*(?P<h>\d{2,5})(?!\d)", re.I)
_SMALL_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_WORD_VALUE_UNIT_RE = re.compile(
    r"\b(?P<value>" + "|".join(_SMALL_NUMBER_WORDS) + r")\s+"
    r"(?P<unit>times?|repetitions?|reps?|seconds?|minutes?|hours?|frames?)\b",
    re.IGNORECASE,
)


def _canonical_unit(raw: str) -> str:
    return _UNIT_ALIASES[raw.casefold() if raw.casefold() in _UNIT_ALIASES else raw]


def extract_numeric_facts(text: str) -> tuple[NumericFact, ...]:
    facts: list[NumericFact] = []
    occupied: list[tuple[int, int]] = []
    for match in _RESOLUTION_RE.finditer(text):
        value = f"{int(match.group('w'))}x{int(match.group('h'))}"
        facts.append(NumericFact(value, "resolution", match.group(0)))
        occupied.append(match.span())
    for match in _RANGE_UNIT_RE.finditer(text):
        unit = _canonical_unit(match.group("unit"))
        facts.extend(
            (
                NumericFact(_normal_decimal(match.group("left")), unit, match.group(0)),
                NumericFact(_normal_decimal(match.group("right")), unit, match.group(0)),
            )
        )
        occupied.append(match.span())

    def overlaps(span: tuple[int, int]) -> bool:
        return any(span[0] < end and span[1] > start for start, end in occupied)

    for match in _VALUE_UNIT_RE.finditer(text):
        if overlaps(match.span()):
            continue
        facts.append(
            NumericFact(
                _normal_decimal(match.group("value")),
                _canonical_unit(match.group("unit")),
                match.group(0),
            )
        )
    word_unit_aliases = {
        "time": "times",
        "times": "times",
        "repetition": "times",
        "repetitions": "times",
        "rep": "times",
        "reps": "times",
        "second": "seconds",
        "seconds": "seconds",
        "minute": "minutes",
        "minutes": "minutes",
        "hour": "hours",
        "hours": "hours",
        "frame": "frames",
        "frames": "frames",
    }
    for match in _WORD_VALUE_UNIT_RE.finditer(text):
        facts.append(
            NumericFact(
                str(_SMALL_NUMBER_WORDS[match.group("value").casefold()]),
                word_unit_aliases[match.group("unit").casefold()],
                match.group(0),
            )
        )
    # Preserve order but collapse repeated statements of the same fact.
    unique: list[NumericFact] = []
    seen_keys: set[tuple[str, str]] = set()
    for fact in facts:
        if fact.key not in seen_keys:
            unique.append(fact)
            seen_keys.add(fact.key)
    return tuple(unique)


def _bind_rounded_duration_fact(
    facts: Sequence[NumericFact], duration_seconds: float | None
) -> tuple[tuple[NumericFact, ...], SourceWarning | None]:
    """Remove only a rounded whole-clip duration already enforced by frames.

    The UI describes 345 frames at 24 fps as approximately 14.4 seconds, while
    the authoritative render duration is 14.375 seconds. Requiring Qwen to
    repeat that rounded display value in prose adds no safety: final shot timing
    is validated against the exact frame-derived duration. Per-cut timestamps
    and unrelated quantities remain deterministic numeric requirements.
    """

    if duration_seconds is None:
        return tuple(facts), None
    matched: list[NumericFact] = []
    retained: list[NumericFact] = []
    for fact in facts:
        try:
            value = float(fact.value)
        except ValueError:
            retained.append(fact)
            continue
        if fact.unit == "seconds" and abs(value - duration_seconds) <= 0.051:
            matched.append(fact)
        else:
            retained.append(fact)
    if not matched:
        return tuple(retained), None
    authored = ", ".join(f"{fact.value} seconds" for fact in matched)
    return (
        tuple(retained),
        SourceWarning(
            "SOURCE_DURATION_BOUND_TO_FRAME_COUNT",
            (
                f"Authored whole-clip duration {authored} matches the exact "
                f"{duration_seconds:g}-second frame budget and is enforced by "
                "validated shot timing instead of model prose."
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class _Replacement:
    start: int
    end: int
    value: str


def prepare_planner_input(
    prompt: str,
    *,
    reference_inventory: Sequence[Mapping[str, Any]] | None = None,
    dialogue_texts: Sequence[str] | None = None,
    duration_seconds: float | None = None,
    style_direction: str = "",
    soundscape: str = "",
    audio_preset: str = "auto",
    music_policy: str = "auto",
) -> PreparedPlannerInput:
    """Redact literal dialogue and collect deterministic validation metadata."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise _error("The source prompt is empty.", "EMPTY_SOURCE_PROMPT")
    if not _source_has_shot_content(prompt):
        raise _error(
            "The source contains labelled Shots/Cuts but no recoverable shot content.",
            "EMPTY_SOURCE_SHOT_CONTENT",
        )
    if _D_TAG_RE.search(prompt):
        raise _error("The community workflow does not accept <d> tags.", "D_TAG_FORBIDDEN")
    if not isinstance(style_direction, str) or not isinstance(soundscape, str):
        raise _error(
            "style_direction and soundscape must be strings.",
            "INVALID_AUXILIARY_CONTROL",
        )
    style_direction = style_direction.strip()
    soundscape = soundscape.strip()
    for label, value in (("style_direction", style_direction), ("soundscape", soundscape)):
        if _D_TAG_RE.search(value) or _ANY_REFERENCE_LIKE_RE.search(value):
            raise _error(
                f"{label} may not contain H3 tags.", "AUXILIARY_CONTROL_TAG_FORBIDDEN"
            )
        if any(
            _contains_japanese(_quote_text(match))
            for match in _QUOTE_RE.finditer(value)
        ):
            raise _error(
                f"{label} contains quoted Japanese. Put literal speech in dialogue_texts.",
                "DIALOGUE_IN_AUXILIARY_CONTROL",
            )
    audio_preset = str(audio_preset or "auto").strip().casefold()
    if audio_preset not in _AUDIO_PRESET_DIRECTIONS:
        raise _error(
            f"Unsupported audio_preset {audio_preset!r}.", "INVALID_AUDIO_PRESET"
        )
    music_policy = str(music_policy or "auto").strip().casefold()
    if music_policy not in {"auto", *_MUSIC_POLICY_DIRECTIONS}:
        raise _error(
            f"Unsupported music_policy {music_policy!r}.", "INVALID_MUSIC_POLICY"
        )
    wardrobe_override, wardrobe_direction, wardrobe_required_terms = (
        _wardrobe_override_for(prompt)
    )
    reference_preflight = preflight_source_references(
        prompt,
        reference_inventory=reference_inventory,
    )
    references = reference_preflight.references
    visual_title_literals = _extract_visual_title_literals(prompt)
    if duration_seconds is not None:
        try:
            duration_seconds = float(duration_seconds)
        except (TypeError, ValueError) as exc:
            raise _error("duration_seconds must be numeric.", "INVALID_DURATION") from exc
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise _error("duration_seconds must be positive and finite.", "INVALID_DURATION")

    requested = [
        _normalize_dialogue_text(value)
        for value in (dialogue_texts or ())
        if isinstance(value, str) and value.strip()
    ]
    if dialogue_texts and len(requested) != len(dialogue_texts):
        raise _error("Every dialogue_texts entry must be a non-empty string.", "INVALID_DIALOGUE_LIST")

    replacements: list[_Replacement] = []
    for start, end, reference in reference_preflight.occurrences:
        if reference.kind == "image":
            replacement_text = (
                f"the visible subject from supplied image reference number {reference.index}"
            )
        elif reference.kind == "video":
            replacement_text = f"supplied video reference number {reference.index}"
        else:
            replacement_text = f"supplied audio reference number {reference.index}"
        replacements.append(_Replacement(start, end, replacement_text))
    dialogues: list[DialogueLiteral] = []
    cues: list[NonverbalCue] = []
    consumed_requested: set[int] = set()

    for match in _QUOTE_RE.finditer(prompt):
        literal = _normalize_dialogue_text(_quote_text(match))
        if not _contains_japanese(literal):
            continue
        local = prompt[max(0, match.start() - 72) : min(len(prompt), match.end() + 72)]
        requested_index = next(
            (
                i
                for i, text in enumerate(requested)
                if i not in consumed_requested and text == literal
            ),
            None,
        )
        speech_cue = _has_explicit_speech_cue(prompt, match)
        explicitly_spoken = _is_community_dialogue_quote(
            prompt,
            match,
            requested_literals=requested,
        )
        if (
            requested_index is None
            and not explicitly_spoken
            and _VISUAL_TEXT_CUE_RE.search(local)
        ):
            # Exact Japanese visible text is intentionally outside this
            # speech-safe contract. Leave it for semantic English translation.
            continue
        nonverbal = _nonverbal_meaning(literal)
        if nonverbal is not None and not (requested_index is not None or speech_cue):
            cue = NonverbalCue(len(cues) + 1, literal, nonverbal)
            cues.append(cue)
            replacements.append(_Replacement(match.start(), match.end(), cue.placeholder))
            continue
        if not explicitly_spoken:
            # Japanese quote marks also delimit titles, emphasis, slogans,
            # mood beats, list items, and logo text.  Without a speech cue or
            # explicit dialogue contract, retain the words as control prose.
            continue
        dialogue = DialogueLiteral(
            len(dialogues) + 1,
            literal,
            _voice_direction_for(local, literal),
        )
        dialogues.append(dialogue)
        replacements.append(_Replacement(match.start(), match.end(), dialogue.placeholder))
        if requested_index is not None:
            consumed_requested.add(requested_index)

    # Explicit dialogue fields remain authoritative even when the prose did not
    # repeat them. Their literal text is never included in the model request.
    for index, literal in enumerate(requested):
        if index in consumed_requested:
            continue
        # dialogue_texts is an explicit user contract. Even an interjection or
        # breath-like literal remains exact speech when supplied here.
        position = prompt.find(literal)
        local = (
            prompt[max(0, position - 56) : min(len(prompt), position + len(literal) + 56)]
            if position >= 0
            else ""
        )
        dialogue = DialogueLiteral(
            len(dialogues) + 1,
            literal,
            _voice_direction_for(local, literal),
        )
        dialogues.append(dialogue)
        if position >= 0 and not any(
            position < item.end and position + len(literal) > item.start
            for item in replacements
        ):
            replacements.append(
                _Replacement(position, position + len(literal), dialogue.placeholder)
            )

    if len({item.text for item in dialogues}) != len(dialogues):
        raise _error(
            "The same dialogue literal was supplied more than once; each literal must be unique.",
            "DUPLICATE_DIALOGUE_TEXT",
        )

    # Redact every exact occurrence that could otherwise leak into the model,
    # not only the quote span that first established the dialogue event.  This
    # keeps ordinary prose that repeats a spoken phrase from triggering a false
    # fatal while preserving the exact literal for Python-owned rendering.
    for dialogue in dialogues:
        for occurrence in re.finditer(re.escape(dialogue.text), prompt):
            start, end = occurrence.span()
            if not any(
                start < replacement.end and end > replacement.start
                for replacement in replacements
            ):
                replacements.append(_Replacement(start, end, dialogue.placeholder))

    # A source-authorized visual title is not dialogue and must not be exposed
    # as a Japanese literal to Qwen.  Replace every occurrence in the private
    # model copy so later headings/continuity notes cannot leak the same name.
    # Explicit dialogue replacements win if a title literal is also spoken.
    for literal in visual_title_literals:
        for occurrence in re.finditer(re.escape(literal), prompt):
            start, end = occurrence.span()
            if any(
                start < replacement.end and end > replacement.start
                and replacement.value.startswith("[DIALOGUE_")
                for replacement in replacements
            ):
                continue
            replacements.append(
                _Replacement(start, end, _VISUAL_TITLE_PLACEHOLDER)
            )

    redacted = prompt
    for replacement in sorted(replacements, key=lambda item: item.start, reverse=True):
        redacted = redacted[: replacement.start] + replacement.value + redacted[replacement.end :]

    unexpected_tag = _ANY_REFERENCE_LIKE_RE.search(redacted)
    if unexpected_tag:
        raise _error(
            f"Source prompt contains unsupported reference tag {unexpected_tag.group(0)}.",
            "SOURCE_REFERENCE_TAG_UNSUPPORTED",
        )

    # Convert obvious unquoted exertion vocalizations into semantic physical
    # sound placeholders before Qwen sees them.
    def replace_unquoted(match: re.Match[str]) -> str:
        source = match.group("sound")
        meaning = _nonverbal_meaning(source) or "a nonverbal exertion sound"
        cue = NonverbalCue(len(cues) + 1, source, meaning)
        cues.append(cue)
        return cue.placeholder

    redacted = _UNQUOTED_NONVERBAL_RE.sub(replace_unquoted, redacted)
    if dialogues:
        redacted += "\n\nDialogue events requested: " + ", ".join(
            item.placeholder for item in dialogues
        )
    if cues:
        redacted += "\nNonverbal physical sounds required: " + "; ".join(
            item.placeholder for item in cues
        )

    for dialogue in dialogues:
        if dialogue.text in redacted:
            raise _error(
                "Dialogue redaction failed; literal words would reach the model.",
                "DIALOGUE_REDACTION_FAILED",
            )

    redacted, source_shot_labels, source_shots, shot_numbering_warning = (
        _source_shot_numbering(redacted)
    )
    fact_source = "\n".join(value for value in (redacted, style_direction, soundscape) if value)
    facts, duration_warning = _bind_rounded_duration_fact(
        extract_numeric_facts(fact_source), duration_seconds
    )
    source_warnings = list(_source_warnings(prompt, dialogues))
    source_warnings.extend(reference_preflight.warnings)
    if shot_numbering_warning is not None:
        source_warnings.append(shot_numbering_warning)
    if duration_warning is not None:
        source_warnings.append(duration_warning)
    return PreparedPlannerInput(
        source_prompt=prompt,
        redacted_prompt=redacted.strip(),
        dialogues=tuple(dialogues),
        nonverbal_cues=tuple(cues),
        references=references,
        source_shot_labels=source_shot_labels,
        source_shot_numbers=source_shots,
        numeric_facts=facts,
        visual_title_literals=visual_title_literals,
        duration_seconds=duration_seconds,
        style_direction=style_direction,
        soundscape=soundscape,
        audio_preset=audio_preset,
        music_policy=music_policy,
        wardrobe_override=wardrobe_override,
        wardrobe_direction=wardrobe_direction,
        wardrobe_required_terms=wardrobe_required_terms,
        source_warnings=tuple(source_warnings),
    )


def build_model_messages(prepared: PreparedPlannerInput) -> list[dict[str, str]]:
    """Build the Qwen conversation without exposing literal dialogue or tags."""

    shot_camera_requirements, global_camera_requirements = _source_camera_requirements(
        prepared.redacted_prompt
    )
    camera_requirement_labels: list[str] = []
    for number, requirements in shot_camera_requirements.items():
        if "low_up" in requirements:
            camera_requirement_labels.append(f"Shot {number} = low-angle upward view")
        if "high_down" in requirements:
            camera_requirement_labels.append(f"Shot {number} = high-angle downward view")
    if "low_up" in global_camera_requirements:
        camera_requirement_labels.append("unscoped = low-angle upward view")
    if "high_down" in global_camera_requirements:
        camera_requirement_labels.append("unscoped = high-angle downward view")

    lines = [
        "SOURCE REQUEST (literal dialogue has been redacted):",
        prepared.redacted_prompt,
        "",
        "COMPILER CONSTRAINTS:",
        "- Dialogue IDs: "
        + (", ".join(str(item.dialogue_id) for item in prepared.dialogues) or "none"),
        "- Dialogue voice constraints: "
        + (
            "; ".join(
                f"DIALOGUE {item.dialogue_id} = {item.voice_direction}"
                for item in prepared.dialogues
                if item.voice_direction
            )
            or "none explicitly supplied"
        ),
        f"- Image reference count: {sum(item.kind == 'image' for item in prepared.references)}",
        f"- Video reference count: {sum(item.kind == 'video' for item in prepared.references)}",
        f"- Audio reference count: {sum(item.kind == 'audio' for item in prepared.references)}",
        "- Source Cut/Shot order: "
        + (", ".join(map(str, prepared.source_shot_numbers)) or "not explicitly numbered"),
        "- Required camera direction by source shot: "
        + ("; ".join(camera_requirement_labels) or "none explicitly supplied"),
        "- Required numeric facts: "
        + (
            ", ".join(f"{fact.value} {fact.unit}" for fact in prepared.numeric_facts)
            or "none"
        ),
        "- Required nonverbal foley phrases: "
        + (", ".join(item.english_sound for item in prepared.nonverbal_cues) or "none"),
        "- Additional style direction: " + (prepared.style_direction or "none"),
        "- WARDROBE_OVERRIDE: "
        + (
            prepared.wardrobe_direction
            if prepared.wardrobe_override and prepared.wardrobe_direction
            else (
                "The authored source clothing replaces the image-reference outfit; "
                "translate its garments precisely."
                if prepared.wardrobe_override
                else "none; preserve the reference appearance"
            )
        ),
        "- Additional soundscape direction: " + (prepared.soundscape or "none"),
        "- Audio mix preset: "
        + _AUDIO_PRESET_DIRECTIONS[prepared.audio_preset],
        "- Music policy: "
        + (
            "Choose the most natural option from the source request."
            if prepared.music_policy == "auto"
            else _MUSIC_POLICY_DIRECTIONS[prepared.music_policy]
        ),
    ]
    if prepared.source_shot_numbers:
        shot_count = len(prepared.source_shot_numbers)
        lines.append(
            f"- Explicit source blocks: return exactly {shot_count} compact shot objects "
            f"numbered 1..{shot_count}, one per canonical source block; never merge/drop blocks."
        )
    if _is_long_or_complex_source(prepared):
        lines.extend(
            [
                "OUTPUT BUDGET:",
                f"- The entire raw JSON must finish under {LONG_SOURCE_OUTPUT_TOKEN_BUDGET} generated tokens.",
                "- Use one concise sentence per string field; keep style and scene short.",
                "- Keep ambient and foley to at most 4 short items each; use no repetition or commentary.",
            ]
        )
        if not prepared.source_shot_numbers:
            max_shots = _long_source_max_shots(prepared)
            duration_hint = (
                f"for the exact {_normal_decimal(prepared.duration_seconds)} second duration"
                if prepared.duration_seconds is not None
                else "using the default duration budget"
            )
            lines.append(
                f"- No explicit source blocks: merge adjacent descriptive beats into at most "
                f"{max_shots} shot objects {duration_hint} (hard cap {LONG_SOURCE_MAX_SHOTS}); "
                "preserve essential chronological transitions."
            )
    if prepared.duration_seconds is not None:
        lines.append(f"- Exact video duration: {_normal_decimal(prepared.duration_seconds)} seconds")
    user_message = "\n".join(lines)
    for dialogue in prepared.dialogues:
        if dialogue.text in user_message:
            raise _error("Dialogue leaked into the model request.", "DIALOGUE_REDACTION_FAILED")
    if _ANY_REFERENCE_LIKE_RE.search(user_message):
        raise _error("Reference tags leaked into the model request.", "REFERENCE_TAG_LEAK")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], where: str) -> None:
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise _error(
            f"{where} has the wrong keys (missing={missing}, extra={extra}).",
            "PLAN_SCHEMA_MISMATCH",
        )


def _strict_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(f"{where} must be an integer.", "PLAN_TYPE_ERROR")
    return value


def _strict_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(f"{where} must be numeric.", "PLAN_TYPE_ERROR")
    result = float(value)
    if not math.isfinite(result):
        raise _error(f"{where} must be finite.", "PLAN_TYPE_ERROR")
    return result


def _english_string(value: Any, where: str, *, allow_na: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(f"{where} must be a non-empty string.", "PLAN_TYPE_ERROR")
    result = re.sub(r"\s+", " ", value).strip()
    if _NON_ENGLISH_RE.search(result):
        raise _error(f"{where} contains non-English control text.", "NON_ENGLISH_CONTROL")
    if _ANY_REFERENCE_LIKE_RE.search(result):
        raise _error(f"{where} contains a model-generated reference tag.", "INVENTED_REFERENCE_TAG")
    if _D_TAG_RE.search(result):
        raise _error(f"{where} contains a forbidden <d> tag.", "D_TAG_FORBIDDEN")
    if any(mark in result for mark in ('"', "“", "”", "「", "」", "『", "』")):
        raise _error(f"{where} contains quotation marks.", "MODEL_QUOTED_TEXT")
    if not allow_na and result.casefold() == "n/a":
        raise _error(f"{where} may not be N/A.", "PLAN_EMPTY_FIELD")
    return result


def _string_list(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _error(f"{where} must be an array.", "PLAN_TYPE_ERROR")
    return tuple(_english_string(item, f"{where}[{index}]") for index, item in enumerate(value))


_MODEL_FIELD_ALIASES = {
    "schema_version": ("schema_version", "schema", "version"),
    "style": ("style", "visual_style", "style_direction", "look"),
    "scene": ("scene", "setting", "overview", "description", "world"),
    "shots": ("shots", "shot_list", "shotlist", "cuts", "scenes", "timeline"),
    "ambient": ("ambient", "ambience", "soundscape", "environmental_sound"),
    "foley": ("foley", "sound_effects", "soundeffects", "sfx", "effects"),
    "music": ("music", "bgm", "score"),
    "dialogue_delivery": (
        "dialogue_delivery",
        "dialoguedelivery",
        "dialogue_deliveries",
        "dialogues",
        "dialogue",
    ),
}
_SHOT_FIELD_ALIASES = {
    "number": ("number", "shot", "shot_number", "shotnumber", "cut", "id", "index"),
    "start_seconds": (
        "start_seconds",
        "startseconds",
        "start",
        "start_time",
        "starttime",
        "from",
        "begin",
    ),
    "end_seconds": (
        "end_seconds",
        "endseconds",
        "end",
        "end_time",
        "endtime",
        "to",
        "finish",
    ),
    "framing": ("framing", "composition", "framing_composition", "shot_size"),
    "camera": ("camera", "camera_move", "camera_movement", "movement", "view"),
    "action": ("action", "visual_action", "description", "event", "content", "motion"),
}
_DELIVERY_FIELD_ALIASES = {
    "dialogue_id": ("dialogue_id", "dialogueid", "id", "line", "line_id"),
    "shot": ("shot", "shot_number", "shotnumber", "cut", "scene"),
    "start_seconds": ("start_seconds", "start", "time", "timestamp", "at"),
    "speaker": ("speaker", "character", "subject", "who"),
    "delivery": ("delivery", "direction", "voice", "performance", "tone"),
}


def _normalized_mapping_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", str(value)).casefold())


def _mapping_value(
    value: Mapping[str, Any], aliases: Sequence[str], default: Any = None
) -> Any:
    normalized = {
        _normalized_mapping_key(key): item for key, item in value.items()
    }
    for alias in aliases:
        found = normalized.get(_normalized_mapping_key(alias))
        if found is not None:
            return found
    return default


def _coerce_model_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    normalized = unicodedata.normalize("NFKC", str(value))
    match = re.search(r"-?\d+", normalized)
    return int(match.group(0)) if match else None


def _coerce_model_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    normalized = unicodedata.normalize("NFKC", str(value)).strip()
    # Accept ordinary model variations such as ``0.0 sec``, ``00:02.6``, and
    # ``2.6 seconds``.  The timeline repair below remains authoritative.
    clock = re.fullmatch(r"(?:\d+:)?(?P<minutes>\d+):(?P<seconds>\d+(?:\.\d+)?)", normalized)
    if clock:
        return float(clock.group("minutes")) * 60.0 + float(clock.group("seconds"))
    match = re.search(r"-?(?:\d+(?:\.\d*)?|\.\d+)", normalized)
    if not match:
        return None
    result = float(match.group(0))
    return result if math.isfinite(result) else None


def _model_text_default(field: str) -> str:
    return {
        "style": "Cinematic visual style with coherent subject continuity and clear physical motion.",
        "scene": "The authored environment and requested subjects continue coherently.",
        "framing": "A clear medium-wide composition keeps the subject and action readable.",
        "camera": "The camera follows the subject with smooth, readable movement.",
        "action": "The visible action continues naturally with clear physical cause and effect.",
        "speaker": "the primary visible subject",
        "delivery": "natural, clear delivery",
    }.get(field, "Natural environmental sound.")


def _replace_allowed_visual_title_literals(
    text: str,
    visual_title_literals: Sequence[str],
) -> tuple[str, bool]:
    """Replace only source-authorized title/logo literals in model text."""

    result = text
    replaced = False
    for literal in visual_title_literals:
        normalized_literal = unicodedata.normalize("NFKC", str(literal)).strip()
        if not normalized_literal:
            continue
        updated = result.replace(normalized_literal, _VISUAL_TITLE_PLACEHOLDER)
        if updated != result:
            replaced = True
            result = updated

    if replaced:
        # The model may wrap the echoed title in a quote pair.  Remove only
        # those delimiters around the protected semantic placeholder; unrelated
        # quote marks still go through the normal control-text guard.
        quoted_placeholder = (
            r"[\"“”「」『』]\s*"
            + re.escape(_VISUAL_TITLE_PLACEHOLDER)
            + r"\s*[\"“”「」『』]"
        )
        result = re.sub(quoted_placeholder, _VISUAL_TITLE_PLACEHOLDER, result)
    return result, replaced


def _is_allowed_visual_title_text_match(match: re.Match[str]) -> bool:
    """Allow the model to describe an authored title, never a subtitle/rune."""

    return match.group(0).lstrip().casefold().startswith("title ")


def _remove_model_speech_clauses(text: str) -> tuple[str, bool]:
    """Remove model-invented speech clauses while retaining physical action.

    Qwen occasionally puts a speech-control sentence beside a useful physical
    beat in one English field.  Rejecting the whole field loses the motion the
    user asked for, while simply stripping ``says`` leaves an unsafe fragment.
    Split at ordinary clause boundaries, discard only clauses containing the
    speech-control vocabulary, and rejoin the surviving physical clauses.
    """

    sentence_clauses = [
        clause.strip(" ;,:.-!?")
        for clause in _MODEL_SPEECH_SENTENCE_SPLIT_RE.split(text)
        if clause and clause.strip(" ;,:.-!?")
    ]
    if not sentence_clauses or not any(
        _SPEECH_CONTROL_RE.search(clause) for clause in sentence_clauses
    ):
        return text, False
    retained: list[str] = []
    for clause in sentence_clauses:
        if not _SPEECH_CONTROL_RE.search(clause):
            retained.append(clause)
            continue
        fine_clauses = [
            part.strip(" ;,:.-!?")
            for part in _MODEL_SPEECH_FINE_SPLIT_RE.split(clause)
            if part and part.strip(" ;,:.-!?")
        ]
        retained.extend(
            part for part in fine_clauses if not _SPEECH_CONTROL_RE.search(part)
        )
    return "; ".join(retained), True


def _visual_title_onscreen_matches_are_allowed(
    text: str,
    visual_title_literals: Sequence[str],
) -> bool:
    matches = tuple(_MODEL_ONSCREEN_TEXT_RE.finditer(text))
    return bool(
        matches
        and visual_title_literals
        and all(_is_allowed_visual_title_text_match(match) for match in matches)
    )


def _remove_model_onscreen_clauses(
    text: str,
    *,
    visual_title_literals: Sequence[str] = (),
) -> tuple[str, bool]:
    """Remove invented on-screen-text clauses while retaining physical prose.

    The model sometimes combines a useful visual beat with a prohibited
    subtitle/caption instruction in one field.  Remove only the offending
    sentence or conjunction-delimited clause.  A source-authorized visual
    title remains protected by the same narrow exception used by the final
    on-screen-text gate.
    """

    sentence_clauses = [
        clause.strip(" ;,:.-!?")
        for clause in _MODEL_ONSCREEN_SENTENCE_SPLIT_RE.split(text)
        if clause and clause.strip(" ;,:.-!?")
    ]
    if not sentence_clauses or not any(
        _MODEL_ONSCREEN_TEXT_RE.search(clause)
        and not _visual_title_onscreen_matches_are_allowed(
            clause,
            visual_title_literals,
        )
        for clause in sentence_clauses
    ):
        return text, False

    retained: list[str] = []
    removed = False
    for clause in sentence_clauses:
        if not _MODEL_ONSCREEN_TEXT_RE.search(clause) or (
            _visual_title_onscreen_matches_are_allowed(
                clause,
                visual_title_literals,
            )
        ):
            retained.append(clause)
            continue

        fine_clauses = [
            part.strip(" ;,:.-!?")
            for part in _MODEL_ONSCREEN_FINE_SPLIT_RE.split(clause)
            if part and part.strip(" ;,:.-!?")
        ]
        for part in fine_clauses:
            if _MODEL_ONSCREEN_TEXT_RE.search(part) and not (
                _visual_title_onscreen_matches_are_allowed(
                    part,
                    visual_title_literals,
                )
            ):
                removed = True
                continue
            retained.append(part)

    return "; ".join(retained), removed


def _safe_model_text(
    value: Any,
    field: str,
    warnings: list[SourceWarning],
    *,
    allow_na: bool = False,
    visual_title_literals: Sequence[str] = (),
) -> str:
    """Keep the effective control layer English without rejecting recoverable output."""

    if value is None:
        warnings.append(SourceWarning("MODEL_FIELD_DEFAULTED", f"Missing model field {field!r} was defaulted."))
        return "N/A" if allow_na else _model_text_default(field)
    if isinstance(value, (list, tuple)):
        value = "; ".join(str(item) for item in value)
    result = unicodedata.normalize("NFKC", str(value))
    result = re.sub(r"\s+", " ", result).strip()
    if allow_na and result.casefold() in {"n/a", "na", "none", "null", "なし"}:
        return "N/A"

    changed = False
    cleaned = _ANY_REFERENCE_LIKE_RE.sub(" ", result)
    cleaned = _D_TAG_RE.sub(" ", cleaned)
    if cleaned != result:
        changed = True
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,:")
    cleaned, visual_title_replaced = _replace_allowed_visual_title_literals(
        cleaned,
        visual_title_literals,
    )
    if visual_title_replaced:
        changed = True
        warnings.append(
            SourceWarning(
                "MODEL_VISUAL_TITLE_PRESERVED",
                f"Model field {field!r} echoed a source-authorized visual title; only the literal was replaced with the reference semantic cue and the surrounding control prose was preserved.",
            )
        )
    cleaned, speech_removed = _remove_model_speech_clauses(cleaned)
    if speech_removed:
        warnings.append(
            SourceWarning(
                "MODEL_SPEECH_CONTROL_REMOVED",
                f"Model field {field!r} had speech-control clause(s) removed while preserving remaining physical action.",
            )
        )
    cleaned, onscreen_removed = _remove_model_onscreen_clauses(
        cleaned,
        visual_title_literals=visual_title_literals,
    )
    if onscreen_removed and cleaned:
        warnings.append(
            SourceWarning(
                "MODEL_ONSCREEN_TEXT_CONTROL_REMOVED",
                f"Model field {field!r} had on-screen text control clause(s) removed while preserving remaining physical action.",
            )
        )
    if onscreen_removed and not cleaned:
        warnings.append(
            SourceWarning(
                "MODEL_ONSCREEN_TEXT_REPAIRED",
                f"Model field {field!r} invented a subtitle, caption, label, or readable on-screen text control and was defaulted.",
            )
        )
        return "N/A" if allow_na else _model_text_default(field)
    unsafe = (
        not cleaned
        or bool(_NON_ENGLISH_RE.search(cleaned))
        or any(mark in cleaned for mark in ('"', "“", "”", "「", "」", "『", "』"))
        or bool(_SPEECH_CONTROL_RE.search(cleaned))
    )
    onscreen_matches = tuple(_MODEL_ONSCREEN_TEXT_RE.finditer(cleaned))
    if onscreen_removed and unsafe and not onscreen_matches:
        warnings.append(
            SourceWarning(
                "MODEL_ONSCREEN_TEXT_REPAIRED",
                f"Model field {field!r} contained on-screen text alongside another unsafe control and was defaulted.",
            )
        )
    if onscreen_matches and not _visual_title_onscreen_matches_are_allowed(
        cleaned,
        visual_title_literals,
    ):
        warnings.append(
            SourceWarning(
                "MODEL_ONSCREEN_TEXT_REPAIRED",
                f"Model field {field!r} invented a subtitle, caption, label, or readable on-screen text control and was defaulted.",
            )
        )
        return "N/A" if allow_na else _model_text_default(field)
    if unsafe:
        warnings.append(
            SourceWarning(
                "MODEL_CONTROL_TEXT_REPAIRED",
                f"Model field {field!r} contained unsafe or non-English control text and was defaulted.",
            )
        )
        return "N/A" if allow_na else _model_text_default(field)
    if changed:
        warnings.append(
            SourceWarning(
                "MODEL_CONTROL_TAGS_REMOVED",
                f"Model-generated tags in field {field!r} were removed; Python owns reference and <d> tags.",
            )
        )
    return cleaned


def _extract_model_object(raw: str, warnings: list[SourceWarning]) -> Mapping[str, Any]:
    """Extract a recoverable JSON object from Qwen wrappers and fences."""

    if not isinstance(raw, str) or not raw.strip():
        raise _error("The planner model returned no text.", "EMPTY_MODEL_RESULT")
    text = raw.lstrip("\ufeff").strip()
    lines = text.splitlines()
    if any(line.strip().startswith("```") for line in lines):
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
        warnings.append(
            SourceWarning(
                "MODEL_MARKDOWN_WRAPPER_REMOVED",
                "Markdown code fencing was removed from the model result.",
            )
        )
    start = text.find("{")
    if start < 0:
        raise _error("The planner result contains no JSON object.", "MODEL_JSON_INVALID")
    try:
        value, end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise _error(f"The planner returned invalid JSON: {exc}.", "MODEL_JSON_INVALID") from exc
    if text[:start].strip() or text[start + end :].strip():
        warnings.append(SourceWarning("MODEL_WRAPPER_TEXT_IGNORED", "Model preamble or trailing explanation was ignored."))
    if not isinstance(value, Mapping):
        raise _error("The planner result must be a JSON object.", "PLAN_TYPE_ERROR")

    current: Mapping[str, Any] = value
    for key in ("plan", "result", "data", "output", "payload"):
        nested = _mapping_value(current, (key,))
        if isinstance(nested, Mapping):
            current = nested
            warnings.append(SourceWarning("MODEL_JSON_WRAPPER_UNWRAPPED", f"Nested model JSON field {key!r} was unwrapped."))
            break
    if not any(_mapping_value(current, aliases) is not None for aliases in (_MODEL_FIELD_ALIASES["shots"],)):
        raise _error("The planner result contains no recoverable shots.", "NO_SHOTS")
    if set(current) - {
        item for aliases in _MODEL_FIELD_ALIASES.values() for item in aliases
    }:
        warnings.append(SourceWarning("MODEL_EXTRA_FIELDS_IGNORED", "Unknown model JSON fields were ignored."))
    return current


def _as_model_list(value: Any, field: str, warnings: list[SourceWarning]) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, str) and value.strip():
        warnings.append(SourceWarning("MODEL_LIST_COERCED", f"Model field {field!r} was coerced to a one-item list."))
        return [value]
    warnings.append(SourceWarning("MODEL_LIST_DEFAULTED", f"Invalid model field {field!r} was defaulted to an empty list."))
    return []


def _has_substantive_model_shot_content(raw_shot: Any) -> bool:
    """Check shot content before field defaults can hide an empty result."""

    if not isinstance(raw_shot, Mapping):
        value = str(raw_shot or "")
    else:
        values = [
            _mapping_value(raw_shot, _SHOT_FIELD_ALIASES[field])
            for field in ("framing", "camera", "action")
        ]
        value = " ".join(
            str(item)
            for item in values
            if item is not None and not isinstance(item, (Mapping, list, tuple))
        )
        for item in values:
            if isinstance(item, (list, tuple)):
                value += " " + " ".join(str(part) for part in item)
    cleaned = _D_TAG_RE.sub(" ", _ANY_REFERENCE_LIKE_RE.sub(" ", value))
    return bool(
        re.search(
            r"[A-Za-z\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff\uf900-\ufaff]",
            cleaned,
        )
    )


def _has_explicit_model_shot_content_field(raw_shot: Any) -> bool:
    """Distinguish explicit empty fields from a model omission we can default."""

    if not isinstance(raw_shot, Mapping):
        return True
    aliases = {
        _normalized_mapping_key(alias)
        for field in ("framing", "camera", "action")
        for alias in _SHOT_FIELD_ALIASES[field]
    }
    return any(_normalized_mapping_key(key) in aliases for key in raw_shot)


def _document_numeric_facts(plan: CommunityPromptPlan) -> set[tuple[str, str]]:
    facts = set(
        fact.key
        for fact in extract_numeric_facts(
            "\n".join(
                [
                    plan.style,
                    plan.scene,
                    *(shot.framing for shot in plan.shots),
                    *(shot.camera for shot in plan.shots),
                    *(shot.action for shot in plan.shots),
                    *plan.ambient,
                    *plan.foley,
                    plan.music,
                    *(delivery.speaker for delivery in plan.dialogue_delivery),
                    *(delivery.delivery for delivery in plan.dialogue_delivery),
                ]
            )
        )
    )
    for shot in plan.shots:
        facts.add((_normal_decimal(shot.start_seconds), "seconds"))
        facts.add((_normal_decimal(shot.end_seconds), "seconds"))
    for delivery in plan.dialogue_delivery:
        facts.add((_normal_decimal(delivery.start_seconds), "seconds"))
    return facts


def validate_plan(
    plan: CommunityPromptPlan,
    prepared: PreparedPlannerInput,
    *,
    require_duration: bool = True,
) -> None:
    numbers = tuple(shot.number for shot in plan.shots)
    if not numbers:
        raise _error("The plan must contain at least one shot.", "NO_SHOTS")
    # Source Cut/Shot count is authoring metadata, not a requirement that the
    # model invent missing visual content.  The model plan is canonicalized by
    # appearance order; a count difference is surfaced as a warning earlier.
    expected_numbers = tuple(range(1, len(numbers) + 1))
    if numbers != expected_numbers:
        raise _error(
            f"Shot order {numbers!r} does not match {expected_numbers!r}.",
            "SHOT_ORDER_MISMATCH",
        )
    previous_end = -1.0
    for shot in plan.shots:
        if shot.start_seconds < 0 or shot.end_seconds <= shot.start_seconds:
            raise _error(
                f"Shot {shot.number} has an invalid time range.", "INVALID_SHOT_TIMING"
            )
        if shot.start_seconds + 1e-6 < previous_end:
            raise _error(
                f"Shot {shot.number} overlaps the preceding shot.", "OVERLAPPING_SHOTS"
            )
        previous_end = shot.end_seconds
    if plan.shots[0].start_seconds > 0.05:
        raise _error("The first shot must begin at 0 seconds.", "SHOT_TIMELINE_GAP")
    if require_duration and prepared.duration_seconds is not None:
        if abs(plan.shots[-1].end_seconds - prepared.duration_seconds) > 0.05:
            raise _error(
                "The final shot must end at the requested video duration.",
                "DURATION_MISMATCH",
            )

    expected_dialogue_ids = tuple(item.dialogue_id for item in prepared.dialogues)
    actual_dialogue_ids = tuple(item.dialogue_id for item in plan.dialogue_delivery)
    if actual_dialogue_ids != expected_dialogue_ids:
        raise _error(
            f"Dialogue delivery IDs {actual_dialogue_ids!r} do not match "
            f"{expected_dialogue_ids!r}.",
            "DIALOGUE_DELIVERY_MISMATCH",
        )
    shot_lookup = {shot.number: shot for shot in plan.shots}
    for delivery in plan.dialogue_delivery:
        shot = shot_lookup.get(delivery.shot)
        if shot is None:
            raise _error(
                f"Dialogue {delivery.dialogue_id} targets an unknown shot.",
                "DIALOGUE_SHOT_MISMATCH",
            )
        if not (shot.start_seconds <= delivery.start_seconds <= shot.end_seconds):
            raise _error(
                f"Dialogue {delivery.dialogue_id} is outside Shot {delivery.shot}.",
                "DIALOGUE_TIME_MISMATCH",
            )

    control_fields = [plan.style, plan.scene]
    for shot in plan.shots:
        control_fields.extend((shot.framing, shot.camera, shot.action))
    control_fields.extend((*plan.ambient, *plan.foley, plan.music))
    if any(_SPEECH_CONTROL_RE.search(value) for value in control_fields):
        raise _error(
            "Speech or narration instructions may appear only in dialogue_delivery.",
            "INVENTED_SPEECH_CONTROL",
        )
    if any(_FORBIDDEN_AUDIO_SPEECH_RE.search(value) for value in (*plan.ambient, *plan.foley)):
        raise _error(
            "Ambient/foley may not contain speech or narration.",
            "INVENTED_SPEECH_AUDIO",
        )

    wardrobe_control = " ".join(
        [
            plan.scene,
            *(shot.framing for shot in plan.shots),
            *(shot.action for shot in plan.shots),
        ]
    ).casefold()
    if prepared.wardrobe_override:
        if not re.search(
            r"\b(?:wears?|wearing|outfit|wardrobe|clothing|dress|shirt|top|shorts|"
            r"bikini|swimsuit|uniform|tracksuit|skirt|jeans|trousers)\b",
            wardrobe_control,
            re.IGNORECASE,
        ):
            raise _error(
                "The authored wardrobe override is absent from Scene/Shot controls.",
                "WARDROBE_OVERRIDE_MISSING",
            )
        missing_wardrobe = [
            term
            for term in prepared.wardrobe_required_terms
            if term.casefold() not in wardrobe_control
        ]
        if missing_wardrobe:
            raise _error(
                "The model dropped wardrobe details: " + ", ".join(missing_wardrobe) + ".",
                "WARDROBE_DETAIL_MISSING",
            )

    camera_flags_by_shot = {
        shot.number: _camera_direction_flags(f"{shot.framing} {shot.camera}")
        for shot in plan.shots
    }
    source_camera_by_shot, global_camera_requirements = _source_camera_requirements(
        prepared.redacted_prompt
    )

    def require_camera_direction(
        requirement: str, flags: frozenset[str], *, source_label: str
    ) -> None:
        if requirement == "low_up":
            required = {"low", "up"}
            opposing = {"high", "down"}
            label = "Japanese 仰角/煽り"
            description = "low-angle upward view"
        else:
            required = {"high", "down"}
            opposing = {"low", "up"}
            label = "Japanese 俯瞰"
            description = "high-angle downward view"
        if flags.intersection(opposing):
            raise _error(
                f"{label} in {source_label} was mixed with opposing camera geometry: "
                f"{', '.join(sorted(flags))}.",
                "CAMERA_DIRECTION_CONFLICT",
            )
        if not required.issubset(flags):
            raise _error(
                f"{label} in {source_label} must compile to a coherent "
                f"{description} in that same Shot.",
                "CAMERA_GLOSSARY_MISMATCH",
            )

    for shot_number, requirements in source_camera_by_shot.items():
        # A model may return fewer recoverable shots than the source has
        # labelled blocks.  The difference is advisory; there is no valid
        # generated Shot on which to enforce a missing source block.
        flags = camera_flags_by_shot.get(shot_number)
        if flags is None:
            continue
        requirement = _prioritized_camera_requirement(requirements)
        if requirement is not None:
            require_camera_direction(requirement, flags, source_label=f"Cut/Shot {shot_number}")

    requirement = _prioritized_camera_requirement(global_camera_requirements)
    if requirement is not None:
        if requirement == "low_up":
            satisfied = any(
                {"low", "up"}.issubset(flags)
                and not flags.intersection({"high", "down"})
                for flags in camera_flags_by_shot.values()
            )
        else:
            satisfied = any(
                {"high", "down"}.issubset(flags)
                and not flags.intersection({"low", "up"})
                for flags in camera_flags_by_shot.values()
            )
        if not satisfied:
            require_camera_direction(requirement, frozenset(), source_label="the source request")

    joined_foley = "\n".join(plan.foley).casefold()
    for cue in prepared.nonverbal_cues:
        if cue.english_sound.casefold() not in joined_foley:
            raise _error(
                f"Required nonverbal foley is missing: {cue.english_sound}.",
                "NONVERBAL_FOLEY_MISSING",
            )

    actual_facts = _document_numeric_facts(plan)
    missing = [fact for fact in prepared.numeric_facts if fact.key not in actual_facts]
    if missing:
        summary = ", ".join(f"{fact.value} {fact.unit}" for fact in missing)
        raise _error(f"The model dropped numeric facts: {summary}.", "NUMERIC_FACT_MISSING")


def _normalize_plan_duration(
    plan: CommunityPromptPlan, prepared: PreparedPlannerInput
) -> tuple[CommunityPromptPlan, bool]:
    """Fit model-authored shot times to the actual H3 output duration.

    The model is asked to preserve source timing, but authored prompts often
    omit later Cut timecodes while the UI still has an exact frame budget. A
    proportional fit keeps the model's cut order and relative rhythm, while
    preventing an otherwise valid Japanese prompt from being rejected merely
    because the final generated frame budget is shorter or longer.
    """

    target = prepared.duration_seconds
    if target is None or not plan.shots:
        return plan, False
    source_end = plan.shots[-1].end_seconds
    if source_end <= 0 or abs(source_end - target) <= 0.05:
        return plan, False
    scale = target / source_end
    shots = tuple(
        ShotPlan(
            number=shot.number,
            start_seconds=round(shot.start_seconds * scale, 4),
            end_seconds=round(shot.end_seconds * scale, 4),
            framing=shot.framing,
            camera=shot.camera,
            action=shot.action,
        )
        for shot in plan.shots
    )
    deliveries = tuple(
        DialogueDelivery(
            dialogue_id=delivery.dialogue_id,
            shot=delivery.shot,
            start_seconds=round(delivery.start_seconds * scale, 4),
            speaker=delivery.speaker,
            delivery=delivery.delivery,
        )
        for delivery in plan.dialogue_delivery
    )
    return (
        CommunityPromptPlan(
            schema_version=plan.schema_version,
            style=plan.style,
            scene=plan.scene,
            shots=shots,
            ambient=plan.ambient,
            foley=plan.foley,
            music=plan.music,
            dialogue_delivery=deliveries,
        ),
        True,
    )


def _repair_plan_timeline(
    plan: CommunityPromptPlan, prepared: PreparedPlannerInput
) -> tuple[CommunityPromptPlan, bool]:
    """Repair model-authored timing syntax without rejecting valid source prose.

    Shot timing is compiler bookkeeping, not creative intent. Qwen can preserve
    every visual instruction yet occasionally emit a zero-length final Shot or
    overlap two adjacent ranges. When the UI supplies an authoritative frame
    duration, rebuild only that broken timing layer from the model's positive
    relative shot lengths. The source prompt, Cut order, and all visual/audio
    controls remain untouched. ``validate_plan`` still provides the strict
    post-condition after this deterministic repair.
    """

    target = prepared.duration_seconds
    if not plan.shots:
        return plan, False

    previous_end = -1.0
    structure_valid = (
        plan.shots[0].start_seconds >= 0
        and plan.shots[0].start_seconds <= 0.05
    )
    for shot in plan.shots:
        if (
            shot.start_seconds < 0
            or shot.end_seconds <= shot.start_seconds
            or shot.start_seconds + 1e-6 < previous_end
        ):
            structure_valid = False
            break
        previous_end = shot.end_seconds
    if structure_valid:
        return plan, False

    if target is None:
        positive_ends = [shot.end_seconds for shot in plan.shots if shot.end_seconds > 0]
        target = max(positive_ends, default=float(len(plan.shots)))

    positive_durations = sorted(
        shot.end_seconds - shot.start_seconds
        for shot in plan.shots
        if shot.end_seconds - shot.start_seconds > 1e-6
    )
    if positive_durations:
        middle = len(positive_durations) // 2
        if len(positive_durations) % 2:
            fallback = positive_durations[middle]
        else:
            fallback = (
                positive_durations[middle - 1] + positive_durations[middle]
            ) / 2
    else:
        fallback = target / len(plan.shots)

    # Bound accidental outliers so a neighboring repaired Shot cannot collapse
    # to a rounded zero-length interval. Broad model-authored pacing survives.
    minimum_weight = fallback * 0.25
    maximum_weight = fallback * 4.0
    weights: list[float] = []
    for shot in plan.shots:
        duration = shot.end_seconds - shot.start_seconds
        if duration <= 1e-6:
            duration = fallback
        weights.append(min(max(duration, minimum_weight), maximum_weight))
    total_weight = sum(weights)

    repaired_shots: list[ShotPlan] = []
    cumulative = 0.0
    cursor = 0.0
    for index, (shot, weight) in enumerate(zip(plan.shots, weights)):
        cumulative += weight
        end = (
            target
            if index == len(plan.shots) - 1
            else round(target * cumulative / total_weight, 4)
        )
        repaired_shots.append(
            ShotPlan(
                number=shot.number,
                start_seconds=cursor,
                end_seconds=end,
                framing=shot.framing,
                camera=shot.camera,
                action=shot.action,
            )
        )
        cursor = end

    old_shots = {shot.number: shot for shot in plan.shots}
    new_shots = {shot.number: shot for shot in repaired_shots}
    repaired_deliveries: list[DialogueDelivery] = []
    for delivery in plan.dialogue_delivery:
        old_shot = old_shots.get(delivery.shot)
        new_shot = new_shots.get(delivery.shot)
        if old_shot is None or new_shot is None:
            repaired_deliveries.append(delivery)
            continue
        old_duration = old_shot.end_seconds - old_shot.start_seconds
        if old_duration > 1e-6:
            position = (delivery.start_seconds - old_shot.start_seconds) / old_duration
            position = min(max(position, 0.0), 1.0)
        else:
            position = 0.5
        new_start = round(
            new_shot.start_seconds
            + position * (new_shot.end_seconds - new_shot.start_seconds),
            4,
        )
        repaired_deliveries.append(
            DialogueDelivery(
                dialogue_id=delivery.dialogue_id,
                shot=delivery.shot,
                start_seconds=new_start,
                speaker=delivery.speaker,
                delivery=delivery.delivery,
            )
        )

    return (
        CommunityPromptPlan(
            schema_version=plan.schema_version,
            style=plan.style,
            scene=plan.scene,
            shots=tuple(repaired_shots),
            ambient=plan.ambient,
            foley=plan.foley,
            music=plan.music,
            dialogue_delivery=tuple(repaired_deliveries),
        ),
        True,
    )


def _source_numeric_description(fact: NumericFact) -> str:
    if fact.unit == "resolution":
        return f"{fact.value} resolution"
    return f"{fact.value} {fact.unit}"


def _strip_camera_geometry(text: str) -> str:
    """Remove only conflicting viewpoint words while retaining camera motion."""

    result = text
    for pattern in (
        _CAMERA_LOW_VIEWPOINT_RE,
        _CAMERA_HIGH_VIEWPOINT_RE,
        _CAMERA_UPWARD_VIEW_RE,
        _CAMERA_DOWNWARD_VIEW_RE,
    ):
        result = pattern.sub(" ", result)
    result = re.sub(r"\s+", " ", result).strip(" ;,:")
    return result


def _repair_authored_camera(
    plan: CommunityPromptPlan,
    prepared: PreparedPlannerInput,
    warnings: list[SourceWarning],
) -> CommunityPromptPlan:
    source_by_shot, global_requirements = _source_camera_requirements(
        prepared.redacted_prompt
    )
    required_by_shot: dict[int, set[str]] = {
        number: set(requirements) for number, requirements in source_by_shot.items()
    }
    if global_requirements:
        for number in range(1, len(plan.shots) + 1):
            required_by_shot.setdefault(number, set()).update(global_requirements)
            break

    repaired: list[ShotPlan] = []
    for shot in plan.shots:
        requirements = required_by_shot.get(shot.number, set())
        if not requirements:
            repaired.append(shot)
            continue
        if shot.number > len(plan.shots):
            repaired.append(shot)
            continue
        requirement = _prioritized_camera_requirement(requirements)
        if requirement is None:
            repaired.append(shot)
            continue
        flags = _camera_direction_flags(f"{shot.framing} {shot.camera}")
        required = {"low", "up"} if requirement == "low_up" else {"high", "down"}
        opposing = {"high", "down"} if requirement == "low_up" else {"low", "up"}
        if required.issubset(flags) and not flags.intersection(opposing):
            repaired.append(shot)
            continue
        neutral_framing = _strip_camera_geometry(shot.framing)
        neutral_camera = _strip_camera_geometry(shot.camera)
        if requirement == "low_up":
            framing = "A low-angle composition from below."
            camera = "A low-angle camera looks upward at the subject."
        else:
            framing = "A high-angle composition from above."
            camera = "A high-angle camera looks downward at the subject."
        if neutral_framing:
            framing += f" {neutral_framing}."
        if neutral_camera:
            camera += f" {neutral_camera}."
        repaired.append(replace(shot, framing=framing, camera=camera))
        warnings.append(
            SourceWarning(
                "SOURCE_CAMERA_DIRECTION_REPAIRED",
                f"Explicit source camera geometry for Shot {shot.number} was deterministically restored.",
            )
        )
    return replace(plan, shots=tuple(repaired))


def _repair_model_requirements(
    plan: CommunityPromptPlan,
    prepared: PreparedPlannerInput,
    warnings: list[SourceWarning],
) -> CommunityPromptPlan:
    """Supplement recoverable model omissions with deterministic English controls."""

    if prepared.source_shot_numbers and len(plan.shots) != len(prepared.source_shot_numbers):
        warnings.append(
            SourceWarning(
                "SOURCE_MODEL_SHOT_COUNT_DIFFERENCE",
                f"Source contains {len(prepared.source_shot_numbers)} Cut/Shot blocks but the model returned {len(plan.shots)}; recoverable shots were kept in appearance order.",
            )
        )

    scene = plan.scene
    actual_facts = _document_numeric_facts(plan)
    missing_facts = [fact for fact in prepared.numeric_facts if fact.key not in actual_facts]
    if missing_facts:
        detail = "; ".join(_source_numeric_description(fact) for fact in missing_facts)
        scene = f"{scene} Required source numeric details: {detail}."
        warnings.append(
            SourceWarning(
                "SOURCE_NUMERIC_DETAIL_REPAIRED",
                f"Missing numeric source details were appended deterministically: {detail}.",
            )
        )

    foley = list(plan.foley)
    joined_foley = "\n".join(foley).casefold()
    missing_cues = [
        cue for cue in prepared.nonverbal_cues if cue.english_sound.casefold() not in joined_foley
    ]
    if missing_cues:
        foley.extend(cue.english_sound for cue in missing_cues)
        warnings.append(
            SourceWarning(
                "SOURCE_NONVERBAL_FOLEY_REPAIRED",
                "Required nonverbal physical sounds were appended to foley.",
            )
        )

    if prepared.wardrobe_override:
        wardrobe_detail = prepared.wardrobe_direction or "the authored wardrobe"
        control = " ".join([scene, *(shot.action for shot in plan.shots)]).casefold()
        if not all(term.casefold() in control for term in prepared.wardrobe_required_terms):
            scene = f"{scene} Wardrobe continuity: the primary subject wears {wardrobe_detail}."
            warnings.append(
                SourceWarning(
                    "SOURCE_WARDROBE_DETAIL_REPAIRED",
                    "Authored wardrobe details were appended deterministically.",
                )
            )

    repaired = replace(plan, scene=scene, foley=tuple(foley))
    return _repair_authored_camera(repaired, prepared, warnings)


def _normalize_dialogue_delivery(
    plan: CommunityPromptPlan,
    prepared: PreparedPlannerInput,
    warnings: list[SourceWarning],
) -> CommunityPromptPlan:
    expected = list(prepared.dialogues)
    if not expected:
        if plan.dialogue_delivery:
            warnings.append(
                SourceWarning(
                    "MODEL_DIALOGUE_IGNORED",
                    "Model-generated dialogue delivery was removed because the source supplied no exact dialogue.",
                )
            )
        return replace(plan, dialogue_delivery=())

    candidates = list(plan.dialogue_delivery)
    normalized: list[DialogueDelivery] = []
    for index, literal in enumerate(expected):
        candidate = next(
            (
                item
                for item in candidates
                if item.dialogue_id == literal.dialogue_id
                or item.dialogue_id == index + 1
            ),
            None,
        )
        shot_number = candidate.shot if candidate is not None else index + 1
        if shot_number < 1 or shot_number > len(plan.shots):
            shot_number = min(max(index + 1, 1), len(plan.shots))
            warnings.append(
                SourceWarning(
                    "MODEL_DIALOGUE_SHOT_REPAIRED",
                    f"Dialogue {literal.dialogue_id} was assigned to a valid Shot.",
                )
            )
        shot = plan.shots[shot_number - 1]
        start = candidate.start_seconds if candidate is not None else (shot.start_seconds + shot.end_seconds) / 2
        if not math.isfinite(start) or not (shot.start_seconds <= start <= shot.end_seconds):
            start = (shot.start_seconds + shot.end_seconds) / 2
            warnings.append(
                SourceWarning(
                    "MODEL_DIALOGUE_TIME_REPAIRED",
                    f"Dialogue {literal.dialogue_id} was placed inside Shot {shot_number} without changing its exact text.",
                )
            )
        speaker = candidate.speaker if candidate is not None else "the primary visible subject"
        delivery = candidate.delivery if candidate is not None else (literal.voice_direction or "natural, clear delivery")
        normalized.append(
            DialogueDelivery(
                dialogue_id=literal.dialogue_id,
                shot=shot_number,
                start_seconds=round(start, 4),
                speaker=speaker,
                delivery=delivery,
            )
        )
    if len(candidates) != len(expected) or tuple(item.dialogue_id for item in candidates) != tuple(item.dialogue_id for item in expected):
        warnings.append(
            SourceWarning(
                "MODEL_DIALOGUE_DELIVERY_NORMALIZED",
                "Dialogue delivery IDs/count were normalized by source order; exact dialogue literals remain Python-owned.",
            )
        )
    return replace(plan, dialogue_delivery=tuple(normalized))


def parse_plan_json(
    raw: str,
    prepared: PreparedPlannerInput,
    *,
    normalize_duration: bool = True,
    diagnostics: list[SourceWarning] | None = None,
) -> CommunityPromptPlan:
    """Parse and deterministically normalize one recoverable model result.

    The source authoring prompt is never passed through this function.  The
    model result is the disposable layer: wrappers, aliases, mild omissions,
    numbering drift, unsafe model-owned tags, and broken timing are repaired or
    downgraded to diagnostics before the final English prompt is validated.
    """

    local_warnings: list[SourceWarning] = []
    value = _extract_model_object(raw, local_warnings)
    shots_value = _mapping_value(value, _MODEL_FIELD_ALIASES["shots"])
    shots_raw = _as_model_list(shots_value, "shots", local_warnings)
    if not shots_raw:
        raise _error("The planner result contains no recoverable shots.", "NO_SHOTS")
    if (
        not any(_has_substantive_model_shot_content(raw_shot) for raw_shot in shots_raw)
        and any(
            _has_explicit_model_shot_content_field(raw_shot)
            for raw_shot in shots_raw
        )
    ):
        raise _error(
            "The planner result contains explicitly supplied Shot fields but no substantive visual content.",
            "EMPTY_MODEL_SHOT_CONTENT",
        )

    shots: list[ShotPlan] = []
    for index, raw_shot in enumerate(shots_raw, start=1):
        if not isinstance(raw_shot, Mapping):
            raw_shot = {"action": str(raw_shot)}
            local_warnings.append(
                SourceWarning("MODEL_SHOT_COERCED", f"Shot {index} was coerced from a non-object value.")
            )
        number = _coerce_model_int(_mapping_value(raw_shot, _SHOT_FIELD_ALIASES["number"]))
        start = _coerce_model_float(_mapping_value(raw_shot, _SHOT_FIELD_ALIASES["start_seconds"]))
        end = _coerce_model_float(_mapping_value(raw_shot, _SHOT_FIELD_ALIASES["end_seconds"]))
        if number != index:
            local_warnings.append(
                SourceWarning("MODEL_SHOT_NUMBER_NORMALIZED", f"Model Shot number {number!r} was normalized to appearance order {index}.")
            )
        if start is None or end is None:
            local_warnings.append(
                SourceWarning("MODEL_SHOT_TIMING_DEFAULTED", f"Shot {index} had missing or unreadable timing; the timeline was rebuilt deterministically.")
            )
        shots.append(
            ShotPlan(
                number=index,
                start_seconds=start if start is not None else 0.0,
                end_seconds=end if end is not None else 0.0,
                framing=_safe_model_text(
                    _mapping_value(raw_shot, _SHOT_FIELD_ALIASES["framing"]),
                    "framing",
                    local_warnings,
                    visual_title_literals=prepared.visual_title_literals,
                ),
                camera=_safe_model_text(
                    _mapping_value(raw_shot, _SHOT_FIELD_ALIASES["camera"]),
                    "camera",
                    local_warnings,
                    visual_title_literals=prepared.visual_title_literals,
                ),
                action=_safe_model_text(
                    _mapping_value(raw_shot, _SHOT_FIELD_ALIASES["action"]),
                    "action",
                    local_warnings,
                    visual_title_literals=prepared.visual_title_literals,
                ),
            )
        )

    ambient_value = _mapping_value(value, _MODEL_FIELD_ALIASES["ambient"])
    foley_value = _mapping_value(value, _MODEL_FIELD_ALIASES["foley"])
    ambient = tuple(
        _safe_model_text(
            item,
            "ambient",
            local_warnings,
            visual_title_literals=prepared.visual_title_literals,
        )
        for item in _as_model_list(ambient_value, "ambient", local_warnings)
        if str(item).strip()
    )
    foley = tuple(
        _safe_model_text(
            item,
            "foley",
            local_warnings,
            visual_title_literals=prepared.visual_title_literals,
        )
        for item in _as_model_list(foley_value, "foley", local_warnings)
        if str(item).strip()
    )

    delivery_value = _mapping_value(value, _MODEL_FIELD_ALIASES["dialogue_delivery"])
    deliveries: list[DialogueDelivery] = []
    for index, raw_delivery in enumerate(
        _as_model_list(delivery_value, "dialogue_delivery", local_warnings), start=1
    ):
        if not isinstance(raw_delivery, Mapping):
            local_warnings.append(
                SourceWarning("MODEL_DIALOGUE_DELIVERY_DROPPED", f"Malformed dialogue delivery {index} was ignored and rebuilt from source dialogue order.")
            )
            continue
        dialogue_id = _coerce_model_int(
            _mapping_value(raw_delivery, _DELIVERY_FIELD_ALIASES["dialogue_id"])
        )
        shot = _coerce_model_int(_mapping_value(raw_delivery, _DELIVERY_FIELD_ALIASES["shot"]))
        start = _coerce_model_float(
            _mapping_value(raw_delivery, _DELIVERY_FIELD_ALIASES["start_seconds"])
        )
        deliveries.append(
            DialogueDelivery(
                dialogue_id=dialogue_id if dialogue_id is not None else index,
                shot=shot if shot is not None else index,
                start_seconds=start if start is not None else 0.0,
                speaker=_safe_model_text(
                    _mapping_value(raw_delivery, _DELIVERY_FIELD_ALIASES["speaker"]),
                    "speaker",
                    local_warnings,
                    visual_title_literals=prepared.visual_title_literals,
                ),
                delivery=_safe_model_text(
                    _mapping_value(raw_delivery, _DELIVERY_FIELD_ALIASES["delivery"]),
                    "delivery",
                    local_warnings,
                    visual_title_literals=prepared.visual_title_literals,
                ),
            )
        )

    schema_value = _mapping_value(value, _MODEL_FIELD_ALIASES["schema_version"])
    if schema_value != PLAN_SCHEMA_VERSION:
        local_warnings.append(
            SourceWarning("MODEL_SCHEMA_NORMALIZED", "Model schema/version drift was normalized to the pinned community plan schema.")
        )
    plan = CommunityPromptPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        style=_safe_model_text(
            _mapping_value(value, _MODEL_FIELD_ALIASES["style"]),
            "style",
            local_warnings,
            visual_title_literals=prepared.visual_title_literals,
        ),
        scene=_safe_model_text(
            _mapping_value(value, _MODEL_FIELD_ALIASES["scene"]),
            "scene",
            local_warnings,
            visual_title_literals=prepared.visual_title_literals,
        ),
        shots=tuple(shots),
        ambient=ambient,
        foley=foley,
        music=_safe_model_text(
            _mapping_value(value, _MODEL_FIELD_ALIASES["music"], "N/A"),
            "music",
            local_warnings,
            allow_na=True,
            visual_title_literals=prepared.visual_title_literals,
        ),
        dialogue_delivery=tuple(deliveries),
    )
    plan, repaired = _repair_plan_timeline(plan, prepared)
    normalized = False
    if normalize_duration:
        plan, normalized = _normalize_plan_duration(plan, prepared)
    if repaired:
        local_warnings.append(SourceWarning("PLANNER_TIMELINE_REPAIRED", "Model timing was rebuilt into a contiguous timeline."))
    if normalized:
        local_warnings.append(SourceWarning("PLANNER_DURATION_NORMALIZED", "Model timing was fitted to the authoritative output duration."))
    plan = _repair_model_requirements(plan, prepared, local_warnings)
    plan = _normalize_dialogue_delivery(plan, prepared, local_warnings)
    validate_plan(plan, prepared, require_duration=normalize_duration)
    if diagnostics is not None:
        diagnostics.extend(local_warnings)
    return plan


def _format_seconds(value: float) -> str:
    return _normal_decimal(value)


def _render_references(prepared: PreparedPlannerInput) -> list[str]:
    references = prepared.references
    if not references:
        return []
    lines = ["Reference material:"]
    image_ordinal = 0
    for item in references:
        if item.kind == "image":
            image_ordinal += 1
            if image_ordinal == 1 and prepared.wardrobe_override:
                lines.append(
                    f"{item.label} defines the primary visible subject's identity, face, "
                    "body shape, and hair; the wardrobe must follow the Scene and Shot "
                    "instructions and replaces the reference outfit."
                )
                continue
            if item.role in {"style", "visual_style"}:
                description = "defines the visual style, palette, and rendering language"
            elif item.role in {"composition", "first_frame", "start_frame"}:
                description = "defines the opening composition and subject placement"
            elif image_ordinal == 1:
                description = "defines the primary visible subject identity and appearance"
            else:
                description = "defines an additional visible subject identity and appearance"
        elif item.kind == "video":
            description = "defines the intended motion rhythm and camera behavior"
        elif item.role in {"music", "bgm"}:
            description = "defines the non-diegetic music character"
        elif item.role in {"ambience", "ambient", "environment"}:
            description = "defines the environmental ambience"
        else:
            description = "defines voice timbre and delivery"
        lines.append(f"{item.label} {description}.")
    return lines


def render_prompt(plan: CommunityPromptPlan, prepared: PreparedPlannerInput) -> str:
    """Render the validated model plan into the public-example prompt shape."""

    validate_plan(plan, prepared)
    blocks: list[str] = [f"Style:\n{plan.style}"]
    reference_lines = _render_references(prepared)
    if reference_lines:
        blocks.append("\n".join(reference_lines))
    scene_lines = [plan.scene]
    if prepared.wardrobe_override and prepared.wardrobe_direction:
        scene_lines.append(
            "Wardrobe: The primary visible subject wears "
            f"{prepared.wardrobe_direction}, replacing the reference outfit."
        )
    blocks.append("Scene:\n" + "\n".join(scene_lines))
    for shot in plan.shots:
        blocks.append(
            f"Shot {shot.number} ({_format_seconds(shot.start_seconds)}-"
            f"{_format_seconds(shot.end_seconds)} seconds):\n"
            f"Framing: {shot.framing}\n"
            f"Camera: {shot.camera}\n"
            f"Action: {shot.action}"
        )
    ambient = "; ".join(plan.ambient) if plan.ambient else "N/A"
    foley = "; ".join(plan.foley) if plan.foley else "N/A"
    effective_audio_preset = prepared.audio_preset
    if effective_audio_preset == "dialogue" and not prepared.dialogues:
        effective_audio_preset = "ambience"
    effective_music = (
        plan.music
        if prepared.music_policy == "auto"
        else _MUSIC_POLICY_DIRECTIONS[prepared.music_policy]
    )
    audio_lines = [
        "Audio:",
        f"Mix: {_AUDIO_PRESET_DIRECTIONS[effective_audio_preset]}",
        f"Ambient: {ambient}",
        f"Foley: {foley}",
        f"Music: {effective_music}",
    ]
    literal_by_id = {item.dialogue_id: item for item in prepared.dialogues}
    for delivery in plan.dialogue_delivery:
        literal = literal_by_id[delivery.dialogue_id]
        effective_delivery = literal.voice_direction or delivery.delivery
        audio_lines.append(
            f"At {_format_seconds(delivery.start_seconds)} seconds in Shot {delivery.shot}, "
            f"{delivery.speaker} says in Japanese with {effective_delivery}: \"{literal.text}\""
        )
    blocks.append("\n".join(audio_lines))
    prompt = "\n\n".join(blocks).strip()
    validate_rendered_prompt(prompt, plan, prepared)
    return prompt


def validate_rendered_prompt(
    prompt: str, plan: CommunityPromptPlan, prepared: PreparedPlannerInput
) -> None:
    if _D_TAG_RE.search(prompt):
        raise _error("Rendered prompt contains a forbidden <d> tag.", "D_TAG_FORBIDDEN")
    quoted = re.findall(r'\"([^\"\r\n]*)\"', prompt)
    expected_quotes = [item.text for item in prepared.dialogues]
    if quoted != expected_quotes:
        raise _error(
            f"Rendered dialogue quotes {quoted!r} do not match {expected_quotes!r}.",
            "DIALOGUE_QUOTE_MISMATCH",
        )
    for literal in expected_quotes:
        if prompt.count(f'\"{literal}\"') != 1:
            raise _error(
                "Every dialogue literal must appear in exactly one ordinary quote pair.",
                "DIALOGUE_NOT_EXACTLY_ONCE",
            )
    control_only = prompt
    for literal in expected_quotes:
        control_only = control_only.replace(f'\"{literal}\"', "[EXACT DIALOGUE]")
    if _NON_ENGLISH_RE.search(control_only):
        raise _error(
            "Rendered prompt contains non-English text outside exact dialogue.",
            "NON_ENGLISH_CONTROL",
        )

    expected_tags = [item.label for item in prepared.references]
    actual_tags = [
        f"<{match.group(1).title()} {int(match.group(2))}>"
        for match in _REFERENCE_TAG_RE.finditer(prompt)
    ]
    if actual_tags != expected_tags or any(actual_tags.count(tag) != 1 for tag in expected_tags):
        raise _error(
            f"Rendered reference tags {actual_tags!r} do not match inventory {expected_tags!r}.",
            "REFERENCE_TAG_SET_MISMATCH",
        )
    rendered_shots = tuple(int(value) for value in re.findall(r"(?m)^Shot ([1-9][0-9]*) ", prompt))
    if rendered_shots != tuple(shot.number for shot in plan.shots):
        raise _error("Rendered Shot order changed.", "SHOT_ORDER_MISMATCH")
    if prepared.wardrobe_override:
        if any(item.kind == "image" for item in prepared.references) and (
            "the wardrobe must follow the Scene and Shot instructions and replaces "
            "the reference outfit"
            not in prompt
        ):
            raise _error(
                "Rendered image-reference wardrobe override is missing.",
                "WARDROBE_OVERRIDE_MISSING",
            )
        for term in prepared.wardrobe_required_terms:
            if term.casefold() not in prompt.casefold():
                raise _error(
                    f"Rendered prompt dropped wardrobe detail {term}.",
                    "WARDROBE_DETAIL_MISSING",
                )
    actual_facts = set(fact.key for fact in extract_numeric_facts(prompt))
    missing = [fact for fact in prepared.numeric_facts if fact.key not in actual_facts]
    if missing:
        raise _error("Rendered prompt dropped a numeric fact.", "NUMERIC_FACT_MISSING")


def compile_model_result(raw: str, prepared: PreparedPlannerInput) -> CompiledCommunityPrompt:
    plan_warnings: list[SourceWarning] = []
    plan = parse_plan_json(
        raw,
        prepared,
        normalize_duration=True,
        diagnostics=plan_warnings,
    )
    warning_codes = {warning.code for warning in plan_warnings}
    return CompiledCommunityPrompt(
        render_prompt(plan, prepared),
        plan,
        prepared,
        duration_normalized="PLANNER_DURATION_NORMALIZED" in warning_codes,
        timeline_repaired="PLANNER_TIMELINE_REPAIRED" in warning_codes,
        plan_warnings=tuple(plan_warnings),
    )


def inspect_model_checkout(
    model_path: str | Path,
    *,
    expected_revision: str = MODEL_REVISION,
) -> ModelCheckoutMetadata:
    """Verify the project-owned provenance marker against the pinned lock.

    Hugging Face's private ``.cache`` layout is deliberately ignored: it may be
    removed after download and is not a stable production provenance contract.
    """

    path = Path(model_path).resolve()
    provenance_path = path / MODEL_PROVENANCE_FILENAME
    detected: str | None = None
    marker_lock_sha: str | None = None
    marker_file_count: int | None = None
    marker_total_bytes: int | None = None
    lock_path: Path | None = None
    verified = False
    try:
        # The lock is trusted only from the repository that owns the fixed
        # models/prompt_planner/Qwen3-4B-Instruct-2507 placement.
        repo_root = path.parents[2]
        expected_path = (repo_root / MODEL_RELATIVE_PATH).resolve()
        if path != expected_path:
            raise ValueError("model path is outside the fixed project placement")
        lock_path = (repo_root / MODEL_LOCK_FILENAME).resolve()
        if lock_path.parent != repo_root or not lock_path.is_file():
            raise ValueError("project prompt planner lock is missing")
        if not provenance_path.is_file():
            raise ValueError("project model provenance marker is missing")

        marker = json.loads(provenance_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if not isinstance(marker, dict) or not isinstance(lock, dict):
            raise ValueError("provenance marker and lock must be objects")
        detected_value = marker.get("revision")
        detected = str(detected_value).lower() if detected_value is not None else None
        marker_lock_sha = str(marker.get("lock_sha256") or "").lower() or None
        marker_file_count_value = marker.get("file_count")
        marker_total_bytes_value = marker.get("total_bytes")
        if isinstance(marker_file_count_value, bool) or not isinstance(
            marker_file_count_value, int
        ):
            raise ValueError("marker file_count is invalid")
        if isinstance(marker_total_bytes_value, bool) or not isinstance(
            marker_total_bytes_value, int
        ):
            raise ValueError("marker total_bytes is invalid")
        marker_file_count = marker_file_count_value
        marker_total_bytes = marker_total_bytes_value

        lock_bytes = lock_path.read_bytes()
        actual_lock_sha = hashlib.sha256(lock_bytes).hexdigest()
        source = lock.get("source")
        verification = lock.get("verification")
        files = lock.get("files")
        if not isinstance(source, dict) or not isinstance(verification, dict):
            raise ValueError("lock source/verification is invalid")
        if not isinstance(files, list) or not all(isinstance(item, dict) for item in files):
            raise ValueError("lock files is invalid")
        file_sizes: list[int] = []
        for item in files:
            size = item.get("size")
            relative = item.get("path")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError("lock file size is invalid")
            if not isinstance(relative, str) or not relative:
                raise ValueError("lock file path is invalid")
            locked_target = (repo_root / Path(relative)).resolve()
            try:
                locked_target.relative_to(path)
            except ValueError as exc:
                raise ValueError("lock file escapes the fixed model directory") from exc
            file_sizes.append(size)

        verified = all(
            (
                marker.get("schema_version") == 1,
                marker.get("model_id") == MODEL_ID,
                detected == expected_revision.lower(),
                marker_lock_sha == actual_lock_sha,
                marker_file_count == MODEL_RUNTIME_FILE_COUNT,
                marker_total_bytes == MODEL_RUNTIME_TOTAL_BYTES,
                lock.get("schema_version") == 1,
                source.get("repo_id") == MODEL_ID,
                str(source.get("revision") or "").lower() == expected_revision.lower(),
                verification.get("total_bytes") == MODEL_RUNTIME_TOTAL_BYTES,
                len(files) == MODEL_RUNTIME_FILE_COUNT,
                sum(file_sizes) == MODEL_RUNTIME_TOTAL_BYTES,
            )
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError, IndexError):
        verified = False
    return ModelCheckoutMetadata(
        model_id=MODEL_ID,
        expected_revision=expected_revision,
        detected_revision=detected,
        verified=verified,
        path=str(path),
        provenance_path=str(provenance_path),
        lock_path=str(lock_path) if lock_path is not None else None,
        lock_sha256=marker_lock_sha,
        file_count=marker_file_count,
        total_bytes=marker_total_bytes,
    )


def require_verified_model_checkout(
    model_path: str | Path,
    *,
    expected_revision: str = MODEL_REVISION,
) -> ModelCheckoutMetadata:
    metadata = inspect_model_checkout(model_path, expected_revision=expected_revision)
    if not metadata.verified:
        raise _error(
            "Prompt planner model checkout is missing or does not match the pinned revision "
            f"{expected_revision}; detected={metadata.detected_revision!r}.",
            "MODEL_REVISION_UNVERIFIED",
        )
    required = (
        "config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    )
    missing = [name for name in required if not (Path(metadata.path) / name).is_file()]
    if missing:
        raise _error(
            f"Prompt planner checkout is incomplete; missing {missing!r}.",
            "MODEL_CHECKOUT_INCOMPLETE",
        )
    return metadata


__all__ = [
    "CompiledCommunityPrompt",
    "CommunityPromptPlan",
    "CommunityPromptPlannerError",
    "DialogueDelivery",
    "DialogueLiteral",
    "MODEL_ID",
    "MODEL_RELATIVE_PATH",
    "MODEL_REVISION",
    "ModelCheckoutMetadata",
    "NonverbalCue",
    "PLAN_SCHEMA_VERSION",
    "PreparedPlannerInput",
    "ReferenceItem",
    "SourceReferencePreflight",
    "SYSTEM_PROMPT",
    "SourceWarning",
    "ShotPlan",
    "build_model_messages",
    "compile_model_result",
    "extract_numeric_facts",
    "extract_shot_numbers",
    "has_explicit_source_dialogue",
    "inspect_model_checkout",
    "parse_plan_json",
    "prepare_planner_input",
    "preflight_source_references",
    "render_prompt",
    "require_verified_model_checkout",
    "validate_plan",
    "validate_rendered_prompt",
]
