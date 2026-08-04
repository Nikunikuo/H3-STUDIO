"""Safe Japanese-to-English prompt preparation for MiniMax H3.

Only user-authored control prose is translated.  Native H3 ``<d>`` dialogue
blocks and the canonical formatter line that contains them are immutable.  A
deterministic renderer, rather than the translation model, owns H3's official
six-section prompt structure.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


OFFICIAL_SECTION_HEADERS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
OFFICIAL_BASE_SECTION_HEADERS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)


class GenerationMode(str, Enum):
    T2V = "t2v"
    I2V = "i2v"
    FIRST_LAST = "first_last"
    OMNI = "omni"

_DIALOGUE_RE = re.compile(
    r"<d>\s*\[(?P<language>[A-Za-z][A-Za-z -]{1,31})\]\s*"
    r"(?P<payload>.*?)</d>",
    re.IGNORECASE | re.DOTALL,
)
# Canonical formatter prose legitimately says "each <d> tag".  Only an open
# tag followed by H3's required language marker begins a dialogue payload.
_DIALOGUE_OPEN_RE = re.compile(
    r"<d>(?=\s*\[[A-Za-z][A-Za-z -]{1,31}\]\s*)", re.IGNORECASE
)
_ANY_DIALOGUE_OPEN_RE = re.compile(r"<d(?:\s+[^>\r\n]*)?>", re.IGNORECASE)
_CANONICAL_DIALOGUE_TOKEN_MENTION_RE = re.compile(
    r"\beach\s+<d>\s+tag\b", re.IGNORECASE
)
_DIALOGUE_CLOSE_RE = re.compile(r"</d>", re.IGNORECASE)
_EXACT_DIALOGUE_RE = re.compile(
    r"<d>\[(?P<language>[A-Za-z][A-Za-z -]{1,31})\] "
    r"(?P<payload>.*?)</d>",
    re.DOTALL,
)
_SUPPORTED_DIALOGUE_LANGUAGES = frozenset(
    {
        "Arabic",
        "Chinese",
        "English",
        "French",
        "German",
        "Italian",
        "Japanese",
        "Korean",
        "Portuguese",
        "Russian",
        "Spanish",
    }
)
_RESERVED_DIALOGUE_TOKEN_RE = re.compile(r"<[^>\r\n]{1,96}>")
_REFERENCE_RE = re.compile(
    r"<(?P<kind>Picture|Video|Audio|Subject)\s+(?P<number>[1-9][0-9]*)>",
    re.IGNORECASE,
)
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_QUOTED_TEXT_RE = re.compile(
    r"「(?P<corner>[^」\r\n]+)」|『(?P<double_corner>[^』\r\n]+)』|"
    r"“(?P<curly>[^”\r\n]+)”|\"(?P<ascii>[^\"\r\n]+)\""
)
_ASCII_QUOTED_JAPANESE_RE = re.compile(
    r'"(?P<text>[^"\r\n]*[\u3040-\u30ff\u3400-\u9fff][^"\r\n]*)"'
)
_VISIBLE_TEXT_CUE_RE = re.compile(
    r"(?:看板|字幕|テロップ|文字|標識|ラベル|表示|書かれ|書いて|題名|タイトル|"
    r"ネオン|画面|ポスター|メニュー|キャプション|"
    r"\b(?:sign|subtitle|caption|on[- ]screen\s+text|visible\s+text|label|banner|"
    r"display(?:s|ed)?|written|reads?|title|neon\s+text|poster|menu)\b)",
    re.IGNORECASE,
)
_LYRIC_CUE_RE = re.compile(
    r"(?:歌詞|歌う|歌って|歌声|唱える|\b(?:lyrics?|sings?|sang|singing|chant(?:s|ed|ing)?)\b)",
    re.IGNORECASE,
)
_CUT_RE = re.compile(
    r"^\s*(?:\[(?:Cut|Shot)\s*[1-9][0-9]*\]|(?:Cut|Shot)\s*#?\s*[1-9][0-9]*)\s*(?::.*)?$",
    re.IGNORECASE,
)
_SEPARATOR_RE = re.compile(r"^\s*(?:[-=_*~]{3,}|\*\s*\*\s*\*)\s*$")
_SECTION_RE = re.compile(
    r"^(subject_definitions|summary|retention_analysis|detailed_description|"
    r"integrated_multimodal_description|"
    r"overall_soundscape|non_diegetic_music):\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ANY_SECTION_LINE_RE = re.compile(
    r"^\s*(?:subject_definitions|summary|retention_analysis|detailed_description|"
    r"integrated_multimodal_description|"
    r"overall_soundscape|non_diegetic_music)\s*:\s*$",
    re.IGNORECASE,
)
_AUDIO_LINE_RE = re.compile(r"^\s*Audio\s*:\s*(?P<body>.*)$", re.IGNORECASE)
_MUSIC_LINE_RE = re.compile(r"^\s*Music\s*:\s*(?P<body>.*)$", re.IGNORECASE)

# This fixes the observed harmless typo without touching dialogue bytes.
_COORDINATED_SUBJECT_TYPO_RE = re.compile(
    r"(?P<left>>)[ \t]*と[ \t]*(?P<right><(?:Picture|Video|Audio|Subject)\s+"
    r"[1-9][0-9]*>)[ \t]*のが",
    re.IGNORECASE,
)

_CHARACTER_TERMS_RE = re.compile(
    r"(?:人物|人|女性|男性|女の子|男の子|少女|少年|彼女|彼|キャラクター|"
    r"character|person|woman|man|girl|boy|subject|speaker)",
    re.IGNORECASE,
)
_JAPANESE_ACTOR_PARTICLE_RE = re.compile(r"(?:は|が|も|たちが|たちは)")
_ACTOR_ACTION_RE = re.compile(
    r"(?:歩|走|座|立|振る|振って|見る|見て|向く|笑|泣|踊|食べ|飲|持|開け|閉じ|"
    r"\b(?:walks?|runs?|sits?|stands?|waves?|looks?|smiles?|cries?|dances?|"
    r"eats?|drinks?|holds?|opens?|closes?|turns?|moves?)\b)",
    re.IGNORECASE,
)
_WARDROBE_TERM_RE = re.compile(
    r"(?:服装|衣装|洋服|ワンピース|コート|ドレス|制服|水着|シャツ|ブラウス|"
    r"スカート|ズボン|ジャケット|outfit|wardrobe|clothing|clothes|dress|coat|"
    r"uniform|swimsuit|shirt|blouse|skirt|pants|trousers|jacket)",
    re.IGNORECASE,
)
_WARDROBE_REJECTION_RE = re.compile(
    r"(?:参照(?:画像)?の(?:服装|衣装)[^。！？\n]{0,20}ではなく|"
    r"元の(?:服装|衣装)[^。！？\n]{0,20}ではなく|"
    r"(?:服装|衣装)[^。！？\n]{0,20}(?:ではなく|の代わりに)|"
    r"\b(?:not|instead\s+of|rather\s+than)\b[^.?!\n]{0,48}"
    r"(?:reference|source|original)?\s*(?:outfit|wardrobe|clothing|clothes)|"
    r"(?:reference|source|original)\s+(?:outfit|wardrobe|clothing|clothes)"
    r"[^.?!\n]{0,32}\b(?:not|replaced?|changed?)\b)",
    re.IGNORECASE,
)
_WARDROBE_KEEP_RE = re.compile(
    r"(?:(?:同じ|そのまま|維持|保持)[^。！？\n]{0,24}"
    r"(?:服装|衣装|ワンピース|コート|ドレス|制服|水着)|"
    r"(?:参照画像|元)[^。！？\n]{0,16}(?:と同じ|のまま)[^。！？\n]{0,16}"
    r"(?:服装|衣装)?|"
    r"\b(?:same|unchanged|preserve(?:s|d|ing)?|keep(?:s|ing)?)\b"
    r"[^.?!\n]{0,32}(?:outfit|wardrobe|clothing|clothes|dress|coat)|"
    r"\b(?:reference|source|original)\s+(?:outfit|wardrobe|clothing|clothes)\b)",
    re.IGNORECASE,
)
_WARDROBE_CHANGE_RE = re.compile(
    r"(?:着替|着せ替|着せる|変更|取り替|入れ替|新しい|別の|異なる|"
    r"(?:服装|衣装|ワンピース|コート|ドレス|制服|水着)[^。！？\n]{0,20}"
    r"(?:に変|に替|にする)|"
    r"\b(?:chang(?:e|es|ed|ing)(?:\s+[^.?!\n]{0,16})?\s+into|"
    r"replac(?:e|es|ed|ing)|swap(?:s|ped|ping)?|switch(?:es|ed|ing)?|"
    r"put(?:s|ting)?\s+on|new|different|target|alternate)\b)",
    re.IGNORECASE,
)
_WARDROBE_CHANGE_ACTION_RE = re.compile(
    r"(?:着替|着せ替|着せる|(?:服装|衣装|洋服)[^。！？\n]{0,20}"
    r"(?:変更|取り替|入れ替|に変|に替|にする)|"
    r"(?:ワンピース|コート|ドレス|制服|水着|シャツ|ブラウス|スカート|"
    r"ズボン|ジャケット)[^。！？\n]{0,20}(?:に変|に替|にする)|"
    r"\b(?:chang(?:e|es|ed|ing)(?:\s+[^.?!\n]{0,16})?\s+into|"
    r"replac(?:e|es|ed|ing)|swap(?:s|ped|ping)?|switch(?:es|ed|ing)?|"
    r"put(?:s|ting)?\s+on)\b)",
    re.IGNORECASE,
)
_JAPANESE_EXPLICIT_GARMENT_WEAR_RE = re.compile(
    r"(?:ワンピース|コート|ドレス|制服|水着|シャツ|ブラウス|スカート|ズボン|"
    r"ジャケット)を(?:着て|着る|身につけ)",
)

_VISIBLE_OUTPUT_CUE_RE = re.compile(
    r"\b(?:sign|subtitle|caption|label|banner|screen|display(?:s|ed|ing)?|"
    r"reads?|written|on[- ]screen\s+text|visible\s+text|title|poster|menu|"
    r"neon\s+text)\b",
    re.IGNORECASE,
)
_HUMAN_VOCAL_OUTPUT_RE = re.compile(
    r"\b(?:woman|man|girl|boy|person|character|subject|speaker|narrator|she|he|they)\b"
    r"[^.?!\n]{0,40}\b(?:say(?:s|ing)?|said|speak(?:s|ing)?|spoke|"
    r"sing(?:s|ing)?|sang|narrat(?:e|es|ed|ing)|utter(?:s|ed|ing)?|"
    r"voice(?:s|d|ing)?|whisper(?:s|ed|ing)?|shout(?:s|ed|ing)?|"
    r"read(?:s|ing)?|recit(?:e|es|ed|ing)|explain(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
_TRANSLATOR_SPEECH_ROLE_RE = re.compile(
    r"\b(?:narrat(?:ion|or)|voice[ -]?over|human\s+voice|"
    r"(?:a|an|the)\s+(?:(?:male|female|human|unknown)\s+)?voice|"
    r"spoken\s+(?:dialogue|line|words?)|"
    r"dialogue(?!\s+(?:box|bubble|balloon)))\b",
    re.IGNORECASE,
)
_UNREQUESTED_SPEECH_AUDIO_RE = re.compile(
    r"\b(?:speech|spoken|speak(?:s|ing)?|said|says?|talk(?:s|ing)?|dialogue|"
    r"narrat(?:e|es|ed|ing|ion|or)|voice(?:[ -]?over)?|voices|"
    r"vocal(?:s|ization)?|language|words?|utterance|greeting)\b",
    re.IGNORECASE,
)

_ALIAS_NAMES: tuple[tuple[str, str], ...] = (
    ("アリス", "Alice"), ("ボブ", "Bob"), ("キャロル", "Carol"),
    ("デイビッド", "David"), ("エマ", "Emma"), ("フランク", "Frank"),
    ("グレース", "Grace"), ("ヘンリー", "Henry"), ("アイリス", "Iris"),
    ("ジャック", "Jack"), ("ケイト", "Kate"), ("ルーク", "Luke"),
    ("メアリー", "Mary"), ("ノア", "Noah"), ("オリビア", "Olivia"),
    ("ピーター", "Peter"), ("クイン", "Quinn"), ("ローズ", "Rose"),
    ("サム", "Sam"), ("ティナ", "Tina"), ("ユマ", "Yuma"),
    ("ビクター", "Victor"), ("ウェンディ", "Wendy"), ("ザック", "Zach"),
    ("アーロン", "Aaron"), ("ベラ", "Bella"), ("クレア", "Claire"),
    ("ダニエル", "Daniel"), ("エリン", "Erin"), ("フェリックス", "Felix"),
    ("ジーナ", "Gina"), ("ヒューゴ", "Hugo"), ("イヴ", "Eve"),
    ("ジョナ", "Jonah"), ("キラ", "Kira"), ("レオ", "Leo"),
    ("ミア", "Mia"), ("ニコ", "Nico"), ("オパール", "Opal"),
    ("ポール", "Paul"), ("ルビー", "Ruby"), ("セス", "Seth"),
    ("テス", "Tess"), ("ウーゴ", "Ugo"), ("ヴェラ", "Vera"),
    ("ウィル", "Will"), ("ゼイン", "Zane"),
)

_ALIAS_JAPANESE_PREFIX = {
    "Picture": "参照画像",
    "Video": "参照動画",
    "Audio": "参照音声",
    "Subject": "被写体",
}
_ALIAS_ENGLISH_NOUN = {
    "Picture": r"reference\s+(?:image|picture)",
    "Video": r"reference\s+video",
    "Audio": r"reference\s+(?:audio|voice\s+recording)",
    "Subject": r"subject",
}


class PromptTranslationError(RuntimeError):
    """A fail-closed prompt translation or compilation error."""

    def __init__(self, message: str, *, code: str = "PROMPT_TRANSLATION_FAILED") -> None:
        super().__init__(message)
        self.code = code


class Translator(Protocol):
    def __call__(self, text: str) -> str: ...


@dataclass(frozen=True, slots=True)
class PromptTranslationResult:
    compiled_prompt: str
    translated_detail: str
    translated_line_count: int
    dialogue_blocks: tuple[str, ...]
    source_reference_tags: tuple[str, ...]
    translated_reference_tags: tuple[str, ...]
    visible_text_literals: tuple[str, ...] = ()
    mode: GenerationMode = GenerationMode.OMNI
    already_compiled: bool = False

    def metadata(self) -> dict[str, Any]:
        renderer = (
            "h3-studio-official-six-section-v1"
            if self.mode is GenerationMode.OMNI
            else "h3-studio-official-three-section-v1"
        )
        return {
            "source_language": "ja" if self.translated_line_count else "en",
            "target_language": "en",
            "translated_line_count": self.translated_line_count,
            "dialogue_block_count": len(self.dialogue_blocks),
            "reference_tag_count": len(self.source_reference_tags),
            "visible_text_literals": list(self.visible_text_literals),
            "visible_text_literal_sha256": [
                hashlib.sha256(item.encode("utf-8")).hexdigest()
                for item in self.visible_text_literals
            ],
            "already_compiled": self.already_compiled,
            "mode": self.mode.value,
            "renderer": renderer,
        }


@dataclass(frozen=True, slots=True)
class _Alias:
    original_tag: str
    kind: str
    number: int
    japanese_name: str
    english_name: str

    @property
    def source_phrase(self) -> str:
        return f"{_ALIAS_JAPANESE_PREFIX[self.kind]}{self.japanese_name}"


@dataclass(frozen=True, slots=True)
class _VisibleTextAlias:
    original_text: str
    japanese_name: str
    english_name: str

    @property
    def protected_phrase(self) -> str:
        return f"on-screen text {self.english_name}"


@dataclass(slots=True)
class _TranslationState:
    source_prompt: str
    next_alias: int = 0
    visible_text_literals: list[str] = field(default_factory=list)

    def next_name(self) -> tuple[str, str]:
        while self.next_alias < len(_ALIAS_NAMES):
            japanese_name, english_name = _ALIAS_NAMES[self.next_alias]
            self.next_alias += 1
            if japanese_name not in self.source_prompt and not re.search(
                rf"\b{re.escape(english_name)}\b", self.source_prompt, re.IGNORECASE
            ):
                return japanese_name, english_name
        raise PromptTranslationError(
            "保護対象が多すぎるため、安全な翻訳用エイリアスを割り当てられません。",
            code="TRANSLATION_ALIAS_EXHAUSTED",
        )

    def alias_for(self, match: re.Match[str]) -> _Alias:
        japanese_name, english_name = self.next_name()
        return _Alias(
            original_tag=match.group(0),
            kind=match.group("kind").title(),
            number=int(match.group("number")),
            japanese_name=japanese_name,
            english_name=english_name,
        )


def validate_native_dialogue_blocks(text: str) -> tuple[str, ...]:
    """Validate native H3 dialogue syntax, language, payload, and token safety."""

    structural_text = _CANONICAL_DIALOGUE_TOKEN_MENTION_RE.sub(
        "native dialogue boundary", text
    )
    opens = len(_ANY_DIALOGUE_OPEN_RE.findall(structural_text))
    closes = len(_DIALOGUE_CLOSE_RE.findall(structural_text))
    matches = tuple(_DIALOGUE_RE.finditer(structural_text))
    if opens != closes or opens != len(matches):
        raise PromptTranslationError(
            "<d>台詞タグが閉じられていないか、入れ子になっています。",
            code="MALFORMED_DIALOGUE_TAG",
        )
    blocks: list[str] = []
    for match in matches:
        block = match.group(0)
        exact = _EXACT_DIALOGUE_RE.fullmatch(block)
        if exact is None:
            raise PromptTranslationError(
                "native dialogue blockの大文字小文字または空白が公式形式と一致しません。",
                code="INVALID_DIALOGUE_LANGUAGE_TAG",
            )
        language = exact.group("language")
        if language not in _SUPPORTED_DIALOGUE_LANGUAGES:
            if language.title() in _SUPPORTED_DIALOGUE_LANGUAGES:
                raise PromptTranslationError(
                    "台詞language tagの大文字小文字が公式形式と一致しません。",
                    code="INVALID_DIALOGUE_LANGUAGE_TAG",
                )
            raise PromptTranslationError(
                f"H3が安定対応していない台詞language tagです: {language}",
                code="UNSUPPORTED_DIALOGUE_LANGUAGE",
            )
        payload = exact.group("payload")
        if not payload.strip():
            raise PromptTranslationError(
                "native dialogue blockの発話内容が空です。",
                code="EMPTY_DIALOGUE_PAYLOAD",
            )
        if _RESERVED_DIALOGUE_TOKEN_RE.search(payload):
            raise PromptTranslationError(
                "台詞payload内にH3予約トークンまたはangle tagを含めることはできません。",
                code="RESERVED_TOKEN_IN_DIALOGUE",
            )
        blocks.append(block)
    return tuple(blocks)


def _dialogue_blocks(text: str) -> tuple[str, ...]:
    return validate_native_dialogue_blocks(text)


def _outside_dialogue(text: str) -> str:
    return _DIALOGUE_RE.sub("", text)


def _quoted_payload(match: re.Match[str]) -> str:
    return next(value for value in match.groupdict().values() if value is not None)


def _classify_visible_text_literals(text: str) -> tuple[str, ...]:
    """Classify CJK quoted text outside dialogue, rejecting ambiguous/lyric uses."""

    outside = _outside_dialogue(text)
    literals: list[str] = []
    for line in outside.splitlines():
        matches = tuple(
            match
            for match in _QUOTED_TEXT_RE.finditer(line)
            if _JAPANESE_RE.search(_quoted_payload(match))
        )
        if not matches:
            continue
        if _LYRIC_CUE_RE.search(line):
            raise PromptTranslationError(
                "歌詞は画面内文字として処理できません。歌唱を使う場合はH3 native dialogue "
                "blockとして明示してください。",
                code="UNSUPPORTED_LYRICS",
            )
        if not _VISIBLE_TEXT_CUE_RE.search(line):
            raise PromptTranslationError(
                "引用された日本語が台詞・画面内文字・歌詞のどれか判定できません。",
                code="UNCLASSIFIED_CJK_QUOTED_TEXT",
            )
        literals.extend(_quoted_payload(match) for match in matches)
    return tuple(literals)


def classify_visible_text_literals(text: str) -> tuple[str, ...]:
    """Return classified visible CJK literals after validating prompt structure.

    This public wrapper is intentionally safe for already-compiled official
    prompts as well as source prompts.  Callers can use the returned literals
    to build local integrity metadata without invoking the translator.
    """

    if not isinstance(text, str):
        raise PromptTranslationError(
            "画面内文字を分類するプロンプトは文字列で指定してください。",
            code="INVALID_PROMPT",
        )
    validate_native_dialogue_blocks(text)
    return _classify_visible_text_literals(text)


def _control_without_visible_literals(text: str) -> str:
    outside = _outside_dialogue(text)
    lines: list[str] = []
    for line in outside.splitlines(keepends=True):
        cjk_matches = tuple(
            match
            for match in _QUOTED_TEXT_RE.finditer(line)
            if _JAPANESE_RE.search(_quoted_payload(match))
        )
        if not cjk_matches:
            lines.append(line)
            continue
        if _LYRIC_CUE_RE.search(line):
            raise PromptTranslationError(
                "歌詞は画面内文字として処理できません。",
                code="UNSUPPORTED_LYRICS",
            )
        if not _VISIBLE_TEXT_CUE_RE.search(line):
            raise PromptTranslationError(
                "引用された日本語が台詞・画面内文字・歌詞のどれか判定できません。",
                code="UNCLASSIFIED_CJK_QUOTED_TEXT",
            )
        rebuilt = line
        for match in reversed(cjk_matches):
            rebuilt = rebuilt[: match.start()] + "\"VISIBLE TEXT\"" + rebuilt[match.end() :]
        lines.append(rebuilt)
    return "".join(lines)


def validate_authorized_visible_text_literals(
    text: str,
    authorized_literals: Sequence[str],
) -> tuple[str, ...]:
    """Validate the only CJK allowed outside dialogue: authorized ASCII quotes."""

    _dialogue_blocks(text)
    outside = _outside_dialogue(text)
    matches = tuple(_ASCII_QUOTED_JAPANESE_RE.finditer(outside))
    observed = tuple(match.group("text") for match in matches)
    expected = tuple(str(item) for item in authorized_literals)
    if observed != expected:
        raise PromptTranslationError(
            "画面内文字リテラルの内容・順序・個数がコンパイル中に変わりました。",
            code="VISIBLE_TEXT_LITERAL_MISMATCH",
        )
    scrubbed = outside
    for match in reversed(matches):
        scrubbed = scrubbed[: match.start()] + "\"VISIBLE TEXT\"" + scrubbed[match.end() :]
    if _JAPANESE_RE.search(scrubbed):
        raise PromptTranslationError(
            "画面内文字リテラル以外の日本語が最終H3プロンプトに残っています。",
            code="UNTRANSLATED_JAPANESE",
        )
    return observed


def contains_japanese_outside_dialogue(
    text: str,
    visible_text_literals: Sequence[str] = (),
) -> bool:
    _dialogue_blocks(text)
    try:
        validate_authorized_visible_text_literals(text, visible_text_literals)
    except PromptTranslationError:
        return True
    return False


def requires_translation(prompt: str) -> bool:
    """Return whether Japanese control prose exists outside native dialogue."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise PromptTranslationError("生成プロンプトが空です。", code="EMPTY_PROMPT")
    classify_visible_text_literals(prompt)
    return bool(_JAPANESE_RE.search(_control_without_visible_literals(prompt)))


def parse_generation_mode(value: str | GenerationMode) -> GenerationMode:
    if isinstance(value, GenerationMode):
        return value
    try:
        return GenerationMode(str(value).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in GenerationMode)
        raise PromptTranslationError(
            f"生成モードが不正です。次のいずれかを指定してください: {allowed}",
            code="INVALID_GENERATION_MODE",
        ) from exc


def _normalize_safe_typos(prompt: str) -> str:
    pieces: list[str] = []
    cursor = 0
    for match in _DIALOGUE_RE.finditer(prompt):
        pieces.append(
            _COORDINATED_SUBJECT_TYPO_RE.sub(
                lambda item: f"{item.group('left')}と{item.group('right')}が",
                prompt[cursor : match.start()],
            )
        )
        pieces.append(match.group(0))
        cursor = match.end()
    pieces.append(
        _COORDINATED_SUBJECT_TYPO_RE.sub(
            lambda item: f"{item.group('left')}と{item.group('right')}が",
            prompt[cursor:],
        )
    )
    return "".join(pieces)


def _reference_tags(text: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in _REFERENCE_RE.finditer(text))


def _protect_references(
    text: str, state: _TranslationState
) -> tuple[str, tuple[_Alias, ...]]:
    aliases: list[_Alias] = []

    def replace(match: re.Match[str]) -> str:
        alias = state.alias_for(match)
        aliases.append(alias)
        return alias.source_phrase

    return _REFERENCE_RE.sub(replace, text), tuple(aliases)


def _protect_visible_text_literals(
    text: str,
    state: _TranslationState,
) -> tuple[str, tuple[_VisibleTextAlias, ...]]:
    matches = tuple(
        match
        for match in _QUOTED_TEXT_RE.finditer(text)
        if _JAPANESE_RE.search(_quoted_payload(match))
    )
    if not matches:
        return text, ()
    if _LYRIC_CUE_RE.search(text):
        raise PromptTranslationError(
            "歌詞は画面内文字として処理できません。歌唱を使う場合はH3 native dialogue "
            "blockとして明示してください。",
            code="UNSUPPORTED_LYRICS",
        )
    if not _VISIBLE_TEXT_CUE_RE.search(text):
        raise PromptTranslationError(
            "引用された日本語が台詞・画面内文字・歌詞のどれか判定できません。",
            code="UNCLASSIFIED_CJK_QUOTED_TEXT",
        )
    aliases: list[_VisibleTextAlias] = []
    for match in matches:
        payload = _quoted_payload(match)
        japanese_name, english_name = state.next_name()
        alias = _VisibleTextAlias(payload, japanese_name, english_name)
        aliases.append(alias)
    state.visible_text_literals.extend(alias.original_text for alias in aliases)
    protected = text
    for match, alias in reversed(tuple(zip(matches, aliases, strict=True))):
        protected = (
            protected[: match.start()] + alias.protected_phrase + protected[match.end() :]
        )
    return protected, tuple(aliases)


def _restore_visible_text_literals(
    text: str,
    aliases: Sequence[_VisibleTextAlias],
) -> str:
    restored = text
    for alias in aliases:
        name = re.escape(alias.english_name)
        pattern = re.compile(
            rf"(?<![A-Za-z])(?:the\s+)?(?:(?:on[- ]screen|visible|displayed)\s+"
            rf"(?:text|wording|caption)(?:\s+(?:named|called|of))?\s*[,;:\-]?\s*)?"
            rf"[\"“”]?{name}[\"“”]?(?![A-Za-z])",
            re.IGNORECASE,
        )
        matches = tuple(pattern.finditer(restored))
        if len(matches) != 1:
            raise PromptTranslationError(
                "翻訳後に画面内文字リテラルを一意に復元できませんでした。",
                code="VISIBLE_TEXT_ALIAS_MISMATCH",
            )
        alias_match = matches[0]
        context_start = max(0, alias_match.start() - 160)
        context_end = min(len(restored), alias_match.end() + 160)
        context = restored[context_start:context_end]
        if _HUMAN_VOCAL_OUTPUT_RE.search(context):
            raise PromptTranslationError(
                "画面内文字が翻訳によって人物の発話へ変更されました。",
                code="VISIBLE_TEXT_BECAME_SPEECH",
            )
        if not _VISIBLE_OUTPUT_CUE_RE.search(context):
            raise PromptTranslationError(
                "翻訳後の画面内文字に看板・字幕などの表示文脈がありません。",
                code="VISIBLE_TEXT_CONTEXT_MISMATCH",
            )
        literal = f'"{alias.original_text}"'
        restored = pattern.sub(lambda _: literal, restored, count=1)
    for alias in aliases:
        if re.search(rf"\b{re.escape(alias.english_name)}\b", restored, re.IGNORECASE):
            raise PromptTranslationError(
                "画面内文字用の予約エイリアスが出力に残っています。",
                code="VISIBLE_TEXT_ALIAS_LEAK",
            )
    return restored


def _english_alias_pattern(alias: _Alias) -> re.Pattern[str]:
    noun = _ALIAS_ENGLISH_NOUN[alias.kind]
    name = re.escape(alias.english_name)
    # LFM commonly emits "reference image Alice".  The finite alternatives
    # below accept harmless determiners/link words, but never a bare name.
    return re.compile(
        rf"(?<![A-Za-z])(?:the\s+)?(?:"
        rf"{noun}(?:\s+(?:named|called|of))?\s*[,;:\-]?\s*{name}"
        rf"|{name}(?:'s)?\s+{noun}"
        rf")(?![A-Za-z])",
        re.IGNORECASE,
    )


def _reference_alias_has_visible_text_role(
    text: str,
    match: re.Match[str],
) -> bool:
    """Return whether a protected component alias was turned into screen text."""

    before_character = text[match.start() - 1] if match.start() else ""
    after_character = text[match.end()] if match.end() < len(text) else ""
    if before_character in {'"', "'", "“", "‘"} or after_character in {
        '"',
        "'",
        "”",
        "’",
    }:
        return True

    start = max(0, match.start() - 120)
    end = min(len(text), match.end() + 120)
    window = text[start:match.start()] + " __REFERENCE__ " + text[match.end():end]
    patterns = (
        r"\b(?:words?|text|wording)\s*[\"'“‘]?\s*__REFERENCE__\b",
        r"\b(?:sign|subtitle|caption|label|banner|poster|screen|display)\b"
        r"[^.?!\n]{0,64}\b(?:displays?|reads?|shows?|writes?|renders?)\b"
        r"[^.?!\n]{0,64}__REFERENCE__\b",
        r"__REFERENCE__[^.?!\n]{0,64}\b(?:appears?|shown|displayed|written|rendered)\b"
        r"[^.?!\n]{0,24}\b(?:on[- ]screen|as\s+(?:text|words?|a\s+caption))\b",
        r"\b(?:on[- ]screen|visible)\s+(?:text|words?|caption|label)\b"
        r"[^.?!\n]{0,64}__REFERENCE__\b",
    )
    return any(re.search(pattern, window, re.IGNORECASE) for pattern in patterns)


def _restore_references(text: str, aliases: Sequence[_Alias]) -> str:
    restored = text
    for alias in aliases:
        pattern = _english_alias_pattern(alias)
        matches = tuple(pattern.finditer(restored))
        if len(matches) != 1:
            raise PromptTranslationError(
                f"翻訳後に参照タグ {alias.original_tag} を一意に復元できませんでした。",
                code="REFERENCE_ALIAS_MISMATCH",
            )
        if _reference_alias_has_visible_text_role(restored, matches[0]):
            raise PromptTranslationError(
                f"参照タグ {alias.original_tag} が翻訳によって画面内文字へ変更されました。",
                code="REFERENCE_ALIAS_BECAME_VISIBLE_TEXT",
            )
        restored = pattern.sub(lambda _: alias.original_tag, restored, count=1)
    for alias in aliases:
        if alias.source_phrase in restored or re.search(
            rf"\b{re.escape(alias.english_name)}\b", restored, re.IGNORECASE
        ):
            raise PromptTranslationError(
                "翻訳用の予約エイリアスが出力に残っています。",
                code="REFERENCE_ALIAS_LEAK",
            )
    return restored


def _invoke_translator(translator: Translator | Callable[[str], str], text: str) -> str:
    try:
        value = translator(text)
    except PromptTranslationError:
        raise
    except Exception as exc:
        raise PromptTranslationError(
            f"ローカル翻訳器の実行に失敗しました: {exc}",
            code="TRANSLATOR_RUNTIME_ERROR",
        ) from exc
    if not isinstance(value, str):
        raise PromptTranslationError(
            "ローカル翻訳器が文字列以外を返しました。",
            code="INVALID_TRANSLATOR_OUTPUT",
        )
    value = value.strip()
    if not value:
        raise PromptTranslationError(
            "ローカル翻訳器が空の結果を返しました。",
            code="EMPTY_TRANSLATOR_OUTPUT",
        )
    if "\n" in value or "\r" in value:
        raise PromptTranslationError(
            "ローカル翻訳器が1行の指示を複数行へ変更しました。",
            code="MULTILINE_TRANSLATOR_OUTPUT",
        )
    if any(ord(character) < 32 and character != "\t" for character in value):
        raise PromptTranslationError(
            "ローカル翻訳器の結果に不正な制御文字があります。",
            code="INVALID_TRANSLATOR_OUTPUT",
        )
    maximum = min(8192, max(512, len(text) * 8))
    if len(value) > maximum:
        raise PromptTranslationError(
            "ローカル翻訳器の結果が入力に対して異常に長くなりました。",
            code="ABNORMAL_TRANSLATOR_LENGTH",
        )
    if _ANY_SECTION_LINE_RE.fullmatch(value):
        raise PromptTranslationError(
            "翻訳結果が予約済みのH3セクション見出しと衝突しました。",
            code="SECTION_HEADER_INJECTION",
        )
    return value


def _apply_narrow_semantic_safeguards(source: str, translated: str) -> str:
    """Correct one confirmed LFM hand-count mistranslation, and nothing broader."""

    if "小さく手を振る" not in source:
        return translated
    return re.sub(
        r"\bwaves her hands slightly\b",
        "gives a small wave with one hand",
        translated,
        flags=re.IGNORECASE,
    )


def _reject_translator_invented_speech(source: str, translated: str) -> None:
    """Reject a new human-vocal role outside immutable native dialogue.

    The server isolates every intended spoken line before this compiler runs.
    A translated detail or music line therefore must not introduce a narrator,
    speaker, voice-over, or ordinary dialogue.  Keep the source argument in the
    error contract so callers can report the exact rejected line without ever
    treating model output as authoritative.
    """

    if not (
        _HUMAN_VOCAL_OUTPUT_RE.search(translated)
        or _TRANSLATOR_SPEECH_ROLE_RE.search(translated)
    ):
        return
    raise PromptTranslationError(
        f"ローカル翻訳器が元の制御文にない人物発話を追加しました: {source.strip()}",
        code="TRANSLATOR_INVENTED_SPEECH",
    )


def _translate_fragment(
    fragment: str,
    *,
    translator: Translator | Callable[[str], str] | None,
    state: _TranslationState,
) -> tuple[str, bool]:
    if not _JAPANESE_RE.search(fragment):
        return fragment.strip(), False
    literal_protected, literal_aliases = _protect_visible_text_literals(fragment, state)
    translated_by_model = bool(_JAPANESE_RE.search(literal_protected))
    if translated_by_model:
        if translator is None:
            raise PromptTranslationError(
                "日本語の映像指示を英語化するローカル翻訳モデルが見つかりません。",
                code="TRANSLATOR_REQUIRED",
            )
        protected, aliases = _protect_references(literal_protected, state)
        translated = _invoke_translator(translator, protected)
        restored = _restore_references(translated, aliases)
    else:
        restored = literal_protected.strip()
    restored = _restore_visible_text_literals(restored, literal_aliases)
    restored = _apply_narrow_semantic_safeguards(fragment, restored)
    validate_authorized_visible_text_literals(
        restored, tuple(alias.original_text for alias in literal_aliases)
    )
    if _reference_tags(fragment) != _reference_tags(restored):
        raise PromptTranslationError(
            "翻訳によって参照タグの種類・順序・個数が変わりました。",
            code="REFERENCE_TAG_MISMATCH",
        )
    return restored, translated_by_model


def _is_dialogue_line(line: str, in_dialogue: bool) -> tuple[bool, bool]:
    opens = len(_DIALOGUE_OPEN_RE.findall(line))
    closes = len(_DIALOGUE_CLOSE_RE.findall(line))
    immutable = in_dialogue or opens > 0 or closes > 0
    depth = (1 if in_dialogue else 0) + opens - closes
    if depth not in (0, 1):
        raise PromptTranslationError(
            "<d>台詞タグが不正に入れ子になっています。",
            code="MALFORMED_DIALOGUE_TAG",
        )
    return immutable, depth == 1


def _ambient_only_sound_line(value: str) -> str:
    """Remove a translator-invented vocal role when no native dialogue exists.

    We deliberately reconstruct a small physical-audio contract instead of
    trying to negate the invented role: negative phrases such as "no speech"
    still prime the audio model with the unwanted concept.  Reference tags are
    retained in their original order so component routing remains auditable.
    """

    if not _UNREQUESTED_SPEECH_AUDIO_RE.search(value):
        return value
    references = _reference_tags(value)
    if references:
        labels = " and ".join(references)
        return (
            f"Use {labels} solely for environmental ambience and synchronized "
            "physical effects appropriate to visible action."
        )
    return (
        "Quiet environmental ambience and synchronized physical effects "
        "appropriate to visible action."
    )


def _translate_body(
    prompt: str,
    *,
    translator: Translator | Callable[[str], str] | None,
) -> tuple[
    str,
    tuple[str, ...],
    tuple[str, ...],
    int,
    tuple[str, ...],
    tuple[str, ...],
]:
    state = _TranslationState(prompt)
    detail_lines: list[str] = []
    sound_lines: list[str] = []
    music_lines: list[str] = []
    translated_count = 0
    translated_reference_trace: list[str] = []
    in_dialogue = False

    for line in prompt.splitlines():
        immutable, in_dialogue = _is_dialogue_line(line, in_dialogue)
        if immutable:
            if _JAPANESE_RE.search(_outside_dialogue(line)):
                raise PromptTranslationError(
                    "台詞を含むcanonical行の台詞タグ外に日本語が残っています。",
                    code="JAPANESE_IN_IMMUTABLE_DIALOGUE_LINE",
                )
            detail_lines.append(line)
            translated_reference_trace.extend(_reference_tags(line))
            continue

        audio_match = _AUDIO_LINE_RE.fullmatch(line)
        music_match = _MUSIC_LINE_RE.fullmatch(line)
        if audio_match:
            value, changed = _translate_fragment(
                audio_match.group("body"), translator=translator, state=state
            )
            # Top-level Audio: lines describe the global ambience/effects bed.
            # Native dialogue and its optional voice reference stay in the
            # canonical detail block, so a vocal role here is always invented.
            value = _ambient_only_sound_line(value)
            if value:
                sound_lines.append(value)
            translated_reference_trace.extend(_reference_tags(value))
            translated_count += int(changed)
            continue
        if music_match:
            value, changed = _translate_fragment(
                music_match.group("body"), translator=translator, state=state
            )
            if changed:
                _reject_translator_invented_speech(
                    music_match.group("body"), value
                )
            if value:
                music_lines.append(value)
            translated_reference_trace.extend(_reference_tags(value))
            translated_count += int(changed)
            continue

        if not line.strip() or _CUT_RE.fullmatch(line) or _SEPARATOR_RE.fullmatch(line):
            detail_lines.append(line)
            translated_reference_trace.extend(_reference_tags(line))
            continue
        value, changed = _translate_fragment(line, translator=translator, state=state)
        if changed:
            _reject_translator_invented_speech(line, value)
        detail_lines.append(value)
        translated_reference_trace.extend(_reference_tags(value))
        translated_count += int(changed)

    if in_dialogue:
        raise PromptTranslationError(
            "<d>台詞タグが閉じられていません。", code="MALFORMED_DIALOGUE_TAG"
        )
    detail = "\n".join(detail_lines).strip()
    if not detail:
        raise PromptTranslationError(
            "翻訳後の映像指示が空です。", code="EMPTY_DETAILED_DESCRIPTION"
        )
    return (
        detail,
        tuple(sound_lines),
        tuple(music_lines),
        translated_count,
        tuple(translated_reference_trace),
        tuple(state.visible_text_literals),
    )


def _event_value(event: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(key, default)
    return getattr(event, key, default)


def _reference_ordinals(text: str, kind: str) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(
            int(match.group("number"))
            for match in _REFERENCE_RE.finditer(text)
            if match.group("kind").lower() == kind.lower()
        )
    )


def _inventory_tags(
    reference_inventory: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return canonical labels from the sanitized Omni upload manifest."""

    kind_names = {"image": "Picture", "video": "Video", "audio": "Audio"}
    tags: list[str] = []
    for item in reference_inventory:
        if not isinstance(item, Mapping):
            raise PromptTranslationError(
                "references inventoryの各要素はオブジェクトで指定してください。",
                code="INVALID_REFERENCE_INVENTORY",
            )
        raw_kind = str(item.get("kind") or "").strip().lower()
        kind = kind_names.get(raw_kind)
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            index = 0
        if kind is None or index < 1:
            raise PromptTranslationError(
                "references inventoryのkind/indexが不正です。",
                code="INVALID_REFERENCE_INVENTORY",
            )
        expected = f"<{kind} {index}>"
        supplied = item.get("tag")
        if supplied is not None and str(supplied) != expected:
            raise PromptTranslationError(
                f"references inventoryのtagがkind/indexと一致しません: {supplied}",
                code="INVALID_REFERENCE_INVENTORY",
            )
        tags.append(expected)
    if len(tags) != len(set(tags)):
        raise PromptTranslationError(
            "references inventoryに重複した参照タグがあります。",
            code="INVALID_REFERENCE_INVENTORY",
        )
    return tuple(tags)


def _merged_reference_ordinals(
    source: str,
    kind: str,
    inventory_tags: Sequence[str],
) -> tuple[int, ...]:
    values = list(_reference_ordinals(source, kind))
    for tag in inventory_tags:
        match = _REFERENCE_RE.fullmatch(tag)
        if match and match.group("kind").lower() == kind.lower():
            number = int(match.group("number"))
            if number not in values:
                values.append(number)
    return tuple(values)


def _subject_bindings(
    source: str,
    detail: str,
    dialogue_events: Sequence[Mapping[str, Any] | Any],
) -> tuple[dict[int, int], dict[int, int]]:
    subject_to_picture: dict[int, int] = {}
    audio_to_subject: dict[int, int] = {}

    canonical = re.compile(
        r"<Subject\s+(?P<subject>[1-9][0-9]*)>\s*\(S[1-9][0-9]*\)\s+"
        r"is\s+the\s+visible\s+character\s+shown\s+in\s+"
        r"<Picture\s+(?P<picture>[1-9][0-9]*)>",
        re.IGNORECASE,
    )
    for match in canonical.finditer(detail):
        subject_to_picture[int(match.group("subject"))] = int(match.group("picture"))

    for event in dialogue_events:
        try:
            subject = int(_event_value(event, "speaker_id", 1))
        except (TypeError, ValueError):
            subject = 1
        label = str(_event_value(event, "speaker_label", "") or "")
        picture_match = re.search(r"<Picture\s+([1-9][0-9]*)>", label, re.IGNORECASE)
        if picture_match:
            subject_to_picture[subject] = int(picture_match.group(1))
        # The server records both the user-requested Audio binding and the
        # binding that is actually allowed to reach H3.  An explicit
        # ``audio_reference_id_effective: null`` is meaningful: the safe
        # dialogue-priority policy removed full-waveform conditioning.  Do not
        # fall back to the requested id in that case or the optional official
        # English compiler would silently resurrect a suppressed <Audio N>.
        if isinstance(event, Mapping) and "audio_reference_id_effective" in event:
            audio = event.get("audio_reference_id_effective")
        elif hasattr(event, "audio_reference_id_effective"):
            audio = getattr(event, "audio_reference_id_effective")
        else:
            audio = _event_value(event, "audio_reference_id")
        if audio is not None:
            try:
                audio_to_subject[int(audio)] = subject
            except (TypeError, ValueError):
                pass

    audio_contract = re.compile(
        r"<Audio\s+(?P<audio>[1-9][0-9]*)>\s+(?:"
        r"is\s+the\s+voice-timbre\s+reference\s+for|"
        r"provides\s+the\s+voice\s+timbre\s+and\s+measured\s+delivery\s+for)\s+"
        r"<Subject\s+(?P<subject>[1-9][0-9]*)>",
        re.IGNORECASE,
    )
    for match in audio_contract.finditer(detail):
        audio_to_subject[int(match.group("audio"))] = int(match.group("subject"))

    pictures = _reference_ordinals(source, "Picture")
    # Explicit Japanese actor grammar, including "<Picture 1>と<Picture 2>が".
    for line in source.splitlines():
        line_pictures = _reference_ordinals(line, "Picture")
        if not line_pictures:
            continue
        without_tags = _REFERENCE_RE.sub("", line)
        if _CHARACTER_TERMS_RE.search(line) and _JAPANESE_ACTOR_PARTICLE_RE.search(without_tags):
            for picture in line_pictures:
                subject_to_picture.setdefault(picture, picture)
        elif len(line_pictures) > 1 and re.search(r"と.*(?:が|は)", line):
            for picture in line_pictures:
                subject_to_picture.setdefault(picture, picture)
        else:
            for picture in line_pictures:
                tag = rf"<Picture\s+{picture}>"
                actor_use = re.search(
                    rf"{tag}(?!\s*の\s*(?:服|衣装|背景|色|画風|スタイル|小物))"
                    rf"[^。！？\n]{{0,24}}(?:が|は|も)[^。！？\n]{{0,32}}"
                    rf"(?:{_ACTOR_ACTION_RE.pattern})",
                    line,
                    re.IGNORECASE,
                )
                if actor_use:
                    subject_to_picture.setdefault(picture, picture)

    # English formatter/translation phrasing that explicitly identifies a person.
    for picture in pictures:
        tag = rf"<Picture\s+{picture}>"
        if re.search(
            rf"(?:{_CHARACTER_TERMS_RE.pattern})[^\n]{{0,80}}{tag}|"
            rf"{tag}[^\n]{{0,80}}(?:{_CHARACTER_TERMS_RE.pattern})|"
            rf"{tag}[^\n]{{0,40}}(?:{_ACTOR_ACTION_RE.pattern})",
            detail,
            re.IGNORECASE,
        ):
            subject_to_picture.setdefault(picture, picture)

    return subject_to_picture, audio_to_subject


def _render_subject_definitions(
    source: str,
    detail: str,
    dialogue_events: Sequence[Mapping[str, Any] | Any],
) -> tuple[str, str, tuple[str, ...]]:
    return _render_subject_definitions_with_inventory(source, detail, dialogue_events, ())


def _line_has_wardrobe_override(line: str) -> bool:
    """Classify an explicit wardrobe replacement, not a mere outfit mention."""

    if not _WARDROBE_TERM_RE.search(line):
        return False
    # Explicit rejection of the reference wardrobe is the strongest signal,
    # even when the same clause also contains words such as "reference".
    if _WARDROBE_REJECTION_RE.search(line):
        return True
    if _WARDROBE_CHANGE_ACTION_RE.search(line):
        return True
    # Preservation language wins over generic wear descriptions.  In
    # particular, "wearing the same/reference outfit" is not an override.
    if _WARDROBE_KEEP_RE.search(line):
        return False
    if _WARDROBE_CHANGE_RE.search(line):
        return True
    # Japanese garment-object constructions are an explicit target wardrobe;
    # a generic 服装/衣装 mention alone is intentionally insufficient.
    return bool(_JAPANESE_EXPLICIT_GARMENT_WEAR_RE.search(line))


def _has_wardrobe_override_for_mapping(
    source: str,
    detail: str,
    *,
    picture: int,
    subject: int,
    only_mapping: bool,
) -> bool:
    """Inspect both source and translated detail for a mapped subject."""

    picture_re = re.compile(rf"<Picture\s+{picture}>", re.IGNORECASE)
    subject_re = re.compile(rf"<Subject\s+{subject}>", re.IGNORECASE)
    for text in (source, detail):
        for line in text.splitlines():
            if not only_mapping and not (
                picture_re.search(line) or subject_re.search(line)
            ):
                continue
            if _line_has_wardrobe_override(line):
                return True
    return False


def _render_subject_definitions_with_inventory(
    source: str,
    detail: str,
    dialogue_events: Sequence[Mapping[str, Any] | Any],
    reference_inventory: Sequence[Mapping[str, Any]],
) -> tuple[str, str, tuple[str, ...]]:
    inventory_tags = _inventory_tags(reference_inventory)
    pictures = _merged_reference_ordinals(source, "Picture", inventory_tags)
    videos = _merged_reference_ordinals(source, "Video", inventory_tags)
    audios = _merged_reference_ordinals(source, "Audio", inventory_tags)
    subjects = set(_reference_ordinals(source, "Subject"))
    subject_to_picture, audio_to_subject = _subject_bindings(
        source, detail, dialogue_events
    )
    subjects.update(subject_to_picture)
    subjects.update(audio_to_subject.values())
    for event in dialogue_events:
        value = _event_value(event, "speaker_id")
        try:
            speaker_id = int(value)
        except (TypeError, ValueError):
            continue
        if speaker_id > 0:
            subjects.add(speaker_id)

    definitions: list[str] = []
    retention_by_label: dict[str, str] = {}
    mapped_pictures = set(subject_to_picture.values())

    for subject, picture in sorted(subject_to_picture.items()):
        wardrobe_override = _has_wardrobe_override_for_mapping(
            source,
            detail,
            picture=picture,
            subject=subject,
            only_mapping=len(subject_to_picture) == 1,
        )
        is_speaker = any(
            int(_event_value(event, "speaker_id", -1)) == subject
            for event in dialogue_events
            if str(_event_value(event, "speaker_id", "")).isdigit()
        ) or bool(
            re.search(rf"<Subject\s+{subject}>\s*\(S{subject}\)", detail, re.IGNORECASE)
        )
        speaker_marker = f" (S{subject})" if is_speaker else ""
        if wardrobe_override:
            definitions.append(
                f"<Subject {subject}>{speaker_marker} is the visible character shown in "
                f"<Picture {picture}>. Use <Picture {picture}> for identity, facial appearance, "
                f"hairstyle, body proportions, and non-conflicting accessories. The target "
                "wardrobe explicitly requested in detailed_description takes precedence and "
                "fully replaces conflicting source clothing. Treat the source as a visual "
                "reference only; do not reproduce its concept-sheet layout, panels, captions, "
                "labels, borders, typography, or multiple-view presentation."
            )
            retention_by_label[f"<Picture {picture}>"] = (
                f"<Picture {picture}>: partially_preserved - preserve identity, face, hairstyle, "
                "body proportions, and non-conflicting accessories, while replacing source "
                "clothing with the explicitly requested target wardrobe."
            )
            retention_by_label[f"<Subject {subject}>"] = (
                f"<Subject {subject}>: partially_preserved - preserve identity and body traits "
                "while applying the explicitly requested target wardrobe consistently."
            )
        else:
            definitions.append(
                f"<Subject {subject}>{speaker_marker} is the visible character shown in "
                f"<Picture {picture}>. Use <Picture {picture}> as the identity, facial "
                f"appearance, hairstyle, body-proportion, outfit, color, and accessory "
                f"reference for <Subject {subject}>. Treat the source as a visual reference "
                "only; do not reproduce its concept-sheet layout, panels, captions, labels, "
                "borders, typography, or multiple-view presentation."
            )
            retention_by_label[f"<Picture {picture}>"] = (
                f"<Picture {picture}>: fully_preserved - preserve the identity, face, hairstyle, body "
                f"proportions, clothing, colors, and accessories established by <Picture {picture}> "
                "across every shot."
            )
            retention_by_label[f"<Subject {subject}>"] = (
                f"<Subject {subject}>: fully_preserved - preserve the identity, face, hairstyle, "
                f"body proportions, clothing, colors, and accessories established by <Picture {picture}> "
                "across every shot."
            )

    for picture in pictures:
        if picture in mapped_pictures:
            continue
        definitions.append(
            f"<Picture {picture}> is a visual reference. Use only the appearance, object, "
            "wardrobe, environment, or composition explicitly requested by the shot plan; "
            "do not reproduce source captions, labels, borders, or layout."
        )
        retention_by_label[f"<Picture {picture}>"] = (
            f"<Picture {picture}>: weak_reference - retain only the visual traits explicitly used by the "
            "shot plan, consistently across relevant shots."
        )

    for subject in sorted(subjects - set(subject_to_picture)):
        is_speaker = any(
            int(_event_value(event, "speaker_id", -1)) == subject
            for event in dialogue_events
            if str(_event_value(event, "speaker_id", "")).isdigit()
        ) or bool(
            re.search(rf"<Subject\s+{subject}>\s*\(S{subject}\)", detail, re.IGNORECASE)
        )
        speaker_marker = f" (S{subject})" if is_speaker else ""
        definitions.append(
            f"<Subject {subject}>{speaker_marker} is the consistently identified visible "
            "subject specified in detailed_description."
        )
        retention_by_label[f"<Subject {subject}>"] = (
            f"<Subject {subject}>: fully_preserved - preserve identity and appearance consistently across "
            "every shot."
        )

    for video in videos:
        definitions.append(
            f"<Video {video}> is a visual motion reference. Use only the explicitly "
            "requested visual motion or temporal structure; do not copy its soundtrack "
            "unless detailed_description explicitly requests it."
        )
        retention_by_label[f"<Video {video}>"] = (
            f"<Video {video}>: weak_reference - retain only the requested visual motion relationship."
        )

    for audio in audios:
        subject = audio_to_subject.get(audio)
        explicitly_assigned = subject is not None or audio in _reference_ordinals(
            source, "Audio"
        )
        if subject is not None:
            definitions.append(
                f"<Audio {audio}> provides the voice timbre and measured delivery for "
                f"<Subject {subject}> (S{subject})."
            )
        elif explicitly_assigned:
            definitions.append(
                f"<Audio {audio}> provides the audio characteristic explicitly assigned by "
                "the request."
            )
        else:
            definitions.append(
                f"<Audio {audio}> is inactive because the request assigns no role."
            )
        if subject is not None:
            retention_by_label[f"<Audio {audio}>"] = (
                f"<Audio {audio}>: reference - preserve its voice timbre and measured "
                f"delivery for <Subject {subject}> (S{subject})."
            )
        elif explicitly_assigned:
            retention_by_label[f"<Audio {audio}>"] = (
                f"<Audio {audio}>: reference - preserve its explicitly assigned audio "
                "characteristic."
            )
        else:
            retention_by_label[f"<Audio {audio}>"] = (
                f"<Audio {audio}>: inactive - the request assigns no role."
            )

    if not definitions:
        definitions.append(
            "The visible subjects and scene are defined only by detailed_description."
        )
    if not retention_by_label:
        retention_by_label["global"] = (
            "Preserve all explicitly requested subject, wardrobe, scene, and camera "
            "continuity across the ordered shot plan."
        )
    # Official retention_analysis requires one relationship marker per supplied
    # reference.  Keep manifest order, followed by prompt-only reference labels.
    ordered_labels = [f"<Subject {number}>" for number in sorted(subjects)]
    ordered_labels.extend(label for label in inventory_tags if label not in ordered_labels)
    for kind in ("Picture", "Video", "Audio"):
        for number in _reference_ordinals(source, kind):
            label = f"<{kind} {number}>"
            if label not in ordered_labels:
                ordered_labels.append(label)
    retention = [retention_by_label[label] for label in ordered_labels if label in retention_by_label]
    retention.extend(
        value
        for label, value in retention_by_label.items()
        if label not in ordered_labels
    )
    all_reference_labels = tuple(f"<Subject {number}>" for number in sorted(subjects)) + tuple(
        f"<{kind} {number}>"
        for kind, numbers in (
            ("Picture", pictures), ("Video", videos), ("Audio", audios)
        )
        for number in numbers
    )
    return "\n".join(definitions), "\n".join(retention), all_reference_labels


def _music_text(explicit: Sequence[str], music_policy: str | None) -> str:
    if explicit:
        return " ".join(item.strip() for item in explicit if item.strip()) or "N/A"
    policy = (music_policy or "auto").strip().lower()
    policies = {
        "none": "N/A",
        "n/a": "N/A",
        "auto": "N/A",
        "subtle": "A subtle instrumental score supports the physical ambience and important sound effects.",
        "prominent": "A prominent instrumental score follows the rhythm and emotional arc of the edit.",
    }
    return policies.get(policy, music_policy.strip() if music_policy else "N/A")


def _render_six_sections(
    source: str,
    detail: str,
    sound_lines: Sequence[str],
    music_lines: Sequence[str],
    dialogue_events: Sequence[Mapping[str, Any] | Any],
    music_policy: str | None,
    reference_inventory: Sequence[Mapping[str, Any]],
) -> str:
    definitions, retention, reference_labels = _render_subject_definitions_with_inventory(
        source, detail, dialogue_events, reference_inventory
    )
    has_references = bool(reference_labels)
    _, audio_to_subject = _subject_bindings(source, detail, dialogue_events)
    assigned_audio_numbers = set(_reference_ordinals(source, "Audio")) | set(
        audio_to_subject
    )
    has_audio_reference = bool(assigned_audio_numbers)
    summary_prefix = (
        "reference generation + audio reference"
        if has_audio_reference
        else "reference generation" if has_references else "text-to-video"
    )
    labels_clause = (
        "Use the defined roles of " + ", ".join(reference_labels) + ". "
        if reference_labels
        else ""
    )
    summary = (
        f"[{summary_prefix}] {labels_clause}Generate one coherent video that follows the ordered shot "
        "plan in detailed_description. Preserve requested subjects, wardrobe, staging, "
        "actions, camera behavior, and timing."
    )
    detailed = (
        "Execute the following shot plan exactly. Treat Cut and Shot markers as "
        "chronological instructions. Reference labels identify supplied inputs and are "
        "not visible text in the target video.\n"
        + detail
    )
    sound_parts = [item.strip() for item in sound_lines if item.strip()]
    sound_parts.append(
        "Follow the diegetic environmental ambience and synchronized physical sound "
        "effects described in detailed_description."
    )
    soundscape = " ".join(sound_parts)
    music = _music_text(music_lines, music_policy)
    return (
        f"subject_definitions:\n{definitions}\n\n"
        f"summary:\n{summary}\n\n"
        f"retention_analysis:\n{retention}\n\n"
        f"detailed_description:\n{detailed}\n\n"
        f"overall_soundscape:\n{soundscape}\n\n"
        f"non_diegetic_music:\n{music}"
    )


def _render_three_sections(
    detail: str,
    sound_lines: Sequence[str],
    music_lines: Sequence[str],
    music_policy: str | None,
    mode: GenerationMode,
    duration_seconds: float | None,
    last_shot_index: int,
) -> str:
    if mode is GenerationMode.OMNI:
        raise PromptTranslationError(
            "Omniモードを3セクション形式では描画できません。",
            code="INVALID_GENERATION_MODE",
        )
    opening = {
        GenerationMode.T2V: (
            "Generate the target video directly from the following chronological "
            "multimodal description."
        ),
        GenerationMode.I2V: (
            "Preserve the identity-defining content of the supplied opening image while "
            "following the chronological instructions below."
        ),
        GenerationMode.FIRST_LAST: (
            "Preserve the identity-defining content of both supplied endpoint images and "
            "create one coherent transition between them while following the "
            "chronological instructions below."
        ),
    }[mode]
    description = (
        opening
        + "\nExecute the following shot plan exactly. Treat Cut and Shot markers as "
        "chronological instructions.\n"
        + detail
    )
    sound_parts = [item.strip() for item in sound_lines if item.strip()]
    sound_parts.append(
        "Follow the diegetic environmental ambience and synchronized physical sound effects in "
        "integrated_multimodal_description."
    )
    soundscape = " ".join(sound_parts)
    music = _music_text(music_lines, music_policy)
    core = (
        f"integrated_multimodal_description:\n{description}\n\n"
        f"overall_soundscape:\n{soundscape}\n\n"
        f"non_diegetic_music:\n{music}"
    )
    if mode is GenerationMode.I2V:
        alignment = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
        return f"{alignment}\n\n{core}"
    if mode is GenerationMode.FIRST_LAST:
        if (
            duration_seconds is None
            or isinstance(duration_seconds, bool)
            or not math.isfinite(float(duration_seconds))
            or float(duration_seconds) <= 0
        ):
            raise PromptTranslationError(
                "first_lastの公式alignmentには正のduration_secondsが必要です。",
                code="DURATION_REQUIRED",
            )
        if isinstance(last_shot_index, bool) or last_shot_index < 1:
            raise PromptTranslationError(
                "first_lastのlast_shot_indexが不正です。",
                code="INVALID_LAST_SHOT_INDEX",
            )
        alignment = (
            "How the reference pictures align with the target video — Picture 1 "
            "(from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {last_shot_index}) aligns with the "
            f"{float(duration_seconds):.2f}-second mark of the target video."
        )
        return f"{alignment}\n\n{core}"
    return core


def _last_shot_number(text: str) -> int:
    values = [
        int(match.group(1) or match.group(2))
        for match in re.finditer(
            r"(?im)^\s*(?:\[(?:Cut|Shot)\s*([1-9][0-9]*)\]|"
            r"(?:Cut|Shot)\s*#?\s*([1-9][0-9]*))",
            text,
        )
    ]
    return max(values, default=1)


def _validate_official_sections(text: str, mode: GenerationMode) -> None:
    matches = tuple(_SECTION_RE.finditer(text))
    names = tuple(match.group(1).lower() for match in matches)
    expected = (
        OFFICIAL_SECTION_HEADERS
        if mode is GenerationMode.OMNI
        else OFFICIAL_BASE_SECTION_HEADERS
    )
    if names != expected:
        section_count = 6 if mode is GenerationMode.OMNI else 3
        raise PromptTranslationError(
            f"{mode.value}用H3公式{section_count}セクションの見出し・順序・個数が不正です。",
            code=(
                "INVALID_SIX_SECTION_SCHEMA"
                if mode is GenerationMode.OMNI
                else "INVALID_THREE_SECTION_SCHEMA"
            ),
        )
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if not text[match.end() : end].strip():
            raise PromptTranslationError(
                f"H3セクション {names[index]} が空です。",
                code="EMPTY_SIX_SECTION",
            )


def _validate_six_sections(text: str) -> None:
    _validate_official_sections(text, GenerationMode.OMNI)


def _looks_like_official_sections(text: str) -> bool:
    return bool(_SECTION_RE.search(text))


def _validate_final(
    source: str,
    translated_reference_trace: tuple[str, ...],
    compiled: str,
    source_dialogue: tuple[str, ...],
    mode: GenerationMode,
    visible_text_literals: tuple[str, ...],
) -> None:
    _validate_official_sections(compiled, mode)
    output_dialogue = _dialogue_blocks(compiled)
    if source_dialogue != output_dialogue:
        raise PromptTranslationError(
            "台詞ブロックがコンパイル中に変更・追加・削除されました。",
            code="DIALOGUE_MUTATED",
        )
    source_refs = _reference_tags(source)
    if source_refs != translated_reference_trace:
        raise PromptTranslationError(
            "翻訳によって参照タグの種類・順序・個数が変わりました。",
            code="REFERENCE_TAG_MISMATCH",
        )
    final_ref_counts = Counter(_reference_tags(compiled))
    for tag, expected_count in Counter(source_refs).items():
        if final_ref_counts[tag] < expected_count:
            raise PromptTranslationError(
                f"最終H3プロンプトから参照タグ {tag} が欠落しました。",
                code="REFERENCE_TAG_MISMATCH",
            )
    validate_authorized_visible_text_literals(compiled, visible_text_literals)
    if not compiled.strip() or len(compiled) > max(32768, len(source) * 20):
        raise PromptTranslationError(
            "最終H3プロンプトが空、または入力に対して異常に長い状態です。",
            code="ABNORMAL_COMPILED_LENGTH",
        )


def translate_and_compile_prompt(
    prompt: str,
    translator: Translator | Callable[[str], str] | None = None,
    *,
    dialogue_events: Sequence[Mapping[str, Any] | Any] = (),
    music_policy: str | None = None,
    reference_inventory: Sequence[Mapping[str, Any]] = (),
    mode: str | GenerationMode = GenerationMode.OMNI,
    duration_seconds: float | None = None,
    last_shot_index: int | None = None,
) -> PromptTranslationResult:
    """Translate control prose and build H3's official mode-specific sections.

    ``translator`` is deliberately a tiny callable interface so unit tests and
    alternative local runtimes need no model dependency.  It is never called
    for pure-English input, native dialogue, Cut markers, separators, or ASCII
    control lines.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise PromptTranslationError("生成プロンプトが空です。", code="EMPTY_PROMPT")
    generation_mode = parse_generation_mode(mode)
    source_dialogue = _dialogue_blocks(prompt)
    source_visible_literals = classify_visible_text_literals(prompt)

    if _looks_like_official_sections(prompt):
        _validate_official_sections(prompt, generation_mode)
        if contains_japanese_outside_dialogue(prompt, source_visible_literals):
            raise PromptTranslationError(
                "既存の公式セクション内に未変換の日本語制御文があります。",
                code="UNTRANSLATED_OFFICIAL_PROMPT",
            )
        if _dialogue_blocks(prompt) != source_dialogue:
            raise PromptTranslationError(
                "既存の公式プロンプトの台詞タグを検証できません。",
                code="DIALOGUE_MUTATED",
            )
        return PromptTranslationResult(
            compiled_prompt=prompt,
            translated_detail=prompt,
            translated_line_count=0,
            dialogue_blocks=source_dialogue,
            source_reference_tags=_reference_tags(prompt),
            translated_reference_tags=_reference_tags(prompt),
            visible_text_literals=source_visible_literals,
            mode=generation_mode,
            already_compiled=True,
        )

    normalized = _normalize_safe_typos(prompt)
    (
        detail,
        sound,
        music,
        translated_count,
        translated_reference_trace,
        visible_text_literals,
    ) = _translate_body(normalized, translator=translator)
    if source_visible_literals != visible_text_literals:
        raise PromptTranslationError(
            "画面内文字リテラルの内容・順序・個数がコンパイル中に変わりました。",
            code="VISIBLE_TEXT_LITERAL_MISMATCH",
        )
    if generation_mode is GenerationMode.OMNI:
        compiled = _render_six_sections(
            normalized,
            detail,
            sound,
            music,
            dialogue_events,
            music_policy,
            reference_inventory,
        )
    else:
        if reference_inventory:
            raise PromptTranslationError(
                "references inventoryはOmniモードでのみ使用できます。",
                code="REFERENCE_INVENTORY_MODE_MISMATCH",
            )
        compiled = _render_three_sections(
            detail,
            sound,
            music,
            music_policy,
            generation_mode,
            duration_seconds,
            last_shot_index or _last_shot_number(detail),
        )
    _validate_final(
        normalized,
        translated_reference_trace,
        compiled,
        source_dialogue,
        generation_mode,
        visible_text_literals,
    )
    return PromptTranslationResult(
        compiled_prompt=compiled,
        translated_detail=detail,
        translated_line_count=translated_count,
        dialogue_blocks=source_dialogue,
        source_reference_tags=_reference_tags(normalized),
        translated_reference_tags=translated_reference_trace,
        visible_text_literals=visible_text_literals,
        mode=generation_mode,
    )


# A concise alias for callers that treat this module as a compiler backend.
compile_prompt = translate_and_compile_prompt


__all__ = [
    "GenerationMode",
    "OFFICIAL_BASE_SECTION_HEADERS",
    "OFFICIAL_SECTION_HEADERS",
    "PromptTranslationError",
    "PromptTranslationResult",
    "Translator",
    "classify_visible_text_literals",
    "compile_prompt",
    "contains_japanese_outside_dialogue",
    "parse_generation_mode",
    "requires_translation",
    "translate_and_compile_prompt",
    "validate_authorized_visible_text_literals",
    "validate_native_dialogue_blocks",
]
