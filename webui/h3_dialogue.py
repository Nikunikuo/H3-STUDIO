"""Deterministic, inline MiniMax H3 dialogue formatting.

The user's Cut text remains the source of truth.  This module performs no model
inference and does not build Context-IR.  It isolates only explicit quoted
speech, renders the surrounding speech controls in H3's documented English
form, and keeps unrelated visual actions and physical sounds in their Cut.  The
legacy dialogue form field is supported as an optional override.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace


_CUT_HEADER_RE = re.compile(
    r"^[ \t]*(?:"
    r"\[(?:Cut|Shot)\s*(?P<bracket>[1-9][0-9]*)\]"
    r"|(?:Cut|Shot|カット|ショット)\s*#?\s*(?P<plain>[1-9][0-9]*)"
    r")\s*(?:(?:[:：\-—])[^\r\n]*)?$",
    re.IGNORECASE,
)
_TARGET_RE = re.compile(
    r"(?i)(?:Cut|Shot|カット|ショット)\s*#?\s*(?P<number>[1-9][0-9]*)"
)
_INLINE_CUT_LAYOUT_RE = re.compile(
    r"(?im)(?P<label>\[(?:Cut|Shot)\s*[1-9][0-9]*\]|"
    r"(?:Cut|Shot|カット|ショット)\s*#?\s*[1-9][0-9]*)"
    r"(?:[ \t]*(?::|：|\-|—)[ \t]*|[ \t]+)(?=\S)"
)
_QUOTE_RE = re.compile(
    r"「(?P<corner>[^」\r\n]+)」"
    r"|『(?P<double_corner>[^』\r\n]+)』"
    r"|“(?P<curly>[^”\r\n]+)”"
    r'|"(?P<ascii>[^"\r\n]+)"'
)
_NATIVE_DIALOGUE_RE = re.compile(
    r"<d>\s*\[(?P<language>[A-Za-z][A-Za-z -]{1,31})\]\s*"
    r"(?P<text>.*?)</d>",
    re.IGNORECASE | re.DOTALL,
)
_SPEAKER_RE = re.compile(r"\(S(?P<number>[1-9][0-9]*)\)", re.IGNORECASE)
_SPEECH_CUE_RE = re.compile(
    r"(?:セリフ|台詞|発話|発声|声(?:質|色|で|に|の)|口調|"
    r"(?:と|を)(?:一度だけ|1回だけ)?(?:"
    r"言(?:う|い|った|って)|話(?:す|し|した|して)|"
    r"喋(?:る|り|った|って)|しゃべ(?:る|り|った|って)|"
    r"囁(?:く|き|いた|いて)|ささや(?:く|き|いた|いて)|"
    r"叫(?:ぶ|び|んだ|んで)|つぶや(?:く|き|いた|いて)|"
    r"呟(?:く|き|いた|いて)|尋ね(?:る|た|て)|答え(?:る|た|て))|"
    r"\b(?:says?|said|speaks?|spoke|whispers?|whispered|shouts?|shouted|"
    r"asks?|asked|replies?|replied|murmurs?|murmured|dialogue|spoken\s+line)\b)",
    re.IGNORECASE,
)
_VOICE_DIRECTION_RE = re.compile(
    r"(?:[（(][^）)]*(?:声|口調|voice|tone|delivery)[^）)]*[）)]|"
    r"(?:低|高|明る|暗|太|細|柔らか|かすれ|くぐも|落ち着)[^\r\n]{0,24}声)",
    re.IGNORECASE,
)
_NON_SPEECH_QUOTE_RE = re.compile(
    r"(?:看板|字幕|テロップ|文字|タイトル|標識|ラベル|表示|書かれ|映画|作品名|"
    r"効果音|環境音|BGM|鳴る|音がする|音が鳴る|歌詞|歌う|"
    r"\b(?:sign|subtitle|caption|title|label|banner|poster|menu|screen|"
    r"on[- ]screen(?:\s+(?:text|wording))?|visible\s+(?:text|wording)|"
    r"written|displayed|sound\s*effect|sfx|lyrics?|sings?)\b)",
    re.IGNORECASE,
)
_SPEECH_AFTER_RE = re.compile(
    r"^\s*(?:と|を)?\s*(?:一度だけ|1回だけ|一回だけ)?\s*(?:"
    r"言(?:う|い|った|って)|話(?:す|し|した|して)|"
    r"喋(?:る|り|った|って)|しゃべ(?:る|り|った|って)|"
    r"囁(?:く|き|いた|いて)|ささや(?:く|き|いた|いて)|"
    r"叫(?:ぶ|び|んだ|んで)|つぶや(?:く|き|いた|いて)|"
    r"呟(?:く|き|いた|いて)|尋ね(?:る|た|て)|答え(?:る|た|て))"
    r"(?:\s*(?:と|が|ものの|けれども?|、|,))?",
    re.IGNORECASE,
)
_NON_SPEECH_AFTER_RE = re.compile(
    r"^\s*(?:と|が)?\s*(?:鳴る|響く|聞こえる|表示される|書かれる|"
    r"映る|出る|歌う|歌い始める|sounds?|rings?|appears?|is\s+displayed)",
    re.IGNORECASE,
)
_ENGLISH_SPEECH_BEFORE_RE = re.compile(
    r"\b(?:says?|said|speaks?|spoke|whispers?|whispered|shouts?|shouted|"
    r"asks?|asked|replies?|replied|murmurs?|murmured)\s*(?::|,)?\s*$",
    re.IGNORECASE,
)
_VOICE_CONTROL_RE = re.compile(
    r"(?:[（(]\s*[^）)]{0,48}(?:声|口調|voice|tone|delivery)[^）)]{0,48}[）)]\s*[、,]?)"
    r"|(?:(?:低い|低く|低め|高い|高く|高め|明るい|明るく|暗い|"
    r"柔らかい|柔らかく|かすれた|くぐもった|落ち着いた|甘えた|震えた|"
    r"疲れた)[^、。\r\n]{0,36}(?:声|口調)(?:で|に)?\s*[、,]?)"
    r"|(?:(?:疲れたように|ゆっくり|静かに|小声で|大声で|囁くように|"
    r"ささやくように|叫ぶように)\s*[、,]?)",
    re.IGNORECASE,
)
_SPEECH_CONTROL_RE = re.compile(
    r"(?:と|を)?\s*(?:一度だけ|1回だけ|一回だけ)?\s*(?:"
    r"言(?:う|い|った|って)|話(?:す|し|した|して)|"
    r"喋(?:る|り|った|って)|しゃべ(?:る|り|った|って)|"
    r"囁(?:く|き|いた|いて)|ささや(?:く|き|いた|いて)|"
    r"叫(?:ぶ|び|んだ|んで)|つぶや(?:く|き|いた|いて)|"
    r"呟(?:く|き|いた|いて)|尋ね(?:る|た|て)|答え(?:る|た|て))"
    r"(?:\s*(?:と|が|ものの|けれども?|、|,))?"
    r"|\b(?:says?|said|speaks?|spoke|whispers?|whispered|shouts?|shouted|"
    r"asks?|asked|replies?|replied|murmurs?|murmured)\b"
    r"(?:\s+(?:exactly\s+)?once)?\s*(?::|,)?",
    re.IGNORECASE,
)
_SPEECH_FREQUENCY_BEFORE_RE = re.compile(
    r"(?:一度だけ|1回だけ|一回だけ)\s*[、,]?\s*$",
    re.IGNORECASE,
)
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_HAN_RE = re.compile(r"[\u3400-\u9fff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_NON_ENGLISH_DESCRIPTION_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\u0600-\u06ff\u0400-\u04ff]"
)
_SUBJECT_REFERENCE_RE = re.compile(r"<Subject\s+(?P<number>[1-9][0-9]*)>", re.IGNORECASE)
_PICTURE_REFERENCE_RE = re.compile(r"<Picture\s+(?P<number>[1-9][0-9]*)>", re.IGNORECASE)
_AUDIO_REFERENCE_RE = re.compile(r"<Audio\s+(?P<number>[1-9][0-9]*)>", re.IGNORECASE)
_AUDIO_REFERENCE_DIRECTION_RE = re.compile(
    r"(?:"
    r"[（(]\s*(?:(?:声質|声|話し方|音声|voice|timbre|delivery)\s*)?"
    r"(?:参照|reference)?\s*[:：]?\s*<Audio\s+[1-9][0-9]*>\s*[）)]"
    r"|(?:(?:声質|声|話し方|音声|voice|timbre|delivery)\s*(?:参照|reference)"
    r"|(?:参照|reference)\s*(?:音声|audio)?)\s*[:：]?\s*"
    r"<Audio\s+[1-9][0-9]*>"
    r")",
    re.IGNORECASE,
)

# A native dialogue line may already contain the official post-line lip
# closure.  When the server asks us to canonicalize native context, remove the
# old copy from the visual residual before inserting the one deterministic
# harness owned by this module.
_POST_DIALOGUE_CLOSURE_RE = re.compile(
    r"\b(?:Speaker\s*\(S[1-9][0-9]*\)|She|He|They)\s+"
    r"(?:closes?|close)\s+(?:(?:her|his|their)\s+)?(?:lips|mouth)\s+"
    r"after\s+the\s+line\s*[.!]?",
    re.IGNORECASE,
)

_OVERRIDE_INSTRUCTION_RE = re.compile(
    r"(?:\b(?:Cut|Shot)\s*#?\s*[1-9][0-9]*\b|カット\s*[1-9][0-9]*|"
    r"ショット\s*[1-9][0-9]*|カメラ|一度だけ|1回だけ|一回だけ|"
    r"セリフ|台詞|発話|声質|口調|"
    r"(?:話(?:す|して|せ)|言(?:う|って)|喋(?:る|って)|しゃべ(?:る|って))|"
    r"(?:キャラクター|人物|女性|男性|彼女|彼)[^。！？.!?\r\n]{0,48}"
    r"(?:話(?:す|し|して)|言(?:う|い|って)|喋(?:る|り)|しゃべ(?:る|り))|"
    r"\b(?:camera|dialogue|spoken\s+line|voice\s+direction|"
    r"say|speak|talk|narrate|"
    r"(?:character|speaker|woman|man|she|he)\b[^.?!\r\n]{0,48}"
    r"(?:say|speak|talk|narrate)s?\b)"
    r")",
    re.IGNORECASE,
)


class DialogueOverrideError(ValueError):
    """The legacy override does not identify literal spoken words safely."""


@dataclass(frozen=True, slots=True)
class NativeDialogueEvent:
    source: str
    target_cut: int | None
    original_text: str
    effective_text: str
    language: str
    speaker_id: int = 1
    speaker_label: str = ""
    voice_direction: str = ""

    @property
    def normalized(self) -> bool:
        return self.original_text != self.effective_text

    @property
    def tag(self) -> str:
        return f"<d>[{self.language}] {self.effective_text}</d>"

    @property
    def audio_reference_ids(self) -> tuple[int, ...]:
        return _audio_reference_ids(self.voice_direction)

    @property
    def audio_reference_id(self) -> int | None:
        references = self.audio_reference_ids
        return references[0] if len(references) == 1 else None

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "target_cut": self.target_cut,
            "language": self.language,
            "speaker_id": self.speaker_id,
            "speaker_label": self.speaker_label,
            "original_text": self.original_text,
            "effective_text": self.effective_text,
            "normalized": self.normalized,
            "voice_direction": self.voice_direction,
            "audio_reference_id": self.audio_reference_id,
            "audio_reference_label": (
                f"<Audio {self.audio_reference_id}>"
                if self.audio_reference_id is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class DialogueFormattingResult:
    text: str
    events: tuple[NativeDialogueEvent, ...]
    source: str
    adjustments: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    trusted_fragments: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _QuoteSpan:
    start: int
    end: int
    text: str
    speaker_id: int = 1
    speaker_label: str = ""
    control_context: str = ""


@dataclass(frozen=True, slots=True)
class _LineCandidate:
    line_index: int
    target_cut: int | None
    spans: tuple[_QuoteSpan, ...]


@dataclass(frozen=True, slots=True)
class _OverrideItem:
    original_text: str
    effective_text: str
    language: str
    voice_direction: str


def _split_line_ending(line: str) -> tuple[str, str]:
    body = line.rstrip("\r\n")
    return body, line[len(body) :]


def _quote_text(match: re.Match[str]) -> str:
    return next(value for value in match.groupdict().values() if value is not None).strip()


def _explicit_language_for(context: str) -> str | None:
    """Return a language explicitly requested by the surrounding speech cue."""

    if re.search(r"(?:英語|English)\s*(?:で|voice|speech)?", context, re.IGNORECASE):
        return "English"
    if re.search(r"(?:日本語|Japanese)\s*(?:で|voice|speech)?", context, re.IGNORECASE):
        return "Japanese"
    if re.search(r"(?:中国語|中文|普通话|漢語|汉语|Chinese)", context, re.IGNORECASE):
        return "Chinese"
    if re.search(r"(?:韓国語|한국어|조선말|Korean)", context, re.IGNORECASE):
        return "Korean"
    if re.search(r"(?:アラビア語|العربية|Arabic)", context, re.IGNORECASE):
        return "Arabic"
    if re.search(r"(?:ロシア語|русский|Russian)", context, re.IGNORECASE):
        return "Russian"
    return None


def _high_confidence_script_language(text: str) -> str | None:
    """Classify only scripts whose language mapping is safe for H3 dialogue.

    Han-only text and Latin text intentionally remain unclassified: Han can be
    Japanese or Chinese, while Latin is shared by many supported languages.
    If more than one strong script occurs, the payload is mixed/ambiguous and
    its requested tag is left untouched.
    """

    candidates = {
        language
        for language, pattern in (
            ("Japanese", _KANA_RE),
            ("Korean", _HANGUL_RE),
            ("Arabic", _ARABIC_RE),
            ("Russian", _CYRILLIC_RE),
        )
        if pattern.search(text)
    }
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def _language_for(text: str, *, context: str = "") -> str:
    """Infer H3's language tag from the utterance before its directions."""

    explicit = _explicit_language_for(context)
    if explicit is not None:
        return explicit
    script_language = _high_confidence_script_language(text)
    if script_language is not None:
        return script_language
    if _LATIN_RE.search(text):
        return "English"
    if _HAN_RE.search(text):
        # Kanji-only Japanese is common in this Japanese-first UI (e.g. 了解).
        # Chinese can still be selected explicitly in the surrounding phrase or
        # by supplying an already-native <d>[Chinese] ...</d> tag.
        return "Japanese"
    if _KANA_RE.search(context):
        return "Japanese"
    return "English"


def _correct_event_language_for_script(
    event: NativeDialogueEvent,
) -> tuple[NativeDialogueEvent, bool]:
    """Correct a provably mismatched tag without modifying payload text."""

    script_language = _high_confidence_script_language(event.effective_text)
    if (
        script_language is None
        or event.language.casefold() == script_language.casefold()
    ):
        return event, False
    return replace(event, language=script_language), True


def _normalize_utterance(text: str, *, normalize_decorative: bool) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    if not normalize_decorative:
        return value
    value = re.sub(r"[～〜~]{2,}", "、", value)
    value = re.sub(r"、{2,}", "、", value)
    value = re.sub(r"(?:・{3,}|…{2,}|\.{3,})+[。.!?！？]*$", "。", value)
    value = re.sub(r"。{2,}$", "。", value)
    value = re.sub(r"！{2,}$", "！", value)
    value = re.sub(r"？{2,}$", "？", value)
    return value


def _last_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    matches = tuple(pattern.finditer(text))
    return matches[-1] if matches else None


def _audio_reference_ids(context: str) -> tuple[int, ...]:
    """Return unique Audio ordinals mentioned in one local speech clause."""

    return tuple(
        dict.fromkeys(int(match.group("number")) for match in _AUDIO_REFERENCE_RE.finditer(context))
    )


def _speaker_for_quote(body: str, position: int, previous: str) -> tuple[int, str]:
    """Resolve the closest explicit H3 speaker/ref label for one quote."""

    clause = body[max(0, body.rfind("。", 0, position) + 1) : position]
    context = f"{previous}\n{clause}"
    explicit = _last_match(_SPEAKER_RE, clause) or _last_match(_SPEAKER_RE, previous)
    subject = _last_match(_SUBJECT_REFERENCE_RE, clause) or _last_match(
        _SUBJECT_REFERENCE_RE, previous
    )
    picture = _last_match(_PICTURE_REFERENCE_RE, clause) or _last_match(
        _PICTURE_REFERENCE_RE, previous
    )
    if explicit:
        speaker_id = int(explicit.group("number"))
    elif subject:
        speaker_id = int(subject.group("number"))
    elif picture:
        speaker_id = int(picture.group("number"))
    else:
        speaker_id = 1
    if subject:
        label = f"<Subject {subject.group('number')}>"
    elif picture:
        label = f"The visible character shown in <Picture {picture.group('number')}>"
    else:
        label = _english_speaker(context)
    return speaker_id, label


def _local_control_context(
    body: str,
    start: int,
    end: int,
    previous: str,
) -> str:
    left_boundary = max(
        body.rfind("。", 0, start),
        body.rfind("！", 0, start),
        body.rfind("？", 0, start),
        body.rfind(".", 0, start),
        body.rfind("!", 0, start),
        body.rfind("?", 0, start),
    )
    right_candidates = [
        value
        for value in (
            body.find("。", end),
            body.find("！", end),
            body.find("？", end),
            body.find(".", end),
            body.find("!", end),
            body.find("?", end),
        )
        if value >= 0
    ]
    right_boundary = min(right_candidates) + 1 if right_candidates else len(body)
    local = body[left_boundary + 1 : right_boundary]
    local = _NATIVE_DIALOGUE_RE.sub("[spoken line]", local)
    local = _QUOTE_RE.sub("[spoken line]", local)
    stripped = body.strip()
    if _QUOTE_RE.fullmatch(stripped.rstrip("。.!?！？")) and previous.strip():
        local = f"{previous.strip()} {local}"
    return local.strip()


def _quote_is_spoken(
    body: str,
    match: re.Match[str],
    previous: str,
) -> bool:
    before = body[max(0, match.start() - 64) : match.start()]
    after = body[match.end() : match.end() + 64]
    if _NON_SPEECH_AFTER_RE.search(after):
        return False
    # A display object can grammatically "say" something in both English and
    # Japanese.  Its local role must win over the speech-looking verb or the
    # formatter invents a human S1 and turns visible lettering into audio.
    previous_quotes = tuple(_QUOTE_RE.finditer(body, 0, match.start()))
    local_start = max(0, match.start() - 64)
    if previous_quotes:
        local_start = max(local_start, previous_quotes[-1].end())
    local_before = re.split(
        r"[\r\n。！？.!?、,]", body[local_start : match.start()]
    )[-1]
    if _NON_SPEECH_QUOTE_RE.search(local_before):
        return False
    if _SPEECH_AFTER_RE.search(after) or _ENGLISH_SPEECH_BEFORE_RE.search(before):
        return True
    stripped = body.strip()
    quote_only = bool(_QUOTE_RE.fullmatch(stripped.rstrip("。.!?！？")))
    if quote_only:
        return bool(
            _VOICE_DIRECTION_RE.search(previous) or _SPEECH_CUE_RE.search(previous)
        )
    # A line-level cue is only a fallback when this quote has no nearby marker
    # saying it is a title, visible text, lyric, or physical sound.
    local = body[max(0, match.start() - 48) : match.end() + 48]
    return bool(_SPEECH_CUE_RE.search(local)) and not bool(
        _NON_SPEECH_QUOTE_RE.search(local)
    )


def _cut_number(line: str) -> int | None:
    body, _ = _split_line_ending(line)
    match = _CUT_HEADER_RE.fullmatch(body)
    if not match:
        return None
    return int(match.group("bracket") or match.group("plain"))


def _normalize_inline_cut_layout(text: str) -> str:
    """Keep inline Cut labels while moving their body onto a parseable line."""

    return _INLINE_CUT_LAYOUT_RE.sub(lambda match: f"{match.group('label')}\n", text)


def _normalize_native_dialogue_layout(text: str) -> str:
    def collapse(match: re.Match[str]) -> str:
        utterance = re.sub(r"\s+", " ", match.group("text")).strip()
        return f"<d>[{match.group('language').strip()}] {utterance}</d>"

    return _NATIVE_DIALOGUE_RE.sub(collapse, text)


def _english_speaker(context: str) -> str:
    subject = _SUBJECT_REFERENCE_RE.search(context)
    if subject:
        return f"<Subject {subject.group('number')}>"
    picture = _PICTURE_REFERENCE_RE.search(context)
    if picture:
        return f"The visible character shown in <Picture {picture.group('number')}>"
    if re.search(r"(?:少女|女の子|girl)", context, re.IGNORECASE):
        return "The visible girl in this shot"
    if re.search(r"(?:少年|男の子|boy)", context, re.IGNORECASE):
        return "The visible boy in this shot"
    if re.search(r"(?:女性|彼女|woman|female)", context, re.IGNORECASE):
        return "The visible woman in this shot"
    if re.search(r"(?:男性|彼(?!女)|man|male)", context, re.IGNORECASE):
        return "The visible man in this shot"
    return "The visible character in this shot"


def _english_delivery(context: str) -> tuple[str, str, str]:
    adjectives: list[str] = []
    mappings = (
        (r"(?:低い|低く|低め|low(?:-pitched)?)", "low-pitched"),
        (r"(?:高い|高く|高め|high(?:-pitched)?)", "high-pitched"),
        (r"(?:落ち着|calm)", "calm"),
        (r"(?:明る|bright)", "bright"),
        (r"(?:柔らか|soft)", "soft"),
        (r"(?:くぐも|muffled)", "muffled"),
        (r"(?:かすれ|hoarse|raspy)", "slightly raspy"),
    )
    for pattern, label in mappings:
        if re.search(pattern, context, re.IGNORECASE):
            adjectives.append(label)
    if re.search(r"(?:女性|彼女|少女|女の子|female|woman|girl)", context, re.IGNORECASE):
        voice_noun = "female voice"
    elif re.search(r"(?:男性|彼(?!女)|少年|男の子|male|man|boy)", context, re.IGNORECASE):
        voice_noun = "male voice"
    else:
        voice_noun = "natural voice"
    voice = ", ".join(adjectives)
    voice_clause = f" in a {voice} {voice_noun}" if voice else ""
    if re.search(r"(?:疲れ|tired|weary)", context, re.IGNORECASE):
        voice_clause += " with a tired delivery"
    if re.search(r"(?:ゆっくり|slowly)", context, re.IGNORECASE):
        voice_clause += " at a slow pace"
    if re.search(r"(?:囁|ささや|whisper)", context, re.IGNORECASE):
        verb = "whispers"
    elif re.search(r"(?:叫|shout|yell)", context, re.IGNORECASE):
        verb = "shouts"
    elif re.search(r"(?:つぶや|呟|murmur|mutter)", context, re.IGNORECASE):
        verb = "murmurs"
    else:
        verb = "says"
    gaze = " looks toward the camera and" if re.search(r"(?:カメラ|こちら|camera)", context, re.IGNORECASE) else ""
    return voice_clause, verb, gaze


def _render_official_dialogue_event(
    event: NativeDialogueEvent,
    *,
    include_audio_references: bool = True,
) -> str:
    context = event.voice_direction
    speaker = event.speaker_label or _english_speaker(context)
    voice_clause, verb, gaze = _english_delivery(context)
    audio_reference_id = (
        event.audio_reference_id if include_audio_references else None
    )
    reference_contract = ""
    audio_clause = ""
    if audio_reference_id is not None:
        audio_label = f"<Audio {audio_reference_id}>"
        picture = _PICTURE_REFERENCE_RE.search(speaker)
        subject_label = f"<Subject {event.speaker_id}>"
        if picture:
            reference_contract += (
                f"{subject_label} (S{event.speaker_id}) is the visible character shown in "
                f"<Picture {picture.group('number')}>. "
            )
        elif not re.fullmatch(
            rf"<Subject\s+{event.speaker_id}>", speaker, re.IGNORECASE
        ):
            reference_contract += (
                f"{subject_label} (S{event.speaker_id}) is the visible speaker in this shot. "
            )
        speaker = subject_label
        reference_contract += (
            f"{audio_label} provides the voice timbre and measured delivery for {speaker} "
            f"(S{event.speaker_id}). "
        )
        audio_clause = f", using {audio_label},"
        # A concrete reference signal is more specific than a generic inferred
        # gendered voice.  Keeping both can make the conditioner choose between
        # two competing timbres.  The caller may suppress this entire reference
        # contract when exact dialogue takes priority over full AudioVAE input.
        voice_clause = ""
    return (
        f"{reference_contract}{speaker} (S{event.speaker_id}){audio_clause}{gaze} "
        f"{verb} once{voice_clause}: {event.tag} "
        f"Speaker (S{event.speaker_id}) closes their lips after the line."
    )


def _render_official_dialogue_line(
    events: list[NativeDialogueEvent],
    *,
    include_audio_references: bool = True,
) -> str:
    return " ".join(
        _render_official_dialogue_event(
            event,
            include_audio_references=include_audio_references,
        )
        for event in events
    )


def _speech_control_ranges(body: str, spans: tuple[_QuoteSpan, ...]) -> list[tuple[int, int]]:
    """Return quote/speech-control ranges, never unrelated action or SFX text."""

    ranges: list[tuple[int, int]] = []
    for span in spans:
        ranges.append((span.start, span.end))
        suffix = _SPEECH_AFTER_RE.search(body[span.end :])
        if suffix:
            ranges.append((span.end + suffix.start(), span.end + suffix.end()))
        clause_start = max(
            body.rfind("。", 0, span.start),
            body.rfind("！", 0, span.start),
            body.rfind("？", 0, span.start),
            body.rfind(".", 0, span.start),
            body.rfind("!", 0, span.start),
            body.rfind("?", 0, span.start),
        ) + 1
        prefix = body[clause_start : span.start]
        # A Japanese speech-frequency adverb can sit immediately before the
        # quote ("一度だけ「…」と言う").  It belongs to the utterance.  If it is
        # left in the visual residual, translation can invent an unrelated
        # once-only action from the dangling fragment.
        frequency = _SPEECH_FREQUENCY_BEFORE_RE.search(prefix)
        if frequency:
            ranges.append(
                (clause_start + frequency.start(), clause_start + frequency.end())
            )
        for match in _VOICE_CONTROL_RE.finditer(prefix):
            ranges.append((clause_start + match.start(), clause_start + match.end()))
        right_candidates = [
            value
            for value in (
                body.find("。", span.end),
                body.find("！", span.end),
                body.find("？", span.end),
                body.find(".", span.end),
                body.find("!", span.end),
                body.find("?", span.end),
            )
            if value >= 0
        ]
        clause_end = min(right_candidates) + 1 if right_candidates else len(body)
        clause = body[clause_start:clause_end]
        # An Audio tag in the same explicit speech clause is a voice reference,
        # not visible action or reusable soundtrack text.  Consume both the
        # common parenthetical form and the bare tag so it cannot survive as a
        # broken Japanese fragment beside the canonical dialogue block.
        for match in _AUDIO_REFERENCE_DIRECTION_RE.finditer(clause):
            ranges.append((clause_start + match.start(), clause_start + match.end()))
        if _SPEECH_CUE_RE.search(clause):
            for match in _AUDIO_REFERENCE_RE.finditer(clause):
                ranges.append((clause_start + match.start(), clause_start + match.end()))
    return sorted(ranges)


def _visual_residual(body: str, spans: tuple[_QuoteSpan, ...]) -> str:
    residual = body
    for start, end in reversed(_speech_control_ranges(body, spans)):
        residual = residual[:start] + "" + residual[end:]
    # A final defensive pass removes inflected speech verbs adjacent to an
    # already-removed quote, while leaving visual verbs and physical sounds.
    residual = _SPEECH_CONTROL_RE.sub("", residual)
    residual = _POST_DIALOGUE_CLOSURE_RE.sub("", residual)
    residual = re.sub(r"[、,]\s*[、,]", "、", residual)
    residual = re.sub(r"[、,]\s*([。.!?！？])", r"\1", residual)
    residual = re.sub(r"([。.!?！？])\s*\1+", r"\1", residual)
    residual = re.sub(
        r"(^|[。.!?！？]\s*)(?:<Picture\s+[1-9][0-9]*>の)?"
        r"(?:キャラクター|人物|彼女|彼|女性|男性|少女|少年)(?:は|が)\s*(?:[。.!?！？]|$)",
        lambda match: match.group(1),
        residual,
        flags=re.IGNORECASE,
    )
    residual = re.sub(
        r"^(?P<prefix>(?:\[(?:Cut|Shot)\s*[1-9][0-9]*\]\s*)?)"
        r"(?:(?:The|A|An)\s+)?(?:woman|man|girl|boy|person|character|speaker)"
        r"(?:\s*\(S[1-9][0-9]*\))?\s*[.!]?$",
        lambda match: match.group("prefix"),
        residual,
        flags=re.IGNORECASE,
    )
    residual = re.sub(r"[ \t]{2,}", " ", residual)
    residual = re.sub(r"^[、,\s]+|[、,\s]+$", "", residual)
    return residual.strip()


def _is_pure_voice_direction(line: str) -> bool:
    body, _ = _split_line_ending(line)
    stripped = body.strip()
    if not stripped or not _VOICE_DIRECTION_RE.search(stripped):
        return False
    remainder = _VOICE_CONTROL_RE.sub("", stripped)
    remainder = re.sub(r"[（()）\s、,。.!！?？:：-]", "", remainder)
    return not remainder


def _scan_prompt(
    text: str,
) -> tuple[
    list[str],
    tuple[_LineCandidate, ...],
    dict[int, tuple[NativeDialogueEvent, ...]],
    tuple[int, ...],
    tuple[str, ...],
]:
    lines = text.splitlines(keepends=True)
    candidates: list[_LineCandidate] = []
    native_by_line: dict[int, tuple[NativeDialogueEvent, ...]] = {}
    cut_numbers: list[int] = []
    diagnostics: list[str] = []
    active_cut: int | None = None

    for index, line in enumerate(lines):
        header_cut = _cut_number(line)
        if header_cut is not None:
            active_cut = header_cut
            cut_numbers.append(header_cut)

        body, _ = _split_line_ending(line)
        previous = lines[index - 1] if index else ""
        native_matches = tuple(_NATIVE_DIALOGUE_RE.finditer(body))
        if native_matches:
            native_events: list[NativeDialogueEvent] = []
            for match in native_matches:
                utterance = match.group("text").strip()
                if not utterance:
                    continue
                speaker_id, speaker_label = _speaker_for_quote(
                    body, match.start(), previous
                )
                native_events.append(
                    NativeDialogueEvent(
                        source="prompt_native",
                        target_cut=active_cut,
                        original_text=utterance,
                        effective_text=utterance,
                        language=match.group("language").strip(),
                        speaker_id=speaker_id,
                        speaker_label=speaker_label,
                        voice_direction=_local_control_context(
                            body, match.start(), match.end(), previous
                        ),
                    )
                )
            native_by_line[index] = tuple(native_events)
            continue

        quote_matches = tuple(_QUOTE_RE.finditer(body))
        spoken_matches = tuple(
            match
            for match in quote_matches
            if _quote_text(match) and _quote_is_spoken(body, match, previous)
        )
        if spoken_matches:
            spans_list: list[_QuoteSpan] = []
            for match in spoken_matches:
                speaker_id, speaker_label = _speaker_for_quote(
                    body, match.start(), previous
                )
                spans_list.append(
                    _QuoteSpan(
                        match.start(),
                        match.end(),
                        _quote_text(match),
                        speaker_id,
                        speaker_label,
                        _local_control_context(body, match.start(), match.end(), previous),
                    )
                )
            spans = tuple(spans_list)
            candidates.append(
                _LineCandidate(
                    line_index=index,
                    target_cut=active_cut,
                    spans=spans,
                )
            )

    open_count = len(re.findall(r"<d>", text, flags=re.IGNORECASE))
    close_count = len(re.findall(r"</d>", text, flags=re.IGNORECASE))
    native_count = len(tuple(_NATIVE_DIALOGUE_RE.finditer(text)))
    if open_count != close_count or open_count != native_count:
        diagnostics.append("MALFORMED_DIALOGUE_TAG")

    return (
        lines,
        tuple(candidates),
        native_by_line,
        tuple(dict.fromkeys(cut_numbers)),
        tuple(dict.fromkeys(diagnostics)),
    )


def _remove_override_target(text: str) -> str:
    value = re.sub(
        r"(?im)^[ \t]*(?:Cut|Shot|カット|ショット)\s*#?\s*[1-9][0-9]*"
        r"\s*(?::|：)?\s*$",
        "",
        text,
    )
    return _TARGET_RE.sub("", value, count=1).lstrip(" 　:：で")


def _parse_override(
    text: str,
    *,
    normalize_decorative: bool,
) -> tuple[tuple[_OverrideItem, ...], int | None]:
    value = text.strip()
    if not value:
        return (), None
    target_match = _TARGET_RE.search(value)
    target = int(target_match.group("number")) if target_match else None
    body = _remove_override_target(value).strip()

    native_matches = tuple(_NATIVE_DIALOGUE_RE.finditer(body))
    if native_matches:
        direction = _NATIVE_DIALOGUE_RE.sub("", body).strip()
        return (
            tuple(
                _OverrideItem(
                    original_text=match.group("text").strip(),
                    effective_text=_normalize_utterance(
                        match.group("text"), normalize_decorative=normalize_decorative
                    ),
                    language=match.group("language").strip(),
                    voice_direction=direction,
                )
                for match in native_matches
                if match.group("text").strip()
            ),
            target,
        )

    quote_matches = tuple(_QUOTE_RE.finditer(body))
    if quote_matches:
        direction = body
        for match in reversed(quote_matches):
            direction = direction[: match.start()] + "" + direction[match.end() :]
        direction = re.sub(r"\s+", " ", direction).strip(" 　。.!！")
        return (
            tuple(
                _OverrideItem(
                    original_text=_quote_text(match),
                    effective_text=_normalize_utterance(
                        _quote_text(match), normalize_decorative=normalize_decorative
                    ),
                    language=_language_for(_quote_text(match), context=body),
                    voice_direction=direction,
                )
                for match in quote_matches
                if _quote_text(match)
            ),
            target,
        )

    nonempty = [line.strip() for line in body.splitlines() if line.strip()]
    if not nonempty:
        return (), target
    if (
        len(nonempty) != 1
        or len(nonempty[0]) > 160
        or _OVERRIDE_INSTRUCTION_RE.search(nonempty[0])
        or _VOICE_DIRECTION_RE.search(nonempty[0])
    ):
        raise DialogueOverrideError(
            "台詞を固定する欄では、実際に発音する文字を「……」で囲むか、"
            "公式の <d>[Language] ...</d> 形式で指定してください。"
            "複数行の説明やカメラ・発話指示を台詞本文としては扱いません。"
        )
    utterance = nonempty[-1]
    direction = " ".join(nonempty[:-1])
    return (
        (
            _OverrideItem(
                original_text=utterance,
                effective_text=_normalize_utterance(
                    utterance, normalize_decorative=normalize_decorative
                ),
                language=_language_for(utterance, context=body),
                voice_direction=direction,
            ),
        ),
        target,
    )


def _resolve_override_target(
    requested: int | None,
    candidates: tuple[_LineCandidate, ...],
    native_by_line: dict[int, tuple[NativeDialogueEvent, ...]],
    cuts: tuple[int, ...],
) -> tuple[int | None, str | None]:
    if requested is not None:
        if not cuts or requested in cuts:
            return requested, None
        return cuts[-1], "DIALOGUE_TARGET_DEFAULTED"
    quote_targets = {
        candidate.target_cut
        for candidate in candidates
        if candidate.target_cut is not None and candidate.spans
    }
    quote_targets.update(
        event.target_cut
        for native_events in native_by_line.values()
        for event in native_events
        if event.target_cut is not None
    )
    if len(quote_targets) == 1:
        return next(iter(quote_targets)), None
    if len(cuts) == 1:
        return cuts[0], None
    if cuts:
        return cuts[-1], "DIALOGUE_TARGET_DEFAULTED"
    return None, "DIALOGUE_TARGET_DEFAULTED"


def _render_inserted_event(
    event: NativeDialogueEvent,
    *,
    include_audio_references: bool = True,
) -> str:
    return _render_official_dialogue_event(
        event,
        include_audio_references=include_audio_references,
    )


def _insert_blocks(
    lines: list[str],
    blocks: dict[int | None, list[str]],
) -> list[str]:
    if not blocks:
        return lines
    result = list(lines)
    header_positions = [
        (index, _cut_number(line))
        for index, line in enumerate(result)
        if _cut_number(line) is not None
    ]
    insertions: list[tuple[int, list[str]]] = []
    for target, target_blocks in blocks.items():
        insertion = len(result)
        if target is not None:
            matching = [position for position, number in header_positions if number == target]
            if matching:
                header_index = matching[-1]
                insertion = next(
                    (
                        position
                        for position, _ in header_positions
                        if position > header_index
                    ),
                    len(result),
                )
        prefix = "" if insertion == 0 or result[insertion - 1].endswith(("\n", "\r")) else "\n"
        rendered = [prefix + block + "\n" for block in target_blocks]
        insertions.append((insertion, rendered))
    for insertion, rendered in sorted(insertions, reverse=True):
        result[insertion:insertion] = rendered
    return result


def format_inline_dialogue(
    prompt: str,
    override: str = "",
    *,
    normalize_decorative: bool = False,
    include_audio_references: bool = True,
    canonicalize_native_context: bool = False,
) -> DialogueFormattingResult:
    """Return a raw prompt with explicit spoken quotations formatted for H3."""

    normalized_prompt = _normalize_native_dialogue_layout(
        _normalize_inline_cut_layout(prompt)
    )
    lines, candidates, native_by_line, cuts, scan_diagnostics = _scan_prompt(
        normalized_prompt
    )
    override_items, requested_target = _parse_override(
        override, normalize_decorative=normalize_decorative
    )
    diagnostics = list(scan_diagnostics)
    adjustments: list[str] = []

    def correct_language(
        event: NativeDialogueEvent,
    ) -> tuple[NativeDialogueEvent, bool]:
        corrected_event, corrected = _correct_event_language_for_script(event)
        if corrected:
            diagnostics.append("DIALOGUE_LANGUAGE_SCRIPT_MISMATCH")
            adjustments.append("DIALOGUE_LANGUAGE_TAG_CORRECTED")
        return corrected_event, corrected

    assignments: dict[tuple[int, int], _OverrideItem] = {}
    suppressed: set[tuple[int, int]] = set()
    native_assignments: dict[tuple[int, int], _OverrideItem] = {}
    native_suppressed: set[tuple[int, int]] = set()
    extras: list[_OverrideItem] = []
    override_target: int | None = None
    if override_items:
        override_target, target_diagnostic = _resolve_override_target(
            requested_target, candidates, native_by_line, cuts
        )
        if target_diagnostic:
            diagnostics.append(target_diagnostic)
        target_keys = sorted(
            [
                (candidate.line_index, quote_index, "quote")
                for candidate in candidates
                if candidate.target_cut == override_target
                for quote_index, _ in enumerate(candidate.spans)
            ]
            + [
                (line_index, event_index, "native")
                for line_index, native_events in native_by_line.items()
                for event_index, event in enumerate(native_events)
                if event.target_cut == override_target
            ],
            key=lambda item: (item[0], item[1]),
        )
        for (line_index, event_index, kind), item in zip(
            target_keys, override_items, strict=False
        ):
            destination = native_assignments if kind == "native" else assignments
            destination[(line_index, event_index)] = item
        if len(target_keys) > len(override_items):
            for line_index, event_index, kind in target_keys[len(override_items) :]:
                destination = native_suppressed if kind == "native" else suppressed
                destination.add((line_index, event_index))
            adjustments.append("DIALOGUE_OVERRIDE_REMOVED_EXTRA_LINES")
        extras.extend(override_items[len(target_keys) :])

    events: list[NativeDialogueEvent] = []
    trusted_fragments: list[str] = []
    rendered_lines = list(lines)
    candidate_by_line = {candidate.line_index: candidate for candidate in candidates}
    for index, line in enumerate(lines):
        if index in native_by_line:
            native_events: list[NativeDialogueEvent] = []
            native_changed = False
            for event_index, original_event in enumerate(native_by_line[index]):
                key = (index, event_index)
                if key in native_suppressed:
                    native_changed = True
                    continue
                assigned = native_assignments.get(key)
                if assigned is None:
                    corrected_event, language_corrected = correct_language(original_event)
                    native_events.append(corrected_event)
                    native_changed = native_changed or language_corrected
                    continue
                native_changed = True
                assigned_event, _ = correct_language(
                    NativeDialogueEvent(
                        source="override",
                        target_cut=original_event.target_cut,
                        original_text=assigned.original_text,
                        effective_text=assigned.effective_text,
                        language=assigned.language,
                        speaker_id=original_event.speaker_id,
                        speaker_label=original_event.speaker_label,
                        voice_direction=(
                            assigned.voice_direction or original_event.voice_direction
                        ),
                    )
                )
                native_events.append(assigned_event)
                if assigned.original_text != assigned.effective_text:
                    adjustments.append("DIALOGUE_PUNCTUATION_NORMALIZED")
            events.extend(native_events)
            body, ending = _split_line_ending(line)
            control = _NATIVE_DIALOGUE_RE.sub("[spoken line]", body)
            if (
                native_changed
                or canonicalize_native_context
                or _NON_ENGLISH_DESCRIPTION_RE.search(control)
            ):
                native_spans = tuple(
                    _QuoteSpan(match.start(), match.end(), match.group("text").strip())
                    for match in _NATIVE_DIALOGUE_RE.finditer(body)
                    if match.group("text").strip()
                )
                residual = _visual_residual(body, native_spans)
                harness_fragments = tuple(
                    _render_official_dialogue_event(
                        event,
                        include_audio_references=include_audio_references,
                    )
                    for event in native_events
                )
                trusted_fragments.extend(harness_fragments)
                harness = " ".join(harness_fragments)
                content = f"{residual}\n{harness}" if residual and harness else residual or harness
                rendered_lines[index] = content + ending
                adjustments.append("DIALOGUE_CONTEXT_CANONICALIZED")
            continue
        candidate = candidate_by_line.get(index)
        if candidate is None:
            continue

        body, ending = _split_line_ending(line)
        replacements: list[tuple[_QuoteSpan, NativeDialogueEvent | None]] = []
        for quote_index, span in enumerate(candidate.spans):
            key = (candidate.line_index, quote_index)
            if key in suppressed:
                replacements.append((span, None))
                continue
            assigned = assignments.get(key)
            source = "override" if assigned else "prompt"
            original = assigned.original_text if assigned else span.text
            effective = (
                assigned.effective_text
                if assigned
                else _normalize_utterance(
                    original, normalize_decorative=normalize_decorative
                )
            )
            language = (
                assigned.language
                if assigned
                else _language_for(original, context=span.control_context)
            )
            direction = (
                assigned.voice_direction
                if assigned and assigned.voice_direction
                else span.control_context
            )
            event = NativeDialogueEvent(
                source=source,
                target_cut=candidate.target_cut,
                original_text=original,
                effective_text=effective,
                language=language,
                speaker_id=span.speaker_id,
                speaker_label=span.speaker_label,
                voice_direction=direction,
            )
            event, _ = correct_language(event)
            replacements.append((span, event))
            events.append(event)
            if event.normalized:
                adjustments.append("DIALOGUE_PUNCTUATION_NORMALIZED")

        active_events = [event for _, event in replacements if event is not None]
        if not active_events:
            rendered_lines[index] = ""
            continue
        processed_spans = tuple(span for span, _ in replacements)
        residual = _visual_residual(body, processed_spans)
        harness_fragments = tuple(
            _render_official_dialogue_event(
                event,
                include_audio_references=include_audio_references,
            )
            for event in active_events
        )
        trusted_fragments.extend(harness_fragments)
        harness = " ".join(harness_fragments)
        rendered_lines[index] = (f"{residual}\n{harness}" if residual else harness) + ending
        adjustments.append("DIALOGUE_CONTEXT_CANONICALIZED")
        if index > 0 and _is_pure_voice_direction(lines[index - 1]):
            rendered_lines[index - 1] = ""

    insertion_blocks: dict[int | None, list[str]] = {}
    for item in extras:
        event = NativeDialogueEvent(
            source="override",
            target_cut=override_target,
            original_text=item.original_text,
            effective_text=item.effective_text,
            language=item.language,
            speaker_id=1,
            speaker_label="The visible speaker",
            voice_direction=item.voice_direction,
        )
        event, _ = correct_language(event)
        events.append(event)
        if event.normalized:
            adjustments.append("DIALOGUE_PUNCTUATION_NORMALIZED")
        rendered_event = _render_inserted_event(
            event,
            include_audio_references=include_audio_references,
        )
        trusted_fragments.append(rendered_event)
        insertion_blocks.setdefault(override_target, []).append(rendered_event)
    rendered_lines = _insert_blocks(rendered_lines, insertion_blocks)

    sources = {event.source for event in events}
    if not events:
        source = "none"
    elif sources <= {"prompt", "prompt_native"}:
        source = "prompt"
    elif sources == {"override"}:
        source = "override"
    else:
        source = "mixed"

    rendered_text = "".join(rendered_lines).strip()
    audio_reference_events = [event for event in events if event.audio_reference_ids]
    if any(len(event.audio_reference_ids) > 1 for event in audio_reference_events):
        diagnostics.append("AMBIGUOUS_DIALOGUE_AUDIO_REFERENCE")
    if any(event.audio_reference_id is not None for event in audio_reference_events):
        adjustments.append("AUDIO_REFERENCE_BOUNDARY_CANONICALIZED")

    return DialogueFormattingResult(
        text=rendered_text,
        events=tuple(events),
        source=source,
        adjustments=tuple(dict.fromkeys(adjustments)),
        diagnostics=tuple(dict.fromkeys(diagnostics)),
        trusted_fragments=tuple(dict.fromkeys(trusted_fragments)),
    )


__all__ = [
    "DialogueOverrideError",
    "DialogueFormattingResult",
    "NativeDialogueEvent",
    "format_inline_dialogue",
]
