"""Small deterministic guard for prompts sent directly to H3.

This is intentionally not a Context-IR compiler.  The persisted UI request
remains the audit source of truth.  Explicit inline dialogue is formatted first
with H3's native ``<d>`` boundary; this guard then removes only unresolved or
negative speech cues while leaving tagged dialogue and Cut-local sound intact.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SanitizedGenerationText:
    text: str
    removed_fragments: tuple[str, ...] = ()
    positive_sound_cues: tuple[str, ...] = ()
    rewritten_gaze_cues: int = 0


_SLURP_RE = re.compile(
    r"(?:(?:ズ|ず){2,}(?:ー|ｰ|〜|～|-)*(?:っと)?|"
    r"(?:ゴクゴク|ごくごく)(?:っと)?)"
)

_GAZE_PLACEHOLDER_RULES = (
    (
        re.compile(
            r"(?:こちら|カメラ)(?:の方)?を見て[、,\s]*"
            r"(?:(?:キャラクター|人物|彼女|彼|女性|男性|少女|少年)(?:が|の)?)?"
            r"(?:セリフ|台詞|発話)(?:を(?:言う|話す|喋る|しゃべる))?"
        ),
        "カメラを見る",
    ),
    (
        re.compile(
            r"looks?\s+(?:toward|at)\s+(?:the\s+)?camera\s*(?:,|and|then)*\s*"
            r"(?:delivers?\s+(?:a\s+)?line|speaks?|says?\s+(?:a\s+)?line)",
            re.IGNORECASE,
        ),
        "looks at the camera",
    ),
)

_JAPANESE_SPEECH_CUE_RE = re.compile(
    r"(?:セリフ|台詞|発話|発声|ナレーション|ボイス[・\s-]*オーバー|"
    r"人(?:間)?の声|人声|しゃべ(?:る|ら|り|れ|ろ|っ|ない)|"
    r"喋(?:る|ら|り|れ|ろ|っ)|話(?:す|さ|し|せ|そ)|"
    r"言(?:う|わ|い|え|お|っ)|歌(?:う|わ|い|え|お|っ))"
)

_ENGLISH_SPEECH_CUE_RE = re.compile(
    r"\b(?:dialogue|narrat(?:e|es|ed|ing|ion|or)|voice[ -]?over|"
    r"human\s+voice|vocal(?:s|ization)?|speaks?|speaking|spoken|"
    r"says?|saying|talks?|talking|sings?|singing|lyrics?)\b",
    re.IGNORECASE,
)

# These phrases contain lexical speech words but describe visible UI/text or a
# figurative physical action.  Removing them before cue detection prevents the
# direct path from deleting legitimate visual instructions such as a dialogue
# box, folklore mural, or wind moving trees "as if singing".
_NON_SPEECH_CONTEXT_RE = re.compile(
    r"\b(?:dialogue|dialog|speech)\s+(?:box|bubble|balloon)\b|"
    r"(?:言い伝え|言伝え|歌うように|歌うような)",
    re.IGNORECASE,
)
_DISPLAY_OBJECT_SPEECHLIKE_RE = re.compile(
    r"^\s*(?:(?:the|a|an|this|that)\s+)?(?:[a-z-]+\s+){0,2}"
    r"(?:sign|title|subtitle|caption|label|banner|poster|menu|"
    r"on[- ]screen\s+(?:text|wording)|visible\s+(?:text|wording))\b"
    r"[^\r\n.!?]{0,64}\b(?:says?|said|reads?|states?|displays?|shows?)\b|"
    r"^\s*(?:[^\r\n、。！？]{0,12}の)?"
    r"(?:看板|字幕|テロップ|文字|タイトル|標識|ラベル|表示)"
    r"[^\r\n、。！？]{0,64}(?:と言(?:う|い|った)|と書かれ|表示(?:する|される))",
    re.IGNORECASE | re.MULTILINE,
)

_SENTENCE_RE = re.compile(r"[^\n。！？.!?]+[。！？.!?]*|\n+")
_CLAUSE_SEPARATOR_RE = re.compile(r"([、,;；])")
_LEADING_NEGATIVE_SPEECH_RE = re.compile(
    r"^\s*(?P<prefix>(?:Audio\s*:\s*)?)"
    r"(?:"
    r"(?P<actor>(?:(?:彼女|彼|女性|男性|少女|少年|人物|キャラクター|話者)"
    r"(?:は|が)\s*)?)(?:(?:何も|一言も)\s*)?(?:一切\s*)?"
    r"(?:(?:話さず|しゃべらず|喋らず|発話せず|歌わず)(?:に)?|"
    r"(?:話さない|しゃべらない|喋らない|発話しない|歌わない)(?:まま|で)?)"
    r"|(?:人(?:間)?の声|人声|ナレーション|ボイス[・\s-]*オーバー|"
    r"セリフ|台詞|発話)(?:は|を|が)?\s*(?:一切\s*)?"
    r"(?:入れず|含めず|出さず|ない|なし|入らない|含めない|出さない)"
    r"|(?:without\s+(?:any\s+)?(?:human\s+)?(?:speech|dialogue|narration|"
    r"voice[ -]?over|voice|vocals?)|no\s+(?:human\s+)?(?:speech|dialogue|"
    r"narration|voice[ -]?over|voice|vocals?))"
    r")\s*(?:[、,;；]|で\s*)?\s*",
    re.IGNORECASE,
)
_TRAILING_CONNECTIVE_RE = re.compile(
    r"(?:\s*(?:が|けれど|けど|一方で)|\s+\b(?:and|but|while|although))\s*$",
    re.IGNORECASE,
)
_LEADING_CONNECTIVE_RE = re.compile(r"^(?:and|but|then)\s+", re.IGNORECASE)
_TERMINAL_PUNCTUATION_RE = re.compile(r"[。！？.!?]+$")
_PUNCTUATION_ONLY_RE = re.compile(r"^[\s、,;；。！？.!?]+$")
_ACTOR_PREFIX_RE = re.compile(
    r"^\s*(?P<actor>(?:彼女|彼|女性|男性|少女|少年|人物|キャラクター|話者)(?:は|が))"
)

_INLINE_NEGATIVE_SPEECH_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"手を振りながら(?:(?:何も|一言も)\s*)?"
            r"(?:話さない|しゃべらない|喋らない|発話しない)"
        ),
        "手を振る",
    ),
    (
        re.compile(
            r"(?:(?:何も|一言も)\s*)?(?:しゃべる|喋る|話す|発話する|歌う)ことなく"
        ),
        "",
    ),
    (
        re.compile(
            r"(?:(?:何も|一言も)\s*)?(?:話す|しゃべる|喋る|発話する|歌う)"
            r"のをやめて\s*[、,]?\s*"
        ),
        "",
    ),
    (re.compile(r"(?:(?:何も|一言も)\s*)?声を出さず(?:に)?"), ""),
    (
        re.compile(
            r"(?:(?:何も|一言も)\s*)?(?:話さず|しゃべらず|喋らず|発話せず|歌わず)"
            r"(?:に)?"
        ),
        "",
    ),
    (
        re.compile(
            r"(?:(?:何も|一言も)\s*)?(?:話さない|しゃべらない|喋らない|発話しない|歌わない)"
            r"(?:まま)?"
        ),
        "",
    ),
    (
        re.compile(
            r"\bwithout\s+(?:any\s+)?(?:speaking|talking|singing|speech|dialogue|"
            r"narration|voice[ -]?over|human\s+voice|vocals?)\b",
            re.IGNORECASE,
        ),
        "",
    ),
    (
        re.compile(
            r"\b(?:does\s+not|doesn't|do\s+not|don't)\s+(?:speak|talk|sing|narrate)\b",
            re.IGNORECASE,
        ),
        "",
    ),
)

_POSITIVE_SOUND_RULES = (
    (
        re.compile(r"(?:砂浜|海辺|海|波|ビーチ|ocean|sea|beach|surf)", re.IGNORECASE),
        "Gentle ocean surf provides a continuous natural ambient bed.",
    ),
    (
        re.compile(r"(?:風|そよ風|海風|wind|breeze)", re.IGNORECASE),
        "A light natural breeze moves softly through the scene.",
    ),
    (
        re.compile(r"(?:雨|小雨|rain|drizzle)", re.IGNORECASE),
        "Natural rainfall matches the visible environment and surfaces.",
    ),
    (
        re.compile(r"(?:足音|歩く|走る|footsteps?|walking|running)", re.IGNORECASE),
        "Visible steps produce subtle, surface-matched footfalls.",
    ),
    (
        re.compile(r"(?:焚き火|炎|火|fire|flame)", re.IGNORECASE),
        "Visible fire produces a soft, physically matched crackle.",
    ),
    (
        re.compile(r"(?:鳥|小鳥|birds?)", re.IGNORECASE),
        "Distant birds add sparse, natural calls from the visible environment.",
    ),
)

_SLURP_SOUND = (
    "During the visible drinking action, liquid moving through the straw creates "
    "one brief, natural slurp."
)

_D_TAG_RE = re.compile(
    r"<d>\s*\[[A-Za-z][A-Za-z -]{1,31}\]\s*.*?</d>",
    re.IGNORECASE | re.DOTALL,
)
_DIALOGUE_SENTINEL_PREFIX = "H3STUDIONATIVEDIALOGUE"
_TRUSTED_SPEECH_SENTINEL_PREFIX = "H3STUDIOTRUSTEDSPEECHCONTEXT"
_PROTECTED_SPEECH_TOKEN_RE = re.compile(
    rf"({_DIALOGUE_SENTINEL_PREFIX}[0-9]{{4}}|"
    rf"{_TRUSTED_SPEECH_SENTINEL_PREFIX}[0-9]{{4}})"
)
_QUOTED_LITERAL_RE = re.compile(
    r'"[^"\r\n]+"|“[^”\r\n]+”|「[^」\r\n]+」|『[^』\r\n]+』'
)
_QUOTED_LITERAL_SENTINEL_PREFIX = "H3STUDIOQUOTEDLITERAL"


def contains_speech_cue(text: str) -> bool:
    """Return whether text asks for, names, or negates human vocal content."""

    candidate = _NON_SPEECH_CONTEXT_RE.sub("", text or "")
    candidate = _DISPLAY_OBJECT_SPEECHLIKE_RE.sub("", candidate)
    return bool(
        _JAPANESE_SPEECH_CUE_RE.search(candidate)
        or _ENGLISH_SPEECH_CUE_RE.search(candidate)
    )


def _deduplicate(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value.strip()))


def _remove_leading_negative_speech(text: str) -> tuple[str, str | None]:
    """Remove a leading no-speech policy while preserving the useful clause.

    A common natural-language form combines a negative vocal instruction and a
    positive visual or ambience instruction in one sentence, for example
    ``人の声は入れず、波と風の音だけにする``.  The negative phrase must not
    be sent to H3 as a vocal token, but deleting the whole sentence also deletes
    the user's requested sound.  This deliberately narrow rewrite handles only
    a leading, syntactically explicit no-speech phrase.
    """

    match = _LEADING_NEGATIVE_SPEECH_RE.match(text)
    if match is None:
        return text, None
    remainder = text[match.end() :].lstrip()
    prefix = match.group("prefix") or ""
    actor = match.groupdict().get("actor") or ""
    return prefix + actor + remainder, match.group(0).strip(" 、,;；")


def _remove_inline_negative_speech(text: str) -> tuple[str, tuple[str, ...]]:
    """Remove known negative vocal predicates without deleting nearby actions."""

    rewritten = text
    removed: list[str] = []
    for pattern, replacement in _INLINE_NEGATIVE_SPEECH_RULES:
        def replace(match: re.Match[str], value: str = replacement) -> str:
            removed.append(match.group(0).strip())
            return value

        rewritten = pattern.sub(replace, rewritten)
    rewritten = _clean_speech_removal_artifacts(rewritten)
    return rewritten, _deduplicate(removed)


def _clean_speech_removal_artifacts(text: str) -> str:
    """Repair only punctuation/connectives made dangling by a removed policy."""

    cleaned = re.sub(r"[、,;；]+\s*([。！？.!?])", r"\1", text)
    cleaned = re.sub(r"(?:で|が|けれど|けど)\s*([。！？.!?])", r"\1", cleaned)
    cleaned = re.sub(
        r"((?:彼女|彼|女性|男性|少女|少年|人物|キャラクター|話者)(?:は|が))[、,]\s*",
        r"\1",
        cleaned,
    )
    cleaned = re.sub(r",\s*(?:and|but|then)\s+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*[、,;；]\s*", "", cleaned)
    cleaned = re.sub(
        r"\b(?P<actor>she|he|they)\s+(?:and|but|then)\s+",
        lambda match: match.group("actor") + " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+([.!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned


def _sanitize_speech_fragment(fragment: str) -> tuple[str, tuple[str, ...]]:
    """Drop speech-only clauses without discarding neighbouring user intent."""

    rewritten, inline_removed = _remove_inline_negative_speech(fragment)
    rewritten, leading_removed = _remove_leading_negative_speech(rewritten)
    removed: list[str] = list(inline_removed)
    if leading_removed:
        removed.append(leading_removed)
    if not contains_speech_cue(rewritten):
        cleaned = _clean_speech_removal_artifacts(rewritten)
        if not cleaned or _PUNCTUATION_ONLY_RE.fullmatch(cleaned):
            return "", _deduplicate(removed)
        return cleaned, _deduplicate(removed)

    parts = _CLAUSE_SEPARATOR_RE.split(rewritten)
    clauses = parts[::2]
    separators = parts[1::2]
    kept_clauses: list[str] = []
    carried_actor = ""
    preferred_separator = "、" if "、" in separators or "；" in separators else ", "

    for clause in clauses:
        cleaned, clause_leading_removed = _remove_leading_negative_speech(clause)
        if clause_leading_removed:
            removed.append(clause_leading_removed)
        stripped = cleaned.strip()
        if not stripped:
            continue
        if contains_speech_cue(stripped):
            removed.append(stripped)
            actor_match = _ACTOR_PREFIX_RE.match(stripped)
            if actor_match and not carried_actor:
                carried_actor = actor_match.group("actor")
            continue
        stripped = _TRAILING_CONNECTIVE_RE.sub("", stripped).rstrip()
        stripped = _LEADING_CONNECTIVE_RE.sub("", stripped).strip()
        if stripped and not _PUNCTUATION_ONLY_RE.fullmatch(stripped):
            kept_clauses.append(stripped)

    if not kept_clauses:
        return "", _deduplicate(removed)

    if carried_actor and not _ACTOR_PREFIX_RE.match(kept_clauses[0]):
        kept_clauses[0] = carried_actor + kept_clauses[0]
    result = preferred_separator.join(kept_clauses)
    terminal = _TERMINAL_PUNCTUATION_RE.search(fragment)
    if terminal and not _TERMINAL_PUNCTUATION_RE.search(result):
        result += terminal.group(0)
    return _clean_speech_removal_artifacts(result), _deduplicate(removed)


def sanitize_generation_text(
    text: str,
    *,
    preserve_speech_context: bool = False,
    trusted_speech_fragments: Sequence[str] = (),
) -> SanitizedGenerationText:
    """Remove unresolved speech cues while preserving native inline dialogue.

    The preserve_speech_context keyword remains for compatibility, but no
    longer bypasses validation.  Only exact deterministic harness fragments
    supplied by the dialogue formatter are trusted; prose before or after a
    native dialogue block is always inspected independently.
    """

    if not text or not text.strip():
        return SanitizedGenerationText("")

    protected_dialogue: dict[str, str] = {}
    protected_literals: dict[str, str] = {}
    protected_trusted: dict[str, str] = {}

    def protect_dialogue(match: re.Match[str]) -> str:
        sentinel = f"{_DIALOGUE_SENTINEL_PREFIX}{len(protected_dialogue):04d}"
        protected_dialogue[sentinel] = match.group(0)
        return sentinel

    def protect_literal(match: re.Match[str]) -> str:
        sentinel = f"{_QUOTED_LITERAL_SENTINEL_PREFIX}{len(protected_literals):04d}"
        protected_literals[sentinel] = match.group(0)
        return sentinel

    rewritten = text
    for trusted in sorted(
        dict.fromkeys(
            fragment
            for fragment in trusted_speech_fragments
            if isinstance(fragment, str) and fragment
        ),
        key=len,
        reverse=True,
    ):
        if trusted not in rewritten:
            continue
        sentinel = (
            f"{_TRUSTED_SPEECH_SENTINEL_PREFIX}{len(protected_trusted):04d}"
        )
        protected_trusted[sentinel] = trusted
        rewritten = rewritten.replace(trusted, sentinel)

    rewritten = _D_TAG_RE.sub(protect_dialogue, rewritten)
    rewritten = _QUOTED_LITERAL_RE.sub(protect_literal, rewritten)
    gaze_rewrites = 0
    for pattern, replacement in _GAZE_PLACEHOLDER_RULES:
        rewritten, count = pattern.subn(replacement, rewritten)
        gaze_rewrites += count

    positive_sound_cues: list[str] = []
    rewritten = _SLURP_RE.sub("短く自然な吸引音を伴って", rewritten)

    kept: list[str] = []
    removed: list[str] = []
    for line in rewritten.splitlines(keepends=True):
        for part in _PROTECTED_SPEECH_TOKEN_RE.split(line):
            if not part:
                continue
            if _PROTECTED_SPEECH_TOKEN_RE.fullmatch(part):
                kept.append(part)
                continue
            for fragment in _SENTENCE_RE.findall(part):
                if fragment.startswith("\n"):
                    kept.append(fragment)
                    continue
                cleaned_fragment, removed_clauses = _sanitize_speech_fragment(fragment)
                removed.extend(removed_clauses)
                if cleaned_fragment:
                    kept.append(cleaned_fragment)

    cleaned = "".join(kept)
    for sentinel, trusted in protected_trusted.items():
        cleaned = cleaned.replace(sentinel, trusted)
    for sentinel, dialogue in protected_dialogue.items():
        cleaned = cleaned.replace(sentinel, dialogue)
    for sentinel, literal in protected_literals.items():
        cleaned = cleaned.replace(sentinel, literal)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return SanitizedGenerationText(
        cleaned,
        _deduplicate(removed),
        _deduplicate(positive_sound_cues),
        gaze_rewrites,
    )


def infer_positive_soundscape(*texts: str) -> tuple[str, ...]:
    """Map explicit scene keywords to conservative, positive physical sounds."""

    source = "\n".join(text for text in texts if text)
    sounds = [sentence for pattern, sentence in _POSITIVE_SOUND_RULES if pattern.search(source)]
    if _SLURP_RE.search(source):
        sounds.append(_SLURP_SOUND)
    return _deduplicate(sounds)


__all__ = [
    "SanitizedGenerationText",
    "contains_speech_cue",
    "infer_positive_soundscape",
    "sanitize_generation_text",
]
