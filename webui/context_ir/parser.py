"""Conservative parsing helpers for user-authored H3 instructions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from .model import DialogueEvent, ReferenceKind, ReferenceLabel, Shot


THREE_SECTION_HEADERS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)
OMNI_SECTION_HEADERS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)

_HEADER_RE = re.compile(
    r"^(?P<header>subject_definitions|summary|retention_analysis|"
    r"detailed_description|integrated_multimodal_description|"
    r"overall_soundscape|non_diegetic_music)[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
_REFERENCE_RE = re.compile(
    r"<\s*(?P<kind>Picture|Video|Audio)\s+(?P<index>[1-9][0-9]*)\s*>",
    re.IGNORECASE,
)
_SHOT_MARKER_RE = re.compile(
    r"^[ \t]*(?:"
    r"\[(?P<bracket_kind>Cut|Shot)\s*(?P<bracket_number>[1-9][0-9]*)?\]"
    r"|(?P<plain_kind>Cut|Shot)\s*(?P<plain_number>[1-9][0-9]*)?"
    r")(?P<separator>[ \t]*(?::|：|-|—)[ \t]*)?(?P<inline>.*)$",
    re.IGNORECASE,
)
_DIALOGUE_TARGET_LINE_RE = re.compile(
    r"(?im)^[ \t]*(?:Cut|Shot)\s*([1-9][0-9]*)[ \t]*(?::|：)?[ \t]*$"
)
_DIALOGUE_TARGET_ANY_RE = re.compile(
    r"(?i)(?:^|[\s\[])(?:Cut|Shot)\s*([1-9][0-9]*)(?=$|[\s\]:：])"
)
_JAPANESE_QUOTE_RE = re.compile(r"「(?P<a>[^」]+)」|『(?P<b>[^』]+)』")
_D_TAG_RE = re.compile(
    r"<d>\s*\[(?P<language>[^\]]+)\]\s*(?P<text>.*?)</d>",
    re.IGNORECASE | re.DOTALL,
)
_AUDIO_TAG_RE = re.compile(r"<\s*Audio\s+[1-9][0-9]*\s*>", re.IGNORECASE)
_SUBJECT_RE = re.compile(
    r"(?i)(?:<\s*Subject\s+|\bSubject\s+|\bS)(?P<index>[1-9][0-9]*)\s*>?"
)
_PLAIN_VISUAL_REFERENCE_RE = re.compile(
    r"(?i)(?P<kind>Picture|Video|reference\s+image|image|参照画像|画像|動画)\s*"
    r"(?P<index>[1-9][0-9]*)"
)
_LEADING_DIRECTION_RE = re.compile(
    r"^\s*(?P<direction>(?:(?:（[^）]*）|\([^)]*\)|"
    r"<\s*Audio\s+[1-9][0-9]*\s*>)\s*)+)(?P<utterance>.+)$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class ParsedPrompt:
    preamble: str
    shots: tuple[Shot, ...]
    inferred_timestamps: bool


@dataclass(frozen=True, slots=True)
class ParsedDialogue:
    events: tuple[DialogueEvent, ...]
    requested_target_shot: int | None


def ordered_headers(text: str) -> tuple[str, ...]:
    return tuple(match.group("header").lower() for match in _HEADER_RE.finditer(text))


def _contains_schema(headers: tuple[str, ...], schema: tuple[str, ...]) -> bool:
    cursor = -1
    for expected in schema:
        try:
            cursor = headers.index(expected, cursor + 1)
        except ValueError:
            return False
    return True


def is_h3_context_ir(text: str) -> bool:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    headers = ordered_headers(text)
    return _contains_schema(headers, THREE_SECTION_HEADERS) or _contains_schema(
        headers, OMNI_SECTION_HEADERS
    )


def discover_reference_labels(*texts: str) -> tuple[ReferenceLabel, ...]:
    found: dict[tuple[ReferenceKind, int], ReferenceLabel] = {}
    for text in texts:
        for match in _REFERENCE_RE.finditer(text or ""):
            label = ReferenceLabel.parse(
                f"<{match.group('kind').capitalize()} {int(match.group('index'))}>"
            )
            found[(label.kind, label.index)] = label
    order = {
        ReferenceKind.PICTURE: 0,
        ReferenceKind.VIDEO: 1,
        ReferenceKind.AUDIO: 2,
    }
    return tuple(sorted(found.values(), key=lambda item: (order[item.kind], item.index)))


def _match_shot_marker(line: str) -> tuple[str, int | None, str] | None:
    match = _SHOT_MARKER_RE.fullmatch(line)
    if match is None:
        return None
    kind = match.group("bracket_kind") or match.group("plain_kind")
    number_text = match.group("bracket_number") or match.group("plain_number")
    separator = match.group("separator")
    inline = match.group("inline") or ""
    # Do not treat ordinary prose such as "Shot from below" as a marker.
    if number_text is None and inline.strip() and separator is None:
        return None
    return kind.title(), int(number_text) if number_text else None, inline


def split_prompt_into_shots(prompt: str, duration_ms: int) -> ParsedPrompt:
    lines = prompt.splitlines()
    preamble: list[str] = []
    raw_shots: list[tuple[int | None, str | None, list[str]]] = []
    current: tuple[int | None, str | None, list[str]] | None = None

    for line in lines:
        marker = _match_shot_marker(line)
        if marker is None:
            if current is None:
                preamble.append(line)
            else:
                current[2].append(line)
            continue
        kind, source_number, inline = marker
        source_marker = f"{kind}{source_number if source_number is not None else ''}"
        current = (source_number, source_marker, [])
        if inline.strip():
            current[2].append(inline)
        raw_shots.append(current)

    inferred = bool(raw_shots)
    if not raw_shots:
        raw_shots = [(None, None, lines)]
        preamble = []

    interval = Decimal(duration_ms) / Decimal(len(raw_shots))
    shots = tuple(
        Shot(
            index=index,
            start_ms=int(
                (interval * Decimal(index - 1)).quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP
                )
            ),
            source_marker=source_marker,
            instruction="\n".join(content).strip(),
        )
        for index, (_, source_marker, content) in enumerate(raw_shots, start=1)
    )
    return ParsedPrompt("\n".join(preamble).strip(), shots, inferred)


def _target_shot(text: str) -> int | None:
    line_match = _DIALOGUE_TARGET_LINE_RE.search(text)
    if line_match:
        return int(line_match.group(1))
    any_match = _DIALOGUE_TARGET_ANY_RE.search(text)
    return int(any_match.group(1)) if any_match else None


def _remove_target_marker(text: str) -> str:
    value = _DIALOGUE_TARGET_LINE_RE.sub("", text)
    # Remove one inline target prefix, while leaving ordinary uses of Cut/Shot.
    value = re.sub(
        r"(?i)^[ \t]*(?:Cut|Shot)\s*[1-9][0-9]*[ \t]*(?::|：|-|—)?[ \t]*",
        "",
        value,
        count=1,
    )
    return value.strip()


def _audio_reference(text: str) -> ReferenceLabel | None:
    for label in discover_reference_labels(text):
        if label.kind is ReferenceKind.AUDIO:
            return label
    return None


def _speaker_reference(text: str) -> ReferenceLabel | None:
    for label in discover_reference_labels(text):
        if label.kind in {ReferenceKind.PICTURE, ReferenceKind.VIDEO}:
            return label
    match = _PLAIN_VISUAL_REFERENCE_RE.search(text)
    if match is None:
        return None
    raw_kind = match.group("kind").lower()
    kind = ReferenceKind.VIDEO if raw_kind in {"video", "動画"} else ReferenceKind.PICTURE
    return ReferenceLabel(kind, int(match.group("index")))


def _speaker_id(text: str, reference: ReferenceLabel | None) -> int:
    match = _SUBJECT_RE.search(text)
    if match is not None:
        return int(match.group("index"))
    return reference.index if reference is not None else 1


def _event_prefix(
    value: str,
    *,
    speaker_reference: ReferenceLabel | None = None,
) -> str:
    cleaned = _AUDIO_TAG_RE.sub(" ", value)
    cleaned = _SUBJECT_RE.sub(" ", cleaned)
    if speaker_reference is not None:
        tag = re.compile(
            rf"<\s*{speaker_reference.kind.value}\s+{speaker_reference.index}\s*>",
            re.IGNORECASE,
        )
        cleaned = tag.sub(" ", cleaned)
        kind = (
            r"(?:Picture|reference\s+image|image|参照画像|画像)"
            if speaker_reference.kind is ReferenceKind.PICTURE
            else r"(?:Video|動画)"
        )
        cleaned = re.sub(
            rf"(?i){kind}\s*{speaker_reference.index}(?![0-9])",
            " ",
            cleaned,
        )
    cleaned = cleaned.strip(" \t\r\n:：,、。")
    cleaned = re.sub(r"^(?:(?:だけ|のみ)?(?:が|は))\s*", "", cleaned)
    if re.fullmatch(
        r"(?i)(?:and|then|and\s+then|says?|speaks?|そして|それから)?",
        cleaned,
    ):
        return ""
    return cleaned


def _events_from_matches(
    body: str,
    matches: tuple[re.Match[str], ...],
    *,
    default_language: str,
    target: int | None,
    tagged: bool,
) -> tuple[DialogueEvent, ...]:
    events: list[DialogueEvent] = []
    active_audio: ReferenceLabel | None = None
    active_speaker_reference: ReferenceLabel | None = None
    active_speaker_id = 1
    previous_end = 0
    for match in matches:
        prefix = body[previous_end : match.start()]
        prefix_audio = _audio_reference(prefix)
        if prefix_audio is not None:
            active_audio = prefix_audio
        prefix_speaker = _speaker_reference(prefix)
        if prefix_speaker is not None:
            active_speaker_reference = prefix_speaker
        elif _SUBJECT_RE.search(prefix) is not None:
            active_speaker_reference = None
        active_speaker_id = _speaker_id(prefix, active_speaker_reference)
        if tagged:
            exact_text = match.group("text").strip()
            language = match.group("language").strip() or default_language
        else:
            exact_text = (match.group("a") or match.group("b")).strip()
            language = default_language
        if exact_text:
            events.append(
                DialogueEvent(
                    exact_text=exact_text,
                    language=language,
                    speaker_id=active_speaker_id,
                    target_shot=target,
                    voice_direction=_event_prefix(
                        prefix,
                        speaker_reference=active_speaker_reference,
                    ),
                    audio_reference=active_audio,
                    speaker_reference=active_speaker_reference,
                )
            )
        previous_end = match.end()
    return tuple(events)


def parse_dialogue(text: str, *, default_language: str) -> ParsedDialogue:
    value = text.strip()
    if not value:
        return ParsedDialogue((), None)

    target = _target_shot(value)
    body = _remove_target_marker(value)
    audio_reference = _audio_reference(body)

    tagged = tuple(_D_TAG_RE.finditer(body))
    if tagged:
        events = _events_from_matches(
            body,
            tagged,
            default_language=default_language,
            target=target,
            tagged=True,
        )
        return ParsedDialogue(events, target)

    quoted = tuple(_JAPANESE_QUOTE_RE.finditer(body))
    if quoted:
        events = _events_from_matches(
            body,
            quoted,
            default_language=default_language,
            target=target,
            tagged=False,
        )
        return ParsedDialogue(events, target)

    direction_match = _LEADING_DIRECTION_RE.fullmatch(body)
    if direction_match:
        direction = direction_match.group("direction").strip()
        utterance = direction_match.group("utterance").strip()
    else:
        direction = ""
        utterance = body
    speaker_reference = _speaker_reference(body)
    return ParsedDialogue(
        (
            DialogueEvent(
                exact_text=utterance,
                language=default_language,
                speaker_id=_speaker_id(body, speaker_reference),
                target_shot=target,
                voice_direction=direction,
                audio_reference=audio_reference,
                speaker_reference=speaker_reference,
            ),
        ),
        target,
    )


def assign_dialogue_to_shots(
    shots: tuple[Shot, ...], events: tuple[DialogueEvent, ...]
) -> tuple[tuple[Shot, ...], tuple[tuple[int | None, int], ...]]:
    """Attach each event once and return target adjustments as ``(old, new)``."""

    if not events:
        return shots, ()
    by_source_number: dict[int, int] = {}
    for shot in shots:
        if not shot.source_marker:
            continue
        number = re.search(r"([1-9][0-9]*)$", shot.source_marker)
        if number:
            by_source_number[int(number.group(1))] = shot.index

    allocated: dict[int, list[DialogueEvent]] = {shot.index: [] for shot in shots}
    adjustments: list[tuple[int | None, int]] = []
    for event in events:
        requested = event.target_shot
        if requested is not None and requested in by_source_number:
            target = by_source_number[requested]
        elif requested is not None and 1 <= requested <= len(shots):
            target = requested
        elif len(shots) == 1:
            target = 1
        else:
            target = len(shots)
        if requested != target:
            adjustments.append((requested, target))
        allocated[target].append(
            DialogueEvent(
                exact_text=event.exact_text,
                language=event.language,
                speaker_id=event.speaker_id,
                target_shot=target,
                voice_direction=event.voice_direction,
                audio_reference=event.audio_reference,
                speaker_reference=event.speaker_reference,
            )
        )

    return (
        tuple(
            Shot(
                index=shot.index,
                start_ms=shot.start_ms,
                source_marker=shot.source_marker,
                instruction=shot.instruction,
                dialogue_events=tuple(allocated[shot.index]),
            )
            for shot in shots
        ),
        tuple(adjustments),
    )


__all__ = [
    "OMNI_SECTION_HEADERS",
    "ParsedDialogue",
    "ParsedPrompt",
    "THREE_SECTION_HEADERS",
    "assign_dialogue_to_shots",
    "discover_reference_labels",
    "is_h3_context_ir",
    "ordered_headers",
    "parse_dialogue",
    "split_prompt_into_shots",
]
