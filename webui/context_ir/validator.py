"""Pure schema and safety validation for rendered H3 Context-IR."""

from __future__ import annotations

from collections import Counter
import re

from .model import (
    ContextIRDocument,
    Diagnostic,
    EmbeddedVideoAudioPolicy,
    GenerationMode,
    MusicPolicy,
    ReferenceKind,
    Severity,
)
from .parser import OMNI_SECTION_HEADERS, THREE_SECTION_HEADERS, discover_reference_labels


_HEADER_RE = re.compile(
    r"^(subject_definitions|summary|retention_analysis|detailed_description|"
    r"integrated_multimodal_description|overall_soundscape|non_diegetic_music)[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
_SHOT_RE = re.compile(
    r"\[Shot\s+(?P<index>[1-9][0-9]*)\](?:\s+At\s+"
    r"(?P<minutes>[0-9]{2,}):(?P<seconds>[0-5][0-9])\.(?P<millis>[0-9]{3}))?",
    re.IGNORECASE,
)
_VERBATIM_BLOCK_RE = re.compile(
    r"--- BEGIN H3 STUDIO VERBATIM TEXT ---\n.*?\n--- END H3 STUDIO VERBATIM TEXT ---",
    re.DOTALL,
)
_DIALOGUE_BLOCK_RE = re.compile(r"<d>.*?</d>", re.IGNORECASE | re.DOTALL)
_NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")
_H3_SPECIAL_TOKENS = (
    "<d>",
    "</d>",
    "<|cutoff|>",
    "<|lyrics_start|>",
    "<|lyrics_end|>",
    "<|caption_start|>",
    "<|caption_end|>",
)
# Keep the legacy spellings blocked as defense in depth.  The canonical seven
# entries above must stay identical to the H3-only tokenizer compatibility shim.
_FORBIDDEN_DIALOGUE_TOKENS = _H3_SPECIAL_TOKENS + ("<scenetrans>", "<cutoff>")


def _diag(severity: Severity, code: str, message: str, path: str = "") -> Diagnostic:
    return Diagnostic(severity=severity, code=code, message=message, path=path)


def validate_document(document: ContextIRDocument) -> tuple[Diagnostic, ...]:
    issues: list[Diagnostic] = []
    if not document.shots:
        issues.append(_diag(Severity.FATAL, "IR_NO_SHOTS", "The IR document has no shots.", "/shots"))
        return tuple(issues)

    defined_subject_ids = {
        definition.subject_index
        for definition in document.subject_definitions
        if definition.subject_index is not None
    }

    expected_indices = list(range(1, len(document.shots) + 1))
    actual_indices = [shot.index for shot in document.shots]
    if actual_indices != expected_indices:
        issues.append(
            _diag(
                Severity.FATAL,
                "IR_SHOT_INDEX_SEQUENCE",
                "Shot indices must be consecutive and start at 1.",
                "/shots",
            )
        )
    if document.shots[0].start_ms != 0:
        issues.append(
            _diag(
                Severity.FATAL,
                "IR_SHOT1_START",
                "Shot 1 must start internally at 0 ms.",
                "/shots/0/start_ms",
            )
        )
    previous = -1
    for shot in document.shots:
        if shot.start_ms <= previous:
            issues.append(
                _diag(
                    Severity.FATAL,
                    "IR_SHOT_TIME_ORDER",
                    "Shot start times must be strictly increasing.",
                    f"/shots/{shot.index - 1}/start_ms",
                )
            )
        if shot.start_ms >= document.duration_ms:
            issues.append(
                _diag(
                    Severity.FATAL,
                    "IR_SHOT_AFTER_END",
                    "A shot starts at or after the effective video duration.",
                    f"/shots/{shot.index - 1}/start_ms",
                )
            )
        previous = shot.start_ms
        for event_index, event in enumerate(shot.dialogue_events):
            has_direct_visual_speaker_reference = (
                event.speaker_reference is not None
                and event.speaker_reference.kind in (
                    ReferenceKind.PICTURE,
                    ReferenceKind.VIDEO,
                )
            )
            if (
                not has_direct_visual_speaker_reference
                and event.speaker_id not in defined_subject_ids
            ):
                issues.append(
                    _diag(
                        Severity.FATAL,
                        "IR_UNKNOWN_SPEAKER_SUBJECT",
                        f"Dialogue selects <Subject {event.speaker_id}> "
                        f"(S{event.speaker_id}), but that subject is not defined.",
                        f"/shots/{shot.index - 1}/dialogue_events/{event_index}/speaker_id",
                    )
                )
            for field_name, value in (
                ("exact_text", event.exact_text),
                ("voice_direction", event.voice_direction),
            ):
                if any(
                    token.lower() in value.lower()
                    for token in _FORBIDDEN_DIALOGUE_TOKENS
                ):
                    issues.append(
                        _diag(
                            Severity.FATAL,
                            "IR_DIALOGUE_TOKEN_INJECTION",
                            "Dialogue text or voice direction contains a reserved H3 control "
                            "token and cannot be rendered safely.",
                            f"/shots/{shot.index - 1}/dialogue_events/{event_index}/{field_name}",
                        )
                    )

    seen: set[str] = set()
    for index, reference in enumerate(document.references):
        if reference.label.text in seen:
            issues.append(
                _diag(
                    Severity.FATAL,
                    "IR_DUPLICATE_REFERENCE",
                    f"Reference {reference.label.text} is declared more than once.",
                    f"/references/{index}",
                )
            )
        seen.add(reference.label.text)

    if document.embedded_video_audio_policy is EmbeddedVideoAudioPolicy.AUTO:
        issues.append(
            _diag(
                Severity.FATAL,
                "IR_UNRESOLVED_VIDEO_AUDIO_POLICY",
                "Embedded video audio policy must be resolved before rendering.",
                "/embedded_video_audio_policy",
            )
        )
    if not document.overall_soundscape.strip():
        issues.append(
            _diag(
                Severity.FATAL,
                "IR_EMPTY_SOUNDSCAPE",
                "overall_soundscape must not be empty.",
                "/overall_soundscape",
            )
        )
    if not document.non_diegetic_music.strip():
        issues.append(
            _diag(
                Severity.FATAL,
                "IR_EMPTY_MUSIC",
                "non_diegetic_music must not be empty.",
                "/non_diegetic_music",
            )
        )

    if document.degraded:
        issues.append(
            _diag(
                Severity.WARNING,
                "IR_DEGRADED_LOCAL_SCAFFOLD",
                "No generative Context-IR model was used. The original-language prompt is "
                "preserved inside an English, official-section scaffold.",
            )
        )
    return tuple(issues)


def _text_for_header_scan(text: str) -> str:
    return _VERBATIM_BLOCK_RE.sub("", text)


def _section_headers(text: str) -> tuple[str, ...]:
    scan = _text_for_header_scan(text)
    return tuple(match.group(1).lower() for match in _HEADER_RE.finditer(scan))


def _timestamp_ms(match: re.Match[str]) -> int | None:
    if match.group("minutes") is None:
        return None
    return (
        int(match.group("minutes")) * 60_000
        + int(match.group("seconds")) * 1_000
        + int(match.group("millis"))
    )


def validate_rendered_ir(
    text: str,
    *,
    mode: GenerationMode,
    duration_ms: int,
    document: ContextIRDocument | None = None,
    allowed_reference_labels: tuple[str, ...] | None = None,
) -> tuple[Diagnostic, ...]:
    issues: list[Diagnostic] = []
    if not isinstance(text, str) or not text.strip():
        return (
            _diag(Severity.FATAL, "IR_EMPTY_TEXT", "Rendered Context-IR is empty."),
        )

    expected = OMNI_SECTION_HEADERS if mode is GenerationMode.OMNI else THREE_SECTION_HEADERS
    actual = _section_headers(text)
    if actual != expected:
        issues.append(
            _diag(
                Severity.FATAL,
                "IR_SECTION_SCHEMA",
                f"Expected sections {expected!r} in exact order, got {actual!r}.",
            )
        )

    main_header = (
        "detailed_description:"
        if mode is GenerationMode.OMNI
        else "integrated_multimodal_description:"
    )
    main_start = text.find(main_header)
    shot_scan = text[main_start + len(main_header) :] if main_start >= 0 else text
    shots = tuple(_SHOT_RE.finditer(shot_scan))
    if not shots:
        issues.append(
            _diag(
                Severity.WARNING,
                "IR_NO_RENDERED_SHOTS",
                "No [Shot N] marker was found; a raw user IR may be underspecified.",
            )
        )
    else:
        indices = [int(match.group("index")) for match in shots]
        if indices != list(range(1, len(shots) + 1)):
            issues.append(
                _diag(
                    Severity.FATAL,
                    "IR_RENDERED_SHOT_SEQUENCE",
                    "Rendered shot markers must be consecutive and start at 1.",
                )
            )
        if _timestamp_ms(shots[0]) is not None:
            issues.append(
                _diag(
                    Severity.FATAL,
                    "IR_SHOT1_HAS_TIMESTAMP",
                    "Official H3 format does not put a timestamp after [Shot 1].",
                )
            )
        previous = 0
        for match in shots[1:]:
            timestamp = _timestamp_ms(match)
            if timestamp is None:
                issues.append(
                    _diag(
                        Severity.FATAL,
                        "IR_LATER_SHOT_MISSING_TIMESTAMP",
                        f"[Shot {match.group('index')}] must have an MM:SS.mmm cut time.",
                    )
                )
                continue
            if timestamp <= previous or timestamp >= duration_ms:
                issues.append(
                    _diag(
                        Severity.FATAL,
                        "IR_RENDERED_SHOT_TIME",
                        "Rendered cut times must increase strictly and stay within duration.",
                    )
                )
            previous = timestamp

    open_tags = len(re.findall(r"<d>", text, re.IGNORECASE))
    close_tags = len(re.findall(r"</d>", text, re.IGNORECASE))
    if open_tags != close_tags:
        issues.append(
            _diag(
                Severity.FATAL,
                "IR_UNBALANCED_DIALOGUE_TAGS",
                f"Found {open_tags} <d> tags and {close_tags} </d> tags.",
            )
        )

    if document is not None:
        events = tuple(event for shot in document.shots for event in shot.dialogue_events)
        if open_tags != len(events):
            issues.append(
                _diag(
                    Severity.FATAL,
                    "IR_DIALOGUE_EVENT_COUNT",
                    "Every immutable dialogue event must be rendered in exactly one <d> block.",
                )
            )
        expected_blocks = Counter(event.tag for event in events)
        rendered_blocks = Counter(_DIALOGUE_BLOCK_RE.findall(text))
        if rendered_blocks != expected_blocks:
            issues.append(
                _diag(
                    Severity.FATAL,
                    "IR_DIALOGUE_NOT_EXACTLY_ONCE",
                    "The rendered dialogue blocks do not exactly match the immutable dialogue "
                    "event multiset.",
                    "/dialogue_events",
                )
            )

    if mode is GenerationMode.T2V:
        expected_labels = set()
    elif mode is GenerationMode.I2V:
        expected_labels = {"<Picture 1>"}
    elif mode is GenerationMode.FIRST_LAST:
        expected_labels = {"<Picture 1>", "<Picture 2>"}
    elif allowed_reference_labels is not None:
        expected_labels = set(allowed_reference_labels)
    elif document is not None:
        expected_labels = {item.label.text for item in document.references}
    else:
        expected_labels = None

    if expected_labels is not None:
        used_labels = {item.text for item in discover_reference_labels(text)}
        unknown = sorted(used_labels.difference(expected_labels))
        if unknown:
            issues.append(
                _diag(
                    Severity.FATAL,
                    "IR_UNKNOWN_REFERENCE_LABEL",
                    "Rendered IR uses references unavailable in this mode/request: "
                    + ", ".join(unknown),
                )
            )

    if mode is GenerationMode.I2V and not text.startswith(
        "For the target video, at 0.00 seconds into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced."
    ):
        issues.append(
            _diag(
                Severity.FATAL,
                "IR_I2V_ALIGNMENT",
                "I2V output is missing the canonical first-frame alignment instruction.",
            )
        )
    if mode is GenerationMode.FIRST_LAST and not text.startswith(
        "How the reference pictures align with the target video —"
    ):
        issues.append(
            _diag(
                Severity.FATAL,
                "IR_FIRST_LAST_ALIGNMENT",
                "First/last output is missing the canonical alignment instruction.",
            )
        )

    non_protected = _DIALOGUE_BLOCK_RE.sub("", _VERBATIM_BLOCK_RE.sub("", text))
    if _NON_ASCII_RE.search(non_protected):
        # The em dash in the official FL2VA alignment sentence is expected.
        residue = non_protected.replace("—", "")
        if _NON_ASCII_RE.search(residue):
            issues.append(
                _diag(
                    Severity.WARNING,
                    "IR_NON_ENGLISH_SCAFFOLD_TEXT",
                    "Non-ASCII text exists outside protected dialogue/verbatim blocks; "
                    "review the scaffold if strict English-only output is required.",
                )
            )
    return tuple(issues)


def has_fatal(diagnostics: tuple[Diagnostic, ...]) -> bool:
    return any(item.fatal for item in diagnostics)


__all__ = ["has_fatal", "validate_document", "validate_rendered_ir"]
