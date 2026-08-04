"""Deterministic renderers for official H3 three- and six-section prompts."""

from __future__ import annotations

from .model import ContextIRDocument, DialogueEvent, GenerationMode, Shot


def format_timestamp(milliseconds: int) -> str:
    minutes, remainder = divmod(milliseconds, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def _verbatim(label: str, value: str) -> str:
    return (
        f"{label} is preserved verbatim in the user's original language; it has not "
        "been translated or semantically rewritten:\n"
        "--- BEGIN H3 STUDIO VERBATIM TEXT ---\n"
        f"{value.strip()}\n"
        "--- END H3 STUDIO VERBATIM TEXT ---"
    )


def _render_dialogue(event: DialogueEvent, *, omni: bool) -> str:
    if omni:
        if event.speaker_reference is not None:
            speaker = f"The visible character identified by {event.speaker_reference.text}"
        else:
            speaker = f"<Subject {event.speaker_id}> (S{event.speaker_id})"
    else:
        speaker = f"The visible speaker (S{event.speaker_id})"
    parts = [
        f"{speaker} produces only the following user-specified utterance."
    ]
    if event.voice_direction.strip():
        parts.append(_verbatim("The user-authored voice direction", event.voice_direction))
    if event.audio_reference is not None:
        parts.append(
            f"The delivery uses {event.audio_reference.text} only according to its declared "
            "reference policy."
        )
    parts.append(f"{speaker} says: {event.tag}")
    parts.append(
        "Immediately after this utterance, the speaker's lips close and the jaw ceases "
        "speaking motion. No narrator, voice-over, additional words, or extra human "
        "vocalization is introduced."
    )
    return " ".join(parts)


def _render_shot(shot: Shot, *, omni: bool) -> str:
    if shot.index == 1:
        opening = "[Shot 1]"
    else:
        opening = (
            f"[Shot {shot.index}] At {format_timestamp(shot.start_ms)}, the camera cuts "
            "to the next user-specified shot."
        )
    body: list[str] = [opening]
    if shot.source_marker:
        body.append(f"The original user marker for this shot was {shot.source_marker}.")
    if shot.instruction.strip():
        body.append(_verbatim("The user-authored shot instruction", shot.instruction))
    else:
        body.append(
            "No additional visual content is invented for this shot beyond the global "
            "user instruction and supplied references."
        )
    body.extend(_render_dialogue(event, omni=omni) for event in shot.dialogue_events)
    return "\n".join(body)


def _render_timeline(document: ContextIRDocument, *, omni: bool) -> str:
    parts: list[str] = []
    if document.preamble.strip():
        parts.append(_verbatim("The global user instruction", document.preamble))
    parts.extend(_render_shot(shot, omni=omni) for shot in document.shots)
    if not any(shot.dialogue_events for shot in document.shots):
        parts.append(
            "Do not invent narration, voice-over, dialogue, lyrics, or human vocalization. "
            "Only speech explicitly contained in the verbatim user instruction may occur."
        )
    return "\n\n".join(parts)


def _alignment_instruction(document: ContextIRDocument) -> str | None:
    if document.mode is GenerationMode.I2V:
        return (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
    if document.mode is GenerationMode.FIRST_LAST:
        seconds = document.duration_ms / 1000.0
        return (
            "How the reference pictures align with the target video — Picture 1 "
            "(from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {len(document.shots)}) aligns with the "
            f"{seconds:.2f}-second mark of the target video."
        )
    return None


def render_base(document: ContextIRDocument) -> str:
    if document.mode is GenerationMode.OMNI:
        raise ValueError("render_base does not accept omni documents")
    description: list[str] = []
    if document.mode is GenerationMode.I2V:
        description.append(
            "The supplied first frame is the opening visual anchor. Preserve its "
            "identity-defining content while following the verbatim user instruction."
        )
    elif document.mode is GenerationMode.FIRST_LAST:
        description.append(
            "The supplied first frame is the opening state and the supplied last frame "
            "is the required final state. Use a coherent path between them."
        )
    if document.style_direction.strip():
        description.append(_verbatim("The user-authored visual style direction", document.style_direction))
    description.append(_render_timeline(document, omni=False))
    core = (
        "integrated_multimodal_description:\n"
        + "\n\n".join(description)
        + "\n\noverall_soundscape:\n"
        + document.overall_soundscape.strip()
        + "\n\nnon_diegetic_music:\n"
        + document.non_diegetic_music.strip()
    )
    alignment = _alignment_instruction(document)
    return f"{alignment}\n\n{core}" if alignment else core


def render_ref2va(document: ContextIRDocument) -> str:
    if document.mode is not GenerationMode.OMNI:
        raise ValueError("render_ref2va accepts only omni documents")
    definitions = "\n".join(item.text for item in document.subject_definitions)
    retention = "\n".join(
        f"{item.label}: {item.relationship} - {item.detail}"
        for item in document.retention_entries
    )
    style: list[str] = [
        "This is a deterministic degraded local rewrite: English structural guidance "
        "wraps the user's unchanged original-language instruction."
    ]
    if document.style_direction.strip():
        style.append(_verbatim("The user-authored visual style direction", document.style_direction))
    style.append(_render_timeline(document, omni=True))
    prefix = " + ".join(document.summary_prefix) or "reference generation"
    return (
        "subject_definitions:\n"
        + definitions
        + "\n\nsummary:\n"
        + f"[{prefix}] {document.summary_text.strip()}"
        + "\n\nretention_analysis:\n"
        + retention
        + "\n\ndetailed_description:\n"
        + "\n\n".join(style)
        + "\n\noverall_soundscape:\n"
        + document.overall_soundscape.strip()
        + "\n\nnon_diegetic_music:\n"
        + document.non_diegetic_music.strip()
    )


def render_context_ir(document: ContextIRDocument) -> str:
    if not isinstance(document, ContextIRDocument):
        raise TypeError("document must be a ContextIRDocument")
    if document.mode is GenerationMode.OMNI:
        return render_ref2va(document)
    return render_base(document)


__all__ = ["format_timestamp", "render_base", "render_context_ir", "render_ref2va"]
