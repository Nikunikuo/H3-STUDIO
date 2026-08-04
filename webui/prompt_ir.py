"""Backward-compatible façade for the local H3 Context-IR compiler.

New integrations should call :func:`webui.context_ir.compile_request`, which
returns structured diagnostics and provenance.  This module keeps the original
``PromptIRRequest -> str`` API for existing callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from .context_ir import (
    OMNI_SECTION_HEADERS,
    THREE_SECTION_HEADERS,
    CompilationResult,
    ReferenceLabel,
    compile_request,
    is_h3_context_ir,
)


GenerationMode = Literal["t2v", "i2v", "first_last", "omni"]
MusicPolicy = Literal["auto", "none", "subtle", "prominent"]
SourceOutfitPolicy = Literal["replace_if_specified", "preserve", "as_prompted"]
AudioReferencePolicy = Literal["timbre", "reuse", "as_prompted"]
EmbeddedVideoAudioPolicy = Literal["auto", "ignore", "reference", "reuse"]

SUPPORTED_MODES = frozenset({"t2v", "i2v", "first_last", "omni"})
SUPPORTED_MUSIC_POLICIES = frozenset({"auto", "none", "subtle", "prominent"})
SUPPORTED_OUTFIT_POLICIES = frozenset(
    {"replace_if_specified", "preserve", "as_prompted"}
)
SUPPORTED_AUDIO_REFERENCE_POLICIES = frozenset({"timbre", "reuse", "as_prompted"})
SUPPORTED_EMBEDDED_VIDEO_AUDIO_POLICIES = frozenset(
    {"auto", "ignore", "reference", "reuse"}
)


@dataclass(frozen=True, slots=True)
class PromptIRRequest:
    mode: GenerationMode
    prompt: str
    duration_seconds: float
    style_direction: str = ""
    dialogue: str = ""
    soundscape: str = ""
    audio_direction: str = ""
    music_policy: MusicPolicy = "auto"
    music_direction: str = ""
    reference_labels: tuple[str, ...] = ()
    dialogue_language: str = "Japanese"
    source_outfit_policy: SourceOutfitPolicy = "replace_if_specified"
    audio_reference_policy: AudioReferencePolicy = "timbre"
    embedded_video_audio_policy: EmbeddedVideoAudioPolicy = "ignore"

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(
                f"unsupported mode {self.mode!r}; expected one of: "
                + ", ".join(sorted(SUPPORTED_MODES))
            )
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")
        if not self.prompt.strip():
            raise ValueError("prompt must not be blank")
        if isinstance(self.duration_seconds, bool) or not isinstance(
            self.duration_seconds, (int, float)
        ):
            raise TypeError("duration_seconds must be a finite positive number")
        if not math.isfinite(float(self.duration_seconds)) or self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be a finite positive number")
        if self.duration_seconds > 24 * 60 * 60:
            raise ValueError("duration_seconds must not exceed 24 hours")
        if self.music_policy not in SUPPORTED_MUSIC_POLICIES:
            raise ValueError(f"unsupported music_policy {self.music_policy!r}")
        if self.source_outfit_policy not in SUPPORTED_OUTFIT_POLICIES:
            raise ValueError(
                f"unsupported source_outfit_policy {self.source_outfit_policy!r}"
            )
        if self.audio_reference_policy not in SUPPORTED_AUDIO_REFERENCE_POLICIES:
            raise ValueError(
                f"unsupported audio_reference_policy {self.audio_reference_policy!r}"
            )
        if (
            self.embedded_video_audio_policy
            not in SUPPORTED_EMBEDDED_VIDEO_AUDIO_POLICIES
        ):
            raise ValueError(
                "unsupported embedded_video_audio_policy "
                f"{self.embedded_video_audio_policy!r}"
            )
        if isinstance(self.reference_labels, (str, bytes)) or not isinstance(
            self.reference_labels, Sequence
        ):
            raise TypeError("reference_labels must be a sequence of labels")
        canonical = tuple(ReferenceLabel.parse(value).text for value in self.reference_labels)
        object.__setattr__(self, "reference_labels", canonical)


def compile_prompt_ir_result(request: PromptIRRequest) -> CompilationResult:
    if not isinstance(request, PromptIRRequest):
        raise TypeError("request must be a PromptIRRequest")
    return compile_request(
        {
            "mode": request.mode,
            "prompt": request.prompt,
            "duration_seconds": request.duration_seconds,
            "style_direction": request.style_direction,
            "dialogue": request.dialogue,
            "soundscape": request.soundscape,
            "audio_direction": request.audio_direction,
            "music_policy": request.music_policy,
            "music_direction": request.music_direction,
            "reference_labels": request.reference_labels,
            "dialogue_language": request.dialogue_language,
            "source_outfit_policy": request.source_outfit_policy,
            "audio_reference_policy": request.audio_reference_policy,
            "embedded_video_audio_policy": request.embedded_video_audio_policy,
            "references": (),
        }
    )


def compile_prompt_ir(request: PromptIRRequest) -> str:
    result = compile_prompt_ir_result(request)
    if result.ir_text is None:
        fatal = next((item for item in result.diagnostics if item.fatal), None)
        raise ValueError(fatal.message if fatal else "Context-IR compilation was blocked")
    return result.ir_text


__all__ = [
    "AudioReferencePolicy",
    "EmbeddedVideoAudioPolicy",
    "GenerationMode",
    "MusicPolicy",
    "OMNI_SECTION_HEADERS",
    "PromptIRRequest",
    "SourceOutfitPolicy",
    "THREE_SECTION_HEADERS",
    "compile_prompt_ir",
    "compile_prompt_ir_result",
    "is_h3_context_ir",
]
