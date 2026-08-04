"""Typed data model for the deterministic H3 Context-IR compiler."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


COMPILER_VERSION = "0.3.0"
H3_GUIDE_REVISION = "5d9b308a59ab12e67147f191e184baf704185bd1"
CLI_GUIDE_REVISION = "7ba4460dbd4af24b6cdc6561d3fd6cbb5cd0dfdc"
BASE_GUIDE_SHA256 = "2cfebc096a6e08370f288d468d90b60f7f9bcb938f94bf090816e910e48e75fc"
REF_GUIDE_SHA256 = "1e574f356716ad55612247ffb7bbccbcdb484ad96599d63c7dca1af186b1fab7"


class GenerationMode(str, Enum):
    T2V = "t2v"
    I2V = "i2v"
    FIRST_LAST = "first_last"
    OMNI = "omni"


class MusicPolicy(str, Enum):
    AUTO = "auto"
    NONE = "none"
    SUBTLE = "subtle"
    PROMINENT = "prominent"


class SourceOutfitPolicy(str, Enum):
    REPLACE_IF_SPECIFIED = "replace_if_specified"
    PRESERVE = "preserve"
    AS_PROMPTED = "as_prompted"


class AudioReferencePolicy(str, Enum):
    TIMBRE = "timbre"
    REUSE = "reuse"
    AS_PROMPTED = "as_prompted"


class EmbeddedVideoAudioPolicy(str, Enum):
    AUTO = "auto"
    IGNORE = "ignore"
    REFERENCE = "reference"
    REUSE = "reuse"


class ReferenceKind(str, Enum):
    PICTURE = "Picture"
    VIDEO = "Video"
    AUDIO = "Audio"


class ReferenceOrigin(str, Enum):
    EXPLICIT = "explicit"
    UPLOADED_IMAGE = "uploaded_image"
    UPLOADED_VIDEO = "uploaded_video"
    EMBEDDED_VIDEO_AUDIO = "embedded_video_audio"
    STANDALONE_AUDIO = "standalone_audio"


class Severity(str, Enum):
    FATAL = "fatal"
    WARNING = "warning"
    NOTE = "note"


class CompilationStatus(str, Enum):
    RAW_VALIDATED = "raw_validated"
    COMPILED = "compiled"
    ADJUSTED = "adjusted"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


_REFERENCE_RE = re.compile(
    r"^<?\s*(?P<kind>Picture|Video|Audio)\s+(?P<index>[1-9][0-9]*)\s*>?$",
    re.IGNORECASE,
)
_LANGUAGE_RE = re.compile(r"^[A-Za-z][A-Za-z -]{0,31}$")


@dataclass(frozen=True, slots=True, order=True)
class ReferenceLabel:
    kind: ReferenceKind
    index: int

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 1:
            raise ValueError("reference index must be a positive integer")

    @classmethod
    def parse(cls, value: str) -> "ReferenceLabel":
        if not isinstance(value, str):
            raise TypeError("reference label must be a string")
        match = _REFERENCE_RE.fullmatch(value.strip())
        if match is None:
            raise ValueError(
                f"invalid reference label {value!r}; expected '<Picture N>', "
                "'<Video N>', or '<Audio N>'"
            )
        canonical_kind = match.group("kind").capitalize()
        return cls(ReferenceKind(canonical_kind), int(match.group("index")))

    @property
    def text(self) -> str:
        return f"<{self.kind.value} {self.index}>"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value.lower(), "index": self.index, "label": self.text}


@dataclass(frozen=True, slots=True)
class ReferenceSpec:
    label: ReferenceLabel
    origin: ReferenceOrigin = ReferenceOrigin.EXPLICIT
    source_index: int | None = None
    audio_policy: AudioReferencePolicy | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.text,
            "kind": self.label.kind.value.lower(),
            "origin": self.origin.value,
            "source_index": self.source_index,
            "audio_policy": self.audio_policy.value if self.audio_policy else None,
        }


@dataclass(frozen=True, slots=True)
class DialogueEvent:
    exact_text: str
    language: str = "Japanese"
    speaker_id: int = 1
    target_shot: int | None = None
    voice_direction: str = ""
    audio_reference: ReferenceLabel | None = None
    speaker_reference: ReferenceLabel | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.exact_text, str) or not self.exact_text.strip():
            raise ValueError("dialogue exact_text must not be blank")
        if not _LANGUAGE_RE.fullmatch(self.language.strip()):
            raise ValueError("dialogue language must be a short English language name")
        if self.speaker_id < 1:
            raise ValueError("speaker_id must be positive")
        if self.target_shot is not None and self.target_shot < 1:
            raise ValueError("target_shot must be positive")

    @property
    def tag(self) -> str:
        return f"<d>[{self.language.strip()}] {self.exact_text.strip()}</d>"


@dataclass(frozen=True, slots=True)
class Shot:
    index: int
    start_ms: int
    source_marker: str | None
    instruction: str
    dialogue_events: tuple[DialogueEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.index < 1:
            raise ValueError("shot index must be positive")
        if self.start_ms < 0:
            raise ValueError("shot start_ms must not be negative")


@dataclass(frozen=True, slots=True)
class SubjectDefinition:
    text: str
    subject_index: int | None = None
    source_label: ReferenceLabel | None = None


@dataclass(frozen=True, slots=True)
class RetentionEntry:
    label: str
    relationship: str
    detail: str


@dataclass(frozen=True, slots=True)
class ContextIRDocument:
    mode: GenerationMode
    duration_ms: int
    alignment_instruction: str | None
    preamble: str
    shots: tuple[Shot, ...]
    references: tuple[ReferenceSpec, ...]
    subject_definitions: tuple[SubjectDefinition, ...]
    retention_entries: tuple[RetentionEntry, ...]
    summary_prefix: tuple[str, ...]
    summary_text: str
    style_direction: str
    overall_soundscape: str
    non_diegetic_music: str
    source_outfit_policy: SourceOutfitPolicy
    audio_reference_policy: AudioReferencePolicy
    embedded_video_audio_policy: EmbeddedVideoAudioPolicy
    degraded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class Diagnostic:
    severity: Severity
    code: str
    message: str
    path: str = ""

    @property
    def fatal(self) -> bool:
        return self.severity is Severity.FATAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "fatal": self.fatal,
        }


@dataclass(frozen=True, slots=True)
class AutoAdjustment:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class Provenance:
    compiler_version: str
    source_sha256: str
    output_sha256: str | None
    h3_guide_revision: str = H3_GUIDE_REVISION
    cli_guide_revision: str = CLI_GUIDE_REVISION
    base_guide_sha256: str = BASE_GUIDE_SHA256
    ref_guide_sha256: str = REF_GUIDE_SHA256
    local_only: bool = True
    model_inference: bool = False
    policy: str = "degraded-but-safe"
    embedded_video_audio_policy: str = EmbeddedVideoAudioPolicy.IGNORE.value

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    relative_path: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class CompilationResult:
    ir_text: str | None
    status: CompilationStatus
    document: ContextIRDocument | None
    diagnostics: tuple[Diagnostic, ...]
    auto_adjustments: tuple[AutoAdjustment, ...]
    provenance: Provenance
    artifacts: tuple[ArtifactRecord, ...] = ()
    embedded_video_audio_indices: tuple[int, ...] = ()

    @property
    def fatal(self) -> bool:
        return any(item.fatal for item in self.diagnostics)

    @property
    def degraded(self) -> bool:
        return self.status is CompilationStatus.DEGRADED or bool(
            self.document and self.document.degraded
        )

    @property
    def embedded_video_audio_policy(self) -> str:
        if self.document is not None:
            return self.document.embedded_video_audio_policy.value
        return self.provenance.embedded_video_audio_policy

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "degraded": self.degraded,
            "fatal": self.fatal,
            "local_only": self.provenance.local_only,
            "model_inference": self.provenance.model_inference,
            "embedded_video_audio_policy": self.embedded_video_audio_policy,
            "embedded_video_audio_indices": list(self.embedded_video_audio_indices),
            "provenance": self.provenance.to_dict(),
            "auto_adjustments": [item.to_dict() for item in self.auto_adjustments],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "artifacts": [item.to_dict() for item in self.artifacts],
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_public_dict()
        value.update(
            ir_text=self.ir_text,
            document=self.document.to_dict() if self.document else None,
        )
        return value

    def with_artifacts(self, artifacts: tuple[ArtifactRecord, ...]) -> "CompilationResult":
        return replace(self, artifacts=artifacts)


@dataclass(frozen=True, slots=True)
class CompileRequest:
    mode: GenerationMode
    prompt: str
    duration_ms: int
    style_direction: str = ""
    dialogue: str = ""
    soundscape: str = ""
    audio_direction: str = ""
    music_policy: MusicPolicy = MusicPolicy.AUTO
    music_direction: str = ""
    references: tuple[ReferenceSpec, ...] = ()
    dialogue_language: str = "Japanese"
    source_outfit_policy: SourceOutfitPolicy = SourceOutfitPolicy.REPLACE_IF_SPECIFIED
    audio_reference_policy: AudioReferencePolicy = AudioReferencePolicy.TIMBRE
    embedded_video_audio_policy: EmbeddedVideoAudioPolicy = EmbeddedVideoAudioPolicy.IGNORE

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must not be blank")
        if isinstance(self.duration_ms, bool) or not isinstance(self.duration_ms, int):
            raise TypeError("duration_ms must be an integer")
        if self.duration_ms <= 0 or self.duration_ms > 24 * 60 * 60 * 1000:
            raise ValueError("duration_ms must be positive and no more than 24 hours")
        if not _LANGUAGE_RE.fullmatch(self.dialogue_language.strip()):
            raise ValueError("dialogue_language must be a short English language name")
        for name in (
            "style_direction",
            "dialogue",
            "soundscape",
            "audio_direction",
            "music_direction",
        ):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0


def stable_sha256(value: Any) -> str:
    if isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(
            _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


__all__ = [
    "ArtifactRecord",
    "AudioReferencePolicy",
    "AutoAdjustment",
    "CLI_GUIDE_REVISION",
    "COMPILER_VERSION",
    "CompilationResult",
    "CompilationStatus",
    "CompileRequest",
    "ContextIRDocument",
    "Diagnostic",
    "DialogueEvent",
    "EmbeddedVideoAudioPolicy",
    "GenerationMode",
    "MusicPolicy",
    "Provenance",
    "ReferenceKind",
    "ReferenceLabel",
    "ReferenceOrigin",
    "ReferenceSpec",
    "RetentionEntry",
    "Severity",
    "Shot",
    "SourceOutfitPolicy",
    "SubjectDefinition",
    "stable_sha256",
]
