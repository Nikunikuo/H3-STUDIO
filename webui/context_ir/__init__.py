"""Public API for H3 Studio's local deterministic Context-IR compiler."""

from .compiler import compile_context_ir, compile_request, write_artifacts
from .model import (
    ArtifactRecord,
    AudioReferencePolicy,
    AutoAdjustment,
    CompilationResult,
    CompilationStatus,
    CompileRequest,
    ContextIRDocument,
    Diagnostic,
    DialogueEvent,
    EmbeddedVideoAudioPolicy,
    GenerationMode,
    MusicPolicy,
    Provenance,
    ReferenceKind,
    ReferenceLabel,
    ReferenceOrigin,
    ReferenceSpec,
    Severity,
    SourceOutfitPolicy,
)
from .parser import OMNI_SECTION_HEADERS, THREE_SECTION_HEADERS, is_h3_context_ir
from .renderer import format_timestamp, render_context_ir
from .validator import has_fatal, validate_document, validate_rendered_ir


__all__ = [
    "ArtifactRecord",
    "AudioReferencePolicy",
    "AutoAdjustment",
    "CompilationResult",
    "CompilationStatus",
    "CompileRequest",
    "ContextIRDocument",
    "Diagnostic",
    "DialogueEvent",
    "EmbeddedVideoAudioPolicy",
    "GenerationMode",
    "MusicPolicy",
    "OMNI_SECTION_HEADERS",
    "Provenance",
    "ReferenceKind",
    "ReferenceLabel",
    "ReferenceOrigin",
    "ReferenceSpec",
    "Severity",
    "SourceOutfitPolicy",
    "THREE_SECTION_HEADERS",
    "compile_context_ir",
    "compile_request",
    "format_timestamp",
    "has_fatal",
    "is_h3_context_ir",
    "render_context_ir",
    "validate_document",
    "validate_rendered_ir",
    "write_artifacts",
]
