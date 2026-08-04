"""Compatibility module exposing the compiler pipeline entry points."""

from .compiler import compile_context_ir, compile_request, write_artifacts
from .model import CompilationResult, CompileRequest

__all__ = [
    "CompilationResult",
    "CompileRequest",
    "compile_context_ir",
    "compile_request",
    "write_artifacts",
]
