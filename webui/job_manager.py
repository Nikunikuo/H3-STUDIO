from __future__ import annotations

import copy
import hashlib
import json
import os
import queue
import re
import subprocess
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from .process_guard import ProcessJob
from .prompt_translation import (
    PromptTranslationError,
    classify_visible_text_literals,
    requires_translation,
    validate_authorized_visible_text_literals,
    validate_native_dialogue_blocks,
)


EVENT_PREFIX = "H3EVENT "
TERMINAL_STATES = {"completed", "failed", "cancelled", "interrupted"}
PROMPT_TRANSLATOR_MODULE = "webui.prompt_translation_worker"
PROMPT_TRANSLATOR_TIMEOUT_SECONDS = 180
PROMPT_TRANSLATOR_MODEL_DIR = Path("models") / "prompt_translator"
PROMPT_TRANSLATOR_LOCK = "prompt_translator.lock.json"
COMMUNITY_PLANNER_MODULE = "webui.community_prompt_worker"
COMMUNITY_PLANNER_TIMEOUT_SECONDS = 240
COMMUNITY_PLANNER_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
COMMUNITY_PLANNER_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
COMMUNITY_PLANNER_CONTRACT = "h3-community-plan-v1"
COMMUNITY_COMPILER_REVISION = "2026-08-05-native-clean-v3-camera-geometry"
COMMUNITY_CACHE_SCHEMA_VERSION = 1
COMMUNITY_PLANNER_LOCK = "prompt_planner.lock.json"
COMMUNITY_PLANNER_PROVENANCE_FILE = "h3-studio-provenance.json"
COMMUNITY_PLANNER_PROVENANCE_SCHEMA_VERSION = 1
COMMUNITY_PLANNER_MODEL_DIR = (
    Path("models") / "prompt_planner" / "Qwen3-4B-Instruct-2507"
)
COMMUNITY_PLANNER_REQUIRED_FILES = {
    "config.json": 727,
    "generation_config.json": 238,
    "LICENSE": 11_343,
    "model.safetensors.index.json": 32_819,
    "model-00001-of-00003.safetensors": 3_957_900_840,
    "model-00002-of-00003.safetensors": 3_987_450_520,
    "model-00003-of-00003.safetensors": 99_630_640,
    "tokenizer.json": 11_422_654,
    "tokenizer_config.json": 9_377,
}
_CJK_TEXT_RE = re.compile(
    "["
    "\u3040-\u30ff"  # Hiragana and Katakana.
    "\u31f0-\u31ff"  # Katakana phonetic extensions.
    "\u3400-\u4dbf"  # CJK Unified Ideographs Extension A.
    "\u4e00-\u9fff"  # CJK Unified Ideographs.
    "\uf900-\ufaff"  # CJK Compatibility Ideographs.
    "\uff66-\uff9f"  # Half-width Katakana.
    "]"
)
_NATIVE_DIALOGUE_RE = re.compile(
    r"<d>\s*\[(?P<language>[A-Za-z][A-Za-z -]{1,31})\]\s*"
    r"(?P<text>.*?)\s*</d>",
    re.IGNORECASE | re.DOTALL,
)
_OPEN_DIALOGUE_RE = re.compile(r"<d>", re.IGNORECASE)
_CLOSE_DIALOGUE_RE = re.compile(r"</d>", re.IGNORECASE)
_REFERENCE_TAG_RE = re.compile(
    r"<(?P<kind>Picture|Video|Audio)\s+(?P<index>[1-9][0-9]*)>",
    re.IGNORECASE,
)
_ALL_REFERENCE_TAG_RE = re.compile(
    r"<(?P<kind>Picture|Video|Audio|Subject) (?P<index>[1-9][0-9]*)>",
)
_NONCANONICAL_CONTROL_TOKEN_RE = re.compile(
    r"<(?:/?[A-Za-z]|\||<|>)[^>\r\n]*(?:>|$)"
)
_OFFICIAL_SECTION_RE = re.compile(
    r"^(?P<header>subject_definitions|summary|retention_analysis|"
    r"detailed_description|integrated_multimodal_description|"
    r"overall_soundscape|non_diegetic_music)[ \t]*:[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_OFFICIAL_SIX_SECTION_HEADERS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
_OFFICIAL_THREE_SECTION_HEADERS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:[\\/]|\\\\)[^\r\n]*"
)
_UNREQUESTED_SPEECH_CUE_RE = re.compile(
    r"\b(?:speech|spoken|speak(?:s|ing)?|said|says?|talk(?:s|ing)?|dialogue|"
    r"narrat(?:e|es|ed|ing|ion|or)|voice(?:[ -]?over)?|voices|vocal(?:s|ization)?|"
    r"language|words?|utterance|greeting)\b",
    re.IGNORECASE,
)
_ORDINARY_QUOTED_TEXT_RE = re.compile(r'"[^"\r\n]+"|“[^”\r\n]+”')
_NATIVE_DIALOGUE_TAG_RE = re.compile(r"</?d(?:\s[^>]*)?>", re.IGNORECASE)
PUBLIC_ENGINE_DIAGNOSTIC_FIELDS = frozenset(
    {
        "comfyui_commit",
        "model_revision",
        "variant",
        "async_offload_streams",
        "attention_backend",
        "workflow_profile",
        "h3_token_ids",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_public_text(value: str) -> str:
    """Remove local absolute paths from browser-visible messages/logs."""

    return _WINDOWS_ABSOLUTE_PATH_RE.sub("[local path redacted]", value)


def prompt_translator_status(root: Path) -> dict[str, Any]:
    """Return a cheap, public-safe readiness snapshot for the pinned model.

    SHA-256 verification belongs to setup/explicit verification commands.  The
    frequently-polled capability endpoint only verifies that every lock-file
    entry resolves beneath ``models/prompt_translator`` and has the pinned byte
    size.
    """

    resolved_root = root.resolve()
    lock_path = resolved_root / PROMPT_TRANSLATOR_LOCK
    public: dict[str, Any] = {
        "ready": False,
        "status": "lock_missing",
        "model": None,
        "repo_id": None,
        "revision": None,
        "local_only": True,
        "model_inference": True,
        "total_bytes": None,
        "missing_files": [],
        "invalid_files": [],
    }
    if not lock_path.is_file():
        return public
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        public["status"] = "lock_invalid"
        return public

    source = lock.get("source")
    if isinstance(source, Mapping):
        repo_id = str(source.get("repo_id") or "").strip() or None
        revision = str(source.get("revision") or "").strip() or None
        public.update(model=repo_id, repo_id=repo_id, revision=revision)
    verification = lock.get("verification")
    if isinstance(verification, Mapping):
        try:
            public["total_bytes"] = int(verification.get("total_bytes"))
        except (TypeError, ValueError):
            pass

    expected_root = (resolved_root / PROMPT_TRANSLATOR_MODEL_DIR).resolve()
    entries = lock.get("files")
    if not isinstance(entries, list) or not entries:
        public["status"] = "lock_invalid"
        return public

    missing: list[str] = []
    invalid: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            invalid.append("<invalid-entry>")
            continue
        relative_name = str(entry.get("path") or "").replace("\\", "/")
        try:
            expected_size = int(entry.get("size"))
            candidate = (resolved_root / relative_name).resolve()
            candidate.relative_to(expected_root)
        except (TypeError, ValueError, OSError):
            invalid.append(relative_name or "<missing-path>")
            continue
        if not candidate.is_file():
            missing.append(relative_name)
            continue
        try:
            actual_size = candidate.stat().st_size
        except OSError:
            missing.append(relative_name)
            continue
        if actual_size != expected_size:
            invalid.append(relative_name)

    public["missing_files"] = missing
    public["invalid_files"] = invalid
    if not public["repo_id"] or not public["revision"]:
        public["status"] = "lock_invalid"
    elif invalid:
        public["status"] = "model_invalid"
    elif missing:
        public["status"] = "model_incomplete"
    else:
        public.update(ready=True, status="ready")
    return public


def community_planner_status(root: Path) -> dict[str, Any]:
    """Return a cheap readiness check for the pinned text-only Qwen planner.

    The nine runtime files are intentionally checked by byte size here so the
    frequently-polled capabilities endpoint remains cheap.  Authenticity and
    revision provenance are bound to the repository lock by a small marker
    written only after the normal setup command hashes all nine files.  Do not
    rely on Hugging Face's private ``.cache`` layout: users and cache cleaners
    may legitimately remove it after setup.
    """

    resolved_root = root.resolve()
    model_root = (resolved_root / COMMUNITY_PLANNER_MODEL_DIR).resolve()
    expected_parent = (resolved_root / "models" / "prompt_planner").resolve()
    public: dict[str, Any] = {
        "ready": False,
        "status": "model_incomplete",
        "model": COMMUNITY_PLANNER_MODEL_ID,
        "repo_id": COMMUNITY_PLANNER_MODEL_ID,
        "revision": COMMUNITY_PLANNER_REVISION,
        "local_only": True,
        "model_inference": True,
        "total_bytes": sum(COMMUNITY_PLANNER_REQUIRED_FILES.values()),
        "missing_files": [],
        "invalid_files": [],
    }
    try:
        model_root.relative_to(expected_parent)
    except ValueError:
        public["status"] = "model_path_invalid"
        return public
    missing: list[str] = []
    invalid: list[str] = []
    for name, size in COMMUNITY_PLANNER_REQUIRED_FILES.items():
        candidate = model_root / name
        if not candidate.is_file():
            missing.append(name)
            continue
        try:
            if candidate.stat().st_size != size:
                invalid.append(name)
        except OSError:
            missing.append(name)
    public["missing_files"] = missing
    public["invalid_files"] = invalid
    if invalid:
        public["status"] = "model_invalid"
    elif missing:
        public["status"] = "model_incomplete"
    else:
        lock_path = resolved_root / COMMUNITY_PLANNER_LOCK
        if not lock_path.is_file():
            public["status"] = "lock_missing"
            return public
        try:
            lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        except OSError:
            public["status"] = "lock_invalid"
            return public

        provenance_path = model_root / COMMUNITY_PLANNER_PROVENANCE_FILE
        if not provenance_path.is_file():
            public["status"] = "provenance_missing"
            return public
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            public["status"] = "provenance_invalid"
            return public

        expected_provenance = {
            "schema_version": COMMUNITY_PLANNER_PROVENANCE_SCHEMA_VERSION,
            "model_id": COMMUNITY_PLANNER_MODEL_ID,
            "revision": COMMUNITY_PLANNER_REVISION,
            "lock_sha256": lock_sha256,
            "file_count": len(COMMUNITY_PLANNER_REQUIRED_FILES),
            "total_bytes": sum(COMMUNITY_PLANNER_REQUIRED_FILES.values()),
        }
        provenance_valid = (
            isinstance(provenance, Mapping)
            and set(provenance) == set(expected_provenance)
            and type(provenance.get("schema_version")) is int
            and type(provenance.get("file_count")) is int
            and type(provenance.get("total_bytes")) is int
            and all(provenance.get(key) == value for key, value in expected_provenance.items())
        )
        if not provenance_valid:
            public["status"] = "provenance_invalid"
            return public
        public.update(ready=True, status="ready")
    return public


def _has_cjk_outside_native_dialogue(prompt: str) -> bool:
    without_dialogue = _NATIVE_DIALOGUE_RE.sub("", prompt)
    return _CJK_TEXT_RE.search(without_dialogue) is not None


def _dialogue_events(prompt: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            match.group("language").strip(),
            match.group("text"),
        )
        for match in _NATIVE_DIALOGUE_RE.finditer(prompt)
    )


def _reference_tags(prompt: str) -> tuple[tuple[str, int], ...]:
    return tuple(
        (match.group("kind").casefold(), int(match.group("index")))
        for match in _REFERENCE_TAG_RE.finditer(prompt)
    )


def _translator_reference_manifest(references: list[Any]) -> list[dict[str, Any]]:
    """Expose only stable reference ordinals; never send upload paths/names."""

    ordinals = {"image": 0, "video": 0, "audio": 0}
    tag_kind = {"image": "Picture", "video": "Video", "audio": "Audio"}
    manifest: list[dict[str, Any]] = []
    for source_index, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            continue
        kind = str(reference.get("kind") or "").casefold()
        if kind not in ordinals:
            continue
        ordinals[kind] += 1
        item: dict[str, Any] = {
            "kind": kind,
            "index": ordinals[kind],
            "source_index": source_index,
            "tag": f"<{tag_kind[kind]} {ordinals[kind]}>",
        }
        if kind == "video" and isinstance(reference.get("has_audio"), bool):
            item["has_audio"] = reference["has_audio"]
        manifest.append(item)
    return manifest


def _validate_direct_reference_tags(
    prompt: str,
    *,
    mode: str,
    references: list[Any],
) -> list[dict[str, Any]]:
    """Read-only allowlist check for reference labels in a direct prompt."""

    scrubbed = _ALL_REFERENCE_TAG_RE.sub("", prompt)
    scrubbed = scrubbed.replace("<d>", "").replace("</d>", "")
    if _NONCANONICAL_CONTROL_TOKEN_RE.search(scrubbed):
        return [
            {
                "severity": "error",
                "code": "PROMPT_INVALID_CONTROL_TAG",
                "message": (
                    "H3制御タグの綴り・大文字小文字・空白が公式形式ではないか、"
                    "未対応のangle tagが含まれています。"
                ),
                "fatal": True,
            }
        ]

    observed = {
        (match.group("kind").casefold(), int(match.group("index")))
        for match in _ALL_REFERENCE_TAG_RE.finditer(prompt)
    }
    if mode == "i2v":
        allowed: set[tuple[str, int]] = {("picture", 1)}
    elif mode == "first_last":
        allowed = {("picture", 1), ("picture", 2)}
    elif mode == "omni":
        allowed = {
            (str(item["kind"]).casefold(), int(item["index"]))
            for item in _translator_reference_manifest(references)
        }
        # Manifest kind names are upload kinds while H3 uses Picture labels.
        allowed = {
            ({"image": "picture", "video": "video", "audio": "audio"}[kind], index)
            for kind, index in allowed
        }
    else:
        allowed = set()

    # The formatter introduces <Subject N> only as a stable identity alias for
    # the character in <Picture N>. It must never create a new input ordinal.
    allowed.update(
        ("subject", index)
        for kind, index in tuple(allowed)
        if kind == "picture"
    )
    if observed <= allowed:
        return []
    return [
        {
            "severity": "error",
            "code": "PROMPT_REFERENCE_OUT_OF_RANGE",
            "message": "プロンプトに、添付素材または生成モードへ対応しない参照タグがあります。",
            "fatal": True,
        }
    ]


def _validate_compiled_prompt(
    source: str,
    compiled: str,
    *,
    mode: str,
    references: list[Any],
    compiler_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Validate the worker's complete H3 prompt before GPU inference."""

    diagnostics: list[dict[str, Any]] = []

    def fail(code: str, message: str) -> None:
        diagnostics.append(
            {
                "severity": "error",
                "code": code,
                "message": message,
                "fatal": True,
            }
        )

    if not compiled.strip():
        fail("TRANSLATOR_EMPTY_OUTPUT", "変換後のH3プロンプトが空です。")
        return diagnostics

    section_matches = list(_OFFICIAL_SECTION_RE.finditer(compiled))
    headers = tuple(match.group("header").casefold() for match in section_matches)
    expected_headers = (
        _OFFICIAL_SIX_SECTION_HEADERS
        if mode == "omni"
        else _OFFICIAL_THREE_SECTION_HEADERS
    )
    sections_valid = headers == expected_headers
    if not sections_valid:
        fail(
            "TRANSLATOR_INVALID_SECTIONS",
            "変換結果が生成モードに対応するH3公式セクション構造になっていません。",
        )
    else:
        for index, match in enumerate(section_matches):
            end = section_matches[index + 1].start() if index + 1 < len(section_matches) else len(compiled)
            if not compiled[match.end() : end].strip():
                fail(
                    "TRANSLATOR_EMPTY_SECTION",
                    f"{headers[index]} セクションが空です。",
                )

    metadata = compiler_metadata if isinstance(compiler_metadata, Mapping) else {}
    raw_literals = metadata.get("visible_text_literals", [])
    raw_hashes = metadata.get("visible_text_literal_sha256", [])
    visible_literals: tuple[str, ...] = ()
    visible_metadata_valid = True
    if not isinstance(raw_literals, list) or not all(
        isinstance(item, str) for item in raw_literals
    ):
        visible_metadata_valid = False
    if not isinstance(raw_hashes, list) or not all(
        isinstance(item, str) for item in raw_hashes
    ):
        visible_metadata_valid = False
    if visible_metadata_valid:
        visible_literals = tuple(raw_literals)
        expected_hashes = tuple(
            hashlib.sha256(item.encode("utf-8")).hexdigest()
            for item in visible_literals
        )
        if tuple(raw_hashes) != expected_hashes:
            visible_metadata_valid = False
    if not visible_metadata_valid:
        fail(
            "TRANSLATOR_VISIBLE_TEXT_METADATA_INVALID",
            "画面内文字の検証情報（個数・順序・hash）が不正です。",
        )
        visible_literals = ()

    try:
        source_dialogue = validate_native_dialogue_blocks(source)
    except PromptTranslationError as exc:
        fail(exc.code, str(exc))
        source_dialogue = ()
    try:
        compiled_dialogue = validate_native_dialogue_blocks(compiled)
    except PromptTranslationError as exc:
        fail(exc.code, str(exc))
        compiled_dialogue = ()
    if source_dialogue != compiled_dialogue:
        fail(
            "TRANSLATOR_DIALOGUE_CHANGED",
            "変換処理が指定台詞を変更・削除・追加したため生成を停止しました。",
        )
    if not source_dialogue and sections_valid:
        soundscape_index = headers.index("overall_soundscape")
        soundscape_match = section_matches[soundscape_index]
        soundscape_end = (
            section_matches[soundscape_index + 1].start()
            if soundscape_index + 1 < len(section_matches)
            else len(compiled)
        )
        soundscape = compiled[soundscape_match.end() : soundscape_end]
        if _UNREQUESTED_SPEECH_CUE_RE.search(soundscape):
            fail(
                "TRANSLATOR_UNREQUESTED_SPEECH_AUDIO",
                "台詞指定のない生成で、音響欄へ発話・ナレーション指示が混入しました。",
            )
    try:
        validate_authorized_visible_text_literals(compiled, visible_literals)
    except PromptTranslationError as exc:
        fail(
            (
                "TRANSLATOR_CJK_OUTSIDE_DIALOGUE"
                if exc.code == "UNTRANSLATED_JAPANESE"
                else exc.code
            ),
            str(exc),
        )
    source_tags = set(_reference_tags(source))
    compiled_tags = set(_reference_tags(compiled))
    missing_source_tags = source_tags - compiled_tags
    if missing_source_tags:
        fail(
            "TRANSLATOR_REFERENCE_CHANGED",
            "変換処理が入力内の参照素材タグを削除または変更したため生成を停止しました。",
        )

    expected_tags: set[tuple[str, int]]
    if mode == "omni":
        tag_kind = {"image": "picture", "video": "video", "audio": "audio"}
        expected_tags = {
            (tag_kind[item["kind"]], int(item["index"]))
            for item in _translator_reference_manifest(references)
        }
    elif mode == "i2v":
        expected_tags = {("picture", 1)}
    elif mode == "first_last":
        expected_tags = {("picture", 1), ("picture", 2)}
    else:
        expected_tags = set()

    unexpected_tags = compiled_tags - expected_tags
    if unexpected_tags:
        fail(
            "TRANSLATOR_REFERENCE_OUT_OF_RANGE",
            "変換結果に、この生成モードの入力へ対応しない参照素材タグがあります。",
        )
    if mode == "omni" and headers == _OFFICIAL_SIX_SECTION_HEADERS:
        subject_end = section_matches[1].start()
        subject_tags = set(_reference_tags(compiled[section_matches[0].end() : subject_end]))
        if expected_tags - subject_tags:
            fail(
                "TRANSLATOR_REFERENCE_DEFINITION_MISSING",
                "添付したOmni参照素材の一部がsubject_definitionsに定義されていません。",
            )
    return diagnostics


def _validate_native_clean_prompt(
    prompt: str,
    *,
    mode: str,
    references: list[Any],
) -> list[dict[str, Any]]:
    """Validate the public Comfy prompt contract without rewriting a byte."""

    diagnostics: list[dict[str, Any]] = []

    def fail(code: str, message: str) -> None:
        diagnostics.append(
            {
                "severity": "error",
                "code": code,
                "message": message,
                "fatal": True,
            }
        )

    if not isinstance(prompt, str) or not prompt.strip():
        fail("NATIVE_CLEAN_EMPTY_PROMPT", "H3へ送る英語プロンプトが空です。")
        return diagnostics
    if _NATIVE_DIALOGUE_TAG_RE.search(prompt):
        fail(
            "NATIVE_CLEAN_DIALOGUE_TAG",
            "公開Comfy互換経路では<d>を使いません。実台詞は普通の二重引用符で囲んでください。",
        )
    if "<|" in prompt or "|>" in prompt:
        fail(
            "NATIVE_CLEAN_RESERVED_TOKEN",
            "公開Comfy互換経路へ予約tokenを入力できません。",
        )
    outside_quotes = _ORDINARY_QUOTED_TEXT_RE.sub("", prompt)
    if _CJK_TEXT_RE.search(outside_quotes):
        fail(
            "NATIVE_CLEAN_CJK_CONTROL_PROSE",
            "H3へ送る映像・カメラ・音響の制御文に日本語が残っています。"
            "日本語は普通の二重引用符内の実台詞だけにしてください。",
        )
    diagnostics.extend(
        _validate_direct_reference_tags(
            prompt,
            mode=mode,
            references=references,
        )
    )
    return diagnostics


def _validated_community_cache_response(
    envelope: Any,
    *,
    cache_key: str,
    mode: str,
    references: list[Any],
) -> Mapping[str, Any] | None:
    """Return a fully validated planner response from one versioned cache envelope.

    Cache files are optimization hints, not trusted compiler output.  Old,
    partial, manually edited, or previously rejected entries are cache misses.
    """

    if not isinstance(envelope, Mapping):
        return None
    if envelope.get("schema_version") != COMMUNITY_CACHE_SCHEMA_VERSION:
        return None
    if envelope.get("cache_key") != cache_key:
        return None
    if envelope.get("compiler_revision") != COMMUNITY_COMPILER_REVISION:
        return None
    if envelope.get("planner_contract") != COMMUNITY_PLANNER_CONTRACT:
        return None
    if envelope.get("planner_model") != COMMUNITY_PLANNER_MODEL_ID:
        return None
    if envelope.get("planner_revision") != COMMUNITY_PLANNER_REVISION:
        return None

    response = envelope.get("response")
    if not isinstance(response, Mapping) or response.get("ok") is not True:
        return None
    compiled_prompt = response.get("compiled_prompt")
    if not isinstance(compiled_prompt, str) or not compiled_prompt.strip():
        return None
    if envelope.get("compiled_sha256") != hashlib.sha256(
        compiled_prompt.encode("utf-8")
    ).hexdigest():
        return None
    if not isinstance(response.get("plan"), Mapping):
        return None
    metadata = response.get("planner_metadata")
    if not isinstance(metadata, Mapping):
        return None
    if metadata.get("contract") != COMMUNITY_PLANNER_CONTRACT:
        return None
    if metadata.get("model_id") != COMMUNITY_PLANNER_MODEL_ID:
        return None
    if metadata.get("model_revision") != COMMUNITY_PLANNER_REVISION:
        return None
    diagnostics = response.get("diagnostics")
    if not isinstance(diagnostics, list):
        return None
    if any(
        isinstance(item, Mapping)
        and (
            bool(item.get("fatal"))
            or str(item.get("severity") or "").casefold() in {"error", "fatal"}
        )
        for item in diagnostics
    ):
        return None
    if _validate_native_clean_prompt(
        compiled_prompt,
        mode=mode,
        references=references,
    ):
        return None
    return response


class JobManager:
    """Runs one GPU generation job at a time and persists its visible state."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.data_dir = self.root / "webui_data"
        self.jobs_dir = self.data_dir / "jobs"
        self.outputs_dir = self.root / "outputs" / "webui"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: dict[str, dict[str, Any]] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.RLock()
        self._runner: threading.Thread | None = None
        self._stop = threading.Event()
        self._current_job_id: str | None = None
        self._current_process: subprocess.Popen[str] | None = None
        self._current_process_job: ProcessJob | None = None
        self._engine_variant: str | None = None
        self._current_translation_process: subprocess.Popen[str] | None = None
        self._current_translation_process_job: ProcessJob | None = None
        self._current_translation_job_id: str | None = None
        self._load_existing_jobs()

    def _load_existing_jobs(self) -> None:
        for metadata_path in self.jobs_dir.glob("*/job.json"):
            try:
                job = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("status") not in TERMINAL_STATES:
                job.update(
                    status="interrupted",
                    phase="停止しました",
                    message="Web UIの前回終了により処理が中断されました。もう一度生成してください。",
                    finished_at=utc_now(),
                    progress_updated_at=utc_now(),
                )
                self._save_job(job)
            self._jobs[job["id"]] = job

    def start(self) -> None:
        with self._lock:
            if self._runner and self._runner.is_alive():
                return
            self._stop.clear()
            self._runner = threading.Thread(target=self._run_loop, name="h3-job-runner", daemon=True)
            self._runner.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put("")
        with self._lock:
            owned_translation = self._claim_current_translation_locked()
            owned_engine = self._claim_current_process_locked()
        if owned_translation != (None, None):
            self._cleanup_owned_process(*owned_translation)
        if owned_engine != (None, None):
            self._cleanup_owned_process(*owned_engine)

    def submit(self, job: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._jobs[job["id"]] = job
            self._save_job(job)
            self._queue.put(job["id"])
            return copy.deepcopy(job)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [copy.deepcopy(job) for job in self._jobs.values()]
        return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def current_job_id(self) -> str | None:
        with self._lock:
            return self._current_job_id

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        owned_process: tuple[subprocess.Popen[str] | None, ProcessJob | None] = (None, None)
        owned_translation: tuple[subprocess.Popen[str] | None, ProcessJob | None] = (
            None,
            None,
        )
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.get("status") in TERMINAL_STATES:
                return copy.deepcopy(job) if job else None
            job.update(
                status="cancelled",
                phase="キャンセルしました",
                message="生成をキャンセルしました。モデルや重みは変更されていません。",
                finished_at=utc_now(),
                progress_updated_at=utc_now(),
            )
            self._save_job(job)
            is_current = self._current_job_id == job_id
            if is_current:
                # Claim this job's exact engine/translator in the same critical
                # section as the job-id comparison. A completed old cancel must
                # never kill work installed for the next queued job.
                owned_process = self._claim_current_process_locked()
                owned_translation = self._claim_current_translation_locked(job_id=job_id)
        if owned_translation != (None, None):
            self._cleanup_owned_process(*owned_translation)
        if owned_process != (None, None):
            self._cleanup_owned_process(*owned_process)
        return self.get_job(job_id)

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            job_id = self._queue.get()
            if not job_id or self._stop.is_set():
                continue
            with self._lock:
                job = self._jobs.get(job_id)
                if not job or job.get("status") == "cancelled":
                    continue
                self._current_job_id = job_id
                job.update(
                    status="running",
                    phase="生成を開始しています",
                    message="ローカル生成エンジンを起動しています。",
                    progress=1,
                    started_at=utc_now(),
                    progress_updated_at=utc_now(),
                )
                self._save_job(job)

            request_path = self.jobs_dir / job_id / "request.json"
            try:
                execution_request_path = self._prepare_effective_prompt(job_id, request_path)
                if execution_request_path is None:
                    continue
                with self._lock:
                    current = self._jobs.get(job_id)
                    if not current or current.get("status") == "cancelled":
                        continue
                requested_variant = "ref2va" if job.get("mode") == "omni" else "fl2va"
                process = self._ensure_engine(requested_variant)
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write(
                    json.dumps(
                        {"request": os.fspath(execution_request_path)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                process.stdin.flush()
                terminal_status = None
                while not self._stop.is_set():
                    raw_line = process.stdout.readline()
                    if not raw_line:
                        return_code = process.poll()
                        raise RuntimeError(f"生成エンジンが終了コード {return_code} で停止しました。")
                    line = raw_line.rstrip("\r\n")
                    if line.startswith(EVENT_PREFIX):
                        payload = line[len(EVENT_PREFIX) :]
                        self._handle_event(job_id, payload)
                        try:
                            event = json.loads(payload)
                            if event.get("status") in TERMINAL_STATES:
                                terminal_status = event["status"]
                                break
                        except json.JSONDecodeError:
                            pass
                    elif line.strip():
                        self._append_log(job_id, line.strip())
                if terminal_status != "completed":
                    self._terminate_current_process()
            except Exception as exc:
                with self._lock:
                    current = self._jobs.get(job_id)
                    if current and current.get("status") not in TERMINAL_STATES:
                        current.update(
                            status="failed",
                            phase="生成に失敗しました",
                            message=f"生成エンジンを起動できませんでした: {exc}",
                            finished_at=utc_now(),
                            progress_updated_at=utc_now(),
                        )
                        self._save_job(current)
                self._terminate_current_process()
            finally:
                owned_process: tuple[subprocess.Popen[str] | None, ProcessJob | None] = (None, None)
                with self._lock:
                    if self._current_process and self._current_process.poll() is not None:
                        owned_process = self._claim_current_process_locked()
                    self._current_job_id = None
                self._cleanup_owned_process(*owned_process)

    def _update_prompt_progress(
        self,
        job_id: str,
        *,
        progress: float,
        phase: str,
        message: str,
    ) -> bool:
        """Publish a monotonic pre-generation update without reviving a job."""

        with self._lock:
            current = self._jobs.get(job_id)
            if not current or current.get("status") in TERMINAL_STATES:
                return False
            previous = float(current.get("progress") or 0)
            current.update(
                phase=phase,
                message=message,
                progress=max(previous, progress),
                progress_updated_at=utc_now(),
            )
            self._save_job(current)
            return True

    @staticmethod
    def _worker_json(stdout: str) -> Mapping[str, Any]:
        payload = stdout.strip()
        if not payload:
            raise ValueError("prompt translator returned no JSON")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as whole_error:
            # The contract is one JSON document on stdout, but accepting a
            # final compact JSON line makes the boundary robust to a dependency
            # that writes a harmless startup notice before the result.
            decoded = None
            for line in reversed(payload.splitlines()):
                try:
                    decoded = json.loads(line)
                except json.JSONDecodeError:
                    continue
                break
            if decoded is None:
                raise ValueError("prompt translator returned malformed JSON") from whole_error
        if not isinstance(decoded, Mapping):
            raise ValueError("prompt translator JSON must be an object")
        return decoded

    @staticmethod
    def _public_worker_metadata(value: Any, *, depth: int = 0) -> Any:
        """Keep useful provenance while excluding local paths and debug dumps."""

        if depth > 3:
            return None
        if isinstance(value, str):
            return _redact_public_text(value)
        if value is None or isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, Mapping):
            public: dict[str, Any] = {}
            for raw_key, raw_value in value.items():
                key = str(raw_key)
                folded = key.casefold()
                if any(
                    token in folded
                    for token in ("path", "stdout", "stderr", "traceback", "exception")
                ):
                    continue
                converted = JobManager._public_worker_metadata(raw_value, depth=depth + 1)
                if converted is not None:
                    public[key] = converted
            return public
        if isinstance(value, (list, tuple)):
            return [
                converted
                for item in value
                if (converted := JobManager._public_worker_metadata(item, depth=depth + 1))
                is not None
            ]
        return str(value)

    def _write_prompt_artifacts(
        self,
        job_dir: Path,
        *,
        original_prompt: str,
        source_prompt: str,
        final_prompt: str | None,
        report: Mapping[str, Any],
    ) -> None:
        artifact_dir = job_dir / "prompt_processing"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "original_prompt.txt").write_text(original_prompt, encoding="utf-8")
        (artifact_dir / "source_prompt.txt").write_text(source_prompt, encoding="utf-8")
        final_path = artifact_dir / "final_prompt.txt"
        if final_prompt is None:
            final_path.unlink(missing_ok=True)
        else:
            final_path.write_text(final_prompt, encoding="utf-8")
        self._write_json_atomic(artifact_dir / "report.json", report)

    def _fail_prompt_translation(
        self,
        job_id: str,
        request: Mapping[str, Any],
        job_dir: Path,
        source_prompt: str,
        processing: Mapping[str, Any],
        *,
        code: str,
        detail: str,
        started: float,
        extra_diagnostics: list[Any] | None = None,
        translator: Mapping[str, Any] | None = None,
        translation_required: bool = True,
    ) -> None:
        """Persist an auditable failure and guarantee no stale execution copy."""

        execution_path = job_dir / "execution_request.json"
        execution_path.unlink(missing_ok=True)
        private_dir = job_dir / "prompt_processing"
        private_dir.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(
            private_dir / "translator_failure.private.json",
            {
                "code": code,
                "detail": detail,
                "diagnostics": list(extra_diagnostics or []),
                "translator": dict(translator or {}),
            },
        )
        public_detail = _redact_public_text(detail)
        diagnostics = list(processing.get("diagnostics") or [])
        public_extra = self._public_worker_metadata(extra_diagnostics or [])
        if isinstance(public_extra, list):
            diagnostics.extend(public_extra)
        diagnostics.append(
            {
                "severity": "error",
                "code": code,
                "message": public_detail,
                "fatal": True,
            }
        )
        model_status = (
            dict(translator)
            if isinstance(translator, Mapping)
            else prompt_translator_status(self.root)
        )
        report = dict(processing)
        report.update(
            status=(
                "translation_failed"
                if translation_required
                else "prompt_compilation_failed"
            ),
            mode="raw_guarded",
            context_ir=False,
            translation_required=translation_required,
            translation_status=("failed" if translation_required else "not_required"),
            local_only=True,
            model_inference=translation_required,
            model_repo=model_status.get("repo_id"),
            model_revision=model_status.get("revision"),
            source_sha256=hashlib.sha256(source_prompt.encode("utf-8")).hexdigest(),
            request_source_sha256=hashlib.sha256(
                str(request.get("prompt") or "").encode("utf-8")
            ).hexdigest(),
            output_sha256=None,
            elapsed_ms=round((time.monotonic() - started) * 1000, 2),
            error_code=code,
            diagnostics=diagnostics,
            artifacts=[
                "prompt_processing/original_prompt.txt",
                "prompt_processing/source_prompt.txt",
                "prompt_processing/report.json",
                "prompt_processing/translator_failure.private.json",
            ],
        )
        if translator:
            report["translator"] = self._public_worker_metadata(translator)
        self._write_prompt_artifacts(
            job_dir,
            original_prompt=str(request.get("prompt") or ""),
            source_prompt=source_prompt,
            final_prompt=None,
            report=report,
        )
        if code == "TRANSLATOR_MODEL_NOT_READY":
            public_message = (
                "この日本語プロンプトには任意のローカル翻訳モデルが必要です。"
                "翻訳モデルを含めてセットアップを再実行してから、もう一度生成してください。"
            )
        elif code == "TRANSLATOR_RUNTIME_MISSING":
            public_message = (
                "日本語プロンプト用のローカル変換環境が未準備です。"
                "セットアップを再実行してから、もう一度生成してください。"
            )
        else:
            public_message = (
                "プロンプトを安全なH3公式形式へ準備できなかったため、"
                f"生成を開始しませんでした（{code}）。"
            )
        with self._lock:
            current = self._jobs.get(job_id)
            if current and current.get("status") not in TERMINAL_STATES:
                current.update(
                    status="failed",
                    phase="プロンプト変換に失敗しました",
                    message=public_message,
                    compiler=report,
                    prompt_processing=report,
                    auto_adjustments=report.get("auto_adjustments", []),
                    diagnostics=diagnostics,
                    finished_at=utc_now(),
                    progress_updated_at=utc_now(),
                )
                self._save_job(current)
        self._append_log(
            job_id,
            f"Prompt translator blocked generation: {code}: {public_detail}",
        )

    def _run_tracked_prompt_translator(
        self,
        job_id: str,
        command: list[str],
        input_text: str,
        env: Mapping[str, str],
        *,
        timeout: float,
    ) -> tuple[int, str, str] | None:
        """Run one cancellable translator process owned by ``job_id``.

        ``Popen.communicate`` runs in a short-lived reader thread so stdout and
        stderr are drained without deadlocking while the runner remains able to
        observe cancellation, manager shutdown, and the wall-clock timeout.
        Ownership is installed and claimed under ``_lock``; a late cancel for
        an old job can therefore never terminate a translator belonging to the
        next queued job.
        """

        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process_job = ProcessJob()
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=self.root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=dict(env),
                creationflags=creation_flags,
            )
            process_job.attach(process)
        except Exception:
            process_job.terminate()
            self._cleanup_owned_process(process, None)
            raise

        installed = False
        with self._lock:
            current = self._jobs.get(job_id)
            eligible = (
                not self._stop.is_set()
                and current is not None
                and current.get("status") not in TERMINAL_STATES
                and self._current_job_id == job_id
            )
            if eligible and self._current_translation_process is None:
                self._current_translation_process = process
                self._current_translation_process_job = process_job
                self._current_translation_job_id = job_id
                installed = True
        if not installed:
            self._cleanup_owned_process(process, process_job)
            return None

        communication: dict[str, Any] = {}
        communication_done = threading.Event()

        def communicate() -> None:
            try:
                stdout, stderr = process.communicate(input=input_text)
                communication["stdout"] = stdout or ""
                communication["stderr"] = stderr or ""
            except Exception as exc:  # pragma: no cover - platform pipe failures vary
                communication["error"] = exc
            finally:
                communication_done.set()

        reader = threading.Thread(
            target=communicate,
            name=f"h3-prompt-translator-{job_id}",
            daemon=True,
        )
        reader.start()
        deadline = time.monotonic() + timeout
        while not communication_done.wait(0.05):
            with self._lock:
                current = self._jobs.get(job_id)
                still_owned = (
                    self._current_translation_process is process
                    and self._current_translation_job_id == job_id
                )
                abort_requested = (
                    self._stop.is_set()
                    or not current
                    or current.get("status") in TERMINAL_STATES
                    or self._current_job_id != job_id
                    or not still_owned
                )
            if abort_requested:
                with self._lock:
                    owned = self._claim_current_translation_locked(
                        job_id=job_id,
                        process=process,
                    )
                self._cleanup_owned_process(*owned)
                reader.join(timeout=6)
                return None
            if time.monotonic() >= deadline:
                with self._lock:
                    owned = self._claim_current_translation_locked(
                        job_id=job_id,
                        process=process,
                    )
                self._cleanup_owned_process(*owned)
                reader.join(timeout=6)
                with self._lock:
                    current = self._jobs.get(job_id)
                    cancelled = (
                        self._stop.is_set()
                        or not current
                        or current.get("status") in TERMINAL_STATES
                        or self._current_job_id != job_id
                    )
                if cancelled:
                    return None
                raise subprocess.TimeoutExpired(command, timeout)

        with self._lock:
            owned = self._claim_current_translation_locked(
                job_id=job_id,
                process=process,
            )
            current = self._jobs.get(job_id)
            cancelled = (
                self._stop.is_set()
                or not current
                or current.get("status") in TERMINAL_STATES
                or self._current_job_id != job_id
            )
        if owned[0] is None:
            # cancel()/stop() already owns and reaps this exact process.
            reader.join(timeout=6)
            return None
        try:
            if cancelled:
                return None
            error = communication.get("error")
            if error is not None:
                raise OSError(
                    "ローカル翻訳ワーカーとの入出力に失敗しました: "
                    f"{error.__class__.__name__}"
                ) from error
            return_code = process.returncode
            if return_code is None:
                return_code = process.poll()
            return (
                int(return_code if return_code is not None else -1),
                str(communication.get("stdout") or ""),
                str(communication.get("stderr") or ""),
            )
        finally:
            self._cleanup_owned_process(*owned)

    def _translate_and_compile_prompt(
        self,
        job_id: str,
        request: Mapping[str, Any],
        job_dir: Path,
        source_prompt: str,
        processing: Mapping[str, Any],
        *,
        started: float,
        translation_required: bool,
    ) -> tuple[str, dict[str, Any]] | None:
        """Run the deterministic worker and validate its complete H3 document."""

        model_status = prompt_translator_status(self.root)
        if translation_required and not model_status["ready"]:
            self._fail_prompt_translation(
                job_id,
                request,
                job_dir,
                source_prompt,
                processing,
                code="TRANSLATOR_MODEL_NOT_READY",
                detail=(
                    "固定revisionのローカル翻訳モデルが未配置、不完全、またはサイズ不一致です。"
                ),
                started=started,
                translator=model_status,
                translation_required=True,
            )
            return None

        python = self.root / ".comfy-venv" / "Scripts" / "python.exe"
        if not python.is_file():
            self._fail_prompt_translation(
                job_id,
                request,
                job_dir,
                source_prompt,
                processing,
                code="TRANSLATOR_RUNTIME_MISSING",
                detail="ローカル翻訳用の専用Python環境が見つかりません。",
                started=started,
                translator=model_status,
                translation_required=translation_required,
            )
            return None

        if not self._update_prompt_progress(
            job_id,
            progress=1.25,
            phase="プロンプトを変換しています",
            message="入力をローカルで生成モード対応のH3公式形式へコンパイルしています。",
        ):
            return None

        raw_references = list(request.get("references") or [])
        payload: dict[str, Any] = {
            "prompt": source_prompt,
            "music_policy": str(request.get("music_policy") or "auto"),
            "mode": str(request.get("mode") or "t2v"),
            "references": _translator_reference_manifest(raw_references),
            "dialogue_events": list(processing.get("dialogue_events") or []),
            "prompt_processing": dict(processing),
        }
        try:
            payload["duration_seconds"] = float(request.get("num_frames") or 0) / 24.0
        except (TypeError, ValueError):
            payload["duration_seconds"] = 0.0
        if translation_required:
            payload.update(
                model_path=os.fspath(
                    (self.root / PROMPT_TRANSLATOR_MODEL_DIR).resolve()
                ),
                model_id=model_status.get("repo_id"),
                revision=model_status.get("revision"),
            )
        env = os.environ.copy()
        env.update(
            PYTHONIOENCODING="utf-8",
            PYTHONUNBUFFERED="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        worker_started = time.monotonic()
        try:
            process_result = self._run_tracked_prompt_translator(
                job_id,
                [os.fspath(python), "-m", PROMPT_TRANSLATOR_MODULE],
                json.dumps(payload, ensure_ascii=False),
                env,
                timeout=PROMPT_TRANSLATOR_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            self._fail_prompt_translation(
                job_id,
                request,
                job_dir,
                source_prompt,
                processing,
                code="TRANSLATOR_TIMEOUT",
                detail=f"ローカル変換が{PROMPT_TRANSLATOR_TIMEOUT_SECONDS}秒以内に完了しませんでした。",
                started=started,
                translator=model_status,
                translation_required=translation_required,
            )
            return None
        except OSError as exc:
            self._fail_prompt_translation(
                job_id,
                request,
                job_dir,
                source_prompt,
                processing,
                code="TRANSLATOR_PROCESS_FAILED",
                detail=(
                    "ローカル翻訳プロセスを開始、または安全に読み取ることができませんでした"
                    f"（{exc.__class__.__name__}）。"
                ),
                started=started,
                translator=model_status,
                translation_required=translation_required,
            )
            return None
        if process_result is None:
            return None

        worker_elapsed_ms = round((time.monotonic() - worker_started) * 1000, 2)
        return_code, worker_stdout, worker_stderr = process_result
        private_dir = job_dir / "prompt_processing"
        private_dir.mkdir(parents=True, exist_ok=True)
        if worker_stderr:
            (private_dir / "translator.stderr.log").write_text(
                worker_stderr,
                encoding="utf-8",
            )
        try:
            response = self._worker_json(worker_stdout)
        except ValueError as exc:
            if worker_stdout:
                (private_dir / "translator.stdout.log").write_text(
                    worker_stdout,
                    encoding="utf-8",
                )
            detail = str(exc)
            code = "TRANSLATOR_RESPONSE_INVALID"
            if return_code != 0:
                code = "TRANSLATOR_PROCESS_FAILED"
                detail = (
                    f"ローカル翻訳ワーカーが終了コード{return_code}で停止し、"
                    "有効な失敗レスポンスを返しませんでした。"
                )
            self._fail_prompt_translation(
                job_id,
                request,
                job_dir,
                source_prompt,
                processing,
                code=code,
                detail=detail,
                started=started,
                translator={**model_status, "worker_elapsed_ms": worker_elapsed_ms},
                translation_required=translation_required,
            )
            return None
        self._write_json_atomic(private_dir / "translator_response.json", response)

        worker_diagnostics = list(response.get("diagnostics") or [])
        worker_metadata = response.get("compiler_metadata")
        if not isinstance(worker_metadata, Mapping):
            worker_metadata = {}
        response_code = str(response.get("code") or "TRANSLATOR_WORKER_FAILED")
        if response.get("ok") is not True:
            error = str(response.get("error") or "ローカル翻訳ワーカーが変換を完了できませんでした。")
            self._fail_prompt_translation(
                job_id,
                request,
                job_dir,
                source_prompt,
                processing,
                code=response_code,
                detail=error,
                started=started,
                extra_diagnostics=worker_diagnostics,
                translator={
                    **model_status,
                    "worker_elapsed_ms": worker_elapsed_ms,
                    "compiler_metadata": worker_metadata,
                },
                translation_required=translation_required,
            )
            return None
        if return_code != 0:
            self._fail_prompt_translation(
                job_id,
                request,
                job_dir,
                source_prompt,
                processing,
                code="TRANSLATOR_PROCESS_FAILED",
                detail=(
                    f"ローカル翻訳ワーカーが成功レスポンス後に終了コード"
                    f"{return_code}を返しました。"
                ),
                started=started,
                extra_diagnostics=worker_diagnostics,
                translator={
                    **model_status,
                    "worker_elapsed_ms": worker_elapsed_ms,
                    "compiler_metadata": worker_metadata,
                },
                translation_required=translation_required,
            )
            return None

        compiled_prompt = response.get("compiled_prompt")
        if not isinstance(compiled_prompt, str) or not compiled_prompt.strip():
            self._fail_prompt_translation(
                job_id,
                request,
                job_dir,
                source_prompt,
                processing,
                code="TRANSLATOR_COMPILED_PROMPT_MISSING",
                detail="翻訳ワーカーが完全なcompiled_promptを返しませんでした。",
                started=started,
                extra_diagnostics=worker_diagnostics,
                translator={
                    **model_status,
                    "worker_elapsed_ms": worker_elapsed_ms,
                    "compiler_metadata": worker_metadata,
                },
                translation_required=translation_required,
            )
            return None
        compiled_prompt = compiled_prompt.strip()

        self._update_prompt_progress(
            job_id,
            progress=1.65,
            phase="プロンプトを検証しています",
            message="生成モード別セクション・台詞・参照素材・画面内文字を検証しています。",
        )
        validation_diagnostics = _validate_compiled_prompt(
            source_prompt,
            compiled_prompt,
            mode=str(request.get("mode") or "t2v"),
            references=list(request.get("references") or []),
            compiler_metadata=worker_metadata,
        )
        worker_has_error = any(
            isinstance(item, Mapping)
            and (
                bool(item.get("fatal"))
                or str(item.get("severity") or "").casefold() in {"error", "fatal"}
            )
            for item in worker_diagnostics
        )
        if validation_diagnostics or worker_has_error:
            self._fail_prompt_translation(
                job_id,
                request,
                job_dir,
                source_prompt,
                processing,
                code="TRANSLATOR_VALIDATION_FAILED",
                detail="変換後プロンプトの安全検証に失敗しました。",
                started=started,
                extra_diagnostics=[*worker_diagnostics, *validation_diagnostics],
                translator={
                    **model_status,
                    "worker_elapsed_ms": worker_elapsed_ms,
                    "compiler_metadata": worker_metadata,
                },
                translation_required=translation_required,
            )
            return None

        diagnostics = list(processing.get("diagnostics") or [])
        public_worker_diagnostics = self._public_worker_metadata(worker_diagnostics)
        if isinstance(public_worker_diagnostics, list):
            diagnostics.extend(public_worker_diagnostics)
        adjustments = list(processing.get("auto_adjustments") or [])
        adjustments.append(
            {
                "code": (
                    "LOCAL_PROMPT_TRANSLATED"
                    if translation_required
                    else "LOCAL_PROMPT_COMPILED"
                ),
                "message": (
                    "日本語の制御指示をローカル翻訳し、H3公式形式へコンパイルしました。"
                    if translation_required
                    else "入力を決定論的にH3公式形式へコンパイルしました。"
                ),
            }
        )
        public_metadata = self._public_worker_metadata(worker_metadata)
        processing_public = dict(processing)
        processing_public.update(
            status=("translated_guarded" if translation_required else "compiled_guarded"),
            mode=("translated_guarded" if translation_required else "compiled_guarded"),
            context_ir=False,
            translation_required=translation_required,
            translation_status=("completed" if translation_required else "not_required"),
            local_only=True,
            model_inference=translation_required,
            model_repo=(model_status.get("repo_id") if translation_required else None),
            model_revision=(model_status.get("revision") if translation_required else None),
            source_sha256=hashlib.sha256(source_prompt.encode("utf-8")).hexdigest(),
            request_source_sha256=hashlib.sha256(
                str(request.get("prompt") or "").encode("utf-8")
            ).hexdigest(),
            output_sha256=hashlib.sha256(compiled_prompt.encode("utf-8")).hexdigest(),
            elapsed_ms=round((time.monotonic() - started) * 1000, 2),
            worker_elapsed_ms=worker_elapsed_ms,
            compiler_metadata=public_metadata,
            auto_adjustments=adjustments,
            diagnostics=diagnostics,
            artifacts=[
                "prompt_processing/original_prompt.txt",
                "prompt_processing/source_prompt.txt",
                "prompt_processing/final_prompt.txt",
                "prompt_processing/report.json",
            ],
            provenance={
                "compiler": "h3-studio-local-prompt-translator",
                "model_repo": (
                    model_status.get("repo_id") if translation_required else None
                ),
                "model_revision": (
                    model_status.get("revision") if translation_required else None
                ),
                "local_only": True,
                "model_inference": translation_required,
            },
        )
        self._write_prompt_artifacts(
            job_dir,
            original_prompt=str(request.get("prompt") or ""),
            source_prompt=source_prompt,
            final_prompt=compiled_prompt,
            report=processing_public,
        )
        if not self._update_prompt_progress(
            job_id,
            progress=1.9,
            phase="生成を準備しています",
            message="H3用プロンプトの変換と検証が完了しました。",
        ):
            return None
        return compiled_prompt, processing_public

    def _plan_community_prompt(
        self,
        job_id: str,
        request: Mapping[str, Any],
        job_dir: Path,
        source_prompt: str,
        processing: Mapping[str, Any],
        *,
        started: float,
    ) -> tuple[str, dict[str, Any]] | None:
        """Run the text-only Qwen planner, then enforce the native-clean contract."""

        planner_status = community_planner_status(self.root)
        python = self.root / ".comfy-venv" / "Scripts" / "python.exe"
        if not python.is_file():
            self._fail_prompt_translation(
                job_id,
                request,
                job_dir,
                source_prompt,
                processing,
                code="PLANNER_RUNTIME_MISSING",
                detail="ローカルQwenプロンプト用の専用Python環境が見つかりません。",
                started=started,
                translator=planner_status,
                translation_required=True,
            )
            return None

        raw_references = list(request.get("references") or [])
        payload: dict[str, Any] = {
            "prompt": str(request.get("prompt") or source_prompt),
            "references": _translator_reference_manifest(raw_references),
            "dialogue_texts": [
                str(request.get("dialogue") or "")
            ] if str(request.get("dialogue") or "").strip() else [],
            "duration_seconds": float(request.get("num_frames") or 0) / 24.0,
            "mode": str(request.get("mode") or "t2v"),
            "style_direction": str(request.get("style_direction") or ""),
            "soundscape": str(request.get("soundscape") or ""),
            "audio_preset": str(request.get("audio_preset") or "auto"),
            "music_policy": str(request.get("music_policy") or "auto"),
            "expected_revision": COMMUNITY_PLANNER_REVISION,
            "max_attempts": 2,
        }
        cache_payload = dict(payload)
        cache_payload.update(
            planner_model=COMMUNITY_PLANNER_MODEL_ID,
            planner_revision=COMMUNITY_PLANNER_REVISION,
            planner_contract=COMMUNITY_PLANNER_CONTRACT,
            compiler_revision=COMMUNITY_COMPILER_REVISION,
        )
        cache_key = hashlib.sha256(
            json.dumps(
                cache_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cache_dir = self.data_dir / "prompt_cache"
        cache_path = cache_dir / f"{cache_key}.json"
        response: Mapping[str, Any] | None = None
        cache_hit = False
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                validated_cached = _validated_community_cache_response(
                    cached,
                    cache_key=cache_key,
                    mode=str(request.get("mode") or "t2v"),
                    references=raw_references,
                )
                if validated_cached is not None:
                    response = validated_cached
                    cache_hit = True
            except (OSError, json.JSONDecodeError):
                response = None

        worker_elapsed_ms = 0.0
        private_dir = job_dir / "prompt_processing"
        private_dir.mkdir(parents=True, exist_ok=True)
        if response is None:
            if not planner_status["ready"]:
                self._fail_prompt_translation(
                    job_id,
                    request,
                    job_dir,
                    source_prompt,
                    processing,
                    code="PLANNER_MODEL_NOT_READY",
                    detail=(
                        "固定revisionのローカルQwenプロンプトモデルが未配置、"
                        "不完全、またはサイズ不一致です。"
                    ),
                    started=started,
                    translator=planner_status,
                    translation_required=True,
                )
                return None
            if not self._update_prompt_progress(
                job_id,
                progress=1.25,
                phase="プロンプトを整理しています",
                message="日本語の意図を公開成功例と同じ英語Storyboardへ変換しています。",
            ):
                return None
            payload["model_path"] = os.fspath(
                (self.root / COMMUNITY_PLANNER_MODEL_DIR).resolve()
            )
            env = os.environ.copy()
            env.update(
                PYTHONIOENCODING="utf-8",
                PYTHONUNBUFFERED="1",
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false",
            )
            worker_started = time.monotonic()
            try:
                process_result = self._run_tracked_prompt_translator(
                    job_id,
                    [os.fspath(python), "-m", COMMUNITY_PLANNER_MODULE],
                    json.dumps(payload, ensure_ascii=False),
                    env,
                    timeout=COMMUNITY_PLANNER_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                self._fail_prompt_translation(
                    job_id,
                    request,
                    job_dir,
                    source_prompt,
                    processing,
                    code="PLANNER_TIMEOUT",
                    detail=(
                        f"ローカルQwenプロンプト整理が"
                        f"{COMMUNITY_PLANNER_TIMEOUT_SECONDS}秒以内に完了しませんでした。"
                    ),
                    started=started,
                    translator=planner_status,
                    translation_required=True,
                )
                return None
            except OSError as exc:
                self._fail_prompt_translation(
                    job_id,
                    request,
                    job_dir,
                    source_prompt,
                    processing,
                    code="PLANNER_PROCESS_FAILED",
                    detail=f"ローカルQwenプロンプトプロセスを開始できませんでした（{exc.__class__.__name__}）。",
                    started=started,
                    translator=planner_status,
                    translation_required=True,
                )
                return None
            if process_result is None:
                return None
            worker_elapsed_ms = round((time.monotonic() - worker_started) * 1000, 2)
            return_code, worker_stdout, worker_stderr = process_result
            if worker_stderr:
                (private_dir / "planner.stderr.log").write_text(
                    worker_stderr,
                    encoding="utf-8",
                )
            try:
                response = self._worker_json(worker_stdout)
            except ValueError as exc:
                if worker_stdout:
                    (private_dir / "planner.stdout.log").write_text(
                        worker_stdout,
                        encoding="utf-8",
                    )
                self._fail_prompt_translation(
                    job_id,
                    request,
                    job_dir,
                    source_prompt,
                    processing,
                    code="PLANNER_RESPONSE_INVALID",
                    detail=str(exc),
                    started=started,
                    translator={**planner_status, "worker_elapsed_ms": worker_elapsed_ms},
                    translation_required=True,
                )
                return None
            if return_code != 0 or response.get("ok") is not True:
                self._fail_prompt_translation(
                    job_id,
                    request,
                    job_dir,
                    source_prompt,
                    processing,
                    code=str(response.get("code") or "PLANNER_PROCESS_FAILED"),
                    detail=str(response.get("error") or "Qwenが有効なH3プロンプトを作成できませんでした。"),
                    started=started,
                    extra_diagnostics=list(response.get("diagnostics") or []),
                    translator={**planner_status, "worker_elapsed_ms": worker_elapsed_ms},
                    translation_required=True,
                )
                return None
        assert response is not None
        self._write_json_atomic(private_dir / "planner_response.json", response)
        compiled_prompt = response.get("compiled_prompt")
        if not isinstance(compiled_prompt, str) or not compiled_prompt.strip():
            self._fail_prompt_translation(
                job_id,
                request,
                job_dir,
                source_prompt,
                processing,
                code="PLANNER_COMPILED_PROMPT_MISSING",
                detail="Qwen plannerが完全なcompiled_promptを返しませんでした。",
                started=started,
                translator=planner_status,
                translation_required=True,
            )
            return None
        if not self._update_prompt_progress(
            job_id,
            progress=1.65,
            phase="プロンプトを検証しています",
            message="日本語台詞・参照番号・英語制御文を検証しています。",
        ):
            return None
        validation = _validate_native_clean_prompt(
            compiled_prompt,
            mode=str(request.get("mode") or "t2v"),
            references=raw_references,
        )
        worker_diagnostics = list(response.get("diagnostics") or [])
        if validation or any(
            isinstance(item, Mapping)
            and (bool(item.get("fatal")) or str(item.get("severity") or "").casefold() in {"error", "fatal"})
            for item in worker_diagnostics
        ):
            self._fail_prompt_translation(
                job_id,
                request,
                job_dir,
                source_prompt,
                processing,
                code="PLANNER_VALIDATION_FAILED",
                detail="公開Comfy用プロンプトの安全検証に失敗しました。",
                started=started,
                extra_diagnostics=[*worker_diagnostics, *validation],
                translator=planner_status,
                translation_required=True,
            )
            return None

        if not cache_hit:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._write_json_atomic(
                cache_path,
                {
                    "schema_version": COMMUNITY_CACHE_SCHEMA_VERSION,
                    "cache_key": cache_key,
                    "compiler_revision": COMMUNITY_COMPILER_REVISION,
                    "planner_contract": COMMUNITY_PLANNER_CONTRACT,
                    "planner_model": COMMUNITY_PLANNER_MODEL_ID,
                    "planner_revision": COMMUNITY_PLANNER_REVISION,
                    "compiled_sha256": hashlib.sha256(
                        compiled_prompt.encode("utf-8")
                    ).hexdigest(),
                    "response": response,
                },
            )

        planner_metadata = response.get("planner_metadata")
        public_metadata = self._public_worker_metadata(
            planner_metadata if isinstance(planner_metadata, Mapping) else {}
        )
        diagnostics = list(processing.get("diagnostics") or [])
        public_worker_diagnostics = self._public_worker_metadata(worker_diagnostics)
        if isinstance(public_worker_diagnostics, list):
            diagnostics.extend(public_worker_diagnostics)
        adjustments = list(processing.get("auto_adjustments") or [])
        adjustments.append(
            {
                "code": "COMMUNITY_PROMPT_CACHE_REUSED" if cache_hit else "COMMUNITY_PROMPT_PLANNED",
                "message": (
                    "同一入力の検証済み英語Storyboardを再利用しました。"
                    if cache_hit
                    else "日本語の意図を公開成功例準拠の英語Storyboardへ整理しました。"
                ),
            }
        )
        processing_public = dict(processing)
        processing_public.update(
            status="community_planned",
            mode="community_planned",
            context_ir=False,
            translation_required=True,
            translation_status="cache_hit" if cache_hit else "completed",
            local_only=True,
            model_inference=not cache_hit,
            model_repo=COMMUNITY_PLANNER_MODEL_ID,
            model_revision=COMMUNITY_PLANNER_REVISION,
            workflow_profile="native_clean",
            prompt_cache_key=cache_key,
            prompt_cache_hit=cache_hit,
            source_sha256=hashlib.sha256(source_prompt.encode("utf-8")).hexdigest(),
            request_source_sha256=hashlib.sha256(
                str(request.get("prompt") or "").encode("utf-8")
            ).hexdigest(),
            output_sha256=hashlib.sha256(compiled_prompt.encode("utf-8")).hexdigest(),
            elapsed_ms=round((time.monotonic() - started) * 1000, 2),
            worker_elapsed_ms=worker_elapsed_ms,
            compiler_metadata=public_metadata,
            auto_adjustments=adjustments,
            diagnostics=diagnostics,
            artifacts=[
                "prompt_processing/original_prompt.txt",
                "prompt_processing/source_prompt.txt",
                "prompt_processing/final_prompt.txt",
                "prompt_processing/report.json",
                "prompt_processing/planner_response.json",
            ],
            provenance={
                "compiler": "h3-studio-community-prompt-planner",
                "contract": COMMUNITY_PLANNER_CONTRACT,
                "compiler_revision": COMMUNITY_COMPILER_REVISION,
                "model_repo": COMMUNITY_PLANNER_MODEL_ID,
                "model_revision": COMMUNITY_PLANNER_REVISION,
                "local_only": True,
                "model_inference": not cache_hit,
            },
        )
        self._write_prompt_artifacts(
            job_dir,
            original_prompt=str(request.get("prompt") or ""),
            source_prompt=source_prompt,
            final_prompt=compiled_prompt,
            report=processing_public,
        )
        if not self._update_prompt_progress(
            job_id,
            progress=1.9,
            phase="生成を準備しています",
            message="公開Comfy用プロンプトの変換と検証が完了しました。",
        ):
            return None
        return compiled_prompt, processing_public

    def _finalize_prompt_execution(
        self,
        job_id: str,
        request_path: Path,
        request: Mapping[str, Any],
        *,
        effective_prompt: str,
        compiler_public: Mapping[str, Any],
        compiler_source: str,
    ) -> Path | None:
        """Persist one derived execution request without mutating request.json."""

        execution_request = dict(request)
        execution_request["effective_prompt"] = effective_prompt
        execution_request["effective_prompt_source"] = compiler_source
        execution_request["compiler"] = dict(compiler_public)
        execution_request["prompt_processing"] = dict(compiler_public)
        execution_request["workflow_profile"] = "native_clean"
        execution_request["embedded_video_audio_policy"] = str(
            compiler_public.get("embedded_video_audio_policy") or "ignore"
        )
        execution_request["embedded_video_audio_indices"] = list(
            compiler_public.get("embedded_video_audio_indices") or []
        )
        execution_request_path = request_path.parent / "execution_request.json"
        self._write_json_atomic(execution_request_path, execution_request)
        with self._lock:
            current = self._jobs.get(job_id)
            if not current or current.get("status") == "cancelled":
                execution_request_path.unlink(missing_ok=True)
                return None
            current.update(
                effective_prompt=effective_prompt,
                effective_prompt_source=compiler_source,
                workflow_profile="native_clean",
                compiler=dict(compiler_public),
                prompt_processing=dict(compiler_public),
                auto_adjustments=list(compiler_public.get("auto_adjustments", [])),
                diagnostics=list(compiler_public.get("diagnostics", [])),
            )
            self._save_job(current)
        return execution_request_path

    def _prepare_effective_prompt(self, job_id: str, request_path: Path) -> Path | None:
        """Prepare the immutable UI request for direct H3 execution.

        New H3 Studio requests either use the local community planner followed
        by the native-clean Comfy profile, or a validated English prompt passed
        byte-for-byte to the same profile.  The legacy compiler remains only
        for explicitly persisted compatibility requests.
        """

        request = json.loads(request_path.read_text(encoding="utf-8"))
        job_dir = request_path.parent
        started = time.monotonic()
        effective_prompt = str(request.get("effective_prompt") or request.get("prompt") or "")
        compiler_public: dict[str, Any]
        processing = request.get("prompt_processing")
        processing_mode = (
            str(processing.get("mode", "raw_guarded"))
            if isinstance(processing, Mapping)
            else "raw_guarded"
        )
        if processing_mode != "context_ir":
            if not effective_prompt.strip():
                with self._lock:
                    current = self._jobs.get(job_id)
                    if current and current.get("status") not in TERMINAL_STATES:
                        current.update(
                            status="failed",
                            phase="入力を確認してください",
                            message="H3へ送るプロンプトが空です。",
                            finished_at=utc_now(),
                            progress_updated_at=utc_now(),
                        )
                        self._save_job(current)
                return None
            processing_public = dict(processing) if isinstance(processing, Mapping) else {}
            removed = list(processing_public.get("removed_speech_cues") or [])
            # Preserve formatter metadata created at the HTTP boundary.  This
            # method only adds audit information for fragments that the final
            # raw guard removed; it must not replace native-dialogue events or
            # punctuation-normalisation notes with the old separate-field UX.
            adjustments = list(processing_public.get("auto_adjustments") or [])
            diagnostics = list(processing_public.get("diagnostics") or [])
            if removed:
                has_inline_dialogue = bool(processing_public.get("dialogue_count"))
                removal_message = (
                    "明示台詞は対象Cut内に保持したまま、本文中の曖昧または重複する"
                    f"発話指示を{len(removed)}件、生成用プロンプトから除外しました。"
                    if has_inline_dialogue
                    else "明示台詞として判定できなかった本文中の曖昧な発話指示を"
                    f"{len(removed)}件、生成用プロンプトから除外しました。"
                )
                adjustments.append(
                    {
                        "code": "RAW_SPEECH_CUES_REMOVED",
                        "message": removal_message,
                    }
                )
                diagnostics.append(
                    {
                        "severity": "info",
                        "code": "RAW_SPEECH_CUES_REMOVED",
                        "message": (
                            "除外した原文はrequest.jsonに保持されています。必要な台詞は"
                            "メインプロンプトの対象Cutへ「実際の言葉」と言う形で記述できます。"
                        ),
                        "fatal": False,
                    }
                )
            processing_public.update(
                auto_adjustments=adjustments,
                diagnostics=diagnostics,
            )
            prompt_processing_mode = str(
                request.get("prompt_processing_mode")
                or processing_public.get("processing_mode_requested")
                or "direct"
            ).casefold()
            if prompt_processing_mode not in {
                "community",
                "raw_en",
                "direct",
                "official_en",
            }:
                self._fail_prompt_translation(
                    job_id,
                    request,
                    job_dir,
                    effective_prompt,
                    processing_public,
                    code="INVALID_PROMPT_PROCESSING_MODE",
                    detail=(
                        "プロンプト処理方式はcommunity、raw_en、direct、"
                        "official_enのいずれかで指定してください。"
                    ),
                    started=started,
                    translation_required=False,
                )
                return None
            processing_public.update(
                processing_mode_requested=prompt_processing_mode,
                processing_mode_effective=prompt_processing_mode,
            )
            mode = str(request.get("mode") or "t2v").casefold()
            if prompt_processing_mode == "community":
                planned = self._plan_community_prompt(
                    job_id,
                    request,
                    job_dir,
                    effective_prompt,
                    processing_public,
                    started=started,
                )
                if planned is None:
                    return None
                effective_prompt, compiler_public = planned
                return self._finalize_prompt_execution(
                    job_id,
                    request_path,
                    request,
                    effective_prompt=effective_prompt,
                    compiler_public=compiler_public,
                    compiler_source="community_planned",
                )
            if prompt_processing_mode == "raw_en":
                native_diagnostics = _validate_native_clean_prompt(
                    effective_prompt,
                    mode=mode,
                    references=list(request.get("references") or []),
                )
                if native_diagnostics:
                    first = native_diagnostics[0]
                    self._fail_prompt_translation(
                        job_id,
                        request,
                        job_dir,
                        effective_prompt,
                        processing_public,
                        code=str(first.get("code") or "NATIVE_CLEAN_VALIDATION_FAILED"),
                        detail=str(first.get("message") or "英語H3プロンプトの検証に失敗しました。"),
                        started=started,
                        extra_diagnostics=native_diagnostics[1:],
                        translation_required=False,
                    )
                    return None
                elapsed_ms = round((time.monotonic() - started) * 1000, 2)
                processing_public.update(
                    status="native_raw",
                    mode="native_raw",
                    context_ir=False,
                    translation_required=False,
                    translation_status="not_required",
                    local_only=True,
                    model_inference=False,
                    workflow_profile="native_clean",
                    source_sha256=hashlib.sha256(effective_prompt.encode("utf-8")).hexdigest(),
                    request_source_sha256=hashlib.sha256(
                        str(request.get("prompt") or "").encode("utf-8")
                    ).hexdigest(),
                    output_sha256=hashlib.sha256(effective_prompt.encode("utf-8")).hexdigest(),
                    elapsed_ms=elapsed_ms,
                    artifacts=[
                        "prompt_processing/original_prompt.txt",
                        "prompt_processing/source_prompt.txt",
                        "prompt_processing/final_prompt.txt",
                        "prompt_processing/report.json",
                    ],
                    provenance={
                        "compiler": "public-comfy-raw-prompt",
                        "local_only": True,
                        "model_inference": False,
                    },
                )
                self._write_prompt_artifacts(
                    job_dir,
                    original_prompt=str(request.get("prompt") or ""),
                    source_prompt=effective_prompt,
                    final_prompt=effective_prompt,
                    report=processing_public,
                )
                if not self._update_prompt_progress(
                    job_id,
                    progress=1.9,
                    phase="生成を準備しています",
                    message="公開Comfy互換プロンプトを検証しました。",
                ):
                    return None
                return self._finalize_prompt_execution(
                    job_id,
                    request_path,
                    request,
                    effective_prompt=effective_prompt,
                    compiler_public=processing_public,
                    compiler_source="native_raw",
                )
            try:
                # This runs before either the deterministic compiler or the
                # engine.  In particular, an English prompt is not a bypass for
                # malformed/unsupported native dialogue syntax.
                validate_native_dialogue_blocks(effective_prompt)
            except PromptTranslationError as exc:
                self._fail_prompt_translation(
                    job_id,
                    request,
                    job_dir,
                    effective_prompt,
                    processing_public,
                    code=exc.code,
                    detail=str(exc),
                    started=started,
                    translation_required=False,
                )
                return None

            reference_diagnostics = _validate_direct_reference_tags(
                effective_prompt,
                mode=mode,
                references=list(request.get("references") or []),
            )
            if reference_diagnostics:
                self._fail_prompt_translation(
                    job_id,
                    request,
                    job_dir,
                    effective_prompt,
                    processing_public,
                    code=str(reference_diagnostics[0]["code"]),
                    detail=reference_diagnostics[0]["message"],
                    started=started,
                    translation_required=False,
                )
                return None

            section_headers = tuple(
                match.group("header").casefold()
                for match in _OFFICIAL_SECTION_RE.finditer(effective_prompt)
            )
            if section_headers:
                # An already-official document is the sole worker bypass.  It
                # still receives the same structural/dialogue/reference/CJK
                # validator as worker output, and client data cannot authorize
                # visible CJK literals on the compiler's behalf.
                try:
                    official_literals = classify_visible_text_literals(
                        effective_prompt
                    )
                except PromptTranslationError as exc:
                    self._fail_prompt_translation(
                        job_id,
                        request,
                        job_dir,
                        effective_prompt,
                        processing_public,
                        code=exc.code,
                        detail=str(exc),
                        started=started,
                        translation_required=False,
                    )
                    return None
                official_metadata = {
                    "visible_text_literals": list(official_literals),
                    "visible_text_literal_sha256": [
                        hashlib.sha256(item.encode("utf-8")).hexdigest()
                        for item in official_literals
                    ],
                }
                official_diagnostics = _validate_compiled_prompt(
                    effective_prompt,
                    effective_prompt,
                    mode=mode,
                    references=list(request.get("references") or []),
                    compiler_metadata=official_metadata,
                )
                if official_diagnostics:
                    self._fail_prompt_translation(
                        job_id,
                        request,
                        job_dir,
                        effective_prompt,
                        processing_public,
                        code="PROMPT_VALIDATION_FAILED",
                        detail="入力済みH3公式プロンプトの安全検証に失敗しました。",
                        started=started,
                        extra_diagnostics=official_diagnostics,
                        translation_required=False,
                    )
                    return None
                processing_public.update(
                    status="official_prompt",
                    mode="official_prompt",
                    context_ir=False,
                    translation_required=False,
                    translation_status="not_required",
                    local_only=True,
                    model_inference=False,
                    elapsed_ms=round((time.monotonic() - started) * 1000, 2),
                    source_sha256=hashlib.sha256(effective_prompt.encode("utf-8")).hexdigest(),
                    request_source_sha256=hashlib.sha256(
                        str(request.get("prompt") or "").encode("utf-8")
                    ).hexdigest(),
                    output_sha256=hashlib.sha256(effective_prompt.encode("utf-8")).hexdigest(),
                    artifacts=[
                        "prompt_processing/original_prompt.txt",
                        "prompt_processing/source_prompt.txt",
                        "prompt_processing/final_prompt.txt",
                        "prompt_processing/report.json",
                    ],
                    provenance={
                        "compiler": "user-supplied-official-prompt",
                        "local_only": True,
                        "model_inference": False,
                    },
                    compiler_metadata=official_metadata,
                )
                self._write_prompt_artifacts(
                    job_dir,
                    original_prompt=str(request.get("prompt") or ""),
                    source_prompt=effective_prompt,
                    final_prompt=effective_prompt,
                    report=processing_public,
                )
                compiler_public = processing_public
                compiler_source = "official_prompt"
            elif prompt_processing_mode == "direct":
                # Recommended A/B baseline: H3 accepts natural Japanese control
                # prose directly.  Preserve the formatter/guard output byte for
                # byte and apply only the native dialogue safety boundary here;
                # no optional translator, Python worker, or official wrapper is
                # required for this route.
                processing_public.update(
                    status="direct",
                    mode="raw_guarded",
                    context_ir=False,
                    translation_required=False,
                    translation_status="not_requested",
                    local_only=True,
                    model_inference=False,
                    source_sha256=hashlib.sha256(
                        effective_prompt.encode("utf-8")
                    ).hexdigest(),
                    request_source_sha256=hashlib.sha256(
                        str(request.get("prompt") or "").encode("utf-8")
                    ).hexdigest(),
                    output_sha256=hashlib.sha256(
                        effective_prompt.encode("utf-8")
                    ).hexdigest(),
                    elapsed_ms=round((time.monotonic() - started) * 1000, 2),
                    artifacts=[
                        "prompt_processing/original_prompt.txt",
                        "prompt_processing/source_prompt.txt",
                        "prompt_processing/final_prompt.txt",
                        "prompt_processing/report.json",
                    ],
                    provenance={
                        "compiler": "h3-studio-direct-prompt",
                        "local_only": True,
                        "model_inference": False,
                    },
                )
                self._write_prompt_artifacts(
                    job_dir,
                    original_prompt=str(request.get("prompt") or ""),
                    source_prompt=effective_prompt,
                    final_prompt=effective_prompt,
                    report=processing_public,
                )
                compiler_public = processing_public
                compiler_source = "direct"
            else:
                try:
                    translation_required = requires_translation(effective_prompt)
                except PromptTranslationError as exc:
                    self._fail_prompt_translation(
                        job_id,
                        request,
                        job_dir,
                        effective_prompt,
                        processing_public,
                        code=exc.code,
                        detail=str(exc),
                        started=started,
                        translation_required=False,
                    )
                    return None
                compiled = self._translate_and_compile_prompt(
                    job_id,
                    request,
                    job_dir,
                    effective_prompt,
                    processing_public,
                    started=started,
                    translation_required=translation_required,
                )
                if compiled is None:
                    return None
                effective_prompt, compiler_public = compiled
                compiler_source = (
                    "translated_guarded"
                    if translation_required
                    else "compiled_guarded"
                )
        else:
            compiler_source = "legacy_fallback"
            try:
                from .context_ir import compile_request, write_artifacts

                result = compile_request(request)
                result = write_artifacts(result, job_dir)
                compiler_public = result.to_public_dict()
                compiler_public["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
                if result.fatal or result.ir_text is None:
                    fatal_messages = [
                        item.message
                        for item in result.diagnostics
                        if getattr(item, "fatal", False)
                    ]
                    message = fatal_messages[0] if fatal_messages else "入力をH3用命令へ変換できませんでした。"
                    with self._lock:
                        current = self._jobs.get(job_id)
                        if current and current.get("status") not in TERMINAL_STATES:
                            current.update(
                                status="failed",
                                phase="入力を確認してください",
                                message=message,
                                compiler=compiler_public,
                                auto_adjustments=compiler_public.get("auto_adjustments", []),
                                diagnostics=compiler_public.get("diagnostics", []),
                                finished_at=utc_now(),
                                progress_updated_at=utc_now(),
                            )
                            self._save_job(current)
                    return None
                effective_prompt = result.ir_text
                compiler_source = "context_ir"
            except Exception as exc:
                elapsed_ms = round((time.monotonic() - started) * 1000, 2)
                compiler_public = {
                    "status": "degraded_fallback",
                    "elapsed_ms": elapsed_ms,
                    "provenance": {
                        "compiler": "h3-studio-safe-fallback",
                        "ai_inference": False,
                    },
                    "auto_adjustments": [],
                    "diagnostics": [
                        {
                            "severity": "info",
                            "code": "COMPILER_FALLBACK",
                            "message": (
                                "IR自動変換を完了できなかったため、元の指示を保った最小形式で生成しました。"
                            ),
                            "fatal": False,
                        }
                    ],
                }
                compiler_dir = job_dir / "compiler"
                compiler_dir.mkdir(parents=True, exist_ok=True)
                internal = {
                    **compiler_public,
                    "exception_type": exc.__class__.__name__,
                    "exception": str(exc),
                }
                self._write_json_atomic(compiler_dir / "compiler_result.json", internal)
                (compiler_dir / "final_ir.txt").write_text(effective_prompt, encoding="utf-8")
                self._append_log(job_id, f"Context-IR fallback: {exc.__class__.__name__}: {exc}")

        # Keep request.json as the immutable record of exactly what the user
        # submitted. Only the derived execution copy reaches H3.
        return self._finalize_prompt_execution(
            job_id,
            request_path,
            request,
            effective_prompt=effective_prompt,
            compiler_public=compiler_public,
            compiler_source=compiler_source,
        )

    @staticmethod
    def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _ensure_engine(self, requested_variant: str) -> subprocess.Popen[str]:
        with self._lock:
            process = self._current_process
            if process and process.poll() is None and self._engine_variant == requested_variant:
                if self._current_process_job is not None:
                    self._current_process_job.check()
                return process
            if process and process.poll() is not None:
                # The kernel Job may still own a redirector child or helper
                # even though the root Popen has already exited.
                self._terminate_current_process()
                process = None
            if process and process.poll() is None:
                # A fresh process is intentional here. PyTorch's CPU allocator
                # can retain tens of GiB after deleting a 33B model, so an
                # in-process FL2VA/Ref2VA swap risks paging both checkpoints.
                self._terminate_current_process()
                process = None
            if process:
                if process.stdin:
                    process.stdin.close()
                if process.stdout:
                    process.stdout.close()
            python = self.root / ".comfy-venv" / "Scripts" / "python.exe"
            if not python.is_file():
                raise FileNotFoundError(f"ComfyUI用Python環境が見つかりません: {python}")
            command = [os.fspath(python), "-m", "webui.comfy_engine_worker", "--serve"]
            env = os.environ.copy()
            env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            process_job = ProcessJob()
            process: subprocess.Popen[str] | None = None
            try:
                process = subprocess.Popen(
                    command,
                    cwd=self.root,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env,
                    creationflags=creation_flags,
                )
                process_job.attach(process)
            except Exception:
                process_job.terminate()
                self._cleanup_owned_process(process, None)
                raise
            self._current_process = process
            self._current_process_job = process_job
            self._engine_variant = requested_variant
            return process

    def _handle_event(self, job_id: str, payload: str) -> None:
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            self._append_log(job_id, payload)
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.get("status") == "cancelled":
                return
            for key in (
                "status",
                "phase",
                "message",
                "progress",
                "step",
                "total_steps",
                "result",
                "preview",
                "backend",
                "workflow_profile",
                "attention_backend",
                "acceleration",
                "media",
                "scheduler",
            ):
                if key in event:
                    job[key] = event[key]
            for key in ("cache", "timings"):
                if key not in event:
                    continue
                value = event[key]
                current = job.get(key)
                if isinstance(current, Mapping) and isinstance(value, Mapping):
                    job[key] = {**current, **value}
                else:
                    job[key] = value
            engine_diagnostics = event.get("diagnostics")
            if isinstance(engine_diagnostics, Mapping):
                # Keep local filesystem paths and process identifiers in the
                # on-disk Comfy log, not in the browser-facing job object.
                safe_diagnostics = {
                    key: value
                    for key, value in engine_diagnostics.items()
                    if key in PUBLIC_ENGINE_DIAGNOSTIC_FIELDS
                }
                technical = job.get("technical")
                if not isinstance(technical, Mapping):
                    technical = {}
                job["technical"] = {**technical, "engine": safe_diagnostics}
            if event.get("status") in TERMINAL_STATES:
                job["finished_at"] = utc_now()
            if any(key in event for key in ("progress", "phase", "message")):
                job["progress_updated_at"] = utc_now()
            self._save_job(job)

    def _append_log(self, job_id: str, line: str) -> None:
        public_line = _redact_public_text(line)
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            logs = job.setdefault("logs", [])
            logs.append({"time": utc_now(), "text": public_line[-2000:]})
            del logs[:-160]
            self._save_job(job)

    def _save_job(self, job: dict[str, Any]) -> None:
        job_dir = self.jobs_dir / job["id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        target = job_dir / "job.json"
        temporary = job_dir / "job.json.tmp"
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    def _claim_current_process_locked(
        self,
    ) -> tuple[subprocess.Popen[str] | None, ProcessJob | None]:
        """Atomically detach the current engine; caller must hold `_lock`."""

        process = self._current_process
        process_job = self._current_process_job
        self._current_process = None
        self._current_process_job = None
        self._engine_variant = None
        return process, process_job

    def _claim_current_translation_locked(
        self,
        *,
        job_id: str | None = None,
        process: subprocess.Popen[str] | None = None,
    ) -> tuple[subprocess.Popen[str] | None, ProcessJob | None]:
        """Atomically detach one exact translator; caller must hold ``_lock``."""

        current = self._current_translation_process
        if job_id is not None and self._current_translation_job_id != job_id:
            return None, None
        if process is not None and current is not process:
            return None, None
        process_job = self._current_translation_process_job
        self._current_translation_process = None
        self._current_translation_process_job = None
        self._current_translation_job_id = None
        return current, process_job

    def _terminate_current_process(self) -> None:
        with self._lock:
            owned_process = self._claim_current_process_locked()
        self._cleanup_owned_process(*owned_process)

    @staticmethod
    def _cleanup_owned_process(
        process: subprocess.Popen[str] | None,
        process_job: ProcessJob | None,
    ) -> None:
        if not process and process_job is None:
            return
        try:
            if process is not None:
                try:
                    parent = psutil.Process(process.pid)
                    children = parent.children(recursive=True)
                    for child in reversed(children):
                        child.terminate()
                    parent.terminate()
                    _, alive = psutil.wait_procs([*children, parent], timeout=5)
                    for remaining in alive:
                        remaining.kill()
                except (psutil.Error, OSError):
                    try:
                        process.terminate()
                    except OSError:
                        pass
        finally:
            # The kernel Job is authoritative even if the root Popen exited
            # before psutil could enumerate an orphaned redirector/helper.
            if process_job is not None:
                process_job.terminate()
            if process is not None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                        process.wait(timeout=2)
                    except (OSError, subprocess.SubprocessError):
                        pass
                except (OSError, subprocess.SubprocessError):
                    pass
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
