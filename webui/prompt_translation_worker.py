"""One-request offline CPU worker for H3 Studio prompt translation."""

from __future__ import annotations

import os

# These must be set before importing torch/transformers.  The worker must not
# compete with MiniMax H3 for RTX memory or perform an implicit network fetch.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .prompt_translation import (
    PromptTranslationError,
    parse_generation_mode,
    requires_translation,
    translate_and_compile_prompt,
)


LOGGER = logging.getLogger("h3.prompt_translation_worker")
SYSTEM_PROMPT = "Translate to English."


class LFM2EnglishTranslator:
    """Strict, local-only, greedy LFM2-350M-ENJP-MT adapter."""

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise PromptTranslationError(
                f"翻訳モデルのローカルフォルダが見つかりません: {path}",
                code="TRANSLATOR_MODEL_NOT_FOUND",
            )
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise PromptTranslationError(
                f"翻訳ランタイムを読み込めません: {exc}",
                code="TRANSLATOR_IMPORT_FAILED",
            ) from exc

        self._torch = torch
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(path), local_files_only=True, trust_remote_code=False
            )
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    str(path),
                    local_files_only=True,
                    trust_remote_code=False,
                    dtype=torch.float32,
                )
            except TypeError:
                self._model = AutoModelForCausalLM.from_pretrained(
                    str(path),
                    local_files_only=True,
                    trust_remote_code=False,
                    torch_dtype=torch.float32,
                )
            self._model.to(device="cpu", dtype=torch.float32)
            self._model.eval()
        except Exception as exc:
            raise PromptTranslationError(
                f"翻訳モデルをローカルから読み込めません: {exc}",
                code="TRANSLATOR_MODEL_LOAD_FAILED",
            ) from exc

    def __call__(self, text: str) -> str:
        # Transformers tokenizers accept a mutable conversation sequence here;
        # current versions reject a tuple as a single TextEncodeInput.
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        try:
            try:
                encoded = self._tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            except TypeError:
                input_ids = self._tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                encoded = {"input_ids": input_ids}
            encoded = {
                key: value.to("cpu") if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            input_length = int(encoded["input_ids"].shape[-1])
            max_new_tokens = max(64, min(1024, input_length * 4 + 32))
            pad_token_id = self._tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self._tokenizer.eos_token_id
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    use_cache=True,
                )
            new_tokens = generated[0, input_length:]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as exc:
            raise PromptTranslationError(
                f"LFM2翻訳推論に失敗しました: {exc}",
                code="TRANSLATOR_INFERENCE_FAILED",
            ) from exc


def _payload_events(payload: dict[str, Any]) -> Sequence[dict[str, Any]]:
    value = payload.get("dialogue_events", [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise PromptTranslationError(
            "dialogue_eventsはオブジェクト配列で指定してください。",
            code="INVALID_WORKER_REQUEST",
        )
    return value


def _payload_references(payload: dict[str, Any]) -> Sequence[dict[str, Any]]:
    value = payload.get("references", [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise PromptTranslationError(
            "referencesはオブジェクト配列で指定してください。",
            code="INVALID_WORKER_REQUEST",
        )
    return value


def process_request(
    payload: dict[str, Any],
    *,
    cli_model_path: str | None = None,
    translator: Any | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PromptTranslationError(
            "worker入力はJSONオブジェクトで指定してください。",
            code="INVALID_WORKER_REQUEST",
        )
    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        raise PromptTranslationError(
            "worker入力のpromptは文字列で指定してください。",
            code="INVALID_WORKER_REQUEST",
        )

    started = time.monotonic()
    generation_mode = parse_generation_mode(str(payload.get("mode") or "omni"))
    selected_translator = translator
    model_path_value = cli_model_path or payload.get("model_path")
    duration_value = payload.get("duration_seconds")
    if duration_value is None and payload.get("num_frames") is not None:
        try:
            duration_value = float(payload["num_frames"]) / 24.0
        except (TypeError, ValueError) as exc:
            raise PromptTranslationError(
                "num_framesからduration_secondsを算出できません。",
                code="INVALID_WORKER_REQUEST",
            ) from exc
    if duration_value is not None:
        try:
            duration_value = float(duration_value)
        except (TypeError, ValueError) as exc:
            raise PromptTranslationError(
                "duration_secondsは数値で指定してください。",
                code="INVALID_WORKER_REQUEST",
            ) from exc
    # Pure-English and already-valid official prompts intentionally work with
    # no optional model installation.  The compiler invokes the translator only
    # if Japanese control prose is actually encountered.
    if selected_translator is None and model_path_value and requires_translation(prompt):
        selected_translator = LFM2EnglishTranslator(str(model_path_value))

    result = translate_and_compile_prompt(
        prompt,
        selected_translator,
        dialogue_events=_payload_events(payload),
        reference_inventory=_payload_references(payload),
        music_policy=(
            str(payload["music_policy"])
            if payload.get("music_policy") is not None
            else None
        ),
        mode=generation_mode,
        duration_seconds=duration_value,
        last_shot_index=(
            int(payload["last_shot_index"])
            if payload.get("last_shot_index") is not None
            else None
        ),
    )
    metadata = result.metadata()
    metadata.update(
        model_id=str(payload.get("model_id") or "LiquidAI/LFM2-350M-ENJP-MT"),
        revision=(str(payload["revision"]) if payload.get("revision") else None),
        model_path=(str(model_path_value) if model_path_value else None),
        elapsed_ms=round((time.monotonic() - started) * 1000, 2),
        device="cpu",
        dtype="float32",
        offline=True,
        decoding="greedy",
        system_prompt=SYSTEM_PROMPT,
    )
    return {
        "ok": True,
        "compiled_prompt": result.compiled_prompt,
        "compiler_metadata": metadata,
        "diagnostics": [],
    }


def _error_payload(exc: Exception) -> dict[str, Any]:
    code = exc.code if isinstance(exc, PromptTranslationError) else "WORKER_FAILED"
    return {
        "ok": False,
        "code": code,
        "error": str(exc),
        "diagnostics": [
            {
                "severity": "error",
                "code": code,
                "message": str(exc),
                "fatal": True,
            }
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Local LFM2-350M-ENJP-MT directory")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            raise PromptTranslationError(
                "workerの標準入力が空です。", code="INVALID_WORKER_REQUEST"
            )
        payload = json.loads(raw)
        response = process_request(payload, cli_model_path=args.model_path)
        exit_code = 0
    except Exception as exc:
        LOGGER.error("Prompt worker failed: %s: %s", exc.__class__.__name__, exc)
        response = _error_payload(exc)
        exit_code = 1
    sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["LFM2EnglishTranslator", "SYSTEM_PROMPT", "main", "process_request"]
