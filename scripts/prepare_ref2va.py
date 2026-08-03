from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import diffusers


DIFFUSERS_SHA = "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_converter(path: Path):
    spec = importlib.util.spec_from_file_location("minimax_h3_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load converter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official Ref2VA Transformer into the unified H3 layout.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    source = root / "models" / "official" / "MiniMax-H3" / "Ref2VA"
    unified = root / "models" / "converted" / "MiniMax-H3-FL2VA-diffusers"
    output = unified / "transformer_ref"
    converter_path = root / ".upstream" / "diffusers" / "scripts" / "convert_minimax_h3_to_diffusers.py"
    artifacts = root / "artifacts"

    if not source.is_dir() or not (source / "transformer" / "model.safetensors.index.json").is_file():
        raise RuntimeError("Official Ref2VA snapshot is incomplete. Run scripts/download_ref2va.py first.")
    converted_index = "diffusion_pytorch_model.safetensors.index.json"
    if not (unified / "transformer" / converted_index).is_file():
        raise RuntimeError("Verified FL2VA converted layout is missing. Run scripts/prepare_diffusers.py first.")
    if output.exists():
        if (output / converted_index).is_file():
            print(json.dumps({"status": "already_prepared", "output": os.fspath(output)}, ensure_ascii=False))
            return
        raise RuntimeError(f"Refusing to overwrite an incomplete existing target: {output}")
    if not converter_path.is_file():
        raise RuntimeError("Pinned Diffusers converter is missing. Run scripts/setup.ps1 -WithLegacyDiffusers first.")

    actual_sha = subprocess.check_output(
        ["git", "-C", os.fspath(converter_path.parents[1]), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_sha != DIFFUSERS_SHA:
        raise RuntimeError(f"Diffusers SHA mismatch: expected {DIFFUSERS_SHA}, got {actual_sha}")

    converter = load_converter(converter_path)
    converter.dry_run(os.fspath(source), converter.MINIMAX_H3_TRANSFORMER_CONFIG)
    converter.convert_transformer(
        os.fspath(source),
        os.fspath(output),
        converter.MINIMAX_H3_TRANSFORMER_CONFIG,
        5 * 1024**3,
    )
    converter.write_transformer_config(
        os.fspath(output),
        converter.MINIMAX_H3_TRANSFORMER_CONFIG,
        diffusers.__version__,
    )

    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size": size,
                "sha256": None if args.skip_sha256 else sha256_file(path),
            }
        )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_repo": "MiniMaxAI/MiniMax-H3",
        "source_revision": "af0fe5abe6fd50d632b65a82fef321c4c5c1f249",
        "source_variant": "Ref2VA",
        "diffusers_revision": DIFFUSERS_SHA,
        "output": os.fspath(output),
        "total_bytes": sum(item["size"] for item in files),
        "files": files,
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts / "converted_ref2va_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": os.fspath(manifest_path), "bytes": manifest["total_bytes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
