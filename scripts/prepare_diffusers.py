from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DIFFUSERS_SHA = "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"


def hardlink_tree(source: Path, destination: Path) -> None:
    for src in source.rglob("*"):
        if not src.is_file():
            continue
        relative = src.relative_to(source)
        dst = destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if dst.stat().st_size != src.stat().st_size:
                raise RuntimeError(f"Existing link target has wrong size: {dst}")
            continue
        os.link(src, dst)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official MiniMax H3 FL2VA weights for pinned Diffusers.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    source = root / "models" / "official" / "MiniMax-H3" / "FL2VA"
    output = root / "models" / "converted" / "MiniMax-H3-FL2VA-diffusers"
    converter = root / ".upstream" / "diffusers" / "scripts" / "convert_minimax_h3_to_diffusers.py"
    artifacts = root / "artifacts"

    if not source.is_dir() or not (source / "transformer" / "model.safetensors.index.json").is_file():
        raise RuntimeError("Official FL2VA snapshot is incomplete. Run download_official.py first.")
    if not converter.is_file():
        raise RuntimeError("Pinned Diffusers checkout is missing. Run scripts/setup.ps1 -WithLegacyDiffusers first.")
    actual_sha = subprocess.check_output(
        ["git", "-C", os.fspath(converter.parents[1]), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_sha != DIFFUSERS_SHA:
        raise RuntimeError(f"Diffusers SHA mismatch: expected {DIFFUSERS_SHA}, got {actual_sha}")

    dry_run_command = [
        sys.executable,
        os.fspath(converter),
        "--checkpoint_path",
        os.fspath(source),
        "--output_path",
        os.fspath(output),
        "--dry_run",
    ]
    subprocess.run(dry_run_command, check=True)

    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        os.fspath(converter),
        "--checkpoint_path",
        os.fspath(source),
        "--output_path",
        os.fspath(output),
        "--modular_repo_id",
        os.fspath(output),
    ]
    subprocess.run(command, check=True)

    # The converter intentionally leaves shared Transformers components in place.
    # NTFS hardlinks expose them under the converted layout without duplicating ~62 GiB.
    for component in ("text_encoder", "tokenizer", "processor"):
        hardlink_tree(source / component, output / component)

    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output).as_posix()
        size = path.stat().st_size
        digest = None if args.skip_sha256 else sha256_file(path)
        files.append({"path": relative, "size": size, "sha256": digest})
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_repo": "MiniMaxAI/MiniMax-H3",
        "source_revision": "af0fe5abe6fd50d632b65a82fef321c4c5c1f249",
        "source_variant": "FL2VA",
        "diffusers_revision": DIFFUSERS_SHA,
        "output": os.fspath(output),
        "total_logical_bytes": sum(item["size"] for item in files),
        "files": files,
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts / "converted_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": os.fspath(manifest_path), "bytes": manifest["total_logical_bytes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
