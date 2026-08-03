from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPO_ID = "MiniMaxAI/MiniMax-H3"
REVISION = "af0fe5abe6fd50d632b65a82fef321c4c5c1f249"

# Official FL2VA only: text-to-video+audio and first/last-frame conditioning.
# Python implementation files from the model repository are intentionally excluded.
ALLOW_PATTERNS = [
    "README.md",
    "LICENSE",
    "docs/QA-about-License.md",
    "FL2VA/model_index.json",
    "FL2VA/transformer/*.json",
    "FL2VA/transformer/*.safetensors",
    "FL2VA/text_encoder/*.json",
    "FL2VA/text_encoder/*.txt",
    "FL2VA/text_encoder/*.safetensors",
    "FL2VA/processor/*.json",
    "FL2VA/processor/*.txt",
    "FL2VA/tokenizer/*.json",
    "FL2VA/tokenizer/*.txt",
    "FL2VA/video_vae/config.json",
    "FL2VA/video_vae/source/config.json",
    "FL2VA/video_vae/source/model.safetensors",
    "FL2VA/audio_vae/config.json",
    "FL2VA/audio_vae/config.yaml",
    "FL2VA/audio_vae/metadata.json",
    "FL2VA/audio_vae/model.safetensors",
]


def selected(path: str) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in ALLOW_PATTERNS)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify official MiniMax H3 FL2VA files.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-sha256", action="store_true", help="Only verify byte sizes; not recommended.")
    args = parser.parse_args()

    root = args.root.resolve()
    destination = root / "models" / "official" / "MiniMax-H3"
    artifacts = root / "artifacts"
    destination.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.model_info(REPO_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError(f"Resolved revision mismatch: expected {REVISION}, got {info.sha}")
    if info.private or info.gated:
        raise RuntimeError("Official repository is private or gated; refusing unattended download.")

    remote = []
    for sibling in info.siblings:
        if not selected(sibling.rfilename):
            continue
        lfs = getattr(sibling, "lfs", None)
        remote.append(
            {
                "path": sibling.rfilename,
                "size": int(getattr(sibling, "size", 0) or 0),
                "lfs_sha256": (lfs or {}).get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None),
            }
        )
    if not remote or not any(item["path"].endswith(".safetensors") for item in remote):
        raise RuntimeError("No official Safetensors files matched the fixed manifest.")

    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=destination,
        allow_patterns=ALLOW_PATTERNS,
        max_workers=args.workers,
    )

    verified = []
    errors = []
    for item in sorted(remote, key=lambda value: value["path"]):
        local = destination / Path(item["path"])
        if not local.is_file():
            errors.append(f"missing: {item['path']}")
            continue
        actual_size = local.stat().st_size
        if item["size"] and actual_size != item["size"]:
            errors.append(f"size mismatch: {item['path']} expected={item['size']} actual={actual_size}")
        actual_sha = None if args.skip_sha256 else sha256_file(local)
        if actual_sha and item["lfs_sha256"] and actual_sha != item["lfs_sha256"]:
            errors.append(f"sha256 mismatch: {item['path']}")
        verified.append({**item, "actual_size": actual_size, "sha256": actual_sha})

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license": "MiniMax H3 Community License Agreement",
        "territory_note": "Applicable worldwide except EU, UK, Republic of Korea, and USA; this setup was performed in Japan.",
        "selected_total_bytes": sum(item["actual_size"] for item in verified),
        "files": verified,
        "errors": errors,
    }
    manifest_path = artifacts / "official_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": os.fspath(manifest_path), "bytes": manifest["selected_total_bytes"], "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
