from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode and validate generated MiniMax H3 MP4 media.")
    parser.add_argument("media", type=Path)
    parser.add_argument("--report", type=Path, default=Path("artifacts/media_probe.json"))
    parser.add_argument("--preview", type=Path, default=Path("outputs/smoke/preview.png"))
    args = parser.parse_args()

    media = args.media.resolve()
    if not media.is_file():
        raise FileNotFoundError(media)

    video_frames = 0
    sampled_pixels = []
    preview_written = False
    with av.open(str(media)) as container:
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            video_frames += 1
            if video_frames in (1, 62, 124):
                sampled_pixels.append(rgb.astype(np.float32))
            if not preview_written and video_frames >= 62:
                args.preview.parent.mkdir(parents=True, exist_ok=True)
                frame.to_image().save(args.preview)
                preview_written = True

    audio_frames = 0
    audio_samples = 0
    squared_sum = 0.0
    peak = 0.0
    with av.open(str(media)) as container:
        for frame in container.decode(audio=0):
            data = frame.to_ndarray().astype(np.float64)
            audio_frames += 1
            audio_samples += data.size
            if np.issubdtype(frame.to_ndarray().dtype, np.integer):
                scale = float(np.iinfo(frame.to_ndarray().dtype).max)
                data /= scale
            squared_sum += float(np.square(data).sum())
            peak = max(peak, float(np.abs(data).max(initial=0.0)))

    pixels = np.concatenate([sample.reshape(-1, 3) for sample in sampled_pixels], axis=0)
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "media": str(media),
        "bytes": media.stat().st_size,
        "video_frames_decoded": video_frames,
        "sampled_pixel_mean": float(pixels.mean()),
        "sampled_pixel_std": float(pixels.std()),
        "audio_frames_decoded": audio_frames,
        "audio_samples_decoded": audio_samples,
        "audio_rms": float((squared_sum / audio_samples) ** 0.5) if audio_samples else 0.0,
        "audio_peak": peak,
        "preview": str(args.preview.resolve()),
    }
    if video_frames < 124:
        raise RuntimeError(f"Expected at least 124 decoded video frames, got {video_frames}")
    if report["sampled_pixel_std"] <= 0.0:
        raise RuntimeError("Decoded video is constant-valued.")
    if audio_samples <= 0 or report["audio_rms"] <= 0.0:
        raise RuntimeError("Decoded audio is missing or silent.")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
