from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.engine_worker import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a lightweight end-to-end Ref2VA validation using an existing Studio request."
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1, help="Consecutive requests using the same loaded pipeline")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    request_path = args.request.resolve()
    output_path = args.output.resolve()
    outputs = [
        output_path
        if args.repeat == 1
        else output_path.with_name(f"{output_path.stem}_run{index}{output_path.suffix}")
        for index in range(1, args.repeat + 1)
    ]
    for current_output in outputs:
        preview_path = current_output.with_suffix(".jpg")
        if current_output.exists() or preview_path.exists():
            raise FileExistsError(f"Validation output already exists: {current_output} or {preview_path}")

    base_request = json.loads(request_path.read_text(encoding="utf-8"))
    pipe = None
    loaded_variant = None
    for index, current_output in enumerate(outputs, start=1):
        preview_path = current_output.with_suffix(".jpg")
        request = dict(base_request)
        request.update(
            {
                "id": f"validation-{base_request['id']}-run{index}",
                "request_path": os.fspath(request_path),
                "width": args.width,
                "height": args.height,
                "num_frames": args.frames,
                "steps": args.steps,
                "output_path": os.fspath(current_output),
                "preview_path": os.fspath(preview_path),
                "result_url": os.fspath(current_output),
                "preview_url": os.fspath(preview_path),
            }
        )
        pipe, loaded_variant = run(request, pipe=pipe, loaded_variant=loaded_variant)


if __name__ == "__main__":
    main()
