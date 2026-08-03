from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import torch
from PIL import Image, ImageOps
from transformers import Qwen3VLProcessor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.engine_worker import _quantized_component_configs
from webui.qwen_conditioner import (
    configure_h3_text_encoder_offload,
    load_h3_text_encoder,
    optimize_h3_text_encoder,
    resolve_reference_image_size,
)


def tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(contiguous.view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark only the H3 Qwen Ref2VA conditioner, without the 33B DiT.")
    parser.add_argument("--request", type=Path, required=True, help="Existing H3 Studio request.json")
    parser.add_argument("--repeat", type=int, default=2, help="Consecutive encodes in the same loaded worker")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    request = json.loads(args.request.resolve().read_text(encoding="utf-8"))
    model_path = Path(request["model_path"])
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    image_items = [item for item in request["references"] if item["kind"] == "image"]
    if len(image_items) != len(request["references"]):
        raise ValueError("This focused benchmark currently accepts image-only Ref2VA requests.")

    prepared = []
    prepared_sizes = []
    from diffusers.modular_pipelines.minimax_h3.packing_ref2va import MiniMaxH3PreparedReference

    for item in image_items:
        with Image.open(item["stored_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            height, width = resolve_reference_image_size(*image.size)
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            else:
                image = image.copy()
        prepared.append(MiniMaxH3PreparedReference(kind="image", image=image))
        prepared_sizes.append({"width": width, "height": height})

    processor = Qwen3VLProcessor.from_pretrained(os.fspath(model_path), subfolder="processor")
    _, text_quant = _quantized_component_configs()
    load_started = time.perf_counter()
    text_encoder = load_h3_text_encoder(model_path, text_quant)
    optimization = optimize_h3_text_encoder(text_encoder)
    configure_h3_text_encoder_offload(text_encoder)
    load_seconds = time.perf_counter() - load_started

    vision = processor.image_processor(images=[entry.image for entry in prepared], return_tensors="pt")
    merge = processor.image_processor.merge_size**2
    image_tokens = [int(grid.prod()) // merge for grid in vision["image_grid_thw"]]
    del vision

    components = SimpleNamespace(
        text_encoder=text_encoder,
        processor=processor,
        tokenizer=processor.tokenizer,
        transformer_ref=SimpleNamespace(dtype=torch.bfloat16),
        _execution_device=torch.device("cuda"),
    )
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep

    process = psutil.Process()
    runs = []
    embedding_shape = None
    embedding_dtype = None
    for index in range(args.repeat):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        prompt_embeds, token_tags = MiniMaxH3Ref2VATextEncoderStep.encode_prompt(
            components,
            request["effective_prompt"],
            prepared,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()
        embedding_shape = list(prompt_embeds.shape)
        embedding_dtype = str(prompt_embeds.dtype)
        runs.append(
            {
                "run": index + 1,
                "encode_seconds": round(time.perf_counter() - started, 3),
                "embedding_sha256": tensor_sha256(prompt_embeds),
                "token_tags_sha256": tensor_sha256(token_tags),
                "finite": bool(torch.isfinite(prompt_embeds).all().item()),
                "cuda_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "cuda_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            }
        )
        del prompt_embeds, token_tags

    reproducible = len({(run["embedding_sha256"], run["token_tags_sha256"]) for run in runs}) == 1
    report = {
        "request": os.fspath(args.request.resolve()),
        "prepared_images": prepared_sizes,
        "image_tokens": image_tokens,
        "total_image_tokens": sum(image_tokens),
        "sequence_length": int(embedding_shape[1]),
        "embedding_shape": embedding_shape,
        "embedding_dtype": embedding_dtype,
        "load_seconds": round(load_seconds, 3),
        "runs": runs,
        "reproducible": reproducible,
        "process_rss_gib": round(process.memory_info().rss / 1024**3, 3),
        "process_private_gib": round(process.memory_full_info().private / 1024**3, 3),
        "optimization": optimization,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not all(run["finite"] for run in runs):
        raise RuntimeError("Qwen produced NaN or Inf conditioning values.")
    if not reproducible:
        raise RuntimeError("Consecutive Qwen conditioning runs produced different tensors.")


if __name__ == "__main__":
    main()
