from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import av
import torch
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
from diffusers.hooks import apply_group_offloading
from diffusers.utils.export_utils import encode_video
from torchao.quantization import Int8WeightOnlyConfig
from transformers import TorchAoConfig as TransformersTorchAoConfig


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.qwen_conditioner import (
    configure_h3_text_encoder_offload,
    load_h3_text_encoder,
    optimize_h3_text_encoder,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal MiniMax H3 text-to-video+audio smoke test.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    config_path = (args.config or (root / "config.example.json")).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = root / "models" / "converted" / "MiniMax-H3-FL2VA-diffusers"
    output = root / config["output"]
    artifacts = root / "artifacts"
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    if not model.is_dir():
        raise RuntimeError("Converted model is missing. Run prepare_diffusers.py first.")

    started = time.perf_counter()
    pipe = ModularPipeline.from_pretrained(os.fspath(model))
    text_encoder = load_h3_text_encoder(
        model,
        TransformersTorchAoConfig(
            Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=[
                "model.visual",
                "model.language_model.embed_tokens",
                "model.language_model.norm",
                "lm_head",
            ],
        ),
    )
    conditioner_report = optimize_h3_text_encoder(text_encoder)
    pipe.update_components(
        transformer=MiniMaxH3Transformer3DModel.from_pretrained(
            os.fspath(model),
            subfolder="transformer",
            dtype=torch.bfloat16,
            quantization_config=TorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=[
                    "proj_in",
                    "audio_proj_in",
                    "context_embedder",
                    "time_embedder",
                    "time_proj",
                    "token_refiner",
                    "norm_out",
                    "proj_out",
                    "audio_proj_out",
                ],
            ),
            # The pinned PR documentation currently says False, but the same
            # pinned loader rejects False for every quantizer. Keep the
            # supported/default True path and record this upstream mismatch.
            low_cpu_mem_usage=True,
        ),
        text_encoder=text_encoder,
    )
    pipe.load_components(dtype=torch.bfloat16)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    offload = {
        "onload_device": torch.device("cuda"),
        "offload_device": torch.device("cpu"),
        "use_stream": True,
    }
    pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
    configure_h3_text_encoder_offload(pipe.text_encoder)
    apply_group_offloading(
        pipe.vae,
        offload_type="leaf_level",
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        use_stream=False,
    )
    # Audio decode queries decoder parameter dtype before nested modules run,
    # so leaf-level offload would leave CPU weights against CUDA latents.
    pipe.audio_vae.to("cuda")

    generator = torch.Generator("cpu").manual_seed(int(config["seed"]))
    state = pipe(
        prompt=config["prompt"],
        width=int(config["width"]),
        height=int(config["height"]),
        num_frames=int(config["num_frames"]),
        num_inference_steps=int(config["num_inference_steps"]),
        generator=generator,
    )
    encode_video(
        state.get("videos")[0],
        fps=int(config["fps"]),
        audio=state.get("audio")[0],
        audio_sample_rate=state.get("sampling_rate"),
        output_path=os.fspath(output),
    )

    with av.open(os.fspath(output)) as container:
        video_streams = [stream for stream in container.streams if stream.type == "video"]
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        duration = float(container.duration / av.time_base) if container.duration else None
        probe = {
            "video_streams": len(video_streams),
            "audio_streams": len(audio_streams),
            "duration_seconds": duration,
            "video_codec": video_streams[0].codec_context.name if video_streams else None,
            "audio_codec": audio_streams[0].codec_context.name if audio_streams else None,
        }
    if probe["video_streams"] < 1 or probe["audio_streams"] < 1:
        raise RuntimeError(f"Output stream verification failed: {probe}")

    report = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "qwen_conditioner": conditioner_report,
        "config": config,
        "output": os.fspath(output),
        "output_bytes": output.stat().st_size,
        "probe": probe,
    }
    report_path = artifacts / "smoke_test.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
