from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from .qwen_conditioner import (
    configure_h3_text_encoder_offload,
    install_reference_size_patch,
    load_h3_text_encoder,
    optimize_h3_text_encoder,
)


EVENT_PREFIX = "H3EVENT "
_PROGRESS_BRIDGE_INSTALLED = False
MINIMUM_MODEL_LOAD_AVAILABLE_RAM_GIB = 225.0
MINIMUM_MODEL_LOAD_AVAILABLE_COMMIT_GIB = 300.0
MINIMUM_MODEL_LOAD_FREE_VRAM_GIB = 24.0
HIGH_MEMORY_REQUEST_PIXEL_FRAMES = 250_000_000
HIGH_MEMORY_REQUEST_FREE_VRAM_GIB = 29.0
LOADED_PIPE_VRAM_ALLOWANCE_GIB = 1.0


def emit(**payload: Any) -> None:
    # Keep the subprocess protocol ASCII-only so Windows console encodings can
    # never corrupt Japanese progress text. json.loads restores the characters.
    print(EVENT_PREFIX + json.dumps(payload, ensure_ascii=True), flush=True)


def _load_request(path: Path) -> dict[str, Any]:
    request = json.loads(path.read_text(encoding="utf-8"))
    request["request_path"] = os.fspath(path)
    return request


def _available_commit_gib() -> float:
    """Return Windows commit headroom, including physical RAM and pagefile."""

    if os.name != "nt":
        import psutil

        return (psutil.virtual_memory().available + psutil.swap_memory().free) / 1024**3

    import ctypes
    from ctypes import wintypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return status.ullAvailPageFile / 1024**3


def _system_free_vram_gib() -> float:
    """Return device-wide free VRAM; WDDM makes torch's process view optimistic."""

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=creationflags,
        )
        first_gpu_mib = float(result.stdout.splitlines()[0].strip())
        return first_gpu_mib / 1024
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        import torch

        return torch.cuda.mem_get_info()[0] / 1024**3


def _required_cold_start_free_vram_gib(request: dict[str, Any]) -> float:
    pixel_frames = int(request["width"]) * int(request["height"]) * int(request["num_frames"])
    if pixel_frames >= HIGH_MEMORY_REQUEST_PIXEL_FRAMES:
        return HIGH_MEMORY_REQUEST_FREE_VRAM_GIB
    return MINIMUM_MODEL_LOAD_FREE_VRAM_GIB


def _install_progress_bridge() -> None:
    global _PROGRESS_BRIDGE_INSTALLED
    if _PROGRESS_BRIDGE_INSTALLED:
        return

    import torch
    from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3AudioDecodeStep, MiniMaxH3VideoDecodeStep
    from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3DenoiseLoopWrapper
    from diffusers.modular_pipelines.minimax_h3.encoders import (
        MiniMaxH3KeyframeVaeEncoderStep,
        MiniMaxH3Ref2VAReferenceEncoderStep,
        MiniMaxH3Ref2VATextEncoderStep,
        MiniMaxH3TextEncoderStep,
    )

    original_text_encoder = MiniMaxH3TextEncoderStep.__call__
    original_keyframe_encoder = MiniMaxH3KeyframeVaeEncoderStep.__call__
    original_ref_text_encoder = MiniMaxH3Ref2VATextEncoderStep.__call__
    original_reference_encoder = MiniMaxH3Ref2VAReferenceEncoderStep.__call__
    original_video_decoder = MiniMaxH3VideoDecodeStep.__call__
    original_audio_decoder = MiniMaxH3AudioDecodeStep.__call__

    @torch.no_grad()
    def progressing_text_encoder(self, components, state):
        emit(
            status="running",
            phase="プロンプトを解析しています",
            message="Qwen3-VLでプロンプトを読み取っています。",
            progress=30,
        )
        result = original_text_encoder(self, components, state)
        emit(
            status="running",
            phase="参照フレームを準備しています",
            message="開始・終了フレームを潜在表現へ変換しています。",
            progress=45,
        )
        return result

    @torch.no_grad()
    def progressing_keyframe_encoder(self, components, state):
        result = original_keyframe_encoder(self, components, state)
        emit(
            status="running",
            phase="生成データを組み立てています",
            message="映像と音声の生成レイアウトを準備しています。",
            progress=52,
        )
        return result

    @torch.no_grad()
    def progressing_ref_text_encoder(self, components, state):
        emit(
            status="running",
            phase="参照素材を解析しています",
            message="Qwen3-VLで参照素材とプロンプトを読み取っています。画像は元解像度を保ち、短辺2048pxを上限に処理します。",
            progress=30,
        )
        result = original_ref_text_encoder(self, components, state)
        emit(
            status="running",
            phase="参照素材を圧縮しています",
            message="参照素材をH3の潜在表現へ変換しています。",
            progress=45,
        )
        return result

    @torch.no_grad()
    def progressing_reference_encoder(self, components, state):
        result = original_reference_encoder(self, components, state)
        emit(
            status="running",
            phase="生成データを組み立てています",
            message="参照素材、映像、音声の生成レイアウトを準備しています。",
            progress=52,
        )
        return result

    @torch.no_grad()
    def progressing_video_decoder(self, components, state):
        emit(
            status="running",
            phase="映像を復元しています",
            message="生成した潜在表現を動画フレームへ戻しています。",
            progress=91,
        )
        result = original_video_decoder(self, components, state)
        emit(
            status="running",
            phase="音声を復元しています",
            message="生成した潜在表現をステレオ音声へ戻しています。",
            progress=92,
        )
        return result

    @torch.no_grad()
    def progressing_audio_decoder(self, components, state):
        result = original_audio_decoder(self, components, state)
        emit(
            status="running",
            phase="生成結果を仕上げています",
            message="映像と音声の復元が完了しました。",
            progress=93,
        )
        return result

    @torch.no_grad()
    def progressing_denoise(self, components, state):
        block_state = self.get_block_state(state)
        total = len(block_state.timesteps)
        emit(
            status="running",
            phase="映像と音を生成しています",
            message=f"デノイズ 0 / {total}",
            progress=55,
            step=0,
            total_steps=total,
        )
        for index, timestep in enumerate(block_state.timesteps):
            components, block_state = self.loop_step(
                components,
                block_state,
                i=index,
                t=timestep,
            )
            completed = index + 1
            progress = 55 + 35 * completed / max(total, 1)
            emit(
                status="running",
                phase="映像と音を生成しています",
                message=f"デノイズ {completed} / {total}",
                progress=min(round(progress, 2), 90),
                step=completed,
                total_steps=total,
            )
        self.set_block_state(state, block_state)
        return components, state

    MiniMaxH3DenoiseLoopWrapper.__call__ = progressing_denoise
    MiniMaxH3TextEncoderStep.__call__ = progressing_text_encoder
    MiniMaxH3KeyframeVaeEncoderStep.__call__ = progressing_keyframe_encoder
    MiniMaxH3Ref2VATextEncoderStep.__call__ = progressing_ref_text_encoder
    MiniMaxH3Ref2VAReferenceEncoderStep.__call__ = progressing_reference_encoder
    MiniMaxH3VideoDecodeStep.__call__ = progressing_video_decoder
    MiniMaxH3AudioDecodeStep.__call__ = progressing_audio_decoder
    _PROGRESS_BRIDGE_INSTALLED = True


def _quantized_component_configs():
    from diffusers import TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    transformer = TorchAoConfig(
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
    )
    text_encoder = TransformersTorchAoConfig(
        Int8WeightOnlyConfig(version=2),
        modules_to_not_convert=[
            "model.visual",
            "model.language_model.embed_tokens",
            "model.language_model.norm",
            "lm_head",
        ],
    )
    return transformer, text_encoder


def _configure_offload(pipe, transformer_name: str) -> None:
    import torch
    from diffusers.hooks import apply_group_offloading

    offload = {
        "onload_device": torch.device("cuda"),
        "offload_device": torch.device("cpu"),
        "use_stream": True,
    }
    transformer = getattr(pipe, transformer_name)
    transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
    configure_h3_text_encoder_offload(pipe.text_encoder)
    apply_group_offloading(
        pipe.vae,
        offload_type="leaf_level",
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        use_stream=False,
    )
    # The audio VAE's decoder inspects its own parameter dtype before its
    # nested convolutions run. Leaf hooks leave those weights on CPU at that
    # point and cause a CUDA-input/CPU-weight mismatch. It is small enough to
    # keep resident and this is the verified upstream-compatible path.
    pipe.audio_vae.to("cuda")


def _load_fl2va_pipeline(model: Path):
    import torch
    from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline

    transformer_quant, text_quant = _quantized_component_configs()
    emit(status="running", phase="モデルを準備しています", message="H3の構成を読み込んでいます。", progress=4)
    pipe = ModularPipeline.from_pretrained(os.fspath(model))
    emit(
        status="running",
        phase="モデルを準備しています",
        message="Qwen3-VLテキストエンコーダーを先に読み込み、8bit化しています。",
        progress=7,
    )
    text_encoder = load_h3_text_encoder(model, text_quant)
    conditioner_report = optimize_h3_text_encoder(text_encoder)
    emit(
        status="running",
        phase="モデルを準備しています",
        message="33B Transformerを読み込み、8bit化しています。",
        progress=17,
    )
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        os.fspath(model),
        subfolder="transformer",
        dtype=torch.bfloat16,
        quantization_config=transformer_quant,
        low_cpu_mem_usage=True,
    )
    emit(
        status="running",
        phase="モデルを準備しています",
        message=(
            f"Qwen3-VLをH3専用の{conditioner_report['active_decoder_layers']}層・"
            f"{conditioner_report['attention'].upper()}経路へ最適化しました。"
        ),
        progress=23,
    )
    pipe.update_components(transformer=transformer, text_encoder=text_encoder)
    emit(
        status="running",
        phase="モデルを準備しています",
        message="動画・音声デコーダーを読み込んでいます。",
        progress=26,
    )
    pipe.load_components(dtype=torch.bfloat16)
    _configure_offload(pipe, "transformer")
    return pipe


def _load_ref2va_pipeline(model: Path):
    import torch
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.modular_pipelines import MiniMaxH3Ref2VABlocks

    transformer_quant, text_quant = _quantized_component_configs()
    emit(status="running", phase="Omniモデルを準備しています", message="Ref2VAの構成を読み込んでいます。", progress=4)
    pipe = MiniMaxH3Ref2VABlocks().init_pipeline(os.fspath(model))
    emit(
        status="running",
        phase="Omniモデルを準備しています",
        message="Qwen3-VLテキストエンコーダーを先に読み込み、8bit化しています。",
        progress=7,
    )
    text_encoder = load_h3_text_encoder(model, text_quant)
    conditioner_report = optimize_h3_text_encoder(text_encoder)
    emit(
        status="running",
        phase="Omniモデルを準備しています",
        message="Ref2VA Transformerを読み込み、8bit化しています。",
        progress=17,
    )
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        os.fspath(model),
        subfolder="transformer_ref",
        dtype=torch.bfloat16,
        quantization_config=transformer_quant,
        low_cpu_mem_usage=True,
    )
    emit(
        status="running",
        phase="Omniモデルを準備しています",
        message=(
            f"Qwen3-VLをH3専用の{conditioner_report['active_decoder_layers']}層・"
            f"{conditioner_report['attention'].upper()}経路へ最適化しました。"
        ),
        progress=23,
    )
    pipe.update_components(transformer_ref=transformer, text_encoder=text_encoder)
    emit(
        status="running",
        phase="Omniモデルを準備しています",
        message="参照メディア用デコーダーを読み込んでいます。",
        progress=26,
    )
    pipe.load_components(dtype=torch.bfloat16)
    _configure_offload(pipe, "transformer_ref")
    return pipe


def _prepare_generation_inputs(request: dict[str, Any]) -> dict[str, Any]:
    import torch
    from PIL import Image, ImageOps

    kwargs: dict[str, Any] = {
        "prompt": request["effective_prompt"],
        "width": int(request["width"]),
        "height": int(request["height"]),
        "num_frames": int(request["num_frames"]),
        "num_inference_steps": int(request["steps"]),
        "generator": torch.Generator("cpu").manual_seed(int(request["seed"])),
    }
    mode = request["mode"]
    if mode in {"i2v", "first_last"}:
        first = Path(request["first_image"])
        with Image.open(first) as image:
            kwargs["image"] = ImageOps.exif_transpose(image).convert("RGB").copy()
    if mode == "first_last":
        last = Path(request["last_image"])
        with Image.open(last) as image:
            kwargs["last_image"] = ImageOps.exif_transpose(image).convert("RGB").copy()
    return kwargs


def _prepare_references(request: dict[str, Any]):
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference

    references = []
    for item in request["references"]:
        path = item["stored_path"]
        kind = item["kind"]
        if kind == "image":
            references.append(MiniMaxH3Reference(image=path))
        elif kind == "video":
            references.append(MiniMaxH3Reference(video=path))
        elif kind == "audio":
            references.append(MiniMaxH3Reference(audio=path))
        else:
            raise ValueError(f"Unsupported reference kind: {kind}")
    return references


def _apply_audio_gain(audio, gain_db: float):
    gain_db = float(gain_db)
    if not gain_db:
        return audio
    gain = 10.0 ** (gain_db / 20.0)
    return audio.mul(gain).clamp(-1.0, 1.0)


def _write_video_and_preview(state, request: dict[str, Any]) -> dict[str, Any]:
    import av
    from diffusers.utils.export_utils import encode_video
    from PIL import Image

    output = Path(request["output_path"])
    preview = Path(request["preview_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.parent.mkdir(parents=True, exist_ok=True)
    emit(status="running", phase="動画を書き出しています", message="映像と音声をMP4へまとめています。", progress=96)
    audio_gain_db = float(request.get("audio_gain_db", 0.0))
    audio = _apply_audio_gain(state.get("audio")[0], audio_gain_db)
    encode_video(
        state.get("videos")[0],
        fps=24,
        audio=audio,
        audio_sample_rate=state.get("sampling_rate"),
        output_path=os.fspath(output),
    )

    with av.open(os.fspath(output)) as container:
        video_streams = [stream for stream in container.streams if stream.type == "video"]
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        duration = float(container.duration / av.time_base) if container.duration else None
        if not video_streams or not audio_streams:
            raise RuntimeError("出力MP4に動画または音声ストリームがありません。")
        for frame in container.decode(video=0):
            Image.fromarray(frame.to_ndarray(format="rgb24")).save(preview, quality=88)
            break
        return {
            "bytes": output.stat().st_size,
            "duration_seconds": duration,
            "video_codec": video_streams[0].codec_context.name,
            "audio_codec": audio_streams[0].codec_context.name,
        }


def _run_mock(request: dict[str, Any]) -> None:
    import shutil
    import time

    root = Path(__file__).resolve().parents[1]
    source_video = root / "outputs" / "smoke" / "minimax_h3_smoke.mp4"
    source_preview = root / "outputs" / "smoke" / "preview.png"
    output = Path(request["output_path"])
    preview = Path(request["preview_path"])
    if not source_video.is_file() or not source_preview.is_file():
        raise FileNotFoundError("モック用スモーク出力がありません。")
    events = [
        (7, "モデルを準備しています", "H3 Transformerを確認しています。"),
        (17, "モデルを準備しています", "Qwen3-VLを確認しています。"),
        (26, "モデルを準備しています", "動画・音声デコーダーを確認しています。"),
        (30, "参照素材を解析しています", "Qwen3-VLでプロンプトと参照素材を読み取っています。"),
        (45, "参照素材を圧縮しています", "参照素材をH3の潜在表現へ変換しています。"),
        (52, "生成データを組み立てています", "映像と音声の生成レイアウトを準備しています。"),
        (55, "映像と音を生成しています", "デノイズ 0 / 3"),
        (67, "映像と音を生成しています", "デノイズ 1 / 3"),
        (78, "映像と音を生成しています", "デノイズ 2 / 3"),
        (90, "映像と音を生成しています", "デノイズ 3 / 3"),
        (91, "映像を復元しています", "生成した潜在表現を動画フレームへ戻しています。"),
        (92, "音声を復元しています", "生成した潜在表現をステレオ音声へ戻しています。"),
        (93, "生成結果を仕上げています", "映像と音声の復元が完了しました。"),
        (96, "動画を書き出しています", "映像と音声をMP4へまとめています。"),
    ]
    for progress, phase, message in events:
        emit(status="running", phase=phase, message=message, progress=progress)
        time.sleep(0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_video, output)
    shutil.copy2(source_preview, preview)
    emit(
        status="completed",
        phase="完成しました",
        message="モック生成が完了しました。",
        progress=100,
        result=request["result_url"],
        preview=request["preview_url"],
        media={"bytes": output.stat().st_size, "duration_seconds": 5.175, "video_codec": "h264", "audio_codec": "aac"},
    )


def _external_cli_generation() -> str | None:
    import psutil

    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.info["pid"] == os.getpid():
            continue
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if "inference_smoke.py" in command:
            return command
    return None


def run(request: dict[str, Any], pipe=None, loaded_variant: str | None = None):
    if os.environ.get("H3_WEBUI_MOCK") == "1":
        _run_mock(request)
        return pipe, loaded_variant

    import gc
    import psutil
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDAが利用できません。")
    external = _external_cli_generation()
    if external:
        raise RuntimeError("PowerShellで開始した別のH3生成が実行中です。完了後にもう一度生成してください。")
    _install_progress_bridge()
    install_reference_size_patch()
    mode = request["mode"]
    requested_variant = "ref2va" if mode == "omni" else "fl2va"
    required_free_vram_gib = _required_cold_start_free_vram_gib(request)
    model = Path(request["model_path"])
    if not model.is_dir():
        raise FileNotFoundError(f"変換済みモデルが見つかりません: {model}")

    if pipe is None or loaded_variant != requested_variant:
        if pipe is not None:
            emit(
                status="running",
                phase="モデルを切り替えています",
                message="前のモデルを解放しています。",
                progress=2,
            )
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
        available_gib = psutil.virtual_memory().available / 1024**3
        if available_gib < MINIMUM_MODEL_LOAD_AVAILABLE_RAM_GIB:
            raise RuntimeError(
                "安定したモデル準備に必要な空きRAMが不足しています"
                f"（現在 {available_gib:.1f}GiB、必要 {MINIMUM_MODEL_LOAD_AVAILABLE_RAM_GIB:.0f}GiB以上）。"
                "ページファイルへ退避すると極端に遅くなるため、他の重いアプリを閉じてください。"
            )
        available_commit_gib = _available_commit_gib()
        if available_commit_gib < MINIMUM_MODEL_LOAD_AVAILABLE_COMMIT_GIB:
            raise RuntimeError(
                "モデルの一時展開に必要なWindows commit余力が不足しています"
                f"（現在 {available_commit_gib:.1f}GiB、必要 {MINIMUM_MODEL_LOAD_AVAILABLE_COMMIT_GIB:.0f}GiB以上）。"
                "ページファイルを自動管理へ戻すか、他の重いアプリを閉じてください。"
            )
        free_vram_gib = _system_free_vram_gib()
        if free_vram_gib < required_free_vram_gib:
            raise RuntimeError(
                "安定したH3生成に必要な空きVRAMが不足しています"
                f"（現在 {free_vram_gib:.1f}GiB、今回の設定には {required_free_vram_gib:.0f}GiB以上必要）。"
                "GPUを使うゲーム、3Dツール、別のAI処理を閉じてください。"
            )
        if mode == "omni":
            pipe = _load_ref2va_pipeline(model)
        else:
            pipe = _load_fl2va_pipeline(model)
        loaded_variant = requested_variant
        pipe._h3_cold_start_free_vram_gib = free_vram_gib
    else:
        cold_start_free_vram_gib = getattr(pipe, "_h3_cold_start_free_vram_gib", 0.0)
        if cold_start_free_vram_gib < required_free_vram_gib:
            raise RuntimeError(
                f"今回の高負荷設定には起動時空きVRAM {required_free_vram_gib:.0f}GiB以上が必要です。"
                f"このworkerの起動時は {cold_start_free_vram_gib:.1f}GiBでした。"
                "GPUを使うアプリを閉じ、H3 StudioのPowerShellを再起動してください。"
            )
        gc.collect()
        torch.cuda.empty_cache()
        current_free_vram_gib = _system_free_vram_gib()
        required_current_free_vram_gib = required_free_vram_gib - LOADED_PIPE_VRAM_ALLOWANCE_GIB
        if current_free_vram_gib < required_current_free_vram_gib:
            raise RuntimeError(
                "モデルのロード後に別アプリがVRAMを使用し始めました"
                f"（現在 {current_free_vram_gib:.1f}GiB、今回の設定には"
                f" {required_current_free_vram_gib:.0f}GiB以上必要）。"
                "GPUを使うアプリを閉じてから再実行してください。"
            )
        emit(
            status="running",
            phase="モデルは準備済みです",
            message="前回ロードしたモデルを再利用します。",
            progress=29,
        )
    emit(status="running", phase="入力を準備しています", message="プロンプトと参照素材を処理しています。", progress=30)

    if mode == "omni":
        kwargs = {
            "prompt": request["effective_prompt"],
            "references": _prepare_references(request),
            "width": int(request["width"]),
            "height": int(request["height"]),
            "num_frames": int(request["num_frames"]),
            "num_inference_steps": int(request["steps"]),
            "generator": torch.Generator("cpu").manual_seed(int(request["seed"])),
        }
    else:
        kwargs = _prepare_generation_inputs(request)
    state = pipe(**kwargs)
    emit(status="running", phase="仕上げています", message="生成結果を動画と音声へ変換しています。", progress=93)
    media = _write_video_and_preview(state, request)
    emit(
        status="completed",
        phase="完成しました",
        message="動画と音声の生成が完了しました。",
        progress=100,
        result=request["result_url"],
        preview=request["preview_url"],
        media=media,
    )
    del state
    gc.collect()
    torch.cuda.empty_cache()
    return pipe, loaded_variant


def _emit_failure(exc: Exception) -> None:
    emit(
        status="failed",
        phase="生成に失敗しました",
        message=str(exc) or exc.__class__.__name__,
        progress=0,
    )
    traceback.print_exc()


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--request", type=Path)
    group.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    if args.request:
        try:
            request = _load_request(args.request.resolve())
            run(request)
            return 0
        except Exception as exc:
            _emit_failure(exc)
            return 1

    pipe = None
    loaded_variant = None
    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        try:
            command = json.loads(raw_line)
            request_path = Path(command["request"]).resolve()
            request = _load_request(request_path)
            pipe, loaded_variant = run(request, pipe, loaded_variant)
        except Exception as exc:
            _emit_failure(exc)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
