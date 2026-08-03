"""Build ComfyUI API prompt graphs for the native MiniMax H3 nodes.

The graph shape and node input names in this module are pinned to ComfyUI
``14b05228cef127ce529bc0c08660770d4af3e9a8``.  The model filenames match the
MiniMax H3 workflow templates distributed with that revision.

The returned prompt is the dictionary accepted by ComfyUI's ``/prompt`` API;
it deliberately contains no frontend-only workflow data (positions, links,
widgets, and so on).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


COMFYUI_COMMIT = "14b05228cef127ce529bc0c08660770d4af3e9a8"

FL2VA_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF2VA_MODEL = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER_MODEL = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE_MODEL = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE_MODEL = "minimax_h3_audio_vae_fp32.safetensors"

UNET_NODE_ID = "1"
EASYCACHE_NODE_ID = "2"
CLIP_NODE_ID = "3"
VIDEO_VAE_NODE_ID = "4"
AUDIO_VAE_NODE_ID = "5"
NOISE_NODE_ID = "6"
SAMPLER_SELECT_NODE_ID = "7"
SCHEDULER_NODE_ID = "8"
CONDITIONING_NODE_ID = "9"
GUIDER_NODE_ID = "10"
SAMPLER_NODE_ID = "11"
VIDEO_DECODE_NODE_ID = "12"
AUDIO_DECODE_NODE_ID = "13"
CREATE_VIDEO_NODE_ID = "14"
SAVE_VIDEO_NODE_ID = "15"
AUDIO_VOLUME_NODE_ID = "16"
FIRST_FRAME_NODE_ID = "20"
LAST_FRAME_NODE_ID = "21"

SUPPORTED_MODES = frozenset({"t2v", "i2v", "first_last", "omni"})
EASYCACHE_PRESETS: dict[str, tuple[float, float, float] | None] = {
    "off": None,
    "conservative": (0.20, 0.20, 0.90),
    "balanced": (0.30, 0.20, 0.90),
}
MIN_EASYCACHE_STEPS = 12


PromptGraph = dict[str, dict[str, Any]]
WorkflowEnvelope = dict[str, Any]


def _edge(node_id: str, output_index: int = 0) -> list[str | int]:
    return [node_id, output_index]


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def _file_list(value: Sequence[str] | None, *, name: str, maximum: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of ComfyUI input filenames, not one string")
    files = list(value)
    if len(files) > maximum:
        raise ValueError(f"{name} accepts at most {maximum} files")
    if any(not isinstance(filename, str) or not filename.strip() for filename in files):
        raise ValueError(f"{name} contains an empty or non-string filename")
    return files


def _validate_parameters(
    *,
    mode: str,
    prompt: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    seed: int,
    output_prefix: str,
    easycache: str,
    audio_gain_db: float,
    input_filename: str | None,
    last_frame_filename: str | None,
    ref_image_size: str,
) -> None:
    if mode not in SUPPORTED_MODES:
        choices = ", ".join(sorted(SUPPORTED_MODES))
        raise ValueError(f"unsupported mode {mode!r}; expected one of: {choices}")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not isinstance(width, int) or not isinstance(height, int):
        raise TypeError("width and height must be integers")
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError("width and height must be positive multiples of 32")
    if not isinstance(frames, int) or not 5 <= frames <= 3600:
        raise ValueError("frames must be an integer between 5 and 3600")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if not isinstance(output_prefix, str) or not output_prefix.strip():
        raise ValueError("output_prefix must be a non-empty string")
    if easycache not in EASYCACHE_PRESETS:
        choices = ", ".join(EASYCACHE_PRESETS)
        raise ValueError(f"unsupported EasyCache preset {easycache!r}; expected one of: {choices}")
    if (
        not isinstance(audio_gain_db, (int, float))
        or isinstance(audio_gain_db, bool)
        or not -100 <= audio_gain_db <= 100
        or not float(audio_gain_db).is_integer()
    ):
        raise ValueError("audio_gain_db must be a whole number between -100 and 100")
    if ref_image_size not in {"match", "max"}:
        raise ValueError("ref_image_size must be 'match' or 'max'")

    if mode in {"i2v", "first_last"} and not input_filename:
        raise ValueError(f"{mode} mode requires input_filename")
    if mode == "first_last" and not last_frame_filename:
        raise ValueError("first_last mode requires last_frame_filename")
    for name, filename in (
        ("input_filename", input_filename),
        ("last_frame_filename", last_frame_filename),
    ):
        if filename is not None and (not isinstance(filename, str) or not filename.strip()):
            raise ValueError(f"{name} must be a non-empty string when provided")


def _add_shared_nodes(
    graph: PromptGraph,
    *,
    model_filename: str,
    steps: int,
    seed: int,
    output_prefix: str,
    audio_gain_db: float,
    effective_easycache: str,
) -> list[str | int]:
    graph[UNET_NODE_ID] = _node(
        "UNETLoader",
        unet_name=model_filename,
        weight_dtype="default",
    )

    model_edge = _edge(UNET_NODE_ID)
    cache_values = EASYCACHE_PRESETS[effective_easycache]
    if cache_values is not None:
        reuse_threshold, start_percent, end_percent = cache_values
        graph[EASYCACHE_NODE_ID] = _node(
            "EasyCache",
            model=model_edge,
            reuse_threshold=reuse_threshold,
            start_percent=start_percent,
            end_percent=end_percent,
            verbose=False,
        )
        model_edge = _edge(EASYCACHE_NODE_ID)

    graph[CLIP_NODE_ID] = _node(
        "CLIPLoader",
        clip_name=TEXT_ENCODER_MODEL,
        type="minimax",
        device="default",
    )
    graph[VIDEO_VAE_NODE_ID] = _node("VAELoader", vae_name=VIDEO_VAE_MODEL)
    graph[AUDIO_VAE_NODE_ID] = _node("VAELoader", vae_name=AUDIO_VAE_MODEL)
    graph[NOISE_NODE_ID] = _node("RandomNoise", noise_seed=seed)
    graph[SAMPLER_SELECT_NODE_ID] = _node("KSamplerSelect", sampler_name="res_multistep")
    graph[SCHEDULER_NODE_ID] = _node(
        "BasicScheduler",
        model=model_edge,
        scheduler="simple",
        steps=steps,
        denoise=1.0,
    )

    # Conditioning is inserted by the mode-specific builder below.  The
    # remaining nodes may safely reference it before it appears in the JSON.
    graph[GUIDER_NODE_ID] = _node(
        "BasicGuider",
        model=model_edge,
        conditioning=_edge(CONDITIONING_NODE_ID, 0),
    )
    graph[SAMPLER_NODE_ID] = _node(
        "SamplerCustomAdvanced",
        noise=_edge(NOISE_NODE_ID),
        guider=_edge(GUIDER_NODE_ID),
        sampler=_edge(SAMPLER_SELECT_NODE_ID),
        sigmas=_edge(SCHEDULER_NODE_ID),
        latent_image=_edge(CONDITIONING_NODE_ID, 1),
    )
    graph[VIDEO_DECODE_NODE_ID] = _node(
        "VAEDecode",
        samples=_edge(SAMPLER_NODE_ID, 0),
        vae=_edge(VIDEO_VAE_NODE_ID),
    )
    graph[AUDIO_DECODE_NODE_ID] = _node(
        "VAEDecodeAudio",
        samples=_edge(SAMPLER_NODE_ID, 0),
        vae=_edge(AUDIO_VAE_NODE_ID),
    )
    graph[AUDIO_VOLUME_NODE_ID] = _node(
        "AudioAdjustVolume",
        audio=_edge(AUDIO_DECODE_NODE_ID),
        volume=int(audio_gain_db),
    )
    graph[CREATE_VIDEO_NODE_ID] = _node(
        "CreateVideo",
        images=_edge(VIDEO_DECODE_NODE_ID),
        audio=_edge(AUDIO_VOLUME_NODE_ID),
        fps=24.0,
        bit_depth=8,
    )
    graph[SAVE_VIDEO_NODE_ID] = _node(
        "SaveVideo",
        video=_edge(CREATE_VIDEO_NODE_ID),
        filename_prefix=output_prefix,
        format="auto",
        # Comfy's COMFY_DYNAMICCOMBO_V3 API receives the selected option key
        # as a string.  The execution layer expands it to {"codec": "auto"}.
        codec="auto",
    )
    return model_edge


def _add_fl2va_conditioning(
    graph: PromptGraph,
    *,
    mode: str,
    prompt: str,
    width: int,
    height: int,
    frames: int,
    input_filename: str | None,
    last_frame_filename: str | None,
) -> None:
    inputs: dict[str, Any] = {
        "clip": _edge(CLIP_NODE_ID),
        "vae": _edge(VIDEO_VAE_NODE_ID),
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": frames,
    }
    if mode in {"i2v", "first_last"}:
        assert input_filename is not None
        graph[FIRST_FRAME_NODE_ID] = _node("LoadImage", image=input_filename)
        inputs["first_frame"] = _edge(FIRST_FRAME_NODE_ID)
    if mode == "first_last":
        assert last_frame_filename is not None
        graph[LAST_FRAME_NODE_ID] = _node("LoadImage", image=last_frame_filename)
        inputs["last_frame"] = _edge(LAST_FRAME_NODE_ID)
    graph[CONDITIONING_NODE_ID] = _node("MiniMaxH3ImageToVideo", **inputs)


def _add_omni_conditioning(
    graph: PromptGraph,
    *,
    prompt: str,
    width: int,
    height: int,
    frames: int,
    reference_images: list[str],
    reference_videos: list[str],
    reference_audios: list[str],
    ref_image_size: str,
) -> None:
    inputs: dict[str, Any] = {
        "clip": _edge(CLIP_NODE_ID),
        "vae": _edge(VIDEO_VAE_NODE_ID),
        "audio_vae": _edge(AUDIO_VAE_NODE_ID),
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": frames,
        "ref_image_size": ref_image_size,
    }

    # Use distinct fixed ranges so graph inspection and history debugging stay
    # readable.  The numeric suffixes on the autogrow inputs determine the
    # <Picture i>, <Video k>, and <Audio j> ordering seen by MiniMax H3.
    for index, filename in enumerate(reference_images):
        node_id = str(100 + index)
        graph[node_id] = _node("LoadImage", image=filename)
        inputs[f"ref_images.ref_image_{index}"] = _edge(node_id)

    for index, filename in enumerate(reference_videos):
        load_id = str(200 + index * 2)
        components_id = str(201 + index * 2)
        graph[load_id] = _node("LoadVideo", file=filename)
        graph[components_id] = _node("GetVideoComponents", video=_edge(load_id))
        inputs[f"ref_videos.ref_video_{index}"] = _edge(components_id, 0)
        inputs[f"ref_video_audios.ref_video_audio_{index}"] = _edge(components_id, 1)

    for index, filename in enumerate(reference_audios):
        node_id = str(300 + index)
        graph[node_id] = _node("LoadAudio", audio=filename)
        inputs[f"ref_audios.ref_audio_{index}"] = _edge(node_id)

    graph[CONDITIONING_NODE_ID] = _node("MiniMaxH3ReferenceToVideo", **inputs)


def build_h3_workflow(
    *,
    mode: str,
    prompt: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    seed: int,
    output_prefix: str,
    audio_gain_db: float = 0.0,
    input_filename: str | None = None,
    last_frame_filename: str | None = None,
    reference_image_filenames: Sequence[str] | None = None,
    reference_video_filenames: Sequence[str] | None = None,
    reference_audio_filenames: Sequence[str] | None = None,
    ref_image_size: str = "match",
    easycache: str = "off",
) -> WorkflowEnvelope:
    """Return a JSON-serializable MiniMax H3 ComfyUI workflow envelope.

    ``input_filename`` is the first frame for ``i2v`` and ``first_last``;
    ``last_frame_filename`` is additionally required by ``first_last``.
    Omni filenames are names in ComfyUI's input directory, in the exact order
    that their ``<Picture i>``, ``<Video k>``, and ``<Audio j>`` tags use.

    EasyCache is intentionally disabled below 12 sampling steps: its warm-up
    and non-cache tail leave too few useful steps for a meaningful win there.
    The metadata records both the requested and effective preset.
    """

    _validate_parameters(
        mode=mode,
        prompt=prompt,
        width=width,
        height=height,
        frames=frames,
        steps=steps,
        seed=seed,
        output_prefix=output_prefix,
        easycache=easycache,
        audio_gain_db=audio_gain_db,
        input_filename=input_filename,
        last_frame_filename=last_frame_filename,
        ref_image_size=ref_image_size,
    )
    reference_images = _file_list(
        reference_image_filenames,
        name="reference_image_filenames",
        maximum=9,
    )
    reference_videos = _file_list(
        reference_video_filenames,
        name="reference_video_filenames",
        maximum=3,
    )
    reference_audios = _file_list(
        reference_audio_filenames,
        name="reference_audio_filenames",
        maximum=3,
    )

    effective_easycache = easycache if steps >= MIN_EASYCACHE_STEPS else "off"
    graph: PromptGraph = {}
    model_filename = REF2VA_MODEL if mode == "omni" else FL2VA_MODEL
    _add_shared_nodes(
        graph,
        model_filename=model_filename,
        steps=steps,
        seed=seed,
        output_prefix=output_prefix,
        audio_gain_db=audio_gain_db,
        effective_easycache=effective_easycache,
    )

    if mode == "omni":
        _add_omni_conditioning(
            graph,
            prompt=prompt,
            width=width,
            height=height,
            frames=frames,
            reference_images=reference_images,
            reference_videos=reference_videos,
            reference_audios=reference_audios,
            ref_image_size=ref_image_size,
        )
    else:
        _add_fl2va_conditioning(
            graph,
            mode=mode,
            prompt=prompt,
            width=width,
            height=height,
            frames=frames,
            input_filename=input_filename,
            last_frame_filename=last_frame_filename,
        )

    return {
        "prompt": graph,
        "metadata": {
            "comfyui_commit": COMFYUI_COMMIT,
            "mode": mode,
            "requested_easycache": easycache,
            "effective_easycache": effective_easycache,
            "audio_gain_db": int(audio_gain_db),
            "save_video_node_id": SAVE_VIDEO_NODE_ID,
        },
    }


def build_h3_prompt(**kwargs: Any) -> PromptGraph:
    """Convenience wrapper returning only the ComfyUI ``prompt`` graph."""

    return build_h3_workflow(**kwargs)["prompt"]
