from __future__ import annotations

import os
from types import MethodType
from typing import Any


H3_CONDITIONING_LAYER = 50
REFERENCE_IMAGE_SHORT_EDGE_LIMIT = 2048
REFERENCE_IMAGE_MULTIPLE = 32


def load_h3_text_encoder(model_path: os.PathLike[str] | str, quantization_config: Any):
    """Load only the 50 Qwen decoder layers H3 can consume.

    Constructing all 64 checkpoint layers and deleting the final 14 after
    quantization causes a very large cold-load RAM spike on Windows.  The
    official merged SGLang/ComfyUI implementations construct a 50-layer Qwen
    from the outset.  Restore the checkpoint's 64-layer config value after
    loading solely because the pinned Diffusers compatibility check expects a
    value greater than 50 before it reads ``hidden_states[50]``.
    """

    import torch
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig.from_pretrained(os.fspath(model_path), subfolder="text_encoder")
    checkpoint_layer_count = int(config.text_config.num_hidden_layers)
    if checkpoint_layer_count < H3_CONDITIONING_LAYER:
        raise ValueError(
            f"H3 needs {H3_CONDITIONING_LAYER} Qwen decoder layers, but the checkpoint has {checkpoint_layer_count}."
        )
    config.text_config.num_hidden_layers = H3_CONDITIONING_LAYER
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        os.fspath(model_path),
        subfolder="text_encoder",
        config=config,
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        attn_implementation="sdpa",
    )
    text_encoder._h3_checkpoint_decoder_layers = checkpoint_layer_count
    # The pinned Diffusers block rejects ``num_hidden_layers <= 50`` even
    # though a 50-layer decoder legitimately exposes hidden_states[50]
    # (embedding + 50 layer outputs).  Keep that compatibility sentinel out of
    # the execution path; the actual ModuleList remains exactly 50 layers.
    compatibility_layer_count = max(checkpoint_layer_count, H3_CONDITIONING_LAYER + 1)
    text_encoder.config.text_config.num_hidden_layers = compatibility_layer_count
    text_encoder.model.language_model.config.num_hidden_layers = compatibility_layer_count
    return text_encoder


def resolve_reference_image_size(width: int, height: int) -> tuple[int, int]:
    """Return H3's quality-oriented reference size without synthetic upscaling.

    The merged ComfyUI H3 implementation treats 2048 as a short-edge upper
    bound, not as a target that every source must be enlarged to.  Keeping a
    normal source close to its native resolution preserves all real detail and
    avoids manufacturing thousands of interpolated Qwen vision tokens.
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"A reference image must have a positive size, got {width}x{height}.")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")

    scale = min(1.0, REFERENCE_IMAGE_SHORT_EDGE_LIMIT / min(width, height))
    return (
        max(REFERENCE_IMAGE_MULTIPLE, round(height * scale / REFERENCE_IMAGE_MULTIPLE) * REFERENCE_IMAGE_MULTIPLE),
        max(REFERENCE_IMAGE_MULTIPLE, round(width * scale / REFERENCE_IMAGE_MULTIPLE) * REFERENCE_IMAGE_MULTIPLE),
    )


def install_reference_size_patch() -> None:
    """Apply the cap-only reference policy to both imported Diffusers symbols."""

    from diffusers.modular_pipelines.minimax_h3 import before_encoder, packing_ref2va

    packing_ref2va.resolve_reference_image_size = resolve_reference_image_size
    # before_encoder imports the function into its own module namespace.
    before_encoder.resolve_reference_image_size = resolve_reference_image_size


def optimize_h3_text_encoder(text_encoder: Any) -> dict[str, Any]:
    """Make Qwen3-VL compute exactly the intermediate state H3 consumes.

    H3 conditions on the unnormalized output after decoder layer 50
    (``hidden_states[50]`` in the full 64-layer Transformers model).  Running
    layers 51-64 and retaining all 65 hidden states cannot change that tensor;
    it only adds work and, for long visual sequences, several GiB of VRAM.

    This mirrors the merged SGLang and ComfyUI H3 implementations:

    * retain decoder layers 0..49;
    * replace the final language-model norm with Identity;
    * remove the unused LM head;
    * force ``output_hidden_states=False``;
    * expose the unnormalized ``last_hidden_state`` at index 50 solely for the
      pinned Diffusers block's existing ``outputs.hidden_states[50]`` access.

    The compatibility tuple contains only one tensor.  Its first 50 entries
    are ``None`` and therefore consume no tensor storage.
    """

    if getattr(text_encoder, "_h3_conditioner_optimized", False):
        return dict(text_encoder._h3_conditioner_report)

    import torch

    model = text_encoder.model
    language_model = model.language_model
    layers = list(language_model.layers)
    loaded_layer_count = len(layers)
    checkpoint_layer_count = int(getattr(text_encoder, "_h3_checkpoint_decoder_layers", loaded_layer_count))
    if loaded_layer_count < H3_CONDITIONING_LAYER:
        raise ValueError(
            f"H3 needs {H3_CONDITIONING_LAYER} Qwen decoder layers, but the model loaded {loaded_layer_count}."
        )

    language_model.layers = torch.nn.ModuleList(layers[:H3_CONDITIONING_LAYER])
    language_model.norm = torch.nn.Identity()
    if hasattr(text_encoder, "lm_head"):
        text_encoder.lm_head = torch.nn.Identity()

    original_forward = model.forward

    def h3_conditioning_forward(self, *args, **kwargs):
        # The Transformers output-capturing decorator allocates one full
        # sequence tensor per decoder layer when this is True.  H3 needs only
        # the final tensor of the retained 50-layer stack.
        kwargs["output_hidden_states"] = False
        outputs = original_forward(*args, **kwargs)
        conditioning = outputs.last_hidden_state
        outputs.hidden_states = (None,) * H3_CONDITIONING_LAYER + (conditioning,)
        return outputs

    model.forward = MethodType(h3_conditioning_forward, model)

    # Do not leave attention dispatch to version-dependent auto-selection.
    text_encoder.set_attn_implementation("sdpa")
    actual_attention = language_model.config._attn_implementation
    if actual_attention != "sdpa":
        raise RuntimeError(f"Qwen language attention must use SDPA, got {actual_attention!r}.")

    report = {
        "original_decoder_layers": checkpoint_layer_count,
        "loaded_decoder_layers": loaded_layer_count,
        "active_decoder_layers": len(language_model.layers),
        "conditioning_hidden_state": H3_CONDITIONING_LAYER,
        "final_norm": "identity",
        "lm_head": "removed",
        "output_hidden_states": False,
        "attention": actual_attention,
    }
    text_encoder._h3_conditioner_report = report
    text_encoder._h3_conditioner_optimized = True
    return dict(report)


def configure_h3_text_encoder_offload(text_encoder: Any) -> None:
    """Synchronously offload Qwen by vision/decoder blocks.

    Qwen's parent model calls ``language_model.embed_tokens`` before it calls
    ``language_model.forward``.  A hook on the language-model container is
    therefore too late.  The large embedding is moved just for its own call,
    while a temporary container installs block hooks directly on the decoder
    layers.

    We intentionally do not use a side CUDA stream here.  A detached layer
    container has no ``forward`` call on which Diffusers can trace and join its
    lazy-prefetch chain.  Asynchronous copies from that container could race
    the default compute stream.  One synchronous decoder-block transfer at a
    time is still dramatically faster than leaf-level offload for Qwen and is
    deterministic across first and repeated runs.
    """

    if getattr(text_encoder, "_h3_offload_configured", False):
        return

    import torch
    from diffusers.hooks import apply_group_offloading

    text_encoder.requires_grad_(False)

    visual = text_encoder.model.visual
    apply_group_offloading(
        visual,
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=False,
    )

    # apply_group_offloading discovers ModuleList children.  This temporary
    # holder exposes only the retained decoder layers, so their hooks fire even
    # though Qwen's parent bypasses language_model.forward for token embedding.
    layer_holder = torch.nn.Module()
    layer_holder.layers = text_encoder.model.language_model.layers
    apply_group_offloading(
        layer_holder,
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=False,
    )

    auxiliary_handles = []
    for module in (
        text_encoder.model.language_model.embed_tokens,
        text_encoder.model.language_model.rotary_emb,
    ):

        def onload_auxiliary(active_module, args):
            tensor = next((value for value in args if torch.is_tensor(value)), None)
            if tensor is None:
                raise RuntimeError(f"Cannot determine the execution device for {active_module.__class__.__name__}.")
            active_module.to(tensor.device)

        def offload_auxiliary(active_module, _args, output):
            active_module.to("cpu")
            return output

        auxiliary_handles.append(module.register_forward_pre_hook(onload_auxiliary))
        auxiliary_handles.append(module.register_forward_hook(offload_auxiliary))

    # RemovableHandle is not a Module; keeping it alive prevents accidental
    # cleanup while avoiding duplicate module registration on text_encoder.
    text_encoder._h3_auxiliary_offload_handles = auxiliary_handles
    text_encoder._h3_offload_configured = True


def image_vision_tokens(width: int, height: int) -> int:
    """Estimate Qwen vision tokens after the cap-only H3 reference resize."""

    prepared_height, prepared_width = resolve_reference_image_size(width, height)
    return (prepared_height // 16) * (prepared_width // 16) // (2 * 2)
