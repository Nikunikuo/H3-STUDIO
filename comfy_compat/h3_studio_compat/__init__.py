"""Trusted MiniMax H3 compatibility shim loaded by H3 Studio only.

The pinned ComfyUI tokenizer reuses the generic Qwen tokenizer and currently
misses MiniMax H3's seven additional tokens.  MiniMax explicitly requires the
tokenizer configuration shipped with H3; without this shim ``<d>`` and the
other control markers are split into ordinary punctuation/text tokens.

The generic vocabulary and merge table otherwise match the official H3
tokenizer.  Add the seven official tokens in their published order, verify the
exact IDs, and fail closed if a future upstream tokenizer no longer matches
that contract.  This file is copied into each job's isolated Comfy base and is
the sole whitelisted custom node.  The pinned ComfyUI checkout stays clean.
"""

from __future__ import annotations

from comfy.text_encoders import minimax


H3_SPECIAL_TOKENS = (
    "<d>",
    "</d>",
    "<|cutoff|>",
    "<|lyrics_start|>",
    "<|lyrics_end|>",
    "<|caption_start|>",
    "<|caption_end|>",
)
H3_SPECIAL_TOKEN_IDS = tuple(range(151669, 151676))
VERIFICATION_MARKER = "H3_STUDIO_COMPAT tokenizer_patch=verified ids=151669-151675"


def _verify_tokenizer_contract(wrapper) -> None:  # noqa: ANN001
    """Require the exact H3 extension without touching generic Qwen tokenizers."""

    tokenizer = wrapper.tokenizer
    actual_ids = tuple(tokenizer.convert_tokens_to_ids(list(H3_SPECIAL_TOKENS)))
    if actual_ids != H3_SPECIAL_TOKEN_IDS:
        raise RuntimeError(
            "MiniMax H3 tokenizer compatibility check failed: "
            f"expected {H3_SPECIAL_TOKEN_IDS}, got {actual_ids}"
        )

    all_special_tokens = set(tokenizer.all_special_tokens)
    for token, token_id in zip(H3_SPECIAL_TOKENS, H3_SPECIAL_TOKEN_IDS, strict=True):
        if token not in all_special_tokens:
            raise RuntimeError(f"MiniMax H3 token {token!r} is not registered as special")
        if tokenizer.convert_ids_to_tokens(token_id) != token:
            raise RuntimeError(
                f"MiniMax H3 tokenizer ID {token_id} does not map back to {token!r}"
            )
        encoded = tokenizer(token, add_special_tokens=False)["input_ids"]
        if encoded != [token_id]:
            raise RuntimeError(
                f"MiniMax H3 tokenizer did not preserve {token!r} as ID {token_id}: {encoded}"
            )

    # ComfyUI's untokenize path uses this cached inverse vocabulary.
    wrapper.inv_vocab = {value: key for key, value in tokenizer.get_vocab().items()}
    for token, token_id in zip(H3_SPECIAL_TOKENS, H3_SPECIAL_TOKEN_IDS, strict=True):
        if wrapper.inv_vocab.get(token_id) != token:
            raise RuntimeError(
                f"MiniMax H3 inverse vocabulary check failed for {token!r} at ID {token_id}"
            )


def _install_tokenizer_compatibility() -> None:
    tokenizer_class = minimax.MiniMaxH3Tokenizer
    if getattr(tokenizer_class, "_h3_studio_compat_installed", False):
        return

    original_init = tokenizer_class.__init__

    def patched_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        original_init(self, *args, **kwargs)
        wrapper = self.qwen3vl_32b
        tokenizer = wrapper.tokenizer
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(H3_SPECIAL_TOKENS)},
            # Transformers 5.14 calls these "extra" special tokens.  Append
            # instead of replacing any generic Qwen control-token metadata.
            replace_extra_special_tokens=False,
        )
        _verify_tokenizer_contract(wrapper)

    tokenizer_class.__init__ = patched_init
    tokenizer_class._h3_studio_compat_installed = True


_install_tokenizer_compatibility()

# ComfyUI deliberately catches custom-node import exceptions.  Instantiate the
# tokenizer during import and print a success marker only after the complete
# contract passes; the parent worker requires this marker before submitting a
# prompt, so a swallowed import failure still fails closed.
_startup_probe = minimax.MiniMaxH3Tokenizer()
_verify_tokenizer_contract(_startup_probe.qwen3vl_32b)
print(VERIFICATION_MARKER, flush=True)
del _startup_probe

# ComfyUI accepts a custom-node package with no frontend nodes.  This package
# exists only for the audited import-time compatibility patch above.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
