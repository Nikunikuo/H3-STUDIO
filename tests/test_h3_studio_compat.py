from __future__ import annotations

import importlib.util
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "comfy_compat" / "h3_studio_compat" / "__init__.py"


class FakeTokenizer:
    def __init__(self, first_added_id: int) -> None:
        self._next_id = first_added_id
        self._vocab: dict[str, int] = {"<|endoftext|>": 151643}
        self._special: list[str] = ["<|endoftext|>"]
        self.add_calls = 0

    @property
    def all_special_tokens(self) -> list[str]:
        return list(self._special)

    def add_special_tokens(
        self,
        values: dict[str, list[str]],
        *,
        replace_extra_special_tokens: bool,
    ) -> int:
        self.add_calls += 1
        if replace_extra_special_tokens:
            raise AssertionError("H3 tokens must be appended, not replace generic metadata")
        added = 0
        for token in values["additional_special_tokens"]:
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._next_id += 1
                added += 1
            if token not in self._special:
                self._special.append(token)
        return added

    def convert_tokens_to_ids(self, values):  # noqa: ANN001, ANN201
        if isinstance(values, str):
            return self._vocab[values]
        return [self._vocab[value] for value in values]

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return {value: key for key, value in self._vocab.items()}[token_id]

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        if add_special_tokens:
            raise AssertionError("contract probe must disable automatic special tokens")
        return {"input_ids": [self._vocab[text]]}

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


def fake_minimax_module(first_added_id: int):
    minimax = types.ModuleType("comfy.text_encoders.minimax")

    class MiniMaxH3Tokenizer:
        instances: list[MiniMaxH3Tokenizer] = []

        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            del args, kwargs
            tokenizer = FakeTokenizer(first_added_id)
            self.qwen3vl_32b = types.SimpleNamespace(tokenizer=tokenizer, inv_vocab={})
            self.__class__.instances.append(self)

    class GenericQwenTokenizer:
        pass

    minimax.MiniMaxH3Tokenizer = MiniMaxH3Tokenizer
    minimax.GenericQwenTokenizer = GenericQwenTokenizer
    return minimax, MiniMaxH3Tokenizer, GenericQwenTokenizer


def execute_compat_module(minimax, module_name: str):  # noqa: ANN001, ANN201
    comfy = types.ModuleType("comfy")
    text_encoders = types.ModuleType("comfy.text_encoders")
    comfy.text_encoders = text_encoders
    text_encoders.minimax = minimax
    spec = importlib.util.spec_from_file_location(module_name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    modules = {
        "comfy": comfy,
        "comfy.text_encoders": text_encoders,
        "comfy.text_encoders.minimax": minimax,
        module_name: module,
    }
    with mock.patch.dict(sys.modules, modules, clear=False):
        spec.loader.exec_module(module)
    return module


class H3StudioCompatTests(unittest.TestCase):
    def test_exact_h3_ids_are_added_idempotently_to_only_h3_tokenizer(self):
        minimax, h3_class, generic_class = fake_minimax_module(151669)
        generic_init = generic_class.__init__
        output = io.StringIO()
        with redirect_stdout(output):
            module = execute_compat_module(minimax, "_h3_compat_test_first")
            # A second package import must not wrap the H3 class a second time.
            execute_compat_module(minimax, "_h3_compat_test_second")

        expected_tokens = (
            "<d>",
            "</d>",
            "<|cutoff|>",
            "<|lyrics_start|>",
            "<|lyrics_end|>",
            "<|caption_start|>",
            "<|caption_end|>",
        )
        self.assertEqual(module.H3_SPECIAL_TOKENS, expected_tokens)
        self.assertEqual(module.H3_SPECIAL_TOKEN_IDS, tuple(range(151669, 151676)))
        self.assertEqual(generic_class.__init__, generic_init)
        self.assertEqual(output.getvalue().count(module.VERIFICATION_MARKER), 2)

        instance = h3_class()
        tokenizer = instance.qwen3vl_32b.tokenizer
        self.assertEqual(tokenizer.add_calls, 1)
        self.assertEqual(
            tuple(tokenizer.convert_tokens_to_ids(list(expected_tokens))),
            tuple(range(151669, 151676)),
        )
        for token, token_id in zip(expected_tokens, range(151669, 151676), strict=True):
            self.assertEqual(instance.qwen3vl_32b.inv_vocab[token_id], token)

    def test_wrong_base_vocabulary_fails_before_success_marker(self):
        minimax, _, _ = fake_minimax_module(151670)
        output = io.StringIO()
        with redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "expected .*151669"):
                execute_compat_module(minimax, "_h3_compat_test_wrong_ids")
        self.assertNotIn("tokenizer_patch=verified", output.getvalue())


if __name__ == "__main__":
    unittest.main()
