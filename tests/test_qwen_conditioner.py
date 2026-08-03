from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.qwen_conditioner import H3_CONDITIONING_LAYER, optimize_h3_text_encoder, resolve_reference_image_size
from webui.engine_worker import _required_cold_start_free_vram_gib


class AddLayer(torch.nn.Module):
    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, hidden_states):
        return hidden_states + self.amount


class MultiplyNorm(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 10


class FakeQwenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.layers = torch.nn.ModuleList(AddLayer(index + 1) for index in range(64))
        self.language_model.norm = MultiplyNorm()
        self.language_model.config = SimpleNamespace(_attn_implementation=None)

    def forward(self, hidden_states, output_hidden_states=False):
        captured = [hidden_states] if output_hidden_states else None
        for layer in self.language_model.layers:
            hidden_states = layer(hidden_states)
            if captured is not None:
                captured.append(hidden_states)
        hidden_states = self.language_model.norm(hidden_states)
        return SimpleNamespace(
            last_hidden_state=hidden_states,
            hidden_states=None if captured is None else tuple(captured),
        )


class FakeTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeQwenModel()
        self.lm_head = torch.nn.Linear(1, 1)

    def set_attn_implementation(self, implementation):
        self.model.language_model.config._attn_implementation = implementation


class QwenConditionerTests(unittest.TestCase):
    def test_reference_size_keeps_normal_image_near_native_resolution(self):
        self.assertEqual(resolve_reference_image_size(1448, 1086), (1088, 1440))

    def test_reference_size_caps_large_image_without_changing_aspect_materially(self):
        height, width = resolve_reference_image_size(4000, 3000)
        self.assertEqual((height, width), (2048, 2720))
        self.assertLess(abs(width / height - 4 / 3), 0.01)

    def test_optimized_path_returns_exact_unnormalized_layer_50_output(self):
        encoder = FakeTextEncoder()
        source = torch.tensor([[0.0]])
        full = encoder.model(source, output_hidden_states=True)
        expected = full.hidden_states[H3_CONDITIONING_LAYER]

        report = optimize_h3_text_encoder(encoder)
        optimized = encoder.model(source, output_hidden_states=True)

        self.assertEqual(report["original_decoder_layers"], 64)
        self.assertEqual(report["loaded_decoder_layers"], 64)
        self.assertEqual(report["active_decoder_layers"], 50)
        self.assertEqual(len(encoder.model.language_model.layers), 50)
        self.assertIsInstance(encoder.model.language_model.norm, torch.nn.Identity)
        self.assertIsInstance(encoder.lm_head, torch.nn.Identity)
        self.assertEqual(len(optimized.hidden_states), 51)
        self.assertTrue(all(item is None for item in optimized.hidden_states[:50]))
        self.assertTrue(torch.equal(optimized.hidden_states[50], expected))
        self.assertTrue(torch.equal(optimized.last_hidden_state, expected))

    def test_optimization_is_idempotent_and_stable_across_calls(self):
        encoder = FakeTextEncoder()
        first_report = optimize_h3_text_encoder(encoder)
        second_report = optimize_h3_text_encoder(encoder)
        first = encoder.model(torch.tensor([[3.0]]), output_hidden_states=True).hidden_states[50]
        second = encoder.model(torch.tensor([[3.0]]), output_hidden_states=True).hidden_states[50]
        self.assertEqual(first_report, second_report)
        self.assertTrue(torch.equal(first, second))

    def test_high_output_workload_requires_dedicated_gpu_headroom(self):
        low = {"width": 320, "height": 192, "num_frames": 124}
        high = {"width": 1344, "height": 768, "num_frames": 345}
        self.assertEqual(_required_cold_start_free_vram_gib(low), 24.0)
        self.assertEqual(_required_cold_start_free_vram_gib(high), 29.0)


if __name__ == "__main__":
    unittest.main()
