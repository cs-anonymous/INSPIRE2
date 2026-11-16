"""E4D3 input isolation, cached decoding, and checkpoint compatibility."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.data_process.dinr_dataset import DINRTokenSpec as S
from src.evaluate.evaluate_dinr_absolute import make_model, predict_notes
from src.model.dinr import DINRAbsoluteModel
from src.model.pianoformer import PianoT5GemmaConfig


def make_small(encoding=None, width=32, pitch=False):
    return DINRAbsoluteModel(
        PianoT5GemmaConfig(hidden_size=32, intermediate_size=64,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                          encoder_layers_num=1, decoder_layers_num=1),
        pedal_mode="none", decoder_input_pitch=pitch, decoder_mask_mode="note",
        decoder_mask_probability=0.5, separate_ioi_duration_tables=True,
        numerical_coordinates=encoding is not None,
        coordinate_encoding=encoding or "linear", coordinate_hidden_size=width,
        coordinate_combination="replace" if encoding else "add",
        separate_coordinate_projections=encoding is not None,
    )


def notes():
    return torch.tensor([[[S.pitch_start + 60, S.timing_start + 10, S.velocity_start + 64, S.timing_start + 20],
                          [S.pitch_start + 64, S.timing_start, S.velocity_start + 70, S.timing_start + 25],
                          [S.pitch_start + 67, S.timing_start + 12, S.velocity_start + 80, S.timing_start + 30]]])


class E4D3Test(unittest.TestCase):
    def test_decoder_pitch_is_absent_and_encoder_pitch_remains(self):
        for encoding, width in ((None, 32), ("mlp", 32), ("mlp", 64), ("linear", 32)):
            with self.subTest(encoding=encoding, width=width):
                m = make_small(encoding, width).eval()
                t = notes()
                changed = t.clone()
                changed[..., 0] += 1
                torch.testing.assert_close(m._sum_decoder_tokens(t), m._sum_decoder_tokens(changed), rtol=0, atol=0)
                torch.testing.assert_close(m._sum_decoder_tokens(t), m._sum_decoder_tokens(t[..., 1:]), rtol=0, atol=0)
                self.assertFalse(torch.allclose(m._sum_note_tokens(t), m._sum_note_tokens(changed)))
                for slot in (1, 2, 3):
                    numeric = t.clone()
                    numeric[..., slot] += 1
                    self.assertFalse(torch.allclose(m._sum_decoder_tokens(t), m._sum_decoder_tokens(numeric)))
                result = m(score_tokens=t, decoder_tokens=t)
                other = m(score_tokens=t, decoder_tokens=changed)
                for head in ("grouped_ioi", "chord_ioi", "duration", "velocity"):
                    torch.testing.assert_close(result["logits"][head], other["logits"][head], rtol=0, atol=0)
                # E4D3 changes the information supplied, not the parameter count.
                baseline = make_small(encoding, width, pitch=True)
                self.assertEqual(sum(p.numel() for p in m.parameters()), sum(p.numel() for p in baseline.parameters()))
                with tempfile.TemporaryDirectory() as directory:
                    m.save_pretrained(directory)
                    args = SimpleNamespace(hidden_size=32, intermediate_size=64, attention_heads=2,
                                           key_value_heads=1, head_dim=16, encoder_layers=1, decoder_layers=1,
                                           device="cpu", pedal_mode="none", numerical_coordinates=False,
                                           coordinate_encoding=None, timing_target_mode="absolute", time_input_mode="linear")
                    restored = make_model(args, directory)
                    self.assertFalse(restored.decoder_input_pitch)
                    self.assertEqual(restored.decoder_num_input_slots, 3)
                    for head, value in result["logits"].items():
                        if torch.is_tensor(value):
                            torch.testing.assert_close(value, restored(score_tokens=t, decoder_tokens=t[..., 1:])["logits"][head])

    def test_actual_timestep_mask_and_backward(self):
        m = make_small("mlp").train()
        t = notes()
        t[:, 0] = S.bos_id
        t[:, -1] = S.pad_id
        captured = {}

        def capture(module, args, kwargs):
            captured.update(score=kwargs["inputs_embeds"].detach(), decoder=kwargs["decoder_inputs_embeds"].detach())

        handle = m.backbone.register_forward_pre_hook(capture, with_kwargs=True)
        labels = torch.tensor([[[10, 20, 64], [0, 25, 70], [12, 30, 80]]])
        with patch("torch.rand", side_effect=lambda shape, **kw: torch.zeros(shape, device=kw.get("device"))):
            output = m(score_tokens=notes(), decoder_tokens=t, labels=labels,
                       decoder_attention_mask=torch.tensor([[1, 1, 0]]),
                       score_zero_ioi_mask=torch.tensor([[False, True, False]]))
        handle.remove()
        torch.testing.assert_close(captured["score"], m._sum_note_tokens(notes()))
        torch.testing.assert_close(captured["decoder"][:, 0], m._sum_decoder_tokens(t)[:, 0])
        torch.testing.assert_close(captured["decoder"][:, 1], m.decoder_mask_embedding.view(1, -1))
        torch.testing.assert_close(captured["decoder"][:, -1], torch.zeros_like(captured["decoder"][:, -1]))
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        self.assertGreater(m.timing_coordinate_projection[0].weight.grad.abs().sum().item(), 0)

    def test_cached_generation_feeds_back_only_three_numeric_fields(self):
        m = make_small("linear").eval()
        captured = []

        def capture(module, args, kwargs):
            captured.append(kwargs["decoder_tokens"].clone())

        handle = m.register_forward_pre_hook(capture, with_kwargs=True)
        with patch("src.evaluate.evaluate_dinr_absolute.sample_top_p", return_value=10):
            predicted = predict_notes(m, notes()[0].tolist(), np.zeros((3, 7)),
                                      [False, True, False], torch.device("cpu"), False)
        handle.remove()
        self.assertEqual(len(captured), 3)
        torch.testing.assert_close(captured[0], torch.full((1, 1, 3), S.bos_id))
        for index in (1, 2):
            torch.testing.assert_close(captured[index], torch.tensor([[predicted[index - 1][1:4]]]))
        self.assertEqual([row[0] for row in predicted], notes()[0, :, 0].tolist())
        # The same three-field teacher-forced sequence must match cached inference.
        full = torch.cat(captured, dim=1)
        with torch.no_grad():
            expected = m(score_tokens=notes(), decoder_tokens=full)["logits"]
            past = None
            for index, decoder in enumerate(captured):
                out = m(score_tokens=notes(), decoder_tokens=decoder,
                        decoder_attention_mask=torch.ones((1, index + 1), dtype=torch.long),
                        past_key_values=past, use_cache=True, cache_position=torch.tensor([index]))
                past = out["past_key_values"]
                for head in ("grouped_ioi", "chord_ioi", "duration", "velocity"):
                    torch.testing.assert_close(out["logits"][head][:, 0], expected[head][:, index], atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
