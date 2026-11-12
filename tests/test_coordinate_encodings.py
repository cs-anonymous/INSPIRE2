"""Check numeric semantics, gradients, and inference checkpoint restoration."""

import tempfile
import unittest
from types import SimpleNamespace

import torch

from src.data_process.dinr_dataset import DINRTokenSpec as S
from src.evaluate.evaluate_dinr_absolute import make_model
from src.model.dinr import DINRAbsoluteModel
from src.model.pianoformer import PianoT5GemmaConfig


def model(encoding, separate=True, combination="add"):
    config = PianoT5GemmaConfig(
        hidden_size=32, intermediate_size=64, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, encoder_layers_num=1, decoder_layers_num=1,
    )
    return DINRAbsoluteModel(
        config, pedal_mode="none", numerical_coordinates=True,
        separate_ioi_duration_tables=True, coordinate_encoding=encoding,
        separate_coordinate_projections=separate,
        coordinate_combination=combination,
        decoder_mask_mode="timestep_single", decoder_mask_probability=0.5,
    )


def tokens():
    return torch.tensor([[[S.pitch_start + 60, S.timing_start + 10,
                           S.velocity_start + 64, S.timing_start + 20]]])


class CoordinateEncodingTest(unittest.TestCase):
    def test_all_attributes_and_special_token_masking(self):
        for encoding in ("linear", "mlp", "fourier"):
            with self.subTest(encoding=encoding):
                m = model(encoding)
                with torch.no_grad():
                    m.token_embedding.weight.zero_()
                    m.ioi_token_embedding.weight.zero_()
                    m.duration_token_embedding.weight.zero_()
                t = tokens()
                actual = m._sum_note_tokens(t)
                expected = sum(
                    projection(m._coordinate_features(torch.tensor([[value]])))
                    for projection, value in (
                        (m.timing_coordinate_projection, 0.1),
                        (m.duration_coordinate_projection, 0.2),
                        (m.velocity_coordinate_projection, 64 / 127),
                    )
                )
                torch.testing.assert_close(actual, expected)
                for slot in (1, 2, 3):
                    changed = t.clone()
                    changed[..., slot] += 1
                    self.assertFalse(torch.allclose(m._sum_note_tokens(changed), actual))
                    mask = torch.zeros((1, 1, 3), dtype=torch.bool)
                    mask[..., (1, 2, 3).index(slot)] = True
                    torch.testing.assert_close(m._sum_note_tokens(changed, mask),
                                               m._sum_note_tokens(t, mask))
                # Biases and cos(0) must not inject coordinates into PAD/BOS/pitch.
                special = torch.full_like(t, S.pad_id)
                special[..., 0] = S.pitch_start + 60
                torch.testing.assert_close(m._sum_note_tokens(special), torch.zeros_like(actual))

    def test_full_forward_gradients_and_inference_roundtrip(self):
        cases = [("fourier", False, "add")]
        cases += [(e, True, c) for e in ("linear", "mlp", "fourier") for c in ("add", "replace")]
        for encoding, separate, combination in cases:
            with self.subTest(encoding=encoding, separate=separate, combination=combination):
                m = model(encoding, separate, combination)
                t = tokens().expand(1, 3, 4).clone()
                labels = torch.tensor([[[10, 20, 64], [0, 25, 70], [12, 30, 80]]])
                result = m(score_tokens=t, decoder_tokens=t, labels=labels,
                           score_zero_ioi_mask=torch.tensor([[False, True, False]]))
                self.assertTrue(torch.isfinite(result["loss"]))
                result["loss"].backward()
                for name, parameter in m.named_parameters():
                    if "coordinate_projection" in name:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                with tempfile.TemporaryDirectory() as directory:
                    m.save_pretrained(directory)
                    args = SimpleNamespace(
                        hidden_size=32, intermediate_size=64, attention_heads=2,
                        key_value_heads=1, head_dim=16, encoder_layers=1, decoder_layers=1,
                        device="cpu", pedal_mode="none", numerical_coordinates=False,
                        coordinate_encoding=None, timing_target_mode="absolute", time_input_mode="linear",
                    )
                    loaded = make_model(args, directory)
                    self.assertEqual(loaded.coordinate_encoding, encoding)
                    self.assertEqual(loaded.coordinate_combination, combination)
                    self.assertEqual(loaded.separate_coordinate_projections, separate)
                    self.assertEqual(set(m.state_dict()), set(loaded.state_dict()))
                    for key, value in m.state_dict().items():
                        torch.testing.assert_close(value, loaded.state_dict()[key])
                    torch.testing.assert_close(m._sum_note_tokens(t), loaded._sum_note_tokens(t))

    def test_replacement_has_no_numeric_lookup_and_preserves_masks(self):
        for encoding in ("linear", "mlp", "fourier"):
            with self.subTest(encoding=encoding):
                m = model(encoding, combination="replace")
                self.assertIsNone(m.ioi_token_embedding)
                self.assertIsNone(m.duration_token_embedding)
                self.assertEqual(m.token_embedding.num_embeddings, S.velocity_start)
                t = tokens()
                pitch = m.token_embedding(t[..., 0])
                expected = pitch + sum(
                    projection(m._coordinate_features(torch.tensor([[value]])))
                    for projection, value in (
                        (m.timing_coordinate_projection, 0.1),
                        (m.duration_coordinate_projection, 0.2),
                        (m.velocity_coordinate_projection, 64 / 127),
                    )
                )
                torch.testing.assert_close(m._sum_note_tokens(t), expected)
                for slot in (1, 2, 3):
                    changed = t.clone()
                    changed[..., slot] += 1
                    self.assertFalse(torch.allclose(m._sum_note_tokens(changed), expected))
                    mask = torch.zeros((1, 1, 3), dtype=torch.bool)
                    mask[..., (1, 2, 3).index(slot)] = True
                    torch.testing.assert_close(m._sum_note_tokens(changed, mask), m._sum_note_tokens(t, mask))
                pitch_only = torch.full_like(t, S.pad_id)
                pitch_only[..., 0] = t[..., 0]
                torch.testing.assert_close(m._sum_note_tokens(pitch_only), pitch)
                torch.testing.assert_close(m._sum_note_tokens(torch.zeros_like(t)), torch.zeros_like(pitch))

    def test_pretraining_never_copies_numeric_rows_in_replacement(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        m = model("mlp", combination="replace")
        source = torch.nn.Embedding(5389, 32)
        source.weight.data.fill_(9)
        fake = SimpleNamespace(
            model=SimpleNamespace(state_dict=lambda: {}),
            get_encoder=lambda: SimpleNamespace(embeddings=SimpleNamespace(word_embeddings=source)),
        )
        before = {k: v.clone() for k, v in m.state_dict().items() if "coordinate_projection" in k}
        with patch("src.model.dinr.PianoT5Gemma.from_pretrained", return_value=fake):
            info = m.initialize_from_pt("unused")
        self.assertFalse(info["numerical_lookup_initialized"])
        self.assertTrue(torch.all(m.token_embedding.weight[S.pitch_start:] == 9))
        for k, v in before.items():
            torch.testing.assert_close(m.state_dict()[k], v)

    def test_linear_fixed_deltas_and_mlp_nonlinearity(self):
        x = torch.tensor([0.0, 0.5, 1.0])
        linear = model("linear")
        y = linear.timing_coordinate_projection(linear._coordinate_features(x))
        torch.testing.assert_close(y[1] - y[0], y[2] - y[1])
        mlp = model("mlp")
        with torch.no_grad():
            for layer in (mlp.timing_coordinate_projection[0], mlp.timing_coordinate_projection[2]):
                layer.weight.fill_(1)
                layer.bias.zero_()
        y = mlp.timing_coordinate_projection(mlp._coordinate_features(x))
        self.assertFalse(torch.allclose(y[1] - y[0], y[2] - y[1]))


if __name__ == "__main__":
    unittest.main()
