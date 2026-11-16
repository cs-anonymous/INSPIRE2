"""Check strictly causal gold histories and bounded-memory TF decoding."""

import unittest
from unittest.mock import patch

import torch

from src.data_process.dinr_dataset import DINRTokenSpec as S
from src.evaluate.evaluate_dinr_absolute import predict_teacher_forced_notes, sample_top_p_batch
from src.model.dinr import DINRAbsoluteModel
from src.model.pianoformer import PianoT5GemmaConfig


class TeacherForcingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = DINRAbsoluteModel(
            PianoT5GemmaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=2,
                              num_key_value_heads=1, head_dim=16, encoder_layers_num=1, decoder_layers_num=1),
            pedal_mode="none", numerical_coordinates=True, coordinate_encoding="mlp",
            coordinate_hidden_size=64, coordinate_combination="replace", separate_coordinate_projections=True,
            decoder_mask_mode="timestep_single", decoder_mask_probability=0.5,
        ).eval()
        self.score = [[65, S.timing_start + 10, S.velocity_start + 64, S.timing_start + 20],
                      [69, S.timing_start, S.velocity_start + 70, S.timing_start + 25],
                      [72, S.timing_start + 12, S.velocity_start + 80, S.timing_start + 30]]

    def test_generated_values_never_enter_tf_history(self):
        captured = []
        handle = self.model.register_forward_pre_hook(lambda module, args, kw: captured.append(kw["decoder_tokens"].clone()), with_kwargs=True)
        with patch("src.evaluate.evaluate_dinr_absolute.sample_top_p_batch", side_effect=lambda logits, *args: torch.full(logits.shape[:-1], 7, device=logits.device)):
            predicted = predict_teacher_forced_notes(self.model, self.score, self.score, [False, True, False],
                                                     torch.device("cpu"), chunk_size=1)
        handle.remove()
        torch.testing.assert_close(captured[0], torch.full((1, 1, 4), S.bos_id))
        for index in (1, 2):
            torch.testing.assert_close(captured[index], torch.tensor([[self.score[index - 1]]]))
            self.assertNotEqual(predicted[index - 1], self.score[index - 1])

    def test_chunked_logits_match_full_sequence_and_do_not_leak_current_targets(self):
        logits_by_call = []

        def capture(logits, *args):
            logits_by_call.append(logits.clone())
            return logits.argmax(-1)

        with patch("src.evaluate.evaluate_dinr_absolute.sample_top_p_batch", side_effect=capture):
            predict_teacher_forced_notes(self.model, self.score, self.score, [False, True, False], torch.device("cpu"), chunk_size=3)
        full = logits_by_call[:]
        logits_by_call.clear()
        with patch("src.evaluate.evaluate_dinr_absolute.sample_top_p_batch", side_effect=capture):
            predict_teacher_forced_notes(self.model, self.score, self.score, [False, True, False], torch.device("cpu"), chunk_size=1)
        for head in range(3):
            torch.testing.assert_close(torch.cat(logits_by_call[head::3]), full[head], atol=2e-5, rtol=2e-5)
        changed = [row[:] for row in self.score]
        changed[1][2] += 20
        logits_by_call.clear()
        with patch("src.evaluate.evaluate_dinr_absolute.sample_top_p_batch", side_effect=capture):
            predict_teacher_forced_notes(self.model, self.score, changed, [False, True, False], torch.device("cpu"), chunk_size=3)
        for head in range(3):
            torch.testing.assert_close(full[head][:2], logits_by_call[head][:2], rtol=0, atol=0)
        self.assertFalse(torch.allclose(full[2][2], logits_by_call[2][2]))

    def test_batched_nucleus_respects_support(self):
        logits = torch.tensor([[100., -100., -100.], [-100., -100., 100.]])
        torch.testing.assert_close(sample_top_p_batch(logits, 1., .9), torch.tensor([0, 2]))


if __name__ == "__main__":
    unittest.main()
