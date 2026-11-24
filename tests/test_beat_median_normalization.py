"""Timing normalization invariants and abs/dev label/evaluation consistency."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.data_process.beat_normalization import normalize_performance
from src.data_process.dinr_beat_dataset import DINRBeatDataset, BeatTokenSpec as S
from src.evaluate.evaluate_dinr_beat import quantized_example, onset_feedback_token, absolute_notes
from script.evaluate_exp1117_beat_dev import work_metrics


class BeatMedianTests(unittest.TestCase):
    def test_single_scale_variable_score_tempo_and_idempotence(self):
        ticks = np.arange(5) * 480
        edges = np.arange(6) * 480
        score_ms = np.array([500, 500, 250, 250, 250.])
        # Supported ratios [2,4,1,2]; median log ratio is log(2).
        perf_ms = np.array([1000, 2000, 250, 500.])
        perf = dict(label_shared_raw=np.column_stack(([123, *perf_ms],
                        [700, 300, 800, 20, 150], [41, 52, 63, 74, 85])).tolist())
        result = normalize_performance(perf, ticks, edges, score_ms)
        self.assertAlmostEqual(result['global_timing_scale']['scale'], .5)
        a, b = np.array(perf['label_shared_raw']), np.array(result['label_shared_raw'])
        np.testing.assert_allclose(b[:, :2], a[:, :2] * .5)
        np.testing.assert_array_equal(b[:, 2], a[:, 2])
        again = normalize_performance(result, ticks, edges, score_ms)
        np.testing.assert_allclose(again['label_shared_raw'], result['label_shared_raw'])
        self.assertEqual(result['global_timing_scale']['valid_beat_intervals'], 4)
        self.assertAlmostEqual(result['global_timing_scale']['normalized_median_log_deviation'], 0)

    def test_requires_supported_positive_beats(self):
        with self.assertRaisesRegex(ValueError, 'No supported positive'):
            normalize_performance(dict(label_shared_raw=[[0, 100, 64], [0, 100, 64]]),
                                  [0, 480], [0, 480, 960], [500, 500])

    def test_abs_and_dev_share_history_but_have_different_labels(self):
        events = torch.tensor([[0, -1, 500, -1, -1, 1], [1, 60, 200, .4, 64, -1],
                               [1, 64, 200, .8, 64, -1]], dtype=torch.float32)
        targets = torch.tensor([[500, -1, -1], [200, .1, 70], [200, -.1, 80.]])
        valid = torch.tensor([[True, False, False], [True, True, True], [True, False, True]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.pt'
            torch.save(dict(format='dinr_beat_events_v2', events=events, targets=targets,
                            target_valid=valid, samples=[dict(span=[0, 3])]), path)
            absolute = DINRBeatDataset(path, onset_target='abs')[0]
            dev = DINRBeatDataset(path, onset_target='dev')[0]
            self.assertEqual(absolute['labels'][1, 2], S.quantize(.5, S.onset_input_range, 500))
            self.assertEqual(dev['labels'][1, 2], S.quantize(.1, S.onset_dev_range, 200))
            self.assertEqual(absolute['labels'][2, 2], -100)
            torch.testing.assert_close(absolute['decoder_tokens'], dev['decoder_tokens'])
            run = dict(duration_target='absolute_log', beat_duration_target='absolute_log',
                       note_duration_target='absolute_log', duration_input_tables='shared',
                       measure_start=True, onset_target='abs', sidecar_format='dinr_beat_events_v2')
            evaluated = quantized_example(events, targets, run, valid)
            torch.testing.assert_close(evaluated['labels'], absolute['labels'])
            model = SimpleNamespace(onset_target='abs', onset_head=SimpleNamespace(out_features=500))
            token = absolute['labels'][1, 2]
            self.assertEqual(onset_feedback_token(token, events[1], model), token)
            notes, _ = absolute_notes(events, targets.numpy(), 'dev')
            self.assertAlmostEqual(notes[0, 1], 250, places=4)

    def test_four_metrics_compare_abs_predictions_in_dev_coordinates(self):
        # Exact bin centres in each representation give identical dev=.005.
        beat_durations = np.array([500., 600., 400.])
        events = np.array([[0, -1, d, -1, -1, 1] if i % 2 == 0
                           else [1, 60, 200, .00125, 64, -1]
                           for i, d in enumerate(np.repeat(beat_durations, 2))])
        targets = np.array([[d, -1, -1] if i % 2 == 0 else [200, .005, 70]
                            for i, d in enumerate(np.repeat(beat_durations, 2))])
        pred = np.zeros((6, 3), dtype=int)
        pred[:, 0] = S.quantize(np.log(targets[:, 0]), S.duration_abs_range, 500).numpy()
        pred[1::2, 2] = 70
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tokens.npz'
            def evaluate(mode, onset_token):
                pred[1::2, 1] = onset_token
                np.savez(path, events=events, predictions=pred[None], references=pred[None],
                         reference_continuous=targets[None], reference_valid=np.ones((1, 6, 3), bool))
                return work_metrics(path, dict(duration_target='absolute_log', onset_target=mode))
            dev = evaluate('dev', 100)  # .005
            absolute = evaluate('abs', 102)  # .00625 minus .00125
            self.assertAlmostEqual(dev['onset_w1'], absolute['onset_w1'], places=12)


if __name__ == '__main__':
    unittest.main()
