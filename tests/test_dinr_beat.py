"""Beat event alignment, duration feedback, and causal/cache regression checks."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data_process.dinr_beat_dataset import BeatTokenSpec as S, DINRBeatDataset, DINRBeatCollator
from src.data_process.build_dinr_beat_sidecars import canonical_beat_windows, performance_beat_anchors, load_beat_mapper
from src.model.dinr_beat import DINRBeatModel
from src.model.pianoformer import PianoT5GemmaConfig


class BeatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / 'events.pt'
        events = torch.tensor([[0,-1,500,-1,-1,1], [1,60,250,0,64,-1],
                               [1,64,250,.5,64,-1], [0,-1,500,-1,-1,0],
                               [1,67,250,0,64,-1]], dtype=torch.float32)
        targets = torch.tensor([[600,-1,-1], [200,-.05,72], [300,.55,68],
                                [550,-1,-1], [225,.01,80]], dtype=torch.float32)
        torch.save(dict(format='dinr_beat_events_v1', events=events, targets=targets,
                        samples=[dict(span=[0,5]), dict(span=[0,3])]), cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def model(self, **kwargs):
        torch.manual_seed(42)
        config = PianoT5GemmaConfig(encoder_layers_num=1, decoder_layers_num=1,
                                    hidden_size=32, intermediate_size=64,
                                    num_attention_heads=2, num_key_value_heads=1, head_dim=16)
        return DINRBeatModel(config, decoder_mask_probability=0, **kwargs)

    def test_duration_feedback_and_labels(self):
        absolute = DINRBeatDataset(self.path)[0]
        mixed = DINRBeatDataset(self.path, beat_duration_target='absolute_log',
                                note_duration_target='log_dev')[0]
        self.assertTrue(torch.equal(absolute['decoder_tokens'], mixed['decoder_tokens']))
        self.assertTrue(torch.equal(absolute['labels'][[0,3]], mixed['labels'][[0,3]]))
        self.assertFalse(torch.equal(absolute['labels'][[1,2,4]], mixed['labels'][[1,2,4]]))
        self.assertEqual(mixed['decoder_tokens'][0,0].item(), S.bos_type)
        self.assertEqual(mixed['decoder_tokens'][1,0].item(), S.beat_type)
        self.assertEqual((mixed['labels'] >= 0).sum().item(), 11)
        no_measure = DINRBeatDataset(self.path, input_measure_start=False)[0]
        self.assertTrue((no_measure['encoder_tokens'][:,5] == 0).all())

    def test_v2_onset_dev_target_and_absolute_history(self):
        events = torch.tensor([[0,-1,500,-1,-1,1], [1,60,250,.4,64,-1],
                               [0,-1,500,-1,-1,0], [1,67,250,.2,64,-1]], dtype=torch.float32)
        targets = torch.tensor([[600,-1,-1], [200,.1,72],
                                [550,-1,-1], [225,-.1,80]], dtype=torch.float32)
        path = Path(self.temp.name) / 'events_v2.pt'
        torch.save(dict(format='dinr_beat_events_v2', events=events, targets=targets,
                        target_valid=torch.ones_like(targets, dtype=torch.bool),
                        samples=[dict(span=[0,4])]), path)
        row = DINRBeatDataset(path)[0]
        expected = S.quantize(torch.tensor([.1, -.1]), S.onset_dev_range, S.onset_dev_bins)
        self.assertTrue(torch.equal(row['labels'][[1,3],2], expected))
        expected_ratio = S.quantize(torch.tensor(.5), S.onset_input_range,
                                    S.onset_input_bins)
        self.assertEqual(row['decoder_tokens'][2,3].item(), expected_ratio.item())
        self.assertEqual(S.onset_dev_bins, 200)

    def test_cache_matches_full_and_causal(self):
        batch = DINRBeatCollator()([DINRBeatDataset(self.path)[0]])
        model = self.model().eval()
        with torch.no_grad():
            full = model(**batch)
            encoder = full['encoder_outputs']
            cache = None
            for i in range(5):
                current = model(encoder_tokens=batch['encoder_tokens'],
                                decoder_tokens=batch['decoder_tokens'][:,i:i+1],
                                event_types=batch['event_types'][:,i:i+1],
                                attention_mask=batch['attention_mask'],
                                decoder_attention_mask=torch.ones(1,i+1,dtype=torch.long),
                                encoder_outputs=encoder, past_key_values=cache,
                                use_cache=True, cache_position=torch.tensor([i]))
                cache = current['past_key_values']
                for key in ('duration','onset','velocity'):
                    torch.testing.assert_close(current['logits'][key][:,0], full['logits'][key][:,i], atol=1e-5, rtol=1e-4)
            changed = dict(batch)
            changed['decoder_tokens'] = batch['decoder_tokens'].clone()
            changed['decoder_tokens'][:,3:,2] = 10
            future = model(**changed)
            torch.testing.assert_close(full['logits']['duration'][:,:3], future['logits']['duration'][:,:3])

    def test_token_metrics_ignore_masked_beat_and_onset_labels(self):
        from src.evaluate.evaluate_dinr_beat import token_metrics
        events = torch.tensor([[0,-1,500,-1,-1,1], [1,60,250,.4,64,-1],
                               [0,-1,500,-1,-1,0], [1,67,250,.2,64,-1]])
        reference = np.array([[600,-1,-1],[200,.1,72],[550,-1,-1],[225,-.1,80]])
        continuous = reference.copy()
        continuous[0,0] = 100000  # ignored beat label
        continuous[1,1] = 100000  # ignored onset label
        labels = torch.tensor([[-100,-100,0],[20,-100,72],[21,-100,0],[22,90,80]])
        tokens = labels.clamp_min(0)
        metrics = token_metrics(tokens, labels, events, continuous, reference)
        self.assertEqual(metrics['beat_duration_log_mae'], 0.)
        self.assertEqual(metrics['onset_mae'], 0.)

    def test_direct_metrics_retain_continuous_labels_and_masks(self):
        from script.evaluate_exp1117_beat_dev import work_metrics, decode_numpy
        events = np.array([[0,-1,500,-1,-1,1],[1,60,250,.4,64,-1],
                           [0,-1,500,-1,-1,0],[1,62,250,.2,64,-1],
                           [0,-1,500,-1,-1,0],[1,64,250,.3,64,-1],
                           [0,-1,500,-1,-1,0],[1,65,250,.6,64,-1]], dtype=float)
        predictions = np.array([[310,0,0],[250,110,72],[330,0,0],[260,90,75],
                                [320,0,0],[240,105,70],[340,0,0],[255,115,73]])
        continuous = np.column_stack((
            np.exp(decode_numpy(predictions[:,0], S.duration_abs_range, S.duration_bins)),
            decode_numpy(predictions[:,1], S.onset_dev_range, S.onset_dev_bins),
            predictions[:,2]))[None,:,:]
        valid = np.ones_like(continuous, dtype=bool)
        valid[:,0,0] = False
        valid[:,1,1] = False
        continuous[:,1,1] = 999.  # excluded by mask, not clipped to a target bin
        path = Path(self.temp.name) / 'direct_metrics.npz'
        np.savez(path, events=events, predictions=predictions[None,:,:],
                 references=predictions[None,:,:], reference_continuous=continuous,
                 reference_valid=valid)
        result = work_metrics(path, {'duration_target':'absolute_log'})
        self.assertAlmostEqual(result['tempo_curve_pearson'], 1.)
        for key in ('onset_w1','duration_w1','velocity_w1'):
            self.assertAlmostEqual(result[key], 0.)
        self.assertEqual(result['onset_notes'], 3)

    def test_backward_routing_and_padding(self):
        ds = DINRBeatDataset(self.path, input_duration_tables='separate')
        batch = DINRBeatCollator()([ds[0],ds[1]])
        for head in ('separate','shared_conditioned','shared'):
            model = self.model(duration_input_tables='separate', duration_output_head=head)
            output = model(**batch)
            self.assertTrue(torch.isfinite(output['loss']))
            output['loss'].backward()
            for part in (model.beat_duration_embedding, model.note_duration_embedding, model.onset_head):
                self.assertGreater(part.weight.grad.abs().sum().item(),0)
            if head == 'separate':
                self.assertGreater(model.beat_duration_head.weight.grad.abs().sum().item(),0)
                self.assertGreater(model.note_duration_head.weight.grad.abs().sum().item(),0)

    def test_generation_does_not_read_performance_targets(self):
        from src.evaluate.evaluate_dinr_beat import predict_many, absolute_notes, decode_duration
        ds = DINRBeatDataset(self.path)
        events, targets = ds.data['events'], ds.data['targets']
        run = dict(duration_target='absolute_log', beat_duration_target='absolute_log',
                   note_duration_target='absolute_log', duration_input_tables='shared', measure_start=True)
        model = self.model().eval()
        first = predict_many(model,events,targets,run,torch.device('cpu'),[123,456])
        changed = predict_many(model,events,torch.ones_like(targets)*100,run,torch.device('cpu'),[123,456])
        for a,b in zip(first,changed):
            self.assertTrue(torch.equal(a,b))
            self.assertEqual(a.shape,(5,3))
        # Known continuous beat/note stream reconstructs negative onset and
        # crossing a beat boundary without changing note identity.
        notes,anchors = absolute_notes(events,targets.numpy())
        np.testing.assert_allclose(anchors,[0,600])
        np.testing.assert_allclose(notes[:,1],[-30,330,605.5],atol=1e-5)
        self.assertTrue((notes[:,2]>0).all())

    def test_beat_mapper_units_and_windows(self):
        _, mapper = load_beat_mapper()
        anchors = performance_beat_anchors(np.array([0,480,960]), np.array([0,480,960,1440]),
                                          np.array([0.,500.,1000.]), mapper,480)
        np.testing.assert_allclose(anchors,[0,500,1000,1500])
        counts = [0,3,5,0,4,8,0]
        windows = canonical_beat_windows(counts,10,.5)
        self.assertTrue(windows)
        covered = set()
        for start,end in windows:
            covered.update(range(start,end))
        self.assertTrue({i for i,n in enumerate(counts) if n}.issubset(covered))


if __name__ == '__main__':
    unittest.main()
