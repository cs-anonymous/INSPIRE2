"""Numerical and event-identity checks for the cross-model four-metric audit."""
from pathlib import Path
import tempfile
import unittest

import mido
import numpy as np

from script.evaluate_exp1116_baselines import (
    align_note_ons, beat_notes, note_features, read_note_on_stream, score_features, ticks_to_ms,
)


class BaselineMetricsTests(unittest.TestCase):
    def test_beat_abs_and_dev_decode_to_same_note_times(self):
        events = np.array([[0,-1,500,-1,-1,1],[1,60,200,.00125,64,-1]],dtype=float)
        reference = dict(events=events, note_mask=np.array([False,True]),
                         beat_mask=np.array([True,False]), membership=np.array([0]))
        tokens = np.array([[321,0,0],[229,100,70]])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tokens.npz'
            np.savez(path,events=events,predictions=tokens[None])
            dev=beat_notes(path,dict(duration_target='absolute_log',onset_target='dev'),reference)
            tokens[1,1]=102
            np.savez(path,events=events,predictions=tokens[None])
            absolute=beat_notes(path,dict(duration_target='absolute_log',onset_target='abs'),reference)
            for key in ('start','duration','velocity'):
                np.testing.assert_allclose(absolute[key],dev[key],atol=1e-12)

    def test_sparse_tempo_integration(self):
        times = ticks_to_ms([0, 240, 480, 720, 960], [(0, 500000), (480, 1000000)], 480)
        np.testing.assert_allclose(times, [0, 250, 500, 1000, 1500])

    def test_unpaired_note_on_does_not_shift_later_velocities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'notes.mid'
            midi = mido.MidiFile(ticks_per_beat=500)
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.extend([
                mido.Message('note_on',note=60,velocity=70,time=0),
                mido.Message('note_off',note=60,velocity=70,time=100),
                mido.Message('note_on',note=60,velocity=80,time=100),
                mido.Message('note_on',note=64,velocity=90,time=100),
                mido.Message('note_off',note=64,velocity=90,time=100),
            ])
            midi.save(path)
            stream = read_note_on_stream(path)
            self.assertEqual(stream['audit']['dangling_note_ons'], 1)
            notes = align_note_ons(stream, np.array([60, 60, 64]))
            np.testing.assert_allclose(notes['velocity'], [70, 80, 90])
            np.testing.assert_allclose(notes['start'], [0, 200, 300])
            self.assertTrue(np.isnan(notes['duration'][1]))
            np.testing.assert_allclose(notes['duration'][[0,2]], [100,100])

    def test_repeated_pitch_ties_only_invalidate_unknown_attributes(self):
        stream = dict(pitch=np.array([60,60,64]), start_tick=np.array([0,0,10]),
                      start=np.array([0.,0.,10.]), duration=np.array([20.,30.,40.]),
                      velocity=np.array([70.,80.,90.]), audit={})
        notes = align_note_ons(stream, [60,64,60])
        np.testing.assert_allclose(notes['start'], [0,10,0])
        self.assertEqual(stream['audit']['ambiguous_repeated_pitch_note_ons'], 2)
        self.assertTrue(np.isnan(notes['velocity'][[0,2]]).all())
        self.assertTrue(np.isnan(notes['duration'][[0,2]]).all())
        self.assertEqual(notes['velocity'][1], 90)

    def test_mapping_self_reference_and_scale_invariance(self):
        reference = dict(edges=np.array([0,1,2,3]), ticks=np.array([0,1,2,3]),
                         notes=4, membership=np.array([0,1,2,2]),
                         score_ratio=np.array([0.,0.,0.,1.]),
                         score_log_duration=np.log(np.array([100.,100.,100.,100.])))
        notes = dict(start=np.array([0.,500.,1100.,1500.]),
                     duration=np.array([100.,150.,200.,250.]),
                     velocity=np.array([60.,70.,80.,90.]))
        f = note_features(notes,reference)
        np.testing.assert_allclose(f['onset'], 0., atol=1e-12)
        np.testing.assert_allclose(np.exp(f['beat_log']), [500.,600.,400.])
        reference.update(human_beat_log=f['beat_log'][None,:], human_onset=f['onset'][None,:],
                         human_duration=f['duration'][None,:], human_velocity=f['velocity'][None,:])
        masks = dict(beat=np.ones(3,bool), onset=np.ones(4,bool),
                     duration=np.ones(4,bool), velocity=np.ones(4,bool))
        metrics = score_features(f,reference,masks)
        self.assertAlmostEqual(metrics['tempo_curve_pearson'], 1.)
        for key in ('onset_w1','duration_w1','velocity_w1'):
            self.assertAlmostEqual(metrics[key], 0.)
        scaled = note_features({**notes,'start':notes['start']*2,'duration':notes['duration']*2},reference)
        np.testing.assert_allclose(scaled['onset'],f['onset'],atol=1e-12)
        self.assertAlmostEqual(score_features(scaled,reference,masks)['tempo_curve_pearson'],1.)

    def test_missing_pitch_fails_instead_of_truncating(self):
        stream = dict(pitch=np.array([60]),start_tick=np.array([0]),start=np.array([0.]),
                      duration=np.array([100.]),velocity=np.array([70.]),audit={})
        with self.assertRaises(ValueError):
            align_note_ons(stream,[60,60])


if __name__ == '__main__':
    unittest.main()
