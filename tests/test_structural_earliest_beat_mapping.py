import unittest

import numpy as np

from src.data_process.beat_mapping import map_score_beats_structural_earliest


class StructuralEarliestBeatMappingTests(unittest.TestCase):
    def test_uses_earliest_onset_in_each_score_group_then_interpolates(self):
        # The second note in each chord is rolled later and must not delay its
        # score onset group's anchor. The middle beat edge has no exact note.
        score_ticks = np.array([0, 0, 960, 960])
        performed_onsets = np.array([12.0, 38.0, 1012.0, 1040.0])
        anchors, supported = map_score_beats_structural_earliest(
            np.array([0, 480, 960]), score_ticks, performed_onsets)
        np.testing.assert_allclose(anchors, [12.0, 512.0, 1012.0])
        self.assertTrue(supported.all())

    def test_does_not_extrapolate_outside_observed_score_onsets(self):
        anchors, supported = map_score_beats_structural_earliest(
            np.array([0, 480, 960, 1440]),
            np.array([480, 480, 960]),
            np.array([10.0, 20.0, 510.0]))
        self.assertEqual(supported.tolist(), [False, True, True, False])
        self.assertTrue(np.isnan(anchors[0]))
        self.assertTrue(np.isnan(anchors[-1]))
        np.testing.assert_allclose(anchors[1:3], [10.0, 510.0])


if __name__ == '__main__':
    unittest.main()
