"""Beat-mean velocity curves and per-human correlation aggregation."""
import unittest

import numpy as np

from script.evaluate_exp1117_five_main_metrics import (
    average_reference_correlations, beat_mean_curve, centered_mae, pearson,
)


class FiveMainMetricTests(unittest.TestCase):
    def test_pearson_ignores_positive_affine_velocity_change(self):
        x=np.array([1.,3.,2.,8.,4.])
        mask=np.ones(len(x),dtype=bool)
        self.assertAlmostEqual(pearson(x,3*x+17,mask),1.)
        self.assertAlmostEqual(pearson(x,-x,mask),-1.)

    def test_beat_curve_uses_mean_of_onset_velocities_and_masks_empty_beats(self):
        curve=beat_mean_curve([40,80,100],[0,0,2],[True,True,True],4)
        self.assertEqual(curve[0],60.)
        self.assertTrue(np.isnan(curve[1]))
        self.assertEqual(curve[2],100.)
        self.assertTrue(np.isnan(curve[3]))

    def test_human_correlations_are_computed_individually_then_averaged(self):
        prediction=np.array([0.,1.,3.,2.])
        refs=np.array([[0.,1.,3.,2.],[0.,2.,1.,3.]])
        result=average_reference_correlations(prediction,refs,np.ones(4,bool))
        expected=np.mean([pearson(prediction,ref,np.ones(4,bool)) for ref in refs])
        self.assertAlmostEqual(result['mean'],expected)
        self.assertEqual(result['valid_references'],2)

    def test_mean_mae_compares_centres(self):
        self.assertAlmostEqual(centered_mae([[1.,4.],[3.,6.]],[[1.,4.],[3.,6.]]),0.)
        self.assertAlmostEqual(centered_mae([[2.],[4.]],[[0.],[2.]]),2.)


if __name__=='__main__':
    unittest.main()
