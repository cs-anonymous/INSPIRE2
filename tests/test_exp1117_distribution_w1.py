"""Exact unequal-size transport, score identity and legacy-score distinction."""
import unittest
from itertools import product

import numpy as np
from scipy.stats import wasserstein_distance

from script.evaluate_exp1117_distribution_w1 import empirical_w1_columns, onset_controls


class DistributionW1Tests(unittest.TestCase):
    def test_matches_scipy_with_unequal_sample_counts(self):
        rng=np.random.default_rng(42)
        for samples,references in ((1,7),(3,1),(3,2),(3,5),(3,14),(6,9),(4,4)):
            p=rng.normal(size=(samples,11));q=rng.normal(size=(references,11))
            actual=empirical_w1_columns(p,q)
            expected=[wasserstein_distance(p[:,i],q[:,i]) for i in range(11)]
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)

    def test_perfect_distribution_has_zero_distance_not_cross_mae(self):
        human=np.array([[-1.],[0.],[1.]])
        self.assertEqual(empirical_w1_columns(human,human)[0],0)
        self.assertAlmostEqual(np.abs(human[:,None,:]-human[None,:,:]).mean(),8/9)
        self.assertAlmostEqual(empirical_w1_columns([[0]],human)[0],2/3)

    def test_per_note_identity_and_onset_translation(self):
        p=np.array([[0.,10.],[1.,11.],[2.,12.]])
        swapped=p[:,::-1]
        np.testing.assert_allclose(empirical_w1_columns(p,swapped),[10,10])
        shift=np.array([.13,.72])
        np.testing.assert_allclose(empirical_w1_columns(p+shift,swapped+shift),empirical_w1_columns(p,swapped))

    def test_self_distance_and_repetition_preserve_empirical_measure(self):
        x=np.array([[1.,2.],[3.,5.],[4.,8.]])
        np.testing.assert_allclose(empirical_w1_columns(x,np.repeat(x,4,axis=0)),0,atol=1e-12)
        with self.assertRaises(ValueError):
            empirical_w1_columns([[np.nan]],[[0]])

    def test_single_reference_score_can_prefer_a_collapsed_generator(self):
        # Underlying truth is uniform {-1,0,1}. Each reference set contains
        # only one independent human sample; exhaust all 3^3 x 3 outcomes.
        outcomes=[(sample,human) for sample in product((-1.,0.,1.),repeat=3)
                  for human in (-1.,0.,1.)]
        p=np.array([s for s,h in outcomes]).T
        q=np.array([[h for s,h in outcomes]])
        self.assertAlmostEqual(empirical_w1_columns(p,q).mean(),8/9)
        self.assertAlmostEqual(empirical_w1_columns(np.zeros_like(p),q).mean(),2/3)

    def test_zero_baseline_and_within_note_diversity_are_distinct(self):
        human=np.array([[-1.,0.],[1.,0.],[-1.,0.],[1.,0.]])
        refs=[dict(work='example',human_onset=human)]
        masks=[dict(onset=np.array([True,True]))]
        predictions=dict(pt_base=[dict(onset=np.array([[-1.,0.],[0.,0.],[1.,0.]]))],
                         new_A2=[dict(onset=np.zeros((3,2)))])
        control=onset_controls(refs,masks,predictions)
        group=control['groups']['humans_ge4']
        self.assertAlmostEqual(group['zero_w1'],.5)
        self.assertAlmostEqual(group['rms_within_note_spread']['human'],np.sqrt(2/3))
        self.assertAlmostEqual(group['rms_within_note_spread']['pt_base'],np.sqrt(.5))
        self.assertEqual(group['rms_within_note_spread']['new_A2'],0)


if __name__=='__main__':
    unittest.main()
