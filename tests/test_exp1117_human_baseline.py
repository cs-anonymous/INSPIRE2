"""Human baselines must exclude self and should expose median-collapse bias."""
import unittest

import numpy as np

from script.evaluate_exp1117_human_baseline import leave_one_out


class HumanBaselineTests(unittest.TestCase):
    def test_two_distinct_performances_never_compare_to_self(self):
        np.testing.assert_allclose(leave_one_out([[0, 10], [4, 14]]), [4, 4])
        with self.assertRaises(ValueError):
            leave_one_out([[0, 10]])

    def test_pointwise_median_can_outscore_real_human_draws(self):
        humans = np.array([[-1., 0.], [0., 0.], [1., 0.]])
        np.testing.assert_allclose(leave_one_out(humans), [.75, .5, .75])
        human_score = np.mean(leave_one_out(humans))
        median_score = np.abs(humans - np.median(humans, axis=0)).mean()
        self.assertGreater(human_score, median_score)


if __name__ == '__main__':
    unittest.main()
