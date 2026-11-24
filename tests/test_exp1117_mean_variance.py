"""Centre error and variation across draws should remain separate."""
import unittest

from script.evaluate_exp1117_mean_variance import mean_variance_columns


class MeanVarianceTests(unittest.TestCase):
    def test_zero_centre_error_does_not_hide_variance_collapse(self):
        human=[[-1.,8.],[1.,12.]]
        collapsed=mean_variance_columns([[0.,10.]]*3,human)
        diverse=mean_variance_columns([[-1.,8.],[0.,10.],[1.,12.]],human)
        self.assertEqual(collapsed['mean_mae'],0.)
        self.assertEqual(diverse['mean_mae'],0.)
        self.assertEqual(collapsed['generated_var'],0.)
        self.assertEqual(diverse['generated_var'],2.5)
        self.assertEqual(diverse['human_var'],5.)

    def test_center_mae_cannot_cancel_errors_between_notes(self):
        result=mean_variance_columns([[0.,10.]]*3,[[2.,8.]])
        self.assertEqual(result['mean_mae'],2.)
        self.assertEqual(result['signed_mean_error'],0.)
        self.assertNotIn('human_var',result)


if __name__=='__main__':
    unittest.main()
