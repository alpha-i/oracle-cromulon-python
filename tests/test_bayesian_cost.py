import tensorflow as tf

from alphai_crocubot_oracle.bayesian_cost import BayesianCost


class TestBayesianCost(tf.test.TestCase):

    def test_calc_log_q_prior(self):

        parameters = [
            (1., 0., 1., -1.9866061),
            (2., -1., -1., -33.169685)
        ]
        with self.test_session():
            for test in parameters:
                theta, mu, rho, expected_result = test
                actual_result = BayesianCost.calc_log_q_prior(theta, mu, rho).eval()
                self.assertAlmostEqual(
                    actual_result,
                    expected_result,
                    places=6
                )

