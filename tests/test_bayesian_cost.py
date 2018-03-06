import tensorflow as tf
import numpy as np
from scipy.stats import norm

from alphai_cromulon_oracle.bayesian_cost import BayesianCost
from alphai_cromulon_oracle.topology import Topology
from alphai_cromulon_oracle.cromulon.model import Cromulon
from tests.helpers import get_default_flags


class TestBayesianCost(tf.test.TestCase):

    def test_calc_log_q_prior(self):

        parameters = [
            (1., 0., 1., -1.4813652),
            (2., -1., -1., -45.614418)
        ]
        parameters = np.array(parameters, dtype=np.float32)
        with self.test_session():
            for test in parameters:
                theta, mu, rho, expected_result = test
                actual_result = BayesianCost.calculate_log_q_prior(theta, mu, rho).eval()
                self.assertAlmostEqual(
                    actual_result,
                    expected_result,
                    places=4
                )

    def test_init(self):

        flags = get_default_flags()

        topology = Topology()

        self.cromulon = Cromulon(topology, flags, is_training=True)

        # case1 no error thrown
        use_double_gaussian_weights_prior = True
        slab_std_dvn = 1.2
        spike_std_dvn = 0.05
        spike_slab_weighting = 0.5
        BayesianCost(self.cromulon, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn, spike_slab_weighting)

        # case2 slab_std_dvn < 0
        slab_std_dvn = -1.
        self.assertRaises(ValueError, BayesianCost, bayes_layers=self.cromulon.bayes,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case3 slab_std_dvn > 100
        slab_std_dvn = 101.
        self.assertRaises(ValueError, BayesianCost, bayes_layers=self.cromulon.bayes,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case4 spike_std_dvn < 0
        spike_std_dvn = -1.
        self.assertRaises(ValueError, BayesianCost, bayes_layers=self.cromulon.bayes,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case5 spike_std_dvn > 100
        spike_std_dvn = 101.
        self.assertRaises(ValueError, BayesianCost, bayes_layers=self.cromulon.bayes,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case6 spike_std_dvn > slab_std_dvn
        spike_std_dvn = 5.
        slab_std_dvn = 1.
        self.assertRaises(ValueError, BayesianCost, bayes_layers=self.cromulon.bayes,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case7 spike_slab_weighting < 0
        spike_slab_weighting = -1.
        self.assertRaises(ValueError, BayesianCost, bayes_layers=self.cromulon.bayes,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case8 spike_slab_weighting > 1.
        spike_slab_weighting = 2.
        self.assertRaises(ValueError, BayesianCost, bayes_layers=self.cromulon.bayes,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

    def test_calculate_log_weight_prior(self):

        self.test_init()

        with self.test_session():
            layer = 0

            # case 1 slab prior
            use_double_gaussian_weights_prior = True
            slab_std_dvn = 1.
            spike_std_dvn = 0.05
            spike_slab_weighting = 1.
            bayes_cost = BayesianCost(self.cromulon.bayes, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn,
                                      spike_slab_weighting)
            weights = np.random.normal(size=5)
            weights = weights.astype(np.float32)
            log_prior_value_computed = bayes_cost.calculate_log_weight_prior(weights, layer)
            log_prior_value_expected = np.sum(norm.logpdf(weights))
            self.assertAlmostEqual(log_prior_value_computed.eval(), log_prior_value_expected, places=5)

            # case 2 spike prior
            slab_std_dvn = 2.  # note that we have condition that slab_std_dvn >= spike_std_dvn
            spike_std_dvn = 1.
            spike_slab_weighting = 0.
            bayes_cost = BayesianCost(self.cromulon.bayes, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn,
                                      spike_slab_weighting)
            log_prior_value_computed = bayes_cost.calculate_log_weight_prior(weights, layer)
            self.assertAlmostEqual(log_prior_value_computed.eval(), log_prior_value_expected, places=5)

            # case 3 50/50 spike/slab
            slab_std_dvn = 2.
            spike_std_dvn = 1.
            spike_slab_weighting = 0.5
            bayes_cost = BayesianCost(self.cromulon.bayes, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn,
                                      spike_slab_weighting)
            log_prior_value_computed = bayes_cost.calculate_log_weight_prior(weights, layer)
            log_prior_value_expected = np.sum(np.log(spike_slab_weighting * norm.pdf(weights / slab_std_dvn) /
                                                     slab_std_dvn + (1. - spike_slab_weighting) *
                                                     norm.pdf(weights / spike_std_dvn) / spike_std_dvn))
            self.assertAlmostEqual(log_prior_value_computed.eval(), log_prior_value_expected, places=5)

    def test_calculate_log_bias_prior(self):
        # FIXME the only difference between this and the previous one is the function
        # FIXME calculate_log_weight_prior() changes to calculate_log_weight_prior().
        # FIXME Otherwise, they are identical.

        self.test_init()

        with self.test_session():
            layer = 0

            # case 1 slab prior
            use_double_gaussian_weights_prior = True
            slab_std_dvn = 1.
            spike_std_dvn = 0.05
            spike_slab_weighting = 1.
            bayes_cost = BayesianCost(self.cromulon.bayes, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn,
                                      spike_slab_weighting)
            weights = np.random.normal(size=5)
            weights = weights.astype(np.float32)
            log_prior_value_computed = bayes_cost.calculate_log_bias_prior(weights, layer)
            log_prior_value_expected = np.sum(norm.logpdf(weights))
            self.assertAlmostEqual(log_prior_value_computed.eval(), log_prior_value_expected, places=5)

            # case 2 spike prior
            slab_std_dvn = 2.  # note that we have condition that slab_std_dvn >= spike_std_dvn
            spike_std_dvn = 1.
            spike_slab_weighting = 0.
            bayes_cost = BayesianCost(self.cromulon.bayes, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn,
                                      spike_slab_weighting)
            log_prior_value_computed = bayes_cost.calculate_log_weight_prior(weights, layer)
            self.assertAlmostEqual(log_prior_value_computed.eval(), log_prior_value_expected, places=5)

            # case 3 50/50 spike/slab
            slab_std_dvn = 2.
            spike_std_dvn = 1.
            spike_slab_weighting = 0.5
            bayes_cost = BayesianCost(self.cromulon.bayes, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn,
                                      spike_slab_weighting)
            log_prior_value_computed = bayes_cost.calculate_log_bias_prior(weights, layer)
            log_prior_value_expected = np.sum(np.log(spike_slab_weighting * norm.pdf(weights / slab_std_dvn) /
                                                     slab_std_dvn + (1. - spike_slab_weighting) *
                                                     norm.pdf(weights / spike_std_dvn) / spike_std_dvn))
            self.assertAlmostEqual(log_prior_value_computed.eval(), log_prior_value_expected, places=5)
