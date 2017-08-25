import tensorflow as tf

from alphai_crocubot_oracle.bayesian_cost import BayesianCost
from alphai_crocubot_oracle.topology import Topology


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

    def test_init(self):
        n_input_series = 10
        n_features_per_series = 100
        n_output_series = 10
        n_classification_bins = 12
        topology = Topology(layers=None, n_series=n_input_series, n_features_per_series=n_features_per_series,
                            n_forecasts=n_output_series, n_classification_bins=n_classification_bins)

        # case1 no error thrown
        use_double_gaussian_weights_prior = True
        slab_std_dvn = 1.2
        spike_std_dvn = 0.05
        spike_slab_weighting = 0.5
        BayesianCost(topology, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn, spike_slab_weighting)

        # case2 slab_std_dvn < 0
        slab_std_dvn = -1.
        self.assertRaises(ValueError, BayesianCost, topology=topology,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case3 slab_std_dvn > 100
        slab_std_dvn = 101.
        self.assertRaises(ValueError, BayesianCost, topology=topology,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case4 spike_std_dvn < 0
        spike_std_dvn = -1.
        self.assertRaises(ValueError, BayesianCost, topology=topology,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case5 spike_std_dvn > 100
        spike_std_dvn = 101.
        self.assertRaises(ValueError, BayesianCost, topology=topology,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case6 spike_std_dvn > slab_std_dvn
        spike_std_dvn = 5.
        slab_std_dvn = 1.
        self.assertRaises(ValueError, BayesianCost, topology=topology,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case7 spike_slab_weighting < 0
        spike_slab_weighting = -1.
        self.assertRaises(ValueError, BayesianCost, topology=topology,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case8 spike_slab_weighting > 1.
        spike_slab_weighting = 2.
        self.assertRaises(ValueError, BayesianCost, topology=topology,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)
