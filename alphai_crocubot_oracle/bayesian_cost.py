import tensorflow as tf
import alphai_crocubot_oracle.tensormaths as tm
import alphai_crocubot_oracle.crocubot_model as cr


class BayesianCost(object):
    def __init__(self, topology, use_double_gaussian_weights_prior=True, slab_std_dvn=1.2, spike_std_dvn=0.05,
                 spike_slab_weighting=0.5):
        """
        A class for computing Bayesian cost as described in https://arxiv.org/pdf/1505.05424.pdf .
        :param topology: A topology object that defines the topology of the network.
        :param use_double_gaussian_weights_prior: Enable the double-Gaussian prior?
        :param slab_std_dvn: The standard deviation of the slab (wide) Gaussian. Default value = 1.2.
        :param spike_std_dvn: The standard deviation of the spike (narrow) Gaussian. Default value = 0.5
        :param spike_slab_weighting: The ratio of the spike(0) to slab(1) standard deviations. Default value = 0.5.
        """
        self.topology = topology
        self._use_double_gaussian_weights_prior = use_double_gaussian_weights_prior
        self._slab_std_dvn = slab_std_dvn
        if self._slab_std_dvn <= 0. or self._slab_std_dvn > 100:
            raise ValueError("The value of slab standard deviation, {} is out of range (0,100)."
                             .format(self._slab_std_dvn))
        self._spike_std_dvn = spike_std_dvn
        if self._spike_std_dvn <= 0. or self._spike_std_dvn > 100:
            raise ValueError("The value of spike standard deviation, {} is out of range (0,100)."
                             .format(self._spike_std_dvn))
        if self._spike_std_dvn >= self._slab_std_dvn:
            raise ValueError("Spike standard deviation {} should be less that slab standard deviation {}."
                             .format(self._spike_std_dvn, self._slab_std_dvn))
        self._spike_slab_weighting = spike_slab_weighting
        if self._spike_slab_weighting < 0. or self._spike_slab_weighting > 1.:
            raise ValueError("The value of spike/slab weighting {} should be in the interval [0,1]."
                             .format(self._spike_slab_weighting))

    def get_bayesian_cost(self, prediction, target):
        log_pw, log_qw = self.calculate_priors()
        log_likelihood = self.calculate_likelihood(prediction, target)

        return log_qw - log_pw - log_likelihood

    def calculate_priors(self):
        log_pw = 0.
        log_qw = 0.

        for layer in range(self.topology.n_layers):
            mu_w = cr.get_layer_variable(layer, 'mu_w')
            rho_w = cr.get_layer_variable(layer, 'rho_w')
            mu_b = cr.get_layer_variable(layer, 'mu_b')
            rho_b = cr.get_layer_variable(layer, 'rho_b')

            # Only want to consider independent weights, not the full set, so do_tile_weights=False
            weights = cr.compute_weights(layer, iteration=0, do_tile_weights=False)
            biases = cr.compute_biases(layer, iteration=0)

            log_pw += self.calculate_log_weight_prior(weights, layer)  # not needed if we're using many passes
            log_pw += self.calculate_log_bias_prior(biases, layer)
            log_pw += self.calculate_log_hyperprior(layer)

            log_qw += self.calculate_log_q_prior(weights, mu_w, rho_w)
            log_qw += self.calculate_log_q_prior(biases, mu_b, rho_b)

        return log_pw, log_qw

    def calculate_log_weight_prior(self, weights, layer):  # TODO can we make these two into a single function?
        """
        See Equation 7 in https://arxiv.org/pdf/1505.05424.pdf
        :param weights: The weights of the layer for which the prior value is to be calculated
        :param layer: The layer number for which the weights are given.
        :return: The log-probability value.
        """
        if self._use_double_gaussian_weights_prior:

            p_slab = tm.unit_gaussian(weights / self._slab_std_dvn) / self._slab_std_dvn
            p_spike = tm.unit_gaussian(weights / self._spike_std_dvn) / self._spike_std_dvn

            log_pw = tf.log(self._spike_slab_weighting * p_slab + (1 - self._spike_slab_weighting) * p_spike)
        else:
            # FIXME this may be removed in the future as the double Gaussian is a better way to do things!
            log_alpha = cr.get_layer_variable(layer, 'log_alpha')
            log_pw = tm.log_gaussian_logsigma(weights, 0., log_alpha)

        return tf.reduce_sum(log_pw)

    def calculate_log_bias_prior(self, biases, layer):  # TODO can we make these two into a single function?
        """
        See Equation 7 in https://arxiv.org/pdf/1505.05424.pdf
        :param biases: The biases of the layer for which the prior value is to be calculated
        :param layer: The layer number for which the weights are given.
        :return: The log-probability value.
        """
        if self._use_double_gaussian_weights_prior:

            p_slab = tf.contrib.distributions.Normal(0., 1.).prob(biases / self._slab_std_dvn) / self._slab_std_dvn
            p_spike = tf.contrib.distributions.Normal(0., 1.).prob(biases / self._spike_std_dvn) / self._spike_std_dvn

            log_pw = tf.log(self._spike_slab_weighting * p_slab + (1 - self._spike_slab_weighting) * p_spike)
        else:
            log_alpha = cr.get_layer_variable(layer, 'log_alpha')
            log_pw = tm.log_gaussian_logsigma(biases, 0., log_alpha)

        return tf.reduce_sum(log_pw)

    @staticmethod
    def calculate_log_hyperprior(layer):
        """
        Compute the hyper prior for a layer. Does make any difference to the optimizer in the current form.
        :param layer: The layer number for which the hyper prior is to be calculated.
        :return: The log-probability value.
        """
        return - cr.get_layer_variable(layer, 'log_alpha')  # p(alpha) = 1 / alpha so log(p(alpha)) = - log(alpha)

    @staticmethod
    def calculate_log_q_prior(theta, mu, rho):
        """
        Calculate the log probability at theta, given mu and rho = log(sigma) of a Gaussian distribution.
        :param theta: The point at which the log-probability is to be calculated.
        :param mu: The mean of the Gaussian distribution.
        :param rho: The logarithm of the standard deviation of the Gaussian distribution.
        :return: The log-probability value.
        """

        # sigma = tf.nn.softplus(rho) #
        # sigma = tf.exp(rho)
        # log_qw = tm.log_gaussian(theta, mu, sigma)  # these 2 lines gives better accuracy than the one line below!!
        log_qw = tm.log_gaussian_logsigma(theta, mu, rho)

        return tf.reduce_sum(log_qw)

    @staticmethod
    def calculate_likelihood(truth, forecast):
        """
        Compute the Gaussian likelihood given truth and forecasts.
        :param truth: The true or target values.
        :param forecast: The forecasted values to be compared with truth
        :return: The log-likelihood value.
        """

        true_indices = tf.argmax(truth, axis=2)  # Dimensions [batch_size, N_LABEL_TIMESTEPS, N_LABEL_CLASSES]
        p_forecast = tf.gather(forecast, true_indices)
        log_likelihood = tf.maximum(tf.log(p_forecast), tm.MIN_LOG_LIKELIHOOD)

        return tf.reduce_sum(log_likelihood)
