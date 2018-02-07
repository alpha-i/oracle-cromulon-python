"""
This modules contains the implementation of the BayesianCost which is
the cost function to control the learning process.

It is used during the train of the model, implemented in the module alphai_crocubot_oracle.train:train

"""

import tensorflow as tf
import alphai_cromulon_oracle.tensormaths as tm

from tensorflow.python.ops import math_ops

N_BATCHES_SUPPRESSED_PRIOR = 1  # How many batches over which we gradually introduce the prior
ENTROPIC_COST_STRENGTH = 1e-3  # How strongly the cost function is modified


class BayesianCost(object):

    def __init__(self, bayes_layers, use_double_gaussian_weights_prior=True, slab_std_dvn=1.2, spike_std_dvn=0.05,
                 spike_slab_weighting=0.5, n_batches=100):
        """
        A class for computing Bayesian cost as described in https://arxiv.org/pdf/1505.05424.pdf .
        :param bayes_layers BayesianLayers: An object that defines the fully connected part of the network.
        :param use_double_gaussian_weights_prior: Enable the double-Gaussian prior?
        :param slab_std_dvn: The standard deviation of the slab (wide) Gaussian. Default value = 1.2.
        :param spike_std_dvn: The standard deviation of the spike (narrow) Gaussian. Default value = 0.5
        :param spike_slab_weighting: The ratio of the spike(0) to slab(1) standard deviations. Default value = 0.5.
        """
        self._verify_args(spike_std_dvn, slab_std_dvn, spike_slab_weighting)
        self._model = bayes_layers
        self.topology = bayes_layers.topology
        self._use_double_gaussian_weights_prior = use_double_gaussian_weights_prior
        self._epoch_fraction = 1 / n_batches
        self._slab_std_dvn = tf.cast(slab_std_dvn, tm.DEFAULT_TF_TYPE)
        self._spike_std_dvn = tf.cast(spike_std_dvn,  tm.DEFAULT_TF_TYPE)
        self._spike_slab_weighting = tf.cast(spike_slab_weighting,  tm.DEFAULT_TF_TYPE)

    def get_bayesian_cost(self, log_prediction, truth, global_step=None):
        """

        :param prediction:
        :param truth:
        :param global_step: Used to suppress the prior during the early phases of learning
        :return:
        """
        log_pw, log_qw = self.calculate_priors()
        log_likelihood = self.calculate_likelihood(truth, log_prediction)

        prior_strength = self.calculate_prior_strength(global_step)
        log_prior = (log_qw - log_pw) * self._epoch_fraction * prior_strength

        l2_loss = tf.losses.get_regularization_loss()  # From convolutional kernels

        cost = log_prior - log_likelihood + l2_loss

        return cost, log_likelihood

    def get_entropy_cost(self, prediction, truth, global_step=None):

        log_pw, log_qw = self.calculate_priors()
        log_likelihood = self.calculate_likelihood(truth, prediction)

        prior_strength = self.calculate_prior_strength(global_step)
        entropic_cost = self.calculate_entropic_cost(prediction)

        log_prior = ((log_qw - log_pw) * self._epoch_fraction) * prior_strength

        cost = log_prior - log_likelihood + entropic_cost * ENTROPIC_COST_STRENGTH

        return cost, log_likelihood

    def calculate_entropic_cost(self, log_prediction):
        """ Discourages the network from monotonously predicting a single outcome."""

        log_prediction = tf.squeeze(log_prediction)
        mean, var_samples = tf.nn.moments(tf.exp(log_prediction), axes=0)
        clip_min_variance = 1e-5  # Prevents 1 / 0 errors
        var_samples = math_ops.maximum(var_samples, clip_min_variance)

        penalty = tf.reciprocal(var_samples)  # Small variance across samples should be discouraged

        return tf.reduce_sum(penalty)

    def calculate_log_p_multinomial(self, n_counts, nbins, batch_size):
        """ Log p of drawing x_i balls of colour i with replacement.
        See e.g. https://en.wikipedia.org/wiki/Multinomial_distribution#Probability_mass_function
        """

        n_counts = tf.cast(n_counts, tf.float32)
        nbins = tf.cast(nbins, tf.float32)
        batch_size = tf.cast(batch_size, tf.float32)

        term_a = tf.lgamma(batch_size + 1)
        term_b = tf.reduce_sum(tf.lgamma(n_counts + 1))
        term_c = batch_size * tf.log(nbins)

        return term_a - term_b - term_c

    def calculate_prior_strength(self, train_steps):

        if train_steps is None:
            prior_strength = tf.cast(1.0, tf.float32)
        else:
            step_ratio = train_steps / N_BATCHES_SUPPRESSED_PRIOR
            one = tf.cast(1.0, tf.float32)
            step = tf.cast(step_ratio, tf.float32)
            prior_strength = tf.minimum(one, step)

        return prior_strength

    def get_hellinger_cost(self, features, truth, n_passes, estimator):
        """ Perform similar sum to bayesian cost, but different weighting over different passes. """

        costs = []
        for i in range(n_passes):
            prediction = estimator.forward_pass(features, iteration=i)
            prediction = tf.nn.log_softmax(prediction, dim=-1)

            log_likelihood = self.calculate_likelihood(truth, prediction)
            log_pw, log_qw = self.calculate_priors(iteration=i)

            single_pass_cost = (log_qw - log_pw) * self._epoch_fraction - log_likelihood
            costs.append(single_pass_cost)

        stack_of_passes = tf.stack(costs, axis=0)
        return - tf.reduce_logsumexp(- 0.5 * stack_of_passes, axis=0)

    def calculate_output_prior(self, prediction):
        """

        :param prediction:
        :return:
        """
        log_py = 0.5 * tf.reduce_sum(prediction + tf.log(1.0 - tf.exp(prediction)))

        return log_py

    def calculate_priors(self, iteration=0):
        log_pw = 0.
        log_qw = 0.

        for layer in range(self.topology.n_layers):
            layer_type = self.topology.layers[layer]["type"]
            if layer_type == 'full':
                mu_w = self._model.get_variable(layer, self._model.VAR_WEIGHT_MU)
                rho_w = self._model.get_variable(layer, self._model.VAR_WEIGHT_RHO)
                mu_b = self._model.get_variable(layer, self._model.VAR_BIAS_MU)
                rho_b = self._model.get_variable(layer, self._model.VAR_BIAS_RHO)

                if self._model._flags.n_train_passes == 1:  # Exploit common random numbers
                    weights = self._model.compute_weights(layer)
                    biases = self._model.compute_biases(layer)

                    log_qw += self.calculate_log_q_prior(weights, mu_w, rho_w)
                    log_qw += self.calculate_log_q_prior(biases, mu_b, rho_b)

                else:  # Use mean and rho to estimate expectation of weights and biases
                    mu_w = self._model.get_variable(layer, self._model.VAR_WEIGHT_MU)
                    mu_b = self._model.get_variable(layer, self._model.VAR_BIAS_MU)
                    sigma_w = tf.nn.softplus(self._model.get_variable(layer, self._model.VAR_WEIGHT_RHO))
                    sigma_b = tf.nn.softplus(self._model.get_variable(layer, self._model.VAR_BIAS_RHO))
                    weights = tf.abs(mu_w) + sigma_w
                    biases = tf.abs(mu_b) + sigma_b

                log_pw += self.calculate_log_weight_prior(weights, layer)
                log_pw += self.calculate_log_bias_prior(biases, layer)
                log_pw += self.calculate_log_hyperprior(layer)

        return log_pw, log_qw

    def calculate_log_weight_prior(self, weights, layer):
        """
        See Equation 7 in https://arxiv.org/pdf/1505.05424.pdf
        :param weights: The weights of the layer for which the prior value is to be calculated
        :param layer: The layer number for which the weights are given.
        :return: The log-probability value.
        """

        if self._use_double_gaussian_weights_prior:
            log_p_spike = tf.log(1 - self._spike_slab_weighting) + tm.log_gaussian(weights, 0., self._spike_std_dvn)
            log_p_slab = tf.log(self._spike_slab_weighting) + tm.log_gaussian(weights, 0., self._slab_std_dvn)

            p_total = tf.stack([log_p_spike, log_p_slab], axis=0)
            log_pw = tf.reduce_logsumexp(p_total, axis=0)
        else:
            # FIXME this may be removed in the future as the double Gaussian is a better way to do things!
            log_alpha = self._model.get_variable(layer, self._model.VAR_LOG_ALPHA)
            log_pw = tm.log_gaussian_logsigma(weights, 0., log_alpha)

        return tf.reduce_sum(log_pw)

    def calculate_log_bias_prior(self, biases, layer):
        """
        At present we impose the same prior on the biases as is imposed on the weights
        :param biases: The biases of the layer for which the prior value is to be calculated
        :param layer: The layer number for which the weights are given.
        :return: The log-probability value.
        """

        return self.calculate_log_weight_prior(biases, layer)

    def calculate_log_hyperprior(self, layer):
        """
        Compute the hyper prior for a layer. Does make any difference to the optimizer in the current form.
        :param layer: The layer number for which the hyper prior is to be calculated.
        :return: The log-probability value.
        """
        return - self._model.get_variable(layer, self._model.VAR_LOG_ALPHA)

    @staticmethod
    def calculate_log_q_prior(theta, mu, rho):
        """
        Calculate the log probability at theta, given mu and rho = log(sigma) of a Gaussian distribution.
        :param theta: The point at which the log-probability is to be calculated.
        :param mu: The mean of the Gaussian distribution.
        :param rho: The logarithm of the standard deviation of the Gaussian distribution.
        :return: The log-probability value.
        """

        sigma = tf.nn.softplus(rho)
        log_qw = tm.log_gaussian(theta, mu, sigma)
        return tf.reduce_sum(log_qw)

    @staticmethod
    def calculate_likelihood(truth, log_forecast):
        """
        Compute the Gaussian likelihood given truth and forecasts.
        :param truth: The true or target distributions.
        :param log_forecast: The log forecast to be compared with truth
        :return: The total log likelihood value.
        """

        return tf.reduce_sum(truth * log_forecast)   # Dimensions [batch_size, N_LABEL_TIMESTEPS, N_LABEL_CLASSES]

    @staticmethod
    def _verify_args(spike_std_dvn, slab_std_dvn, spike_slab_weighting):

        if slab_std_dvn <= 0. or slab_std_dvn > 100:
            raise ValueError("The value of slab standard deviation, {} is out of range (0,100)."
                             .format(slab_std_dvn))
        if spike_std_dvn <= 0. or spike_std_dvn > 100:
            raise ValueError("The value of spike standard deviation, {} is out of range (0,100)."
                             .format(spike_std_dvn))
        if spike_std_dvn >= slab_std_dvn:
            raise ValueError("Spike standard deviation {} should be less that slab standard deviation {}."
                             .format(spike_std_dvn, slab_std_dvn))
        if spike_slab_weighting < 0. or spike_slab_weighting > 1.:
            raise ValueError("The value of spike/slab weighting {} should be in the interval [0,1]."
                             .format(spike_slab_weighting))
