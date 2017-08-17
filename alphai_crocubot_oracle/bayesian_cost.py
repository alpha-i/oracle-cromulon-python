import tensorflow as tf
import alphai_crocubot_oracle.tensormaths as tm

# TODO
# 1. Consult with Fergus about the interface with the network
# 2. Ask Fergus/Daniele about a subclass for classification
# 3. Fully implement tensormaths module
# 4. merge calc_log_weight_prior/calc_log_bias_prior into one?
# 5. move the constant pi to private variable ?
# 6. move LOG_TWO_PI into tensormaths
# 7. what is the purpose of calc_log_hyperprior function?

class BayesianCost(object):
    def __init__(self, network, double_gaussian_weights_prior, wide_prior_std, narrow_prior_std, do_softmax):
        self._network = network
        self._double_gaussian_weights_prior = double_gaussian_weights_prior
        self._wide_prior_std = wide_prior_std
        self._narrow_prior_std = narrow_prior_std

    def get_bayesian_cost(self, prediction, target, variances):
        log_pw, log_qw = self.calculate_priors()
        log_likelihood = self.calculate_likelihood(prediction, target, variances)

        return log_qw - log_pw - log_likelihood

    def calculate_priors(self):
        log_pw = 0.
        log_qw = 0.

        for layer in range(self._network.n_layers):  # TODO n_layers has to come from the network
            mu_w = self._network.get_layer_variable(layer, 'mu_w')
            rho_w = self._network.get_layer_variable(layer, 'rho_w')
            mu_b = self._network.get_layer_variable(layer, 'mu_b')
            rho_b = self._network.get_layer_variable(layer, 'rho_b')

            # Only want to consider independent weights, not the full set, so do_tile_weights=False
            weights = self._network.compute_weights(layer, iteration=0, do_tile_weights=False)
            biases = self._network.compute_biases(layer, iteration=0)

            log_pw += self.calc_log_weight_prior(weights, layer)  # not needed if we're using many passes
            log_pw += self.calc_log_bias_prior(biases, layer)
            log_pw += self.calc_log_hyperprior(layer)

            log_qw += self.calc_log_q_prior(weights, mu_w, rho_w)
            log_qw += self.calc_log_q_prior(biases, mu_b, rho_b)

        return log_pw, log_qw

    def calculate_likelihood(self, prediction, target, variances):
        pass

    def calc_log_weight_prior(self, weights, layer):

        if self._double_gaussian_weights_prior:
            pi = 0.6  # pi is prior that weight is large  # TODO 0.6 needs to be constant set somewhere
            # raw_pi = get_layer_variable(layer, 'pi')
            # pi = tf.clip_by_value(raw_pi, ZERO, ONE)

            pwide = tm.unit_gaussian(weights / self._wide_prior_std) / self._wide_prior_std
            pnarrow = tm.unit_gaussian(weights / self._narrow_prior_std) / self._narrow_prior_std

            log_pw = tf.log(pi * pwide + (1 - pi) * pnarrow)
        else:
            log_alpha = self._network.get_layer_variable(layer, 'log_alpha')
            log_pw = tm.log_gaussian_logsigma(weights, 0., log_alpha)

        return tf.reduce_sum(log_pw)

    def calc_log_bias_prior(self, biases, layer):  # TODO can we make these two into a single function?

        if self._double_gaussian_weights_prior:
            # pi = get_layer_variable(layer, 'pi')
            pi = 0.5 # TODO 0.5 needs to be constant set somewhere. change the name pi to something more informative

            pwide = tf.contrib.distributions.Normal(0., 1.).prob(biases / self._wide_prior_std) / self._wide_prior_std
            pnarrow = tf.contrib.distributions.Normal(0., 1.).prob(biases / self._narrow_prior_std) / self._narrow_prior_std

            log_pw = tf.log(pi * pwide + (1 - pi) * pnarrow)
        else:
            log_alpha = self._network.get_layer_variable(layer, 'log_alpha')
            log_pw = tm.log_gaussian_logsigma(biases, 0., log_alpha)

        return tf.reduce_sum(log_pw)

    def calc_log_hyperprior(self, layer):  # TODO what is purpose of this function?
        return - self._network.get_layer_variable(layer, 'log_alpha') # p(alpha) = 1 / alpha so log(p(alpha)) = - log(alpha)

    @staticmethod
    def calc_log_q_prior(theta, mu, rho):

        # sigma = tf.nn.softplus(rho) #
        sigma = tf.exp(rho)
        log_qw = tm.log_gaussian(theta, mu, sigma)

        return tf.reduce_sum(log_qw)

    def calculate_likelihood(self, truth, forecast, variance):
        if DO_SOFTMAX:  # Corresponds to probability assigned to true outcome
            MIN_LOG_LIKELI = -10  # Avoid numerical issues  # TODO make sure that this is properly defined somewhere
            true_indices = tf.argmax(truth, # TODO create a new class for claaification proproblme????
                                     axis=CLASS_DIM)  # Dimensions [batch_size, N_LABEL_TIMESTEPS, N_LABEL_CLASSES]
            p_forecast = tf.gather(forecast, true_indices)
            log_likelihood = tf.maximum(tf.log(p_forecast), MIN_LOG_LIKELI)
        else:
            chi_squared = (truth - forecast) ** 2 / variance
            log_likelihood = - 0.5 * (chi_squared + tf.log(variance) + LOG_TWO_PI)  # TODO move LOG_PWO_PI to somewhere?

        return tf.reduce_sum(log_likelihood)

