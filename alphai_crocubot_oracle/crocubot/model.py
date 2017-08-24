# Defines the network in tensorflow and allows access to its variables
# Used by crocubot_train and crocubot_eval

import logging

import numpy as np
import tensorflow as tf

import alphai_crocubot_oracle.tensormaths as tm

FLAGS = tf.app.flags.FLAGS


class CrocuBotModel:

    VAR_RHO_BIAS = 'rho_b'
    VAR_RHO_WEIGHT = 'rho_w'

    VAR_MU_BIAS = 'mu_b'
    VAR_MU_WEIGHT = 'mu_w'

    VAR_WEIGHT_NOISE = 'weight_noise'
    VAR_BIAS_NOISE = 'bias_noise'

    VAR_LOG_ALPHA = 'log_alpha'

    def __init__(self, topology, flags):

        self._topology = topology
        self._graph = tf.get_default_graph()
        self._flags = flags
        self._noise_seed = flags.noise_seed

    @property
    def graph(self):
        return self._graph

    def initialize(self):
        
        weight_uncertainty = self._flags.INITIAL_WEIGHT_UNCERTAINTY
        bias_uncertainty = self._flags.INITIAL_BIAS_UNCERTAINTY
        weight_displacement = self._flags.INITIAL_WEIGHT_DISPLACEMENT
        bias_displacement = self._flags.INITIAL_BIAS_DISPLACEMENT

        initial_rho_weights = np.log(weight_uncertainty)
        initial_rho_bias = np.log(bias_uncertainty)
        initial_alpha = self._flags.INITIAL_ALPHA

        for layer_number in range(self._topology.n_layers):

            w_shape = self._topology.get_weight_shape(layer_number)
            b_shape = self._topology.get_bias_shape(layer_number)

            self._create_variable_for_layer(
                layer_number,
                self.VAR_MU_WEIGHT,
                tm.centred_gaussian(w_shape, weight_displacement)
            )

            self._create_variable_for_layer(
                layer_number,
                self.VAR_RHO_WEIGHT,
                initial_rho_weights + tm.centred_gaussian(w_shape, np.abs(initial_rho_weights) / 10 )
            )

            self._create_variable_for_layer(
                layer_number,
                self.VAR_MU_BIAS,
                tm.centred_gaussian(b_shape, bias_displacement)
            )

            self._create_variable_for_layer(
                layer_number,
                self.VAR_RHO_BIAS,
                initial_rho_bias + tm.centred_gaussian(b_shape, np.abs(initial_rho_bias) / 10)
            )

            self._create_variable_for_layer(
                layer_number,
                self.VAR_LOG_ALPHA,
                np.log(initial_alpha).astype(self._flags.d_type),
                False
            )  # Hyperprior on the distribution of the weights

            self._create_noise(layer_number, self.VAR_WEIGHT_NOISE, w_shape)
            self._create_noise(layer_number, self.VAR_BIAS_NOISE, b_shape)

    def _create_variable_for_layer(self, layer_number, variable_name, initializer, is_trainable=True):

        assert isinstance(layer_number, int)
        scope_name = str(layer_number)
        with tf.variable_scope(scope_name):  # TODO check if this is the correct
            tf.get_variable(variable_name, initializer=initializer, trainable=is_trainable, dtype=tm.DEFAULT_TF_TYPE)

    def _create_noise(self, layer_number, variable_name, shape):

        if self._flags.USE_PERFECT_NOISE:
            noise_vector = tm.perfect_centred_gaussian(shape)
        else:
            noise_vector = tm.centred_gaussian(shape)

        self._create_variable_for_layer(layer_number, variable_name, noise_vector, False)

    def get_variable(self, layer_number, variable_name, reuse=True):

        scope_name = str(layer_number)
        with tf.variable_scope(scope_name, reuse=reuse):
            v = tf.get_variable(variable_name, dtype=tm.DEFAULT_TF_TYPE)

        return v

    def get_weight_noise(self, layer_number, iteration):
        noise = self.get_variable(layer_number, self.VAR_WEIGHT_NOISE)
        return tm.roll_noise(
            noise,
            iteration
        )

    def get_bias_noise(self, layer_number, iteration):
        noise = self.get_variable(layer_number, self.VAR_BIAS_NOISE)
        return tm.roll_noise(
            noise,
            iteration
        )

    def compute_weights(self, layer_number, iteration=0):

        mean = self.get_variable(layer_number, self.VAR_MU_WEIGHT)
        rho = self.get_variable(layer_number, self.VAR_RHO_WEIGHT)
        noise = self.get_weight_noise(layer_number, iteration)

        return mean + tf.exp(rho) * noise

    def compute_biases(self, layer_number, iteration):
        """Bias is Gaussian distributed"""
        mean = self.get_variable(layer_number, self.VAR_MU_BIAS)
        rho = self.get_variable(layer_number, self.VAR_RHO_BIAS)
        noise = self.get_bias_noise(layer_number, iteration)

        return mean + tf.exp(rho) * noise

    def reset(self):
        self._graph.reset()

    def increment_noise_seed(self):
        self._noise_seed += 1

def get_layer_variable(layer_number, var_name, reuse=True):
    """

    :param layer_number:
    :param var_name:
    :param reuse:
    :return:
    """

    assert isinstance(layer_number, int)
    scope_name = str(layer_number)
    with tf.variable_scope(scope_name, reuse=reuse):
        v = tf.get_variable(var_name, dtype=tm.DEFAULT_TF_TYPE)
    return v


def compute_weights(layer, iteration=0, do_tile_weights=True):

    mean = get_layer_variable(layer, 'mu_w')
    rho = get_layer_variable(layer, 'rho_w')
    noise = get_noise(layer, iteration)

    return mean + tf.exp(rho) * noise


def compute_biases(layer, iteration):
    """Bias is Gaussian distributed"""

    mean = get_layer_variable(layer, 'mu_b')
    rho = get_layer_variable(layer, 'rho_b')
    noise = get_noise(layer, iteration, is_weight=False)

    # biases = mean + tf.nn.softplus(rho) * noise
    biases = mean + tf.exp(rho) * noise

    return biases


def get_noise(layer, iteration, is_weight=True):

    if is_weight:
        noise_type = 'weight_noise'
    else:
        noise_type = 'bias_noise'

    noise = get_layer_variable(layer, noise_type)
    noise = tm.roll_noise(noise, iteration)

    return noise

def average_multiple_passes(data, number_of_passes, topology):
    """  Multiple passes allow us to estimate the posterior distribution.
    :param data: Mini-batch to be fed into the network
    :param number_of_passes: How many random realisations of the weights should be sampled
    :return: Means and variances of the posterior. NB this is not the covariance - see network_covariance.py
    """

    collated_outputs = collate_multiple_passes(data, topology, number_of_passes=number_of_passes)
    mean, variance = tf.nn.moments(collated_outputs, axes=[0])

    if number_of_passes == 1:
        logging.warning("Using default variance")
        variance = FLAGS.DEFAULT_FORECAST_VARIANCE

    return mean, variance


def collate_multiple_passes(x, topology, number_of_passes=50):
    """Collate outputs from many realisations of weights from a bayesian network.
    :return 4D tensor with dimensions [n_passes, batch_size, n_label_timesteps, n_categories]
    """

    outputs = []
    for iter in range(number_of_passes):
        result = forward_pass(x, topology, iteration=iter)
        outputs.append(result)

    stacked_output = tf.stack(outputs, axis=0)

    # Make sure we softmax across the 'bin' dimension, but not across all series!
    stacked_output = tf.nn.softmax(stacked_output, dim=-1)

    return stacked_output


def forward_pass(signal, topology, iteration=0):
    """ Takes input data and returns predictions

    :param tensor signal: signal[i,j] holds input j from sample i, so data.shape = [batch_size, n_inputs]
    or if classification then   [batch_size, n_series, n_classes]
    :return: tensor: predictions of length N_OUTPUTS
    """

    for layer in range(topology.n_layers):

        weights = compute_weights(layer, iteration)
        biases = compute_biases(layer, iteration)

        signal = tf.tensordot(signal, weights, axes=2) + biases
        activation_function = topology.get_activation_function(layer, signal)
        signal = activation_function(signal)

    return signal



