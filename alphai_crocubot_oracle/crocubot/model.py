"""
This modules contains two classes

1 CrocuBotModel
    class which wraps all the methods needed to initialize the graph and retrieve its variables
        The graph in which the class operates is the default tf graph
2 Estimator
  This class is an service class which permits to visit the graph of a given estimator and retrieve different passes
   algrithm


"""

import numpy as np
import tensorflow as tf

import alphai_crocubot_oracle.tensormaths as tm

CONVOLUTIONAL_LAYER_1D = 'conv1d'
CONVOLUTIONAL_LAYER_3D = 'conv3d'
FULLY_CONNECTED_LAYER = 'full'
RESIDUAL_LAYER = 'res'
POOL_LAYER_2D = 'pool2d'
POOL_LAYER_3D = 'pool3d'
DEFAULT_PADDING = 'same'  # TBC: add 'valid', will need to add support in topology.py
DATA_FORMAT = 'channels_last'
TIME_DIMENSION = 3  # Tensor dimensions defined as: [batch, series, time, features, filters]
FORECAST_OVERLAP = 26  # How many timesteps in features correspond to forecast interval. Assumes 15min interval and forecast of 1 trading day


class CrocuBotModel:
    VAR_WEIGHT_RHO = 'rho_w'
    VAR_WEIGHT_MU = 'mu_w'
    VAR_WEIGHT_NOISE = 'weight_noise'

    VAR_BIAS_RHO = 'rho_b'
    VAR_BIAS_MU = 'mu_b'
    VAR_BIAS_NOISE = 'bias_noise'

    VAR_LOG_ALPHA = 'log_alpha'
    VAR_PENALTY = 'penalty_vector'

    def __init__(self, topology, flags, is_training):
        """

        :param Topology topology:
        :param flags flags:
        :param tf.bool is_training: Whether the model will be training or evaluating
        """

        self._topology = topology
        self._graph = tf.get_default_graph()
        self._flags = flags
        self._is_training = is_training

    @property
    def graph(self):
        return self._graph

    @property
    def topology(self):
        return self._topology

    @property
    def number_of_layers(self):
        return self._topology.n_layers

    @property
    def layer_variables_list(self):
        return [
            self.VAR_BIAS_RHO,
            self.VAR_BIAS_MU,
            self.VAR_BIAS_NOISE,
            self.VAR_WEIGHT_RHO,
            self.VAR_WEIGHT_MU,
            self.VAR_WEIGHT_NOISE,
            self.VAR_LOG_ALPHA
        ]

    def build_layers_variables(self):

        weight_uncertainty = self._flags.INITIAL_WEIGHT_UNCERTAINTY
        bias_uncertainty = self._flags.INITIAL_BIAS_UNCERTAINTY
        weight_displacement = self._flags.INITIAL_WEIGHT_DISPLACEMENT
        bias_displacement = self._flags.INITIAL_BIAS_DISPLACEMENT

        initial_rho_weights = tf.contrib.distributions.softplus_inverse(weight_uncertainty)
        initial_rho_bias = tf.contrib.distributions.softplus_inverse(bias_uncertainty)
        initial_alpha = self._flags.INITIAL_ALPHA

        for layer_number in range(self._topology.n_layers):
            layer_type = self._topology.layers[layer_number]["type"]
            if layer_type == 'full':  # No point building weights for conv or pool layers
                w_shape = self._topology.get_weight_shape(layer_number)
                b_shape = self._topology.get_bias_shape(layer_number)
                penalty_tensor = self.calculate_penalty_vector(FORECAST_OVERLAP, w_shape[1])

                self._create_variable_for_layer(
                    layer_number,
                    self.VAR_WEIGHT_MU,
                    tm.centred_gaussian(w_shape, weight_displacement)
                )

                self._create_variable_for_layer(
                    layer_number,
                    self.VAR_WEIGHT_RHO,
                    initial_rho_weights + tf.zeros(w_shape, tm.DEFAULT_TF_TYPE)
                )

                self._create_variable_for_layer(
                    layer_number,
                    self.VAR_BIAS_MU,
                    tm.centred_gaussian(b_shape, bias_displacement)
                )

                self._create_variable_for_layer(
                    layer_number,
                    self.VAR_BIAS_RHO,
                    initial_rho_bias + tf.zeros(b_shape, tm.DEFAULT_TF_TYPE)
                )

                self._create_variable_for_layer(
                    layer_number,
                    self.VAR_PENALTY,
                    penalty_tensor,
                    False
                )

                self._create_variable_for_layer(
                    layer_number,
                    self.VAR_LOG_ALPHA,
                    np.log(initial_alpha).astype(self._flags.d_type),
                    False
                )  # Hyperprior on the distribution of the weights

                self._create_noise(layer_number, self.VAR_WEIGHT_NOISE, w_shape)
                self._create_noise(layer_number, self.VAR_BIAS_NOISE, b_shape)

    def calculate_penalty_vector(self, overlap,  total_length):
        """ Penalise nodes in the distant past using Delta t / t"""

        penalty_vector = [1.0] * total_length
        for i in range(total_length):
            if i > overlap:
                index = total_length - i - 1  # Since t=0 appears at end of vector
                penalty_vector[index] = overlap / i

        penalty_tensor = tf.zeros(total_length, tm.DEFAULT_TF_TYPE) + penalty_vector
        penalty_shape = [1, 1, total_length, 1, 1]

        return tf.reshape(penalty_tensor, shape=penalty_shape)

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
        # noise = tf.cond(self._is_training, lambda: tf.random_shuffle(noise, seed=iteration),
        #                 lambda: tf.random_shuffle(noise))
        try:
            noise = tf.random_shuffle(noise, seed=iteration)
        except:
            noise = tf.random_shuffle(noise)

        return noise

    def get_bias_noise(self, layer_number, iteration):
        noise = self.get_variable(layer_number, self.VAR_BIAS_NOISE)
        return tf.random_shuffle(noise, seed=iteration)

    def compute_weights(self, layer_number, iteration=0):

        mean = self.get_variable(layer_number, self.VAR_WEIGHT_MU)
        rho = self.get_variable(layer_number, self.VAR_WEIGHT_RHO)
        noise = self.get_weight_noise(layer_number, iteration)

        return mean + tf.nn.softplus(rho) * noise

    def compute_biases(self, layer_number, iteration):
        """Bias is Gaussian distributed"""
        mean = self.get_variable(layer_number, self.VAR_BIAS_MU)
        rho = self.get_variable(layer_number, self.VAR_BIAS_RHO)
        noise = self.get_bias_noise(layer_number, iteration)

        return mean + tf.nn.softplus(rho) * noise


class Estimator:
    def __init__(self, crocubot_model, flags):
        """

        :param CrocuBotModel crocubot_model:
        :param flags:
        :return:
        """
        self._model = crocubot_model
        self._flags = flags

    def average_multiple_passes(self, data, number_of_passes):
        """
        Multiple passes allow us to estimate the posterior distribution.

        :param data: Mini-batch to be fed into the network
        :param number_of_passes: How many random realisations of the weights should be sampled
        :return: Estimate of the posterior distribution.
        """

        collated_outputs = self.collate_multiple_passes(data, number_of_passes)

        return tf.reduce_logsumexp(collated_outputs, axis=[0])

    def collate_multiple_passes(self, x, number_of_passes):
        """
        Collate outputs from many realisations of weights from a bayesian network.

        :param tensor x:
        :param int number_of_passes:
        :return 4D tensor with dimensions [n_passes, batch_size, n_label_timesteps, n_categories]:
        """

        outputs = []
        for iteration in range(number_of_passes):
            result = self.forward_pass(x, iteration)
            outputs.append(result)

        stacked_output = tf.stack(outputs, axis=0)
        # Make sure we softmax across the 'bin' dimension, but not across all series!
        stacked_output = tf.nn.log_softmax(stacked_output, dim=-1)

        return stacked_output

    def efficient_multiple_passes(self, input_signal, number_of_passes=50):
        """
        Collate outputs from many realisations of weights from a bayesian network.

        :param tensor x:
        :param int number_of_passes:
        :return 4D tensor with dimensions [n_passes, batch_size, n_label_timesteps, n_categories]:
        """

        output_signal = tf.nn.log_softmax(self.forward_pass(input_signal, int(0)), dim=-1)
        index_summation = (int(0), output_signal)

        def condition(index, _):
            return tf.less(index, number_of_passes)

        def body(index, summation):
            raw_output = self.forward_pass(input_signal, iteration=0)
            log_p = tf.nn.log_softmax(raw_output, dim=-1)

            stacked_p = tf.stack([log_p, summation], axis=0)
            log_p_total = tf.reduce_logsumexp(stacked_p, axis=0)

            return tf.add(index, 1), log_p_total

        # We do not care about the index value here, return only the signal
        output = tf.while_loop(condition, body, index_summation, parallel_iterations=1)[1]
        return tf.expand_dims(output, axis=0)

    def forward_pass(self, signal, iteration=0):
        """
        Takes input data and returns predictions

        :param tensor signal: signal[i,j] holds input j from sample i, so data.shape = [batch_size, n_inputs]
        or if classification then   [batch_size, n_series, n_classes]
        :param int iteration: number of iteration
        :return:
        """

        if self._model.topology.get_layer_type(0) in {'conv2d', 'conv3d'}:
            signal = tf.expand_dims(signal, axis=-1)

        input_signal = tf.identity(signal, name='input')

        for layer_number in range(self._model.topology.n_layers):
            signal = self.single_layer_pass(signal, layer_number, iteration, input_signal)

        return signal

    def single_layer_pass(self, signal, layer_number, iteration, input_signal):
        """

        :param signal:
        :param layer_number:
        :param iteration:
        :param input_signal: Used by residual layers
        :return:
        """

        layer_type = self._model.topology.get_layer_type(layer_number)
        activation_function = self._model.topology.get_activation_function(layer_number)

        if self._model.topology.layers[layer_number]['reshape']:
            if self._flags.apply_temporal_suppression:  # Prepare transition from conv/pool layers to fully connected
                penalty = self._model.get_variable(layer_number, self._model.VAR_PENALTY, reuse=True)
                signal = tf.multiply(signal, penalty)
                signal = self.batch_normalisation(signal, 'penalty')

            signal = self.flatten_last_dimension(signal)

        if layer_type == CONVOLUTIONAL_LAYER_1D:
            signal = self.convolutional_layer_1d(signal)
        elif layer_type == CONVOLUTIONAL_LAYER_3D:
            signal = self.convolutional_layer_3d(signal, layer_number)
            signal = self.batch_normalisation(signal, layer_number)
        elif layer_type == FULLY_CONNECTED_LAYER:
            signal = self.fully_connected_layer(signal, layer_number, iteration)
        elif layer_type == POOL_LAYER_2D:
            signal = self.pool_layer_2d(signal)
        elif layer_type == POOL_LAYER_3D:
            signal = self.pool_layer_3d(signal)
        elif layer_type == RESIDUAL_LAYER:
            signal = self.residual_layer(signal, input_signal)
        else:
            raise ValueError('Unknown layer type')

        return activation_function(signal)

    def fully_connected_layer(self, signal, layer_number, iteration):
        """ Propoagates signal through a fully connected set of weights

        :param signal:
        :param layer_number:
        :param iteration:
        :return:
        """

        weights = self._model.compute_weights(layer_number, iteration)
        biases = self._model.compute_biases(layer_number, iteration)

        return tf.tensordot(signal, weights, axes=3) + biases

    def residual_layer(self, signal, input_signal):
        """ TBC: add learnable weight

        :param signal:
        :param input_signal:
        :return:
        """

        return signal + input_signal

    def convolutional_layer_1d(self, signal, iteration):
        """ Sets a convolutional layer with a one-dimensional kernel. """

        reuse_kernel = (iteration > 0)  # later passes reuse kernel

        signal = tf.layers.conv1d(
            inputs=signal,
            filters=6,
            kernel_size=(5,),
            padding=DEFAULT_PADDING,
            data_format=DATA_FORMAT,
            activation=None,
            reuse=reuse_kernel)

        return signal

    def batch_normalisation(self, signal, layer_number):
        """ Normalises the signal to unit variance and zero mean.

        :param signal:
        :return:
        """

#         signal = tf.cond(self._model._is_training,
#                         lambda: tf.contrib.layers.batch_norm(signal, is_training=True),
#                        lambda: tf.contrib.layers.batch_norm(signal, is_training=False))
        NORM_NAME = "batch_norm_" + str(layer_number)
        try:
            signal = tf.layers.batch_normalization(signal, training=self._model._is_training, reuse=True, name=NORM_NAME)
        except:
            signal = tf.layers.batch_normalization(signal, training=self._model._is_training, reuse=False, name=NORM_NAME)

        return signal

    def convolutional_layer_3d(self, signal, layer_number):
        """  Sets a convolutional layer with a three-dimensional kernel.
        The ordering of the dimensions in the inputs: DATA_FORMAT = `channels_last` corresponds to inputs with shape
        `(batch, depth, height, width, channels)` while DATA_FORMAT = `channels_first`
        corresponds to inputs with shape `(batch, channels, depth, height, width)`.

        :param signal: A rank 5 tensor of dimensions [batch, series, time, features, filters]
        :return:  A rank 5 tensor of dimensions [batch, series, time, features, filters]
        """

        n_kernels = self._model._topology.n_kernels
        dilation_rate = self._model._topology.dilation_rates
        strides = self._model._topology.strides

        kernel_size = self.calculate_3d_kernel_size()
        op_name = CONVOLUTIONAL_LAYER_3D + str(layer_number)

        try:
            signal = tf.layers.conv3d(
                inputs=signal,
                filters=n_kernels,
                kernel_size=kernel_size,
                padding=DEFAULT_PADDING,
                activation=None,
                data_format=DATA_FORMAT,
                dilation_rate=dilation_rate,
                strides=strides,
                name=op_name,
                reuse=True)
        except:
            signal = tf.layers.conv3d(
                inputs=signal,
                filters=n_kernels,
                kernel_size=kernel_size,
                padding=DEFAULT_PADDING,
                activation=None,
                data_format=DATA_FORMAT,
                dilation_rate=dilation_rate,
                strides=strides,
                name=op_name,
                reuse=False)

        return signal

    def pool_layer_2d(self, signal):
        """ Pools evenly across dimensions

        :param signal:
        :return:
        """

        return tf.layers.max_pooling3d(inputs=signal, pool_size=[2, 1, 2], strides=2, data_format='channels_first',)

    def pool_layer_3d(self, signal):
        """ Usually follows conv_3d layer to reduce the dimensionality. Currently only targets the timestep dimension

        :param signal:
        :return:
        """

        return tf.layers.max_pooling3d(inputs=signal, pool_size=[1, 4, 1], strides=[1, 4, 1], data_format=DATA_FORMAT)

    def flatten_last_dimension(self, signal):
        """ Takes a tensor and squishes its last dimension into the penultimate dimension.

        :param signal: Tensor of N dimensions, with N >= 2
        :return:  Tensor of N-1 dimensions
        """

        partial_new_shape = tf.shape(signal)[0:3]
        end_value = tf.multiply(tf.shape(signal)[-1], tf.shape(signal)[-2])
        end_tiled = tf.expand_dims(end_value, axis=0)
        new_shape = tf.concat([partial_new_shape, end_tiled], 0)

        return tf.reshape(signal, new_shape)

    def calculate_3d_kernel_size(self):
        """ Computes the desired kernel size based on the shape of the signal

        :param signal:
        :return:
        """

        target_kernel_size = self._model._topology.kernel_size
        input_layer = self._model._topology.layers[0]

        k_depth = min(target_kernel_size[0], input_layer['depth'])
        k_height = min(target_kernel_size[1], input_layer['height'])
        k_width = min(target_kernel_size[2], input_layer['width'])

        return [k_depth, k_height, k_width]
