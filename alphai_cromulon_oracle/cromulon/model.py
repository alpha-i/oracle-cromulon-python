"""
This modules contains two classes

1 Cromulon
  This class builds the network structure and uses a combination of convolutional layers and BayesLayers.

2 BayesLayers
    class which wraps all the methods needed to initialize the graph and retrieve its variables
        The graph in which the class operates is the default tf graph
"""

import logging
import numpy as np
import tensorflow as tf

import alphai_cromulon_oracle.tensormaths as tm

DEFAULT_N_TRANSITION_KERNELS = 2
LAYER_CONVOLUTIONAL = 'conv'
LAYER_POOL = 'pool'
LAYER_FULLY_CONNECTED = 'full'
LAYER_RESIDUAL = 'res'
DEFAULT_PADDING = 'same'  # TBC: add 'valid', will need to add support in topology.py
DATA_FORMAT = 'channels_first'  # Slightly faster for cuDNN to use CHW format
RANDOM_SEED = None  # Set to None for best performance, but lacks reproducibility

if RANDOM_SEED:  # How many passes over the bayes layers can be performed in parallel. Default is 10.
    N_PARALLEL_PASSES = 1  # Must go sequentially
else:
    N_PARALLEL_PASSES = 10


class Cromulon:
    def __init__(self, topology, flags, is_training):
        """

        :param Topology topology: Outlines the architecture of the network
        :param flags: TensorFlow flags
        :param bool is_training:
        """
        self._topology = topology
        self._flags = flags
        self._is_training = is_training
        self.bayes = BayesLayers(topology, flags)
        self.intialise_variables()

    def show_me_what_you_got(self, x):
        """ Predict the future outcome of the temporal signal x.

        :param x: A 3D tensor of dimensions [samples, time, features]
        :return: A 2D tensor representing the probabiity distribution of dimensions [samples, classification_bins]
        """

        # Process input in accordance with https://www.gwern.net/docs/rl/2017-silver.pdf
        x = self.convolutional_layer(x, layer_label='input', layer_number=0)

        if self._flags.do_batch_norm:
            batch_norm_label = 'input_batch_norm_'
            x = self.batch_normalisation(x, batch_norm_label, is_conv_layer=True)

        x = tf.nn.relu(x)

        # Bulk of the network consists of residual blocks
        for block_number in range(self._flags.n_res_blocks):
            x = self.residual_block(x, block_number)

        # Reduce dimensionality with 1x1 convolutional layer
        x = self.convolutional_layer(x, layer_label='reduction', layer_number=99, kernel_size=1, n_kernels=DEFAULT_N_TRANSITION_KERNELS)
        if self._flags.do_batch_norm:
            batch_norm_label = 'output_batch_norm_'
            x = self.batch_normalisation(x, batch_norm_label, is_conv_layer=True)

        x = tf.nn.relu(x)

        # Add final Bayesian layer(s)
        x = self.looped_passes(x)

        return x

    def residual_block(self, x, block_number):

        """ Residual Layer based on alphaZero architecture

        :param x: Tensor
        :param int block_number: Used to assign variable names
        :return:
        """

        residual_name = 'res_' + str(block_number)
        identity = tf.identity(x, name=residual_name)
        x = self.convolutional_layer(x, 'a', block_number)

        if self._flags.do_batch_norm:
            batch_norm_label = 'batch_norm_' + str(block_number) + 'a'
            x = self.batch_normalisation(x, batch_norm_label, is_conv_layer=True)

        x = tf.nn.relu(x)

        x = self.convolutional_layer(x, 'b', block_number)

        if self._flags.do_batch_norm:
            batch_norm_label = 'batch_norm_' + str(block_number) + 'b'
            x = self.batch_normalisation(x, batch_norm_label, is_conv_layer=True)

        x = x + identity

        return tf.nn.relu(x)

    def looped_passes(self, input_signal):
        """ Collate outputs from many realisations of weights from a bayesian network.
        Uses tf.while for improved memory efficiency

        :param tensor input_signal: Output from the conv layers
        :param int start_layer: the start of the bayesian layers
        :return: 4D tensor with dimensions [n_passes, batch_size, n_label_timesteps, n_categories]
        """

        start_index = tf.constant(0)
        n_passes = tf.cond(self._is_training, lambda: self._flags.n_train_passes,
                           lambda: self._flags.n_eval_passes)

        def condition(index, _):
            return tf.less(index, n_passes)

        def body(index, multipass):
            single_output = self.bayes_forward_pass(input_signal)
            multipass = tf.concat([multipass, [single_output]], axis=0)
            index += 1
            return index, multipass

        dummy_output = self.calculate_dummy_output(input_signal)
        loop_shape = [start_index.get_shape(), dummy_output.get_shape()]

        output_list = tf.while_loop(condition, body, [start_index, dummy_output],
                                    parallel_iterations=N_PARALLEL_PASSES, shape_invariants=loop_shape)[1]
        output_signal = tf.stack(output_list[1:], axis=0)  # Ignore dummy first entry
        output_signal = tf.nn.softmax(output_signal, dim=-1)  # Create discrete PDFs

        # Average probability over multiple passes
        output_signal = tf.reduce_mean(output_signal, axis=0)

        return tf.expand_dims(output_signal, axis=0)

    def bayes_forward_pass(self, signal):
        """  Propagate only through the fully connected layers.

        :param signal:
        :param iteration:
        :return:
        """

        for bayes_layer in range(self._topology.n_bayes_layers):
            layer_number = bayes_layer + 2
            signal = self.fully_connected_layer(signal, layer_number)

        return signal

    def fully_connected_layer(self, x, layer_number):
        """ Propoagates signal through a fully connected set of weights

        :param x: 3D tensor representing a temporal signal. Dimensions [samples, timesteps, features]
        :param layer_number:
        :param iteration:
        :param input_signal: Used by residual layers
        :return:
        """

        weights = self.bayes.compute_weights(layer_number)
        biases = self.bayes.compute_biases(layer_number)
        x = tf.tensordot(x, weights, axes=3) + biases

        activation_function = self._topology.get_activation_function(layer_number)

        return activation_function(x)

    def batch_normalisation(self, signal, norm_name, is_conv_layer):
        """ Normalises the signal to unit variance and zero mean.

        :param signal:
        :param str norm_name: Name for reference
        :return:
        """

        axis = 1 if is_conv_layer else -1

        try:
            signal = tf.layers.batch_normalization(signal, training=self._is_training,
                                                   reuse=True, name=norm_name, axis=axis)
        except:
            signal = tf.layers.batch_normalization(signal, training=self._is_training,
                                                   reuse=False, name=norm_name, axis=axis)

        return signal

    def convolutional_layer(self, signal, layer_label, layer_number, kernel_size=None, n_kernels=None):
        """  Sets a convolutional layer with a two-dimensional kernel.
        The ordering of the dimensions in the inputs: DATA_FORMAT = `channels_last` corresponds to inputs with shape
        `(batch, depth, height, width, channels)` while DATA_FORMAT = `channels_first`
        corresponds to inputs with shape `(batch, channels, depth, height, width)`.

        :param signal: A rank 4 tensor of dimensions [batch, channels, time, features]
        :param str layer_label: Label to identify the layer
        :param int layer_number: Which block it belongs to
                :param int layer_number: Which block it belongs to

        :param int layer_number: Which block it belongs to

        :return:  A rank 4 tensor of dimensions [batch, channels, time, features]
        """

        op_name = LAYER_CONVOLUTIONAL + layer_label + str(layer_number)
        if kernel_size is None:
            kernel_size = self._topology.kernel_size

        if n_kernels is None:
            n_kernels = self._topology.n_kernels

        if self._flags.do_kernel_regularisation:
            regulariser = tf.contrib.layers.l2_regularizer(scale=0.1)
        else:
            regulariser = None

        try:
            signal = tf.layers.conv2d(
                inputs=signal,
                filters=n_kernels,
                kernel_size=kernel_size,
                padding=DEFAULT_PADDING,
                activation=None,
                data_format=DATA_FORMAT,
                dilation_rate=self._topology.dilation_rates,
                strides=self._topology.strides,
                name=op_name,
                kernel_regularizer=regulariser,
                use_bias=False,
                reuse=True)
        except:
            signal = tf.layers.conv2d(
                inputs=signal,
                filters=n_kernels,
                kernel_size=kernel_size,
                padding=DEFAULT_PADDING,
                activation=None,
                data_format=DATA_FORMAT,
                dilation_rate=self._topology.dilation_rates,
                strides=self._topology.strides,
                name=op_name,
                kernel_regularizer=regulariser,
                use_bias=False,
                reuse=False)

        return signal

    def pool_layer(self, signal):
        """ Pools evenly across dimensions

        :param signal:
        :return:
        """

        return tf.layers.max_pooling2d(inputs=signal, pool_size=[2, 2], strides=2, data_format=DATA_FORMAT)

    def calculate_dummy_output(self, input_signal):
        """ Need a tensor which mimics the shape of the output of the network

        :param input_signal:
        :return:
        """

        partial_new_shape = tf.shape(input_signal)[0:1]
        one = tf.expand_dims(tf.constant(int(1)), axis=0)
        nbins = tf.expand_dims(tf.constant(self._topology.n_classification_bins), axis=0)

        # Will need to update this if using multiple forecasts
        dummy_shape = tf.concat([one, partial_new_shape, one, one, nbins], 0)

        return tf.zeros(dummy_shape)

    def calculate_kernel_size(self):
        """ Computes the desired kernel size based on the shape of the signal

        :param signal:
        :return:
        """

        target_kernel_size = self._topology.kernel_size
        input_layer = self._topology.layers[0]

        k_height = min(target_kernel_size[0], input_layer['height'])
        k_width = min(target_kernel_size[1], input_layer['width'])

        return [k_height, k_width]

    def intialise_variables(self):
        """ Sets variables to initial conditions, if not already initialised

        :return:
        """

        try:
            self.bayes.initialise_bayesian_parameters()
        except:
            logging.debug('Variables already initialised')


class BayesLayers:
    VAR_WEIGHT_RHO = 'rho_w'
    VAR_WEIGHT_MU = 'mu_w'
    VAR_WEIGHT_NOISE = 'weight_noise'

    VAR_BIAS_RHO = 'rho_b'
    VAR_BIAS_MU = 'mu_b'
    VAR_BIAS_NOISE = 'bias_noise'

    VAR_LOG_ALPHA = 'log_alpha'

    def __init__(self, topology, flags):
        """

        :param Topology topology:
        :param flags flags:
        :param tf.bool is_training: Whether the model will be training or evaluating
        """

        self._topology = topology
        self._graph = tf.get_default_graph()
        self._flags = flags

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

    def initialise_bayesian_parameters(self):

        weight_uncertainty = self._flags.INITIAL_WEIGHT_UNCERTAINTY
        bias_uncertainty = self._flags.INITIAL_BIAS_UNCERTAINTY
        weight_displacement = self._flags.INITIAL_WEIGHT_DISPLACEMENT
        bias_displacement = self._flags.INITIAL_BIAS_DISPLACEMENT

        initial_rho_weights = tf.contrib.distributions.softplus_inverse(weight_uncertainty)
        initial_rho_bias = tf.contrib.distributions.softplus_inverse(bias_uncertainty)
        initial_alpha = self._flags.INITIAL_ALPHA

        for layer_number in range(self._topology.n_layers):
            layer_type = self._topology.layers[layer_number]["type"]
            if layer_type == LAYER_FULLY_CONNECTED:  # No point building weights for conv or pool layers
                w_shape = self._topology.get_weight_shape(layer_number)
                b_shape = self._topology.get_bias_shape(layer_number)

                self._create_variable_for_layer(
                    layer_number,
                    self.VAR_WEIGHT_MU,
                    tm.centred_gaussian(w_shape, weight_displacement, seed=1)
                )

                self._create_variable_for_layer(
                    layer_number,
                    self.VAR_WEIGHT_RHO,
                    initial_rho_weights + tf.zeros(w_shape, tm.DEFAULT_TF_TYPE)
                )

                self._create_variable_for_layer(
                    layer_number,
                    self.VAR_BIAS_MU,
                    tm.centred_gaussian(b_shape, bias_displacement, seed=1)
                )

                self._create_variable_for_layer(
                    layer_number,
                    self.VAR_BIAS_RHO,
                    initial_rho_bias + tf.zeros(b_shape, tm.DEFAULT_TF_TYPE)
                )

                self._create_variable_for_layer(
                    layer_number,
                    self.VAR_LOG_ALPHA,
                    np.log(initial_alpha).astype(self._flags.d_type),
                    False
                )  # Hyperprior on the distribution of the weights

    def _create_variable_for_layer(self, layer_number, variable_name, initializer, is_trainable=True):

        assert isinstance(layer_number, int)
        scope_name = str(layer_number)
        with tf.variable_scope(scope_name):
            tf.get_variable(variable_name, initializer=initializer, trainable=is_trainable, dtype=tm.DEFAULT_TF_TYPE)

    def get_variable(self, layer_number, variable_name, reuse=True):

        scope_name = str(layer_number)
        with tf.variable_scope(scope_name, reuse=reuse):
            v = tf.get_variable(variable_name, dtype=tm.DEFAULT_TF_TYPE)

        return v

    def get_weight_noise(self, layer_number):
        return self._get_layer_noise(layer_number, self.VAR_WEIGHT_NOISE)

    def get_bias_noise(self, layer_number):
        return self._get_layer_noise(layer_number, self.VAR_BIAS_NOISE)

    def _get_layer_noise(self, layer_number, var_name):

        if var_name == self.VAR_WEIGHT_NOISE:
            noise_shape = self._topology.get_weight_shape(layer_number)
        else:
            noise_shape = self._topology.get_bias_shape(layer_number)

        if RANDOM_SEED:
            seed = layer_number * 1000 + RANDOM_SEED
        else:
            seed = None
        noise = tf.random_normal(shape=noise_shape, seed=seed)
        # noise = tf.Print(noise, [noise[0, 0, :]], message="Bias noise: ")
        # tf.contrib.stateless.stateless_random_normal will allow seed set by a tensor
        return noise

    def compute_weights(self, layer_number):

        mean = self.get_variable(layer_number, self.VAR_WEIGHT_MU)
        rho = self.get_variable(layer_number, self.VAR_WEIGHT_RHO)
        noise = self.get_weight_noise(layer_number)

        return mean + tf.nn.softplus(rho) * noise

    def compute_biases(self, layer_number):
        """Bias is Gaussian distributed"""
        mean = self.get_variable(layer_number, self.VAR_BIAS_MU)
        rho = self.get_variable(layer_number, self.VAR_BIAS_RHO)
        noise = self.get_bias_noise(layer_number)
        # noise = tf.Print(noise, [layer_number], message="layer_number: ") # Useful debugging statement

        return mean + tf.nn.softplus(rho) * noise
