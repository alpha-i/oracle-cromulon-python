# Defines the layout of the cromulon network:
# Holds a list of layers: [0] Conv layer [1] Residual blocks [2.....N] Bayesian Layers

import logging

import tensorflow as tf

import alphai_cromulon_oracle.tensormaths as tm
from alphai_cromulon_oracle.cromulon.model import LAYER_CONVOLUTIONAL, LAYER_POOL, LAYER_FULLY_CONNECTED, LAYER_RESIDUAL, DEFAULT_N_TRANSITION_KERNELS

logging.getLogger(__name__).addHandler(logging.NullHandler())

ACTIVATION_FN_LINEAR = "linear"
ACTIVATION_FN_SELU = "selu"
ACTIVATION_FN_RELU = "relu"

ALLOWED_ACTIVATION_FN = [ACTIVATION_FN_RELU, ACTIVATION_FN_SELU, ACTIVATION_FN_LINEAR]
ALLOWED_LAYER_TYPES = [LAYER_CONVOLUTIONAL, LAYER_POOL, LAYER_FULLY_CONNECTED, LAYER_RESIDUAL]

DEFAULT_N_KERNELS = 64
DEFAULT_TIMESTEPS = 28
DEFAULT_N_FEATURES = 28
DEFAULT_BINS = 10
DEFAULT_N_FORECASTS = 1
DEFAULT_HIDDEN_LAYERS = 2
DEFAULT_HEIGHT = 400
DEFAULT_WIDTH = 1
DEFAULT_DEPTH = 1
DEFAULT_KERNEL_SIZE = [3, 3]
DEFAULT_N_OUTPUT_SERIES = 1
DEFAULT_ACT_FUNCTION = ACTIVATION_FN_RELU
DEFAULT_LAYER_TYPE = LAYER_FULLY_CONNECTED

DEFAULT_LAYER_WIDTHS = [DEFAULT_TIMESTEPS] + [DEFAULT_WIDTH] * DEFAULT_HIDDEN_LAYERS + [DEFAULT_BINS]
DEFAULT_LAYER_HEIGHTS = [DEFAULT_N_FEATURES] + [DEFAULT_HEIGHT] * DEFAULT_HIDDEN_LAYERS + [DEFAULT_N_FORECASTS]
DEFAULT_LAYER_DEPTHS = [DEFAULT_N_FEATURES] + [DEFAULT_DEPTH] * DEFAULT_HIDDEN_LAYERS + [DEFAULT_N_OUTPUT_SERIES]
DEFAULT_LAYER_ACTIVATION_FUNCTIONS = [ACTIVATION_FN_LINEAR] + [DEFAULT_ACT_FUNCTION] * DEFAULT_HIDDEN_LAYERS + [ACTIVATION_FN_LINEAR]


class Topology(object):
    """
    A class for containing the information that defines the topology of the neural network.
    Run checks on the user input to verify that it defines a valid topology.
    """

    def __init__(self,
                 n_timesteps=DEFAULT_TIMESTEPS,
                 n_features=DEFAULT_N_FEATURES,
                 n_forecasts=DEFAULT_N_FORECASTS,
                 n_classification_bins=DEFAULT_BINS,
                 layer_heights=None,
                 layer_widths=None,
                 layer_depths=None,
                 activation_functions=None,
                 layer_types=None,
                 conv_config=None
        ):
        """
        Following info is required to construct a topology object

        :param n_timesteps: Length of timesteps dimension; defines height of input layer
        :param n_features:
        :param n_forecasts: Number of forecasts; defines width of output layer
        :param n_classification_bins:
        :param layer_heights:
        :param layer_widths:
        :param layer_depths:
        :param activation_functions:
        :param layer_types:
        :param conv_config:
        """

        if layer_heights is None:
            assert layer_widths is None and activation_functions is None and layer_depths is None
            layer_depths, layer_heights, layer_widths, activation_functions = self.get_default_layers()
        elif layer_depths is None:
            layer_depths = [DEFAULT_DEPTH] * len(layer_heights)
        else:
            assert len(layer_depths) == len(layer_heights), "Length of depths array does not match height array"
        assert len(layer_widths) == len(layer_heights), "Length of widths array does not match height array"
        assert len(activation_functions) == len(layer_heights), "Length of act fns does not match height array"

        # Setup convolution params if specified
        if conv_config:
            logging.debug("Convolution config", conv_config)
            self.kernel_size = conv_config['kernel_size']
            self.n_kernels = conv_config["n_kernels"]  # kernels used in first conv layer.
            self.dilation_rates = conv_config["dilation_rates"]
            self.strides = conv_config["strides"]
        else:
            logging.debug("******** No convolution config found ********")
            self.kernel_size = DEFAULT_KERNEL_SIZE
            self.n_kernels = DEFAULT_N_KERNELS
            self.dilation_rates = 1
            self.strides = 1

        # How many kernels to use when transitioning from conv to full
        self.n_transition_kernels = DEFAULT_N_TRANSITION_KERNELS

        # First two Cromulon layers, and final layers, must be consistent with data

        # first layer
        layer_depths[0] = 1
        layer_heights[0] = n_timesteps
        layer_widths[0] = n_features

        # second layer
        layer_depths[1] = self.n_kernels
        layer_heights[1] = n_timesteps
        layer_widths[1] = n_features

        # last layer
        layer_depths[-1] = 1
        layer_heights[-1] = n_forecasts
        layer_widths[-1] = n_classification_bins

        self.layers = self._build_layers(layer_depths, layer_heights, layer_widths, activation_functions, layer_types)

        self._verify_layers()
        self.n_layers = len(self.layers) - 1  # n layers of neurons are connected by n-1 sets of weights
        self.n_bayes_layers = self.n_layers - 2

        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.n_forecasts = n_forecasts
        self.n_classification_bins = n_classification_bins
        self.n_parameters = self._calculate_number_of_parameters()

    def _verify_layers(self):
        """
        A function that checks each layer to ensure that it is valid i.e., expected activation function, trainable
        flag, etc.
        :param layers:
        :return: None
        """
        for i, layer in enumerate(self.layers):

            if layer["activation_func"] not in ALLOWED_ACTIVATION_FN:
                raise ValueError('Unexpected activation function ' + str(layer["activation_func"]))

            if layer["type"] not in ALLOWED_LAYER_TYPES:
                raise ValueError('Unexpected layer type ' + str(layer["type"]))

            for key in ['height', 'width']:
                x = layer[key]
                if not (isinstance(x, int) and x > 0):
                    raise ValueError(
                        'Layer {} {} should be a positive integer'.format(i, key)
                    )

            if not isinstance(layer["trainable"], bool):
                raise ValueError('Layer {} trainable should be a boolean'.format(i))

    def _calculate_number_of_parameters(self):
        """ Returns total number of connections, assuming layers are fully connected"""

        number_of_parameters = 0

        def _calculate_layer_n_parameters(layer_number):
            the_layer = self.layers[layer_number]
            return the_layer["width"] * the_layer["height"]

        for i in range(self.n_layers):
            number_of_parameters += _calculate_layer_n_parameters(i) * _calculate_layer_n_parameters(i + 1)

        return number_of_parameters

    def get_cell_shape(self, layer_number):
        """
        returns the shape of the cells in a layer specified by the layer number
        :param layer_number: int
        :return: [int, int]
        """

        cell_height = self.layers[layer_number]["cell_height"]
        cell_width = self.layers[layer_number]["width"]

        return [cell_height, cell_width]

    def get_weight_shape(self, layer_number):

        if not layer_number < self.n_layers:
            raise ValueError('layer_number should be strictly less than the number of layers')

        input_depth = self.layers[layer_number]["depth"]
        input_height = self.layers[layer_number]["height"]
        input_width = self.layers[layer_number]["width"]

        output_depth = self.layers[layer_number + 1]["depth"]
        output_height = self.layers[layer_number + 1]["height"]
        output_width = self.layers[layer_number + 1]["width"]

        weight_shape = [input_depth, input_height, input_width, output_depth, output_height, output_width]

        return weight_shape

    def get_bias_shape(self, layer_number):
        """
        returns the shape of the biases in a layer specified by layer number as an array
        :param layer_number:
        :return:
        """
        if not layer_number < self.n_layers:
            raise ValueError('layer_number should be strictly less the number of layers')

        depth = self.layers[layer_number + 1]["depth"]
        height = self.layers[layer_number + 1]["height"]
        width = self.layers[layer_number + 1]["width"]

        bias_shape = [depth, height, width]

        return bias_shape

    def get_layer_type(self, layer_number):

        return self.layers[layer_number]["type"]

    def get_activation_function(self, layer_number):

        function_name = self.layers[layer_number + 1]["activation_func"]

        if function_name == 'linear':
            return lambda x: x
        elif function_name == 'selu':
            return tm.selu
        elif function_name == 'relu':
            return tf.nn.relu
        elif function_name == 'kelu':
            return tm.kelu
        else:
            raise NotImplementedError

    def _build_layers(self, layer_depths, layer_heights, layer_widths, activation_functions, layer_types=None):
        """
        :param activation_functions:
        :param n_series:
        :param n_features_per_series:
        :param n_forecasts:
        :param n_classification_bins:
        :param layer_heights:
        :param layer_widths:
        :return:
        """

        layers = []
        n_layers = len(activation_functions)
        current_n_kernels = self.n_kernels

        for i in range(n_layers):
            layer = dict()
            layer["activation_func"] = activation_functions[i]
            layer["trainable"] = True  # Just hardcode for now, will be configurable in future
            layer["cell_height"] = 1  # Just hardcode for now, will be configurable in future
            layer["depth"] = layer_depths[i]
            layer["height"] = layer_heights[i]
            layer["width"] = layer_widths[i]

            if layer_types is None:
                if i > 1:
                    layer["type"] = LAYER_FULLY_CONNECTED
                else:
                    layer["type"] = LAYER_CONVOLUTIONAL
            else:
                assert layer_types[0] == 'conv', "Cromulon intro layer should be conv"
                assert layer_types[1] == 'res', "Cromulon first layer should be res"
                layer["type"] = layer_types[i]

            if i > 0:  # Enforce consistent dimensions for subsequent layers
                prev_layer = layers[i - 1]
                previous_layer_type = prev_layer["type"]

                if previous_layer_type == LAYER_POOL:  # Pooling will rescale size of last layer
                    layer["height"] = max(1, int(prev_layer['height'] / 2))
                    layer["width"] = max(1, int(prev_layer['width'] / 2))
                    current_n_kernels *= 2
                elif previous_layer_type == LAYER_CONVOLUTIONAL:
                    # This will depend on choice of padding. Default for now is same, so easier.
                    layer["depth"] = self.n_kernels
                    layer["height"] = int(prev_layer["height"])
                    layer["width"] = int(prev_layer["width"])
                elif previous_layer_type == LAYER_RESIDUAL or layer["type"] == LAYER_RESIDUAL:
                    input_layer = layers[0]
                    layer["height"] = int(input_layer["height"])
                    layer["width"] = int(input_layer["width"])
                if previous_layer_type in {LAYER_CONVOLUTIONAL, LAYER_RESIDUAL} and \
                                layer["type"] == LAYER_FULLY_CONNECTED:
                    layer["depth"] = self.n_transition_kernels

            layer["n_kernels"] = current_n_kernels

            layers.append(layer)

        return layers

    @staticmethod
    def get_default_layers():
        """ Compiles the list of layer heights, widths and activation funcs to be used if none are provided

        :return:
        """
        return DEFAULT_LAYER_DEPTHS, DEFAULT_LAYER_HEIGHTS, DEFAULT_LAYER_WIDTHS, DEFAULT_LAYER_ACTIVATION_FUNCTIONS

    def get_network_input_shape(self):
        """ Returns the required shape of input data.

        :return: 2D Numpy array
        """

        input_layer = self.layers[0]
        input_shape = (input_layer["depth"], input_layer["height"], input_layer["width"])

        return input_shape
