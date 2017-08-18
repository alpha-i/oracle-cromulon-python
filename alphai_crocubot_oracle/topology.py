ACTIVATION_FN_LINEAR = "linear"
ACTIVATION_FN_SELU = "selu"
ACTIVATION_FN_RELU = "relu"

ALLOWED_ACTIVATION_FN = [ACTIVATION_FN_RELU, ACTIVATION_FN_SELU, ACTIVATION_FN_LINEAR]


class Topology(object):
    """
    A class for containing the information that defines the topology of the neural network.
    Run checks on the user input to verify that it defines a valid topology.
    """

    def __init__(self, layers):
        self.activation_funcs = ALLOWED_ACTIVATION_FN
        self._verify_layers(layers)
        self.layers = layers
        self.n_layers = len(layers)
        self.n_inputs = layers[0]["height"] * layers[0]["width"]
        self.n_outputs = layers[-1]["height"] * layers[-1]["width"]

    def _verify_layers(self, layers):
        """
        A function that checks each layer to ensure that it is valid i.e., expected activation function, trainable
        flag, etc.
        :param layers:
        :return: None
        """
        for i, layer in enumerate(layers):

            if layer["activation_func"] not in self.activation_funcs:
                raise ValueError('Unexpected activation function ' + str(layer["activation_func"]))

            for key in ['height', 'width']:
                x = layer[key]
                if not (isinstance(x, int) and x > 0):
                    raise ValueError(
                        'Layer {} {} should be a positive integer'.format(i, key)
                    )

            if not isinstance(layer["trainable"], bool):
                raise ValueError('Layer {} trainable should be a boolean'.format(i))

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

        if layer_number >= self.n_layers:
            raise ValueError('layer_number should be strictly less the number of layers')

        input_height = self.layers[layer_number]["height"]
        input_width = self.layers[layer_number]["width"]

        output_height = self.layers[layer_number + 1]["height"]
        output_width = self.layers[layer_number + 1]["width"]

        weight_shape = [input_height, input_width, output_height, output_width]

        return weight_shape

    def get_bias_shape(self, layer_number):
        """
        returns the shape of the biases in a layer specified by layer number as an array
        :param layer_number:
        :return:
        """
        if layer_number >= self.n_layers:
            raise ValueError('layer_number should be strictly less the number of layers')

        height = self.layers[layer_number + 1]["height"]
        width = self.layers[layer_number + 1]["width"]

        bias_shape = [height, width]

        return bias_shape


if __name__ == "__main__":

    layers = [
        {"activation_func": "relu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
        {"activation_func": "relu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
        {"activation_func": "linear", "trainable": False, "height": 20, "width": 10, "cell_height": 1}
        ]

    topology = Topology(layers)
