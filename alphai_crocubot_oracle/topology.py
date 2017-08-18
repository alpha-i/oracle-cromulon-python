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
        self.n_inputs = layers[0]["input"]
        self.n_outputs = layers[-1]["output"]

    def _verify_layers(self, layers):
        """
        A function that checks each layer to ensure that it is valid i.e., expected activation function, trainable
        flag, etc.
        :param layers:
        :return: None
        """
        for i, layer in enumerate(layers):
            self._verify_single_layer(layer, i)

        for i in range(len(layers) - 1):
            self.verify_pair_of_layers(layers[i], layers[i+1], i)

    def _verify_single_layer(self, layer, layer_number):
        """
        A function that checks individual layers to ensure that the activation function is one that is expected,
        that the input and output dimensions are positive integers and that the trainable flag is a boolean.
        :param layer: dict
        :param layer_number: int
        :return: None
        """

        if layer["activation_func"] not in self.activation_funcs:
            raise ValueError('Unexpected activation function ' + str(layer["activation_func"]))

        for key in ['input', 'output']:
            x = layer[key]
            if not (isinstance(x, int) and x > 0):
                raise ValueError(
                    'Layer {} {} should be a positive integer'.format(layer_number, key)
                )

        if not isinstance(layer["trainable"], bool):
            raise ValueError('Layer {} trainable should be a boolean'.format(layer_number))

    def verify_pair_of_layers(self, primary_layer, secondary_layer, layer_number):
        """
        A function that checks that the output dimension of one layer matches the input dimension in the layer above
        :param primary_layer: dict defining a layer
        :param secondary_layer: dict defining layer above
        :param layer_number: int
        :return: None
        """

        if primary_layer["output"] != secondary_layer["input"]:
            raise ValueError(
                'The output of layer {} and the input of layer {} do not match.'.format(layer_number, layer_number+1)
            )
