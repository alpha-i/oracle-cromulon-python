import tensorflow as tf
import numpy as np

import alphai_crocubot_oracle.tensormaths as tm
import alphai_crocubot_oracle.topology as tp

FLAGS = tf.app.flags.FLAGS

class LayeredNetwork(object):
    """

    """

    def __init__(self, topology, scope):
        self.verify_scope(scope)
        self.scope = scope
        self.topology = topology
        self.n_inputs = topology.n_inputs
        self.n_outputs = topology.n_outputs
        self.input = tf.placeholder(tf.float32, shape=[None, topology.n_inputs])
        self.output = tf.placeholder(tf.float32, shape=[None, topology.n_outputs])

    def verify_scope(self, scope):

        if not isinstance(scope, str):
            raise ValueError('The variable scope should be a string but got: {} '.format(scope))


# class BayesMLP(LayeredNetwork):
#     """
#     Initialises and provides access to the network's tensorflow variables
#     """
#
#     # To build the graph when instantiated
#     def __init__(self, parameters):
#         self.random_seed = 0
#         self.graph = tf.Graph()
#
#         with self.graph.as_default():
#             self.prediction = average_multiple_passes(x)
#             self.cost = cost.get_cost(prediction, truth)
#             self.optimizer = set_optimiser(parameters)
#
#         LayeredNetwork.__init__(self, layers, scope)
#         self.build_graph()


def increment_noise_seed():
    """ Called whenever we wish to change the random realisation of the weights. """

    FLAGS.random_seed += 1

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

def initialize_layer_variable(layer_number, var_name, initializer, trainable=True):

    assert isinstance(layer_number, int)
    scope_name = str(layer_number)
    with tf.variable_scope(scope_name) as scope:
        v = tf.get_variable(var_name, initializer=initializer, trainable=trainable, dtype=tm.DEFAULT_TF_TYPE)

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

def reset():

    tf.reset_default_graph()

def average_multiple_passes(data, number_of_passes, topology):
    """  Multiple passes allow us to estimate the posterior distribution.
    :param data: Mini-batch to be fed into the network
    :param number_of_passes: How many random realisations of the weights should be sampled
    :return: Means and variances of the posterior. NB this is not the covariance - see network_covariance.py
    """

    collated_outputs = collate_multiple_passes(data, topology, number_of_passes=number_of_passes)
    mean, variance = tf.nn.moments(collated_outputs, axes=[0])

    if number_of_passes == 1:
        print("warning - using default variance")
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
        signal = activation_function(signal, topology.layers[layer + 1]["activation_func"])

    return signal


def activation_function(signal, function):
    """ Select desired activation function.
    """

    if function == 'linear':
        return signal
    elif function == 'selu':
        return tm.selu(signal)
    elif function == 'relu':
        return tf.nn.relu(signal)
    elif function == 'kelu':
        return tm.kelu(signal)
    else:
        raise NotImplementedError


def initialise_parameters(topology):
    """Sets all parameter values for each layer.
    """

    for layer_number in range(topology.n_layers):

        w_shape = topology.get_weight_shape(layer_number)
        b_shape = topology.get_bias_shape(layer_number)

        initial_rho_weights = np.log(FLAGS.INITIAL_WEIGHT_UNCERTAINTY)
        initial_rho_bias = np.log(FLAGS.INITIAL_BIAS_UNCERTAINTY)

        # random_signs = 2*(np.random.randint(2, size=w_shape) - 0.5)
        # initial_weight_means = INITIAL_WEIGHT_DISPLACEMENT * random_signs

        initialize_layer_variable(layer_number, 'mu_w', tm.centred_gaussian(w_shape, FLAGS.INITIAL_WEIGHT_DISPLACEMENT))
        initialize_layer_variable(layer_number, 'rho_w', initial_rho_weights + tm.centred_gaussian(w_shape, np.abs(initial_rho_weights) / 10))

        initialize_layer_variable(layer_number, 'mu_b', tm.centred_gaussian(b_shape, FLAGS.INITIAL_BIAS_DISPLACEMENT))
        initialize_layer_variable(layer_number, 'rho_b', initial_rho_bias + tm.centred_gaussian(b_shape, np.abs(initial_rho_bias) / 10))

        is_alpha_trainable = False
        initialize_layer_variable(layer_number, 'log_alpha', np.log(FLAGS.INITIAL_ALPHA).astype(FLAGS.D_TYPE), trainable=is_alpha_trainable) # Hyperprior on the distribution of the weights

        initialise_noise('weight_noise', w_shape, layer_number)
        initialise_noise('bias_noise', b_shape, layer_number)



def initialise_noise(var_name, shape, layer):

    if FLAGS.USE_PERFECT_NOISE:
        noise_vector = tm.perfect_centred_gaussian(shape)
    else:
        noise_vector = tm.centred_gaussian(shape)

    initialize_layer_variable(layer, var_name, noise_vector, trainable=False)


if __name__ == "__main__":

    layers = [
        {"activation_func": "input", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
        {"activation_func": "relu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
        {"activation_func": "linear", "trainable": False, "height": 20, "width": 10, "cell_height": 1}
        ]

    topology = tp.Topology(layers)
