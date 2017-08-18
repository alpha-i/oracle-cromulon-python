import tensorflow as tf


class LayeredNetwork(object):

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


class MLP(LayeredNetwork):

    def __init__(self, layers, scope='mlp'):
        LayeredNetwork.__init__(self, layers, scope)
        self.build_graph()

    def build_graph(self):

        output = self.input
        with tf.variable_scope(self.scope):
            for i, layer in enumerate(self.topology.layers):
                with tf.variable_scope('layer' + str(i)):
                    weights = tf.get_variable('weights',
                                              shape=[layer["input"], layer["output"]],
                                              trainable=layer["trainable"]
                                              )
                    biases = tf.get_variable('biases',
                                             shape=[layer["output"]],
                                             trainable=layer["trainable"]
                                             )
                    output = tf.matmul(output, weights) + biases

                    if layer["activation_func"] == "relu":
                        output = tf.nn.relu(output)
                    elif layer["activation_func"] == "linear":
                        pass


        self.output = output

    def get_weights_dict(self):

        with tf.variable_scope(self.scope):
            for i, layer in enumerate(self.topology.layers):
                with tf.variable_scope('layer' + str(i)):
                    weights = tf.get_variable('weights')


        return weights_dict
