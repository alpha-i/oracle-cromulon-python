import numpy as np
import tensorflow as tf


class LayeredNetwork(object):

    def __init__(self, layers, scope):
        self.layers = layers
        self.scope = scope
        self.check_layers()
        self.input = tf.placeholder(tf.float32, shape=[None, layers[0]["input"]])
        self.output = tf.placeholder(tf.float32, shape=[None, layers[-1]["output"]])

    def check_layers(self):
        for i in range(len(self.layers) - 1):
            if self.layers[i]["output"] != self.layers[i + 1]["input"]:
                raise Exception(
                    'The output of layer ' + str(i) + ' and the input of layer ' + str(i + 1) + ' do not match.')


class MLP(LayeredNetwork):

    def __init__(self, layers, scope='mlp'):
        LayeredNetwork.__init__(self, layers, scope)
        self.build_graph()

    def build_graph(self):

        count = 0
        output = self.input
        with tf.variable_scope(self.scope):
            for layer in self.layers:
                with tf.variable_scope('layer' + str(count)):
                    weights = tf.get_variable('weights', shape=[layer["input"], layer["output"]])
                    biases = tf.get_variable('biases', shape=[layer["output"]])
                    output = tf.matmul(output, weights) + biases
                    if layer["activation_func"] == "relu":
                        output = tf.nn.relu(output)
                count += 1

        self.output = output


class GRU(LayeredNetwork):

    def __init__(self, layers, scope='gru'):
        LayeredNetwork.__init__(self, layers, scope)
        self.build_graph()

    def build_graph(self):

        count = 0
        output = self.input
        with tf.variable_scope(self.scope):
            for i, layer in enumerate(self.layers):
                with tf.variable_scope('layer_' + str(count)):
                    key = "layer_" + str(i)
                    state_ph = tf.placeholder(tf.float32, shape=[None, layer["state"]], name="layer_state")

                    B = tf.get_variable('B', shape=[layer["input"], layer["state"]])
                    new_state_1 = tf.tanh(state_ph)
                    new_state_2 = tf.tanh(tf.matmul(output, B))

                    C = tf.get_variable('C', shape=[layer["state"], layer["output"]])
                    new_output_1 = tf.tanh(tf.matmul(state_ph, C))

                    D = tf.get_variable('D', shape=[layer["input"], layer["output"]])
                    new_output_2 = tf.tanh(tf.matmul(output, D))

                    for gate in layer["gates"]:
                        if gate == "forget":
                            new_state_1 = self.make_gated_signal(gate, layer, state_ph, output, new_state_1)
                        elif gate == "input":
                            new_state_2 = self.make_gated_signal(gate, layer, state_ph, output, new_state_2)
                        elif gate == "output":
                            new_output_1 = self.make_gated_signal(gate, layer, state_ph, output, new_output_1)
                        elif gate == "pass":
                            new_output_2 = self.make_gated_signal(gate, layer, state_ph, output, new_output_2)
                        else:
                            raise Exception('Unexpected gate: ' + str(gate))

                    output = new_output_1 + new_output_2
                    # self.layer_states[key] = new_state_1 + new_state_2
                count += 1
        self.output = output

    def make_gated_signal(self, gate_name, layer, state, input, signal):

        A = tf.get_variable('A_' + str(gate_name), shape=[layer["state"], signal.shape[1]])
        B = tf.get_variable('B_' + str(gate_name), shape=[layer["input"], signal.shape[1]])
        b = tf.get_variable('b_' + str(gate_name), shape=signal.shape[1])
        gate = tf.sigmoid(tf.matmul(state, A) + tf.matmul(input, B) + b)

        return gate * signal


if __name__ == "__main__":

    mlp_layers = [{"input":200,"output":50, "activation_func":"relu"},
                  {"input":50,"output":40, "activation_func":"relu"},
                  {"input":40, "output": 20, "activation_func": "linear"}]

    mlp = MLP(mlp_layers, scope='phi')

    gru_layers = [{"input": 20, "state":70, "output": 30, "gates":["input","output","forget","pass"] },
                  {"input": 30, "state": 50, "output": 20, "gates":["input","output","forget"]}]


    gru = GRU(gru_layers, scope='encoder')

    mlp_input = np.zeros([1,mlp_layers[0]["input"]])

    mlp_feed_dict = {mlp.input: mlp_input}

    gru_feed_dict = {}
    for i, layer in enumerate(gru_layers):
        key = "layer_" + str(i)
        gru_feed_dict[key] = np.zeros([1, layer["state"]])

    gru_input = np.zeros([1, gru_layers[0]["input"]])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(mlp.output, feed_dict=mlp_feed_dict))

        # print(sess.run(gru.output, feed_dict=)
















