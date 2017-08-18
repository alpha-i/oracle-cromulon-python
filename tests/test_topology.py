import unittest

from alphai_crocubot_oracle.topology import Topology


class TestTopology(unittest.TestCase):

    def setUp(self):

        self.layers = [{"input": 200, "output": 50, "activation_func": "relu", "trainable": False},
                       {"input": 50, "output": 40, "activation_func": "relu", "trainable": False},
                       {"input": 40, "output": 20, "activation_func": "linear", "trainable": False}]

        self.topology = Topology(self.layers)

    def test_positive_num_units(self):

        bad_layers = self.topology.layers
        bad_layers[0]["output"] = -1
        self.assertRaises(
            ValueError,
            self.topology._verify_layers,
            bad_layers
        )

    def test_activation_func(self):

        bad_activation_func = "crocubot will destroy thee"
        bad_layers = self.topology.layers
        bad_layers[0]["activation_func"] = bad_activation_func
        self.assertRaises(
            ValueError,
            self.topology._verify_layers,
            bad_layers
        )

    def test_trainable_is_bool(self):

        bad_layers = self.topology.layers
        bad_layers[0]["trainable"] = "surprise motherfucker"
        self.assertRaises(
            ValueError,
            self.topology._verify_layers,
            bad_layers
        )

    def test_pair_of_layers(self):

        bad_layers = self.topology.layers
        bad_layers[0]["output"] = bad_layers[1]["input"] + 10
        self.assertRaises(
            ValueError,
            self.topology._verify_layers,
            bad_layers
        )
