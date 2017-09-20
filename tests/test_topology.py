import unittest

from alphai_crocubot_oracle.topology import Topology, DEFAULT_HEIGHTS, DEFAULT_WIDTHS


class TestTopology(unittest.TestCase):

    def setUp(self):
        self.topology = Topology()

    def test_layer_contructor(self):
        # There are currently two methods for constructing a topology.
        # Here we test passing the predefined list of layers
        layers = [{"activation_func": "relu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
                  {"activation_func": "relu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
                  {"activation_func": "linear", "trainable": False, "height": 20, "width": 10, "cell_height": 1}]

        new_topology = Topology(layers)
        assert isinstance(new_topology, Topology)

    def test_positive_num_units(self):

        bad_layers = self.topology.layers
        bad_layers[0]["height"] = -1
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

    def test_get_cell_shape(self):

        cell_shape = self.topology.get_cell_shape(0)
        assert cell_shape == [1, 10]

    def test_get_weight_shape(self):

        weight_shape = self.topology.get_weight_shape(0)
        assert weight_shape == [DEFAULT_HEIGHTS[0], DEFAULT_WIDTHS[0], DEFAULT_HEIGHTS[1], DEFAULT_WIDTHS[1]]

    def test_get_bias_shape(self):

        bias_shape = self.topology.get_bias_shape(0)
        assert bias_shape == [DEFAULT_HEIGHTS[1], DEFAULT_WIDTHS[1]]