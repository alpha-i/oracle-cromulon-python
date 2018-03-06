import unittest

import pytest

from alphai_cromulon_oracle.topology import (
    Topology,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_N_FEATURES,
    DEFAULT_DEPTH,
    DEFAULT_LAYER_WIDTHS, DEFAULT_LAYER_HEIGHTS, DEFAULT_LAYER_DEPTHS)


class TestTopology(unittest.TestCase):

    def test_positive_num_units(self):
        topology = Topology()
        topology.layers[0]["height"] = -1
        self.assertRaises(
            ValueError,
            topology._verify_layers
        )

    def test_activation_func(self):
        topology = Topology()
        bad_activation_func = "crocubot will destroy thee"
        topology.layers[0]["activation_func"] = bad_activation_func
        self.assertRaises(
            ValueError,
            topology._verify_layers
        )

    def test_trainable_is_bool(self):
        topology = Topology()
        topology.layers[0]["trainable"] = "surprise motherfucker"
        self.assertRaises(
            ValueError,
            topology._verify_layers
        )

    def test_get_cell_shape(self):
        topology = Topology()
        cell_shape = topology.get_cell_shape(0)
        assert cell_shape == [1, DEFAULT_N_FEATURES]

    def test_get_weight_shape(self):
        topology = Topology()
        weight_shape = topology.get_weight_shape(0)
        input_shape = [DEFAULT_LAYER_DEPTHS[0], DEFAULT_LAYER_HEIGHTS[0], DEFAULT_LAYER_WIDTHS[0]]
        output_shape = [DEFAULT_LAYER_DEPTHS[1], DEFAULT_LAYER_HEIGHTS[1], DEFAULT_LAYER_WIDTHS[1]]
        expected_weight_shape = input_shape + output_shape
        assert weight_shape == expected_weight_shape

    def test_get_bias_shape(self):
        topology = Topology()
        bias_shape = topology.get_bias_shape(0)
        assert bias_shape == [DEFAULT_LAYER_DEPTHS[1], DEFAULT_LAYER_HEIGHTS[1], DEFAULT_LAYER_WIDTHS[1]]
