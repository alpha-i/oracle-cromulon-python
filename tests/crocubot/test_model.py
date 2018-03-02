import tensorflow as tf
import numpy as np

from alphai_cromulon_oracle.cromulon.model import Cromulon

from alphai_cromulon_oracle.topology import (
    Topology,
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_DEPTH,
    DEFAULT_TIMESTEPS,
    DEFAULT_BINS,
    DEFAULT_N_FORECASTS
)
from tests.helpers import get_default_flags

DEFAULT_BATCH_SIZE = 100


class TestCromulonModel(tf.test.TestCase):

    def test_create_model(self):

        flags = get_default_flags()

        topology = Topology()

        model = Cromulon(topology, flags, is_training=True)
        n_connections_between_layers = DEFAULT_HIDDEN_LAYERS + 1
        self.assertEqual(model.number_of_layers, n_connections_between_layers)
        self.assertEqual(model.topology, topology)

        with self.test_session() as session:
            for layer_number in range(model.number_of_layers):
                for variable_name in model.layer_variables_list:
                    self.assertRaises(
                        ValueError,
                        model.get_variable,
                        layer_number, variable_name
                    )

        with self.test_session() as session:

            model.build_layers_variables()
            session.run(tf.global_variables_initializer())

            self.assertEqual(len(session.run(tf.report_uninitialized_variables())), 0)

            for layer_number in range(model.number_of_layers):
                for variable_name in model.layer_variables_list:
                    variable = model.get_variable(layer_number, variable_name)
                    self.assertIsInstance(variable, tf.Variable)
                    self.assertIsNotNone(variable.eval())
