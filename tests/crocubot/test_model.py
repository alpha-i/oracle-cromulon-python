import tensorflow as tf

from alphai_crocubot_oracle.crocubot.model import CrocuBotModel
from alphai_crocubot_oracle.topology import Topology

FLAGS = tf.app.flags.FLAGS

class TestCrocubotModel(tf.test.TestCase):

    def test_create_model(self):
        layers = [{"activation_func": "relu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
                  {"activation_func": "relu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
                  {"activation_func": "linear", "trainable": False, "height": 20, "width": 10, "cell_height": 1}]
        topology = Topology(layers)

        model = CrocuBotModel(topology, FLAGS)

        self.assertEqual(model.layers, 3)