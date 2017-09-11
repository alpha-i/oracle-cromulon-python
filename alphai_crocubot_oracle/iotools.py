# Used for retrieving non-financial data, and saving/retrieving non-financial models
# Will not be used by quant workflow

import tensorflow as tf

from alphai_data_sources.generator import BatchGenerator
import alphai_crocubot_oracle.classifier as cl

FLAGS = tf.app.flags.FLAGS
batch_generator = BatchGenerator()


def load_batch(batch_options, data_source, bin_edges=None):

    features, labels = batch_generator.get_batch(batch_options, data_source)

    if bin_edges is not None:
        labels = cl.classify_labels(bin_edges, labels)

    return features, labels


def load_file_name(data_source, topology):
    """ File used for storing the network parameters.

    :param str data_source: Identify the data on which the network was trained: MNIST, low_noise, randomwalk, etc
    :param Topology topology: Info on network shape
    :param str path:
    :return:
    """

    depth_string = str(topology.n_layers)
    breadth_string = str(topology.n_features_per_series)
    series_string = str(topology.n_series)

    bitstring = str(FLAGS.TF_TYPE)
    path = FLAGS.model_save_path

    return path + bitstring[-2:] + "model_" + data_source + "_" + series_string + '_' + depth_string + "x" + breadth_string + ".ckpt"
