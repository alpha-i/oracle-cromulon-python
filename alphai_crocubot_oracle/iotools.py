# Used for retrieving non-financial data, and saving/retrieving non-financial models
# Will not be used by quant workflow

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

import alphai_time_series.performance_trials as pt
import alphai_crocubot_oracle.classifier as cl

FLAGS = tf.app.flags.FLAGS

N_TEST_SAMPLES = 100
DEFAULT_SAVE_PATH = "/tmp/"
MNIST = None
DO_DIFF = True
RESHAPE_MNIST = True


def initialise_MNIST():
    global MNIST
    MNIST = input_data.read_data_sets("MNIST_data/", one_hot=True)


def load_training_batch(data_source="MNIST", batch_number=0, batch_size=100, labels_per_series=1, do_diff=DO_DIFF,
                        bin_edges=None):
    """Load a subsample of examples for training"""

    if data_source == "low_noise":
        do_diff = False

    if data_source == "MNIST":
        if MNIST is None:
            initialise_MNIST()

        features, labels = MNIST.train.next_batch(batch_size)

        if RESHAPE_MNIST:
            N_CLASSIFICATION_BINS = 10
            features = features.reshape(features.shape[0], 28, 28)
            labels = labels.reshape(labels.shape[0], 1, N_CLASSIFICATION_BINS)

    else:
        features, labels = pt.get_training_batch(series_name=data_source, batch_size=batch_size, batch_number=batch_number,
                                                 label_timesteps=labels_per_series, do_differential_forecast=do_diff, dtype=FLAGS.d_type)

    if bin_edges is not None:
        labels = cl.classify_labels(bin_edges, labels)

    return features, labels


def load_test_samples(data_source="MNIST", labels_per_series=1, do_differential_forecast=DO_DIFF):
    """Load a subsample of examples for testing. """

    if data_source == "low_noise":
        do_differential_forecast = False

    if data_source == "MNIST":
        if MNIST is None:
            initialise_MNIST()

        features = MNIST.test.images
        labels = MNIST.test.labels

        if RESHAPE_MNIST:
            N_CLASSIFICATION_BINS = 10
            features = features.reshape(features.shape[0], 28, 28)
            labels = labels.reshape(labels.shape[0], 1, N_CLASSIFICATION_BINS)
    else:
        features, labels = pt.get_test_batch(batch_size=N_TEST_SAMPLES, series_name=data_source, do_differential_forecast=do_differential_forecast,
                                             label_timesteps=labels_per_series, dtype=FLAGS.d_type)

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
