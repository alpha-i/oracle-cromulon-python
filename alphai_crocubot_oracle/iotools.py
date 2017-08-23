import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

import alphai_time_series.performance_trials as pt

FLAGS = tf.app.flags.FLAGS

N_TEST_SAMPLES = 100
DEFAULT_SAVE_PATH = "/tmp/"
MNIST = None
DO_DIFF = True
RESHAPE_MNIST = True


def initialise_MNIST():
    global MNIST
    MNIST = input_data.read_data_sets("MNIST_data/", one_hot=True)


def load_save_state(data_source, architecture='full'):
    """Restore previous state of tensorflow variables"""

    saver = tf.train.Saver()
    save_file = load_file_name(data_source, architecture=architecture)

    with tf.Session() as sess:
        saver.restore(sess, save_file)
        print("Model restored.")


def load_training_batch(data_source="MNIST", batch_number=0, batch_size=100, labels_per_series=1, do_diff=DO_DIFF,
                        dtype=None):
    """Load a subsample of examples for training"""

    if dtype is None:
        dtype = FLAGS.D_TYPE

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

        np_dtype = dtype_from_tf_type(dtype)
        features, labels = pt.get_training_batch(series_name=data_source, batch_size=batch_size, batch_number=batch_number,
                                                 label_timesteps=labels_per_series, do_differential_forecast=do_diff, dtype=np_dtype)
        # features = np.squeeze(features)
        # labels = np.squeeze(labels)

    return features, labels


def load_test_samples(data_source="MNIST", labels_per_series=1, tf_type=None, do_differential_forecast=DO_DIFF):
    """Load a subsample of examples for testing. """

    if tf_type is None:
        if FLAGS.TF_TYPE == 32:
            tf_type = tf.float32
        else:
            tf_type = tf.float64

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
        dtype = dtype_from_tf_type(tf_type)
        features, labels = pt.get_test_batch(batch_size=N_TEST_SAMPLES, series_name=data_source, do_differential_forecast=do_differential_forecast,
                                             label_timesteps=labels_per_series, dtype=dtype)

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
    path = FLAGS.save_path

    return path + bitstring[-2:] + "model_" + data_source + "_" + series_string + '_' + depth_string + "x" + breadth_string + ".ckpt"


def dtype_from_tf_type(tf_dtype):
    if tf_dtype == tf.float64:
        return 'float64'
    elif tf_dtype == tf.float32:
        return 'float32'
    else:
        raise NotImplementedError
