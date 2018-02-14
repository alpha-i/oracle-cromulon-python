# This module is used to make predictions
# Only used by oracle.py

import logging
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf

from alphai_cromulon_oracle.cromulon.model import Cromulon
from alphai_cromulon_oracle.cromulon.train import log_network_confidence

PRINT_KERNEL = True


def eval_neural_net(data, topology, tf_flags, last_train_file):
    """ Multiple passes allow us to estimate the posterior distribution.

    :param data:  Mini-batch to be fed into the network
    :param topology: Specifies layout of network, also used to identify save file
    :param tf_flags:
    :param last_train_file:

    :return: 3D array with dimensions [n_passes, n_samples, n_labels, n_bins]
    """

    is_training = tf.placeholder(tf.bool, name='is_training')
    cromulon = Cromulon(topology, tf_flags, is_training)

    saver = tf.train.Saver()
    x = tf.placeholder(tf_flags.d_type, shape=data.shape, name="x")
    logging.info("Evaluating {} passes with shape {}".format(tf_flags.n_eval_passes, data.shape))

    y = cromulon.show_me_what_you_got(x)

    with tf.Session() as sess:
        logging.info("Attempting to recover trained network: {}".format(last_train_file))
        start_time = timer()
        saver.restore(sess, last_train_file)
        end_time = timer()
        delta_time = end_time - start_time
        logging.info("Loading the model from disk took:{}".format(delta_time))

        posterior = sess.run(y, feed_dict={x: data, is_training: False})

        log_network_confidence(posterior, None)

    return posterior


def forecast_means_and_variance(outputs, bin_distribution):
    """ Each forecast comprises a mean and variance. NB not the covariance matrix
    Oracle will perform this externally, but this function is useful for testing purposes

    :param nparray outputs: Raw output from the network, a 4D array of shape [n_passes, n_samples, n_series, classes]
    :param bin_distribution: Characterises the binning used to perform the classification task

    :return: Means and variances of the posterior.
    """

    n_samples = outputs.shape[1]
    n_series = outputs.shape[2]

    mean = np.zeros(shape=(n_samples, n_series))
    variance = np.zeros(shape=(n_samples, n_series))

    for i in range(n_samples):
        for j in range(n_series):
            discrete_pdf = outputs[:, i, j, :]
            temp_mean, temp_variance = bin_distribution.declassify_single_pdf(discrete_pdf)
            mean[i, j] = temp_mean
            variance[i, j] = temp_variance

    return mean, variance
