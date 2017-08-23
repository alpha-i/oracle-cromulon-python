import tensorflow as tf
import numpy as np

from alphai_time_series.calculator import make_diagonal_covariance_matrices

import alphai_crocubot_oracle.network as nt
import alphai_crocubot_oracle.classifier as cl

FLAGS = tf.app.flags.FLAGS


def eval_neural_net(data, topology, save_file):
    """ Multiple passes allow us to estimate the posterior distribution.

    :param data:  Mini-batch to be fed into the network
    :param topology: Specifies layout of network, also used to identify save file
    :return: 3D array with dimensions [n_passes, n_samples, n_labels]
    """

    print("Evaluating with shape", data.shape)

    try:
        nt.initialise_parameters(topology)
    except:
        print('Variables already initialised')

    saver = tf.train.Saver()
    y = nt.collate_multiple_passes(data, topology, number_of_passes=FLAGS.num_eval_passes)

    with tf.Session() as sess:
        print("Attempting to recover trained network:", save_file)
        saver.restore(sess, save_file)

        return y.eval()

def forecast_means_and_variance(outputs, bin_distribution):
    """ Each forecast comprises a mean and variance. NB not the covariance matrix
    Oracle will perform this outside, but this function is useful for testing purposes

    :param nparray outputs: Raw output from the network, a 4D array of shape [n_passes, n_samples, n_series, classes]
    :param bin_distribution: Characterises the binning used to perform the classification task
    :return: Means and variances of the posterior.
    """

    assert outputs.shape[0] == FLAGS.num_eval_passes, 'unexpected output shape'
    n_samples = outputs.shape[1]
    n_series = outputs.shape[2]

    mean = np.zeros(shape=(n_samples, n_series))
    variance = np.zeros(shape=(n_samples, n_series))

    for i in range(n_samples):
        for j in range(n_series):
            bin_passes = outputs[:,i,j, :]
            temp_mean, temp_variance = cl.declassify_labels(bin_distribution, bin_passes)
            mean[i, j] = temp_mean
            variance[i ,j] = temp_variance

    if n_series > 1:
        variance = make_diagonal_covariance_matrices(variance)

    return mean, variance
