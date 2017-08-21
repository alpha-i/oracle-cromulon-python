import tensorflow as tf
import numpy as np

import alphai_crocubot_oracle.network as nt
import alphai_crocubot_oracle.iotools as io
import alphai_crocubot_oracle.classifier as cl

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_eval_passes', 50,
                            """Number of passes to average over.""")


def eval_neural_net(data_source, data, topology, number_of_passes=FLAGS.num_eval_passes):
    """ Multiple passes allow us to estimate the posterior distribution.

    :param data_source: will load network trained on this dataset
    :param data:  Mini-batch to be fed into the network
    :param number_of_passes: How many random realisations of the weights should be sampled
    :return: 3D array with dimensions [n_passes, n_samples, n_labels] NB this is not the covariance - see network_covariance.py
    """

    print("Evaluating", data_source, "with shape", data.shape)

    try:
        nt.initialise_parameters(topology)
    except:
        print('Variables already initialised')

    saver = tf.train.Saver()
    save_file = io.load_file_name(data_source, topology)
    y = nt.collate_multiple_passes(data, topology, number_of_passes=number_of_passes)

    with tf.Session() as sess:
        print("Attempting to recover trained network:", save_file)
        saver.restore(sess, save_file)
        first_output = y.eval()

        if number_of_passes == 1:
            return first_output
        else:
            return y.eval()

def forecast_means_and_variance(data_source, data, topology, number_of_passes=30, bin_distribution=None):
    """ Each forecast comprises a mean and variance. NB not the covariance matrix - see network_covariance.py

    :param data_source:
    :param data:
    :param number_of_passes:
    :return: Means and variances of the posterior.
    """

    outputs = eval_neural_net(data_source, data, topology, number_of_passes)
    assert outputs.shape[0] == number_of_passes, 'unexpected output shape'
    # Expect 4D array of shape [n_passes, n_samples, n_series, classes]

    if bin_distribution is None:
        # Take mean and variance across different passes
        mean = np.mean(outputs, axis=FLAGS.ENSEMBLE_DIMENSION)
        variance = np.var(outputs, axis=FLAGS.ENSEMBLE_DIMENSION)
    else:
        n_samples = outputs.shape[1]
        n_series = outputs.shape[2]

        # Find mean and variance associated with each sample forecast
        mean = np.zeros(shape=(n_samples, n_series))
        variance = np.zeros(shape=(n_samples, n_series))

        for i in range(n_samples):
            for j in range(n_series):
                bin_passes = outputs[:,i,j, :]
                temp_mean, temp_variance = cl.declassify_labels(bin_distribution, bin_passes)
                mean[i, j] = temp_mean
                variance[i ,j] = temp_variance

    return mean, variance
