# Example usage of crocubot
# Acts as a useful test that training & inference still works!

from timeit import default_timer as timer

import logging

import alphai_time_series.performance_trials as pt
import numpy as np
import tensorflow as tf

import alphai_crocubot_oracle.classifier as cl
import alphai_crocubot_oracle.crocubot.train as crocubot
import alphai_crocubot_oracle.crocubot.evaluate as eval
from alphai_crocubot_oracle.crocubot.model import CrocuBotModel
import alphai_crocubot_oracle.flags as fl
import alphai_crocubot_oracle.iotools as io
import alphai_crocubot_oracle.topology as topo

FLAGS = tf.app.flags.FLAGS
DEFAULT_DATA_SOURCE = 'MNIST'
TIME_LIMIT = 600


def run_timed_performance_benchmark(flags, data_source=DEFAULT_DATA_SOURCE, do_training=True, n_labels_per_series=1):

    topology = load_default_topology(data_source, flags)

    #  First need to establish bin edges using full training set
    template_sample_size = np.minimum(flags.n_training_samples, 10000)
    _, training_labels = io.load_training_batch(data_source, batch_number=0, batch_size=template_sample_size,
                                                labels_per_series=n_labels_per_series)
    #  Ideally may use a template for each series
    if data_source == 'MNIST':
        bin_distribution = None
        bin_edges = None
    else:
        bin_distribution = cl.make_template_distribution(training_labels, topology.n_classification_bins)
        bin_edges = bin_distribution["bin_edges"]

    start_time = timer()

    if do_training:
        crocubot.train(topology, data_source=data_source, flags=flags, bin_edges=bin_edges)
    else:
        tf.reset_default_graph()
        model = CrocuBotModel(topology)
        model.build_layers_variables()

    mid_time = timer()
    train_time = mid_time - start_time
    print("Training complete.")

    metrics = evaluate_network(topology, data_source, bin_distribution)
    eval_time = timer() - mid_time

    print('Metrics:')
    if data_source == 'MNIST':
        accuracy = print_MNIST_accuracy(metrics)
    else:
        pt.print_performance_summary(metrics)

    print('Training took', str.format('{0:.2f}', train_time), "seconds")
    print('Evaluation took', str.format('{0:.2f}', eval_time), "seconds")

    if train_time > TIME_LIMIT:
        print('** Training took ', str.format('{0:.2f}', train_time - TIME_LIMIT), ' seconds too long - DISQUALIFIED! **')

    if data_source == 'MNIST':
        return accuracy


def evaluate_network(topology, data_source, bin_distribution):

    # Get the test data
    test_features, test_labels = io.load_test_samples(data_source=data_source)
    save_file = io.load_file_name(data_source, topology)

    binned_outputs = eval.eval_neural_net(test_features, topology, save_file)

    if data_source == 'MNIST':
        binned_outputs = np.mean(binned_outputs, axis=0)  # Average over passes
        predicted_indices = np.argmax(binned_outputs, axis=2)
        true_indices = np.argmax(test_labels, axis=2)

        metrics = np.equal(predicted_indices, true_indices)

    else:
        estimated_means, estimated_covariance = eval.forecast_means_and_variance(binned_outputs, bin_distribution)
        metrics = pt.evaluate_sample_performance(truth=test_labels, estimation=estimated_means, test_name=data_source, estimated_covariance=estimated_covariance)

    return metrics


def load_default_topology(data_source, flags):
    """The input and output layers must adhere to the dimensions of the features and labels.
    """

    if data_source == 'low_noise':
        n_input_series = 1
        n_features_per_series = 100
        n_classification_bins = 12
        n_output_series = 1
    elif data_source == 'stochasticwalk':
        n_input_series = 10
        n_features_per_series = 100
        n_classification_bins = 12
        n_output_series = 10
    elif data_source == 'MNIST':
        n_input_series = 1
        n_features_per_series = 784
        n_classification_bins = 10
        n_output_series = 1
    else:
        raise NotImplementedError

    return topo.Topology(layers=None, n_series=n_input_series, n_features_per_series=n_features_per_series, n_forecasts=n_output_series,
                         n_classification_bins=n_classification_bins)


def print_MNIST_accuracy(metrics):

    total_tests = len(metrics)
    correct = np.sum(metrics)

    accuracy = correct / total_tests * 100

    print('MNIST accuracy of ', accuracy, '%')

    return accuracy


def run_MNIST_test():

    config = fl.load_default_config()
    config["n_epochs"] = 1
    config["learning_rate"] = 3e-3   # Use high learning rate for testing purposes
    config["cost_type"] = 'softmax'  # 'bayes'; 'softmax'; 'hellinger'
    config['batch_size'] = 200
    config['n_training_samples'] = 50000
    config['n_series'] = 1
    config['n_features_per_series'] = 784
    config['resume_training'] = False  # Make sure we start from scratch
    config['activation_functions'] = ['linear', 'selu', 'selu']

    fl.set_training_flags(config)
    print("Epochs to evaluate:", FLAGS.n_epochs)
    accuracy = run_timed_performance_benchmark(FLAGS, data_source=DEFAULT_DATA_SOURCE, do_training=True)

    return accuracy


def run_stochastic_test():
    config = fl.load_default_config()
    fl.set_training_flags(config)
    print("Epochs to evaluate:", FLAGS.n_epochs)
    run_timed_performance_benchmark(FLAGS, data_source='stochasticwalk', do_training=True)


if __name__ == '__main__':

    logger = logging.getLogger('tipper')
    logger.addHandler(logging.StreamHandler())
    # run_stochastic_test()
    run_MNIST_test()
