# Example usage of crocubot
# Acts as a useful test that training & inference still works!

from timeit import default_timer as timer
import datetime
import logging
import numpy as np
import tensorflow as tf

import alphai_crocubot_oracle.classifier as cl
import alphai_crocubot_oracle.crocubot.evaluate as eval
from alphai_crocubot_oracle.crocubot.model import CrocuBotModel
import alphai_crocubot_oracle.flags as fl
import alphai_crocubot_oracle.iotools as io
import alphai_crocubot_oracle.topology as topo
from alphai_crocubot_oracle.crocubot import train as crocubot_train

from alphai_data_sources.data_sources import DataSourceGenerator
from alphai_data_sources.generator import BatchOptions, BatchGenerator
from alphai_time_series.performance_trials.performance import Metrics

data_source_generator = DataSourceGenerator()
batch_generator = BatchGenerator()
model_metrics = Metrics()


FLAGS = tf.app.flags.FLAGS
TIME_LIMIT = 600


def run_timed_benchmark_mnist(series_name, flags, do_training):

    topology = load_default_topology(series_name)

    batch_options = BatchOptions(batch_size=200,
                                 batch_number=0,
                                 train=do_training,
                                 dtype='float32')

    data_source = data_source_generator.make_data_source(series_name)

    _, labels = io.load_batch(batch_options, data_source)

    start_time = timer()

    execution_time = datetime.datetime.now()

    if do_training:
        crocubot_train.train(topology, series_name, execution_time)
    else:
        tf.reset_default_graph()
        model = CrocuBotModel(topology)
        model.build_layers_variables()

    mid_time = timer()
    train_time = mid_time - start_time
    print("Training complete.")

    metrics = evaluate_network(topology, series_name, bin_distribution=None)
    eval_time = timer() - mid_time

    print('Metrics:')
    print_MNIST_accuracy(metrics)
    print_time_info(train_time, eval_time)


def print_time_info(train_time, eval_time):

    print('Training took', str.format('{0:.2f}', train_time), "seconds")
    print('Evaluation took', str.format('{0:.2f}', eval_time), "seconds")

    if train_time > TIME_LIMIT:
        print('** Training took ', str.format('{0:.2f}', train_time - TIME_LIMIT),
              ' seconds too long - DISQUALIFIED! **')


def run_timed_benchmark_time_series(series_name, flags, do_training=True):

    topology = load_default_topology(series_name)

    #  First need to establish bin edges using full training set
    template_sample_size = np.minimum(flags.n_training_samples_benchmark, 10000)

    batch_options = BatchOptions(batch_size=template_sample_size,
                                 batch_number=0,
                                 train=do_training,
                                 dtype='float32')

    data_source = data_source_generator.make_data_source(series_name)

    _, labels = io.load_batch(batch_options, data_source)

    bin_distribution = cl.make_template_distribution(labels, topology.n_classification_bins)
    bin_edges = bin_distribution["bin_edges"]

    start_time = timer()

    execution_time = datetime.datetime.now()

    if do_training:
        crocubot_train.train(topology, series_name, execution_time, bin_edges=bin_edges)
    else:
        tf.reset_default_graph()
        model = CrocuBotModel(topology)
        model.build_layers_variables()

    mid_time = timer()
    train_time = mid_time - start_time
    print("Training complete.")

    evaluate_network(topology, series_name, bin_distribution)
    eval_time = timer() - mid_time

    print('Metrics:')
    print_time_info(train_time, eval_time)


def evaluate_network(topology, series_name, bin_distribution):

    # Get the test data
    batch_options = BatchOptions(batch_size=100,
                                 batch_number=1,
                                 train=False,
                                 dtype='float32')

    data_source = data_source_generator.make_data_source(series_name)

    test_features, test_labels = io.load_batch(batch_options, data_source)
    save_file = io.load_file_name(series_name, topology)

    binned_outputs = eval.eval_neural_net(test_features, topology, save_file)

    if series_name == 'mnist':
        binned_outputs = np.mean(binned_outputs, axis=0)  # Average over passes
        predicted_indices = np.argmax(binned_outputs, axis=2)
        true_indices = np.argmax(test_labels, axis=2)

        metric = np.equal(predicted_indices, true_indices)
        return metric

    else:
        estimated_means, estimated_covariance = eval.forecast_means_and_variance(binned_outputs, bin_distribution)
        test_labels = np.squeeze(test_labels)

        model_metrics.evaluate_sample_performance(data_source, test_labels, estimated_means, estimated_covariance)


def load_default_topology(series_name):
    """The input and output layers must adhere to the dimensions of the features and labels.
    """

    if series_name == 'low_noise':
        n_input_series = 1
        n_features_per_series = 100
        n_classification_bins = 12
        n_output_series = 1
    elif series_name == 'stochastic_walk':
        n_input_series = 10
        n_features_per_series = 100
        n_classification_bins = 12
        n_output_series = 10
    elif series_name == 'mnist':
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


def run_mnist_test(train_path, tensorboard_log_path):

    config = load_default_config()
    config["n_epochs"] = 10
    config["learning_rate"] = 3e-3   # Use high learning rate for testing purposes
    config["cost_type"] = 'bayes'  # 'bayes'; 'softmax'; 'hellinger'
    config['batch_size'] = 200
    config['n_training_samples_benchmark'] = 50000
    config['n_series'] = 1
    config['n_features_per_series'] = 784
    config['resume_training'] = False  # Make sure we start from scratch
    config['activation_functions'] = ['linear', 'selu', 'selu']
    config['tensorboard_log_path'] = tensorboard_log_path
    config['train_path'] = train_path
    config['model_save_path'] = train_path
    config['n_retrain_epochs'] = 5

    fl.set_training_flags(config)
    print("Epochs to evaluate:", FLAGS.n_epochs)
    run_timed_benchmark_mnist(series_name="mnist", flags=FLAGS, do_training=True)


def run_stochastic_test(train_path, tensorboard_log_path):
    config = load_default_config()

    config["n_epochs"] = 10   # -3 per sample after 10 epochs
    config["learning_rate"] = 3e-3   # Use high learning rate for testing purposes
    config["cost_type"] = 'bayes'  # 'bayes'; 'softmax'; 'hellinger'
    config['batch_size'] = 200
    config['n_training_samples_benchmark'] = 1000
    config['n_series'] = 10
    config['n_features_per_series'] = 100
    config['resume_training'] = False  # Make sure we start from scratch
    config['activation_functions'] = ['linear', 'selu', 'selu', 'selu']
    config["layer_heights"] = 200
    config["layer_widths"] = 1
    config['tensorboard_log_path'] = tensorboard_log_path
    config['train_path'] = train_path
    config['model_save_path'] = train_path
    config['n_retrain_epochs'] = 5

    fl.set_training_flags(config)
    print("Epochs to evaluate:", FLAGS.n_epochs)
    run_timed_benchmark_time_series(series_name='stochastic_walk', flags=FLAGS, do_training=True)


def load_default_config():
    configuration = {
        'data_transformation': {
            'feature_config_list': [
                {
                    'name': 'close',
                    'order': 'log-return',
                    'normalization': 'standard',
                    'nbins': 12,
                    'is_target': True,
                },
            ],
            'exchange_name': 'NYSE',
            'features_ndays': 10,
            'features_resample_minutes': 15,
            'features_start_market_minute': 60,
            'prediction_frequency_ndays': 1,
            'prediction_market_minute': 60,
            'target_delta_ndays': 1,
            'target_market_minute': 60,
        },
        'train_path': '/tmp/crocubot/',
        'covariance_method': 'NERCOME',
        'covariance_ndays': 9,
        'model_save_path': '/tmp/crocubot/',
        'd_type': 'float32',
        'tf_type': 32,
        'random_seed': 0,
        'predict_single_shares': False,

        # Training specific
        'n_epochs': 1,
        'n_training_samples_benchmark': 1000,
        'learning_rate': 2e-3,
        'batch_size': 100,
        'cost_type': 'bayes',
        'n_train_passes': 30,
        'n_eval_passes': 30,
        'resume_training': False,

        # Topology
        'n_series': 1,
        'n_features_per_series': 271,
        'n_forecasts': 1,
        'n_classification_bins': 12,
        'layer_heights': [200, 200, 200],
        'layer_widths': [1, 1, 1],
        'activation_functions': ['relu', 'relu', 'relu'],

        # Initial conditions
        'INITIAL_ALPHA': 0.2,
        'INITIAL_WEIGHT_UNCERTAINTY': 0.05,
        'INITIAL_BIAS_UNCERTAINTY': 0.005,
        'INITIAL_WEIGHT_DISPLACEMENT': 0.02,
        'INITIAL_BIAS_DISPLACEMENT': 0.0005,
        'USE_PERFECT_NOISE': False,

        # Priors
        'double_gaussian_weights_prior': False,
        'wide_prior_std': 1.2,
        'narrow_prior_std': 0.05,
        'spike_slab_weighting': 0.5
    }

    return configuration


if __name__ == '__main__':

    logger = logging.getLogger('tipper')
    logger.addHandler(logging.StreamHandler())
    logging.basicConfig(level=logging.DEBUG)

    # change the following lines according to your machine
    train_path = 'D:\\tmp'
    tensorboard_log_path = 'D:\\tmp\\tensorboard'

    run_stochastic_test(train_path, tensorboard_log_path)
    run_mnist_test(train_path, tensorboard_log_path)
