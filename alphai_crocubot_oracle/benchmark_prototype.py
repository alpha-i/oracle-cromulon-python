from timeit import default_timer as timer
import numpy as np
import tensorflow as tf

import alphai_crocubot_oracle.crocubot_train as crocubot
import alphai_crocubot_oracle.crocubot_eval as eval
import alphai_crocubot_oracle.topology as topo
import alphai_crocubot_oracle.classifier as cl
import alphai_crocubot_oracle.iotools as io
import alphai_crocubot_oracle.network as nt
import alphai_time_series.performance_trials as pt


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_path', '/tmp/', """Path to save graph.""")
tf.app.flags.DEFINE_string('D_TYPE', 'float32', """Data type for numpy.""")
tf.app.flags.DEFINE_integer('TF_TYPE', 32, """Data type for tensorflow.""")
tf.app.flags.DEFINE_integer('n_training_samples', 1000, """Number of data samples to use for training.""")
tf.app.flags.DEFINE_integer('random_seed', 0, """Seed used to identify random noise realisiation.""")
tf.app.flags.DEFINE_float('INITIAL_WEIGHT_UNCERTAINTY', 0.4, """Initial standard deviation on weights.""")
tf.app.flags.DEFINE_float('INITIAL_BIAS_UNCERTAINTY', 0.4, """Initial standard deviation on bias.""")
tf.app.flags.DEFINE_float('INITIAL_WEIGHT_DISPLACEMENT', 0.1, """Initial offset on weight distributions.""")
tf.app.flags.DEFINE_float('INITIAL_BIAS_DISPLACEMENT', 0.4, """Initial offset on bias distributions.""")
tf.app.flags.DEFINE_float('INITIAL_ALPHA', 0.2, """Prior on weights.""")
tf.app.flags.DEFINE_boolean('USE_PERFECT_NOISE', True, """Whether to set noise such that its mean and std are exactly the desired values""")
tf.app.flags.DEFINE_boolean('double_gaussian_weights_prior', False, """Whether to impose a double Gaussian prior.""")
tf.app.flags.DEFINE_float('wide_prior_std', 1.2, """Initial standard deviation on weights.""")
tf.app.flags.DEFINE_float('narrow_prior_std', 0.05, """Initial standard deviation on weights.""")
tf.app.flags.DEFINE_float('spike_slab_weighting', 0.5, """Initial standard deviation on weights.""")
FLAGS._parse_flags()

DEFAULT_DATA_SOURCE = 'stochasticwalk'
DEFAULT_ARCHITECTURE = 'full'
EVAL_NUMBER_PASSES = 100
TRAIN_NUMBER_PASSES = 50
EVAL_COVARIANCE_PASSES = 100
TIME_LIMIT = 600
DEFAULT_N_EPOCHS = 1


def run_timed_performance_benchmark(data_source=DEFAULT_DATA_SOURCE, n_epochs=DEFAULT_N_EPOCHS, do_training=True, n_labels_per_series=1):

    topology = load_default_topology(data_source)

    # First need to establish bin edges using full training set
    template_sample_size = np.minimum(FLAGS.n_training_samples, 10000)
    _, training_labels = io.load_training_batch(data_source, batch_number=0, batch_size=template_sample_size,
                                                labels_per_series=n_labels_per_series)
    # Ideally may use a template for each series
    bin_distribution = cl.make_template_distribution(training_labels, topology.n_classification_bins)

    start_time = timer()
    if do_training:
        crocubot.train(topology, data_source=data_source, n_epochs=n_epochs, do_load_model=False,
                       bin_distribution=bin_distribution, n_passes=TRAIN_NUMBER_PASSES)
    else:
        nt.reset()
        nt.initialise_parameters(topology)

    mid_time = timer()
    train_time = mid_time - start_time
    print("Training complete.")

    metrics = evaluate_network(topology, data_source, bin_distribution)

    eval_time = timer() - mid_time
    print('Training took', str.format('{0:.2f}', train_time), "seconds")
    print('Evaluation took', str.format('{0:.2f}', eval_time), "seconds")
    if train_time > TIME_LIMIT:
        print('** Training took ', str.format('{0:.2f}', train_time - TIME_LIMIT), ' seconds too long - DISQUALIFIED! **')

    print('Metrics:')
    pt.print_performance_summary(metrics)


def evaluate_network(topology, data_source, bin_distribution):

    # Get the test data
    test_features, test_labels = io.load_test_samples(data_source=data_source)
    save_file = io.load_file_name(data_source, topology)

    binned_outputs = eval.eval_neural_net(test_features, topology, save_file)
    estimated_means, estimated_covariance = eval.forecast_means_and_variance(binned_outputs, bin_distribution)

    return pt.evaluate_sample_performance(truth=test_labels, estimation=estimated_means, test_name=data_source, estimated_covariance=estimated_covariance)


def load_default_topology(data_source):
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
    else:
        raise NotImplementedError

    return topo.Topology(n_series=n_input_series, n_features_per_series=n_features_per_series, n_forecasts=n_output_series,
                         n_classification_bins=n_classification_bins)


if __name__ == '__main__':

    # Data_source:  'low_noise' 'randomwalk' 'weightedwalk' 'correlatedwalk' 'stochasticwalk
    run_timed_performance_benchmark(data_source=DEFAULT_DATA_SOURCE, n_epochs=DEFAULT_N_EPOCHS, do_training=True)
