import tensorflow as tf
import logging

from alphai_crocubot_oracle import flags as fl
from examples.benchmark.mnist import run_timed_benchmark_mnist
from examples.helpers import load_default_config, FLAGS


def run_mnist_test(train_path, tensorboard_log_path, method='GDO', use_full_train_set=True, do_convolution=True,
    quick_test=True):

    config = load_default_config()
    if quick_test:
        config["n_epochs"] = 10  # 98.91 after 10 epochs and only 6 layers
        config["learning_rate"] = 1e-3   # Use high learning rate for testing purposes
    else:
        config["n_epochs"] = 100  # Scored 98.99% after 100 epochs; 98.5 after 10
        config["learning_rate"] = 1e-3   # 1e-3 gest 98.95  in 10 epochs; 99.08 after 100; n_layers=10
        # 21 layer res network. 10 epoch: 98.86; 100 epoch: 99.21%

    config["cost_type"] = 'bayes'  # 'bayes'; 'softmax'; 'bbalpha'
    config['batch_size'] = 200
    config['n_series'] = 1
    config['optimisation_method'] = method
    config['n_features_per_series'] = 784
    config['resume_training'] = False  # Make sure we start from scratch
    config['activation_functions'] = ['linear', 'relu', 'relu']
    config['layer_types'] = ['conv3d', 'conv3d', 'full']
    config['tensorboard_log_path'] = tensorboard_log_path
    config['train_path'] = train_path
    config['model_save_path'] = train_path
    config['n_retrain_epochs'] = 5
    config['n_train_passes'] = 1
    config['n_eval_passes'] = 50
    config['use_convolution'] = do_convolution

    fl.build_tensorflow_flags(config)

    # this flag is only used in benchmark.
    tf.app.flags.DEFINE_integer('n_training_samples_benchmark', 60000, """Number of samples for benchmarking.""")
    tf.app.flags.DEFINE_integer('n_prediction_sample', 10000, """Number of samples for benchmarking.""")

    FLAGS._parse_flags()
    print("Epochs to evaluate:", FLAGS.n_epochs)

    run_timed_benchmark_mnist("mnist_reshaped", FLAGS, do_training=True)


if __name__ == '__main__':

    logger = logging.getLogger('tipper')
    logger.addHandler(logging.StreamHandler())
    logging.basicConfig(level=logging.DEBUG)

    # change the following lines according to your machine
    train_path = '/tmp/'
    tensorboard_log_path = '/tmp/'

    do_quick_test = True

    run_mnist_test(train_path, tensorboard_log_path,  use_full_train_set=True, quick_test=do_quick_test)
