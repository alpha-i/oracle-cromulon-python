
import logging


from examples.benchmark.mnist import run_timed_benchmark_mnist
from examples.helpers import load_default_config, FLAGS
from examples.benchmark_flags import set_benchmark_flags

MNIST_RESHAPED = "mnist_reshaped"


def run_mnist_test(update_config):

    config = load_default_config()

    do_quick_test = update_config.get('quick_test', True)
    if do_quick_test:
        config["n_epochs"] = 10  # 98.91 after 10 epochs and only 6 layers
        config["learning_rate"] = 1e-3   # Use high learning rate for testing purposes
    else:
        config["n_epochs"] = 100  # Scored 98.99% after 100 epochs; 98.5 after 10
        config["learning_rate"] = 1e-4   # 1e-3 gest 98.95  in 10 epochs; 99.08 after 100; n_layers=10
        # 21 layer res network. 10 epoch: 98.86; 100 epoch: 99.21%
        # 400 noise test: 1e-4 gets 28.23 % after 100 epochs
        # 1e-4 gets 11.35 after 10; 10
        # 1e-3 got 11; 11 again after 10; after 100: still 11. :(
        # 1e-5 got 4% acc after 100!!!
        # 30 layer res monster: 1e-4:

        # Batch size of 200 and 1e-3 after 100:
    config["cost_type"] = 'bayes'  # 'bayes'; 'softmax'; 'bbalpha'
    config['batch_size'] = 400
    config['do_batch_norm'] = False
    config['n_series'] = 1
    config['n_features_per_series'] = 784
    config['resume_training'] = False  # Make sure we start from scratch
    config['tensorboard_log_path'] = '/tmp/'
    config['train_path'] = '/tmp/'
    config['model_save_path'] = '/tmp/'
    config['n_retrain_epochs'] = 0
    config['n_eval_passes'] = 1
    config['apply_temporal_suppression'] = False
    config.update(update_config)
    set_benchmark_flags(config)

    multi_eval_passes = config.get('multi_eval_passes', None)
    eval_time, accuracy = run_timed_benchmark_mnist(MNIST_RESHAPED, FLAGS, True, config, multi_eval_passes)

    return eval_time, accuracy


if __name__ == '__main__':

    logger = logging.getLogger('tipper')
    logger.addHandler(logging.StreamHandler())
    logging.basicConfig(level=logging.DEBUG)

    # change the following lines according to your machine
    train_path = '/tmp/'
    tensorboard_log_path = '/tmp/'

    do_quick_test = False

    run_mnist_test(train_path, tensorboard_log_path,  use_full_train_set=True, quick_test=do_quick_test)


    #  With batch norm now get  99.31%  with  9 layers and 6 eval per passes (25 min runtime)
    #  Repeat for 12 eval per passes:
    #  Repeat for 1 eval per passes:

    # Repeat for noisy mnist:
    # 1 eval per pass:
    # 6 eval per pass:
    # 12 eval per pass:

