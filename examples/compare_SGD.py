# Performance comparison: Adam vs SGD
import tensorflow as tf
import numpy as np
import logging

from examples.run_mnist import run_mnist_test

logger = logging.getLogger('tipper')
logger.addHandler(logging.StreamHandler())
logging.basicConfig(level=logging.DEBUG)

FLAGS = tf.app.flags.FLAGS
N_CYCLES = 1   # 20
NOISE_AMPLITUDE = 0
TRAIN_PASSES = [8]
DEFAULT_EVAL_PASSES = [8]
DEFAULT_RANDOM_SEED = 42

# One res block holds 2 convolutional layers and a skipped connection
N_RES_BLOCKS = [2]  # AlphaGo Zero uses 40 blocks
N_BAYES_LAYERS = 1
OPT_METHODS = ['Adam']  # GDO Adam: Adam performs better in noisy domain perhaps due to effectively large batch size
N_NETWORKS = 1
TF_LOG_PATH = '/tmp/'
TRAIN_PATH = '/mnt/pika/Networks/'
SAVE_FILE = '/mnt/pika/MNIST/mnist_results.txt'
ADAM_FILE = '/mnt/pika/MNIST/adam_results.txt'
QUICK_TEST = False


def run_mnist_tests():

    for method in OPT_METHODS:
        accuracy_list = []
        config = build_config(method)
        for n_blocks in N_RES_BLOCKS:
            config['n_res_blocks'] = n_blocks
            for train_pass in TRAIN_PASSES:
                config['n_train_passes'] = train_pass
                temp_acc_list = []
                seed = DEFAULT_RANDOM_SEED
                for i in range(N_CYCLES):
                    np.random.seed(seed)
                    config['random_seed'] = seed
                    eval_time, accuracy = run_mnist_test(config)
                    temp_acc_list.extend(accuracy)
                    seed += 1

                average_accuracy = np.mean(np.asarray(temp_acc_list))
                accuracy_list.append(average_accuracy)

        accuracy_array = np.asarray(accuracy_list)
        print(method, 'accuracy:', accuracy_list)
        print('Mean accuracy:', np.mean(accuracy_array))

        if method == 'Adam':
            filename = ADAM_FILE
        else:
            filename = SAVE_FILE

        with open(filename, 'w') as f:
            print(method, 'accuracy:', accuracy_list, file=f)
            print('Mean accuracy:', np.mean(accuracy_array), file=f)


def build_config(optimisation_method):

    config={}
    config["train_path"] = TRAIN_PATH
    config["tensorboard_log_path"] = TF_LOG_PATH
    config["optimisation_method"] = optimisation_method
    config["use_convolution"] = True
    config["quick_test"] = QUICK_TEST
    config["multi_eval_passes"] = DEFAULT_EVAL_PASSES
    config['noise_amplitude'] = NOISE_AMPLITUDE
    config['n_networks'] = N_NETWORKS
    return config


run_mnist_tests()

# Cromulon results:
## NOISE-FREE
# 98.93 4 blocks; 1 pass; 100 epoch; 32 kernels
# 99.22 6 blocks; 8 pass; 100 epoch; 32 kernels
