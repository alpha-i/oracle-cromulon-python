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
TRAIN_PASSES = [64]
DEFAULT_EVAL_PASSES = [64]
DEFAULT_RANDOM_SEED = 42
BATCH_NORM = False

# One res block holds 2 convolutional layers and a skipped connection
N_RES_BLOCKS = [20]  # AlphaGo Zero uses 40 blocks
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
    config['do_batch_norm'] = BATCH_NORM  # Suspect this still isnt working correctly
    return config

run_mnist_tests()

# Cromulon results:
##  NOISE-FREE
# OLD log probability signal
# 98.93 4 blocks; 1 pass; 100 epoch; 32 kernels
# 99.22; 99.0; LL= -0.0528; 6 blocks; 8 pass; 100 epoch; 32 kernels. Eval time:
# Median probability assigned to true outcome: 1.0
# Mean probability assigned to forecasts: 0.9897998851909904
# Mean probability assigned to successful forecast: 0.9982460123035122
# Mean probability assigned to unsuccessful forecast: 0.1275112584352944
# Min probability assigned to unsuccessful forecast: 2.5215371240658313e-13

# Now with linear output:
# 98.00; 98.78; LL -0.119   6 blocks; 8 pass; 100 epoch; 32 kernels; batch norm  Eval time: 16 sec/epoch
# 99.18 ; 0.99.17 LL -0.042  6 blocks; 8 pass; 100 epoch; 32 kernels; no batch norm  Eval time: 12 sec/epoch
# test: 99% with 2 blocks / no batch norm. With batch norm now get: 98 lol
# 98.91; -0.07 w gdo and batch norm
# 98.59; 20 blocks w adam and batch norm; 60 sec eval
# New updated head:  98.64; -0.07  20 blocks w adam and batch norm; 40 sec eval; 100 epoch
# another 100 epoch: 98.79; -0.05

# Median probability assigned to true outcome: 1.0
# Mean probability assigned to forecasts: 0.9885175551935091
# Mean probability assigned to successful forecast: 0.9974925602686172
# Mean probability assigned to unsuccessful forecast: 0.15870703923047016
# Min probability assigned to unsuccessful forecast: 1.1362009830163597e-13

# Now Experimenting with large eval passes:
# 98.9; 98.81  at 1
# 98.9; 98.96; 98.93; 98.95 at 8; 2 sec eval
# 98.99; 98.97; 99.0  at 128
# 98.99; 98.98; 98.99  at 512; 10 sec eval


## NOISE 400
#
