# Performance comparison: Adam vs SGD
import tensorflow as tf
import numpy as np
import logging

from examples.run_mnist import run_mnist_test

logger = logging.getLogger('tipper')
logger.addHandler(logging.StreamHandler())
logging.basicConfig(level=logging.DEBUG)

FLAGS = tf.app.flags.FLAGS
N_CYCLES = 5   # 20
NOISE_AMPLITUDE = 0 # 800  # Rms noise relative to rms signal. 54% achieved on 400 with 64 train passes
TRAIN_PASSES = [64]  # 8 works well [1, 4, 16, 64] # Big influence
DEFAULT_EVAL_PASSES = [64]  # [1, 4, 16, 64]
DEFAULT_RANDOM_SEED = 42

N_RES_BLOCKS = [9]  # [4, 9, 11, 21] # Big Influence. 9, 11 fail at 800 noise, 128 passes
N_BAYES_LAYERS = 1
OPT_METHODS = ['Adam']  # GDO Adam: Adam performs better in noisy domain perhaps due to effectively large batch size
N_NETWORKS = 1
TF_LOG_PATH = '/tmp/'
TRAIN_PATH = '/mnt/pika/Networks/'
SAVE_FILE = '/mnt/pika/MNIST/mnist_results.txt'
ADAM_FILE = '/mnt/pika/MNIST/adam_results.txt'
QUICK_TEST = False

# RESULTS FOR 800 NOISE AMP  20 epoch average.
# Updated prior from 0.8 to 1 and spike-slab from 0.5 to 0.7

# RESULTS FOR 800 NOISE AMP  20 epoch average.
# suppressed priors:
# bayesian cost:  Predicted: 6772 of 10000 in bin 1 of 10  ( 17.93 % accuracy)
# bayesian cost:  Predicted: 7266 of 10000                 (14.24 %)
#   Predicted: 4914 of 10000 predictions fell in bin 1 of 10    (14.36)
#  :   Predicted :5876 of 10000 predictions fell in bin 1 of 10  (17.88)
#   Predicted : 4357 of 10000 predictions fell in bin 1 of 10  ( 22.47 %)
#    Predicted : 6941 of 10000 predictions fell in bin 1 of 10   17.96 %
#   Predicted : 5549 of 10000 predictions fell in bin 1 of 10  (17 %)
#   Predicted : 3880 of 10000 predictions fell in bin 1 of 10. (18.62 % )
# Predicted : 4626 of 10000 predictions fell in bin 1 of 10. (20.77 % )
#  Predicted : 4000 of 10000 predictions fell in bin 1 of 10. (14.08 % )
# 1e4 entropic cost: 44 of 400; 11%! terrible..
# 1 entropic cost: 5445 of 10000 predictions fell in bin 0, 7.68 %
# 1e-3 entropic cost:
# 1e-6 entropic cost: 6319 of 10000 predictions fell in bin 1 of 10  15.09 %



# RESULTS FOR 800 NOISE AMP. 3 cycle, 10 epoch average.
# no supp priors:
# bayesian cost: 12.3
# entropic cost: 12.9 (max 18) # 12.3

# WITH batch norm:
# suppressed priors:
# bayesian cost:
# entropic cost:  10.9;   # 10000, so more than we will take

# no supp priors:
# bayesian cost: 10.45
# entropic cost: 10.6

# WITH no conv batch norm:
# bayesian cost: 12.9
# entropic cost: 17 (!) # repeat with 128: 12; 25 sec per epoch; 14sec per epoch at 64 passes.
# but then not reproducible :/get 11 % twice over! weird!

# WITH no transition batch norm:
# bayesian cost:
# entropic cost: 10.8

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
        # print('Log likelihood:', np.mean(likeli_array))

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

# Noise 400 w res layer

# NOISE 60 RESULTS; 9 layer; 500 epoch
    #
    # MNIST accuracy of  37.94 % @ 1, 1.

# NOISE 40 RESULTS 4 layer; 100 epoch; t = [1, 4, 10] ; e = [1, 4, 10]:
#     Adam accuracy: [0.69140000000000001, 0.20880000000000001, 0.24709999999999999, 0.47260000000000002, 0.46860000000000002,
#                0.4677, 0.29580000000000001, 0.52890000000000004, 0.64200000000000002]
#     GDO accuracy: [0.14630000000000001, 0.15040000000000001, 0.14680000000000001, 0.64539999999999997, 0.66180000000000005,
#                0.66159999999999997, 0.1101, 0.1069, 0.1047]
    # Best: Adam with 1/1
# REPEAT:
#     GDO  accuracy: [0.115, 0.1154, 0.1145, 0.60680000000000001, 0.60270000000000001, 0.59619999999999995,
#               0.65190000000000003, 0.64790000000000003, 0.64459999999999995]    Mean    accuracy: 0.455
#    Adam accuracy: [0.69099999999999995, 0.12720000000000001, 0.1283, 0.69210000000000005, 0.7036, 0.69410000000000005,
#               0.21990000000000001, 0.36509999999999998, 0.5968]    Mean    accuracy: 0.468677777778
# Best: Adam with 4/4;  1/1 2nd best
# REPEAT:
#    GDO accuracy: [0.2266, 0.20430000000000001, 0.21299999999999999, 0.21929999999999999, 0.22919999999999999, 0.2235,
#               0.1368, 0.13669999999999999, 0.1391] Mean accuracy: 0.192055555556
#    Adam    accuracy: [0.65469999999999995, 0.098000000000000004, 0.18859999999999999, 0.64300000000000002, 0.63590000000000002,
#               0.63360000000000005, 0.33279999999999998, 0.50229999999999997, 0.45200000000000001]    Mean    accuracy: 0.4601
# Best:  Adam with 1/1

# Finally repeat with fresh noise each time (should kill the 1/1 performance but may help higher loops):
#     Adam accuracy: [0.1135, 0.1135, 0.1135, 0.68510000000000004, 0.69520000000000004, 0.68310000000000004,
#                0.44359999999999999, 0.4783, 0.46329999999999999] Mean  accuracy: 0.421011111111
#    GDO    accuracy: [0.1215, 0.1239, 0.1232, 0.4173, 0.45989999999999998, 0.47970000000000002, 0.68089999999999995,
#               0.68240000000000001, 0.68400000000000005]    Mean    accuracy: 0.4192
# Best:  Adam with 4/4 but interestingly GDO flies up. Going to retry with t=50, e=50, GDO:
# 9 seconds per epoch. 0.209. Batch size reduced:


# NOISE 20 RESULTS 4 layer; 100 epoch; t = [1, 4, 64] ; e = [1, 4, 64]
   # Adam accuracy: [0.6825, 0.6825, 0.6825, 0.72599999999999998, 0.72599999999999998, 0.72599999999999998,
               #0.59640000000000004, 0.59640000000000004, 0.59640000000000004]
   # GDO  accuracy: [0.64610000000000001, 0.64610000000000001, 0.64610000000000001, 0.66190000000000004, 0.66190000000000004,
               #0.66190000000000004, 0.67000000000000004, 0.67000000000000004, 0.67000000000000004]
# Best performer: Adam on 4, 4, but not much in it

# NOISE 40 RESULTS 4 layer; 100 epoch; t = [1, 4, 10] ; e = [1, 4, 10]
# GDO accuracy: [0.2359, 0.2359, 0.2359, 0.10290000000000001, 0.10290000000000001, 0.10290000000000001, 0.1135, 0.1135, 0.1135]
#   Mean  accuracy: 0.150766666667
#   Adam: [0.65839999999999999, 0.65839999999999999, 0.65839999999999999, 0.6633, 0.6633, 0.6633, 0.37830000000000003, 0.37830000000000003, 0.37830000000000003]
    # #  Mean: 0.566666666667
    # Best performance: Adam 4,4, GDO nowhere
    # FIXME Why is multieval not working?! same answer in each case. need to reassign flags? now fixed.
    # Training took 259.67 seconds for 10 passes
    # now repeat for parallel=1: 280.04

# OLDER TEST RESULTS (no batch normalisation etc)
# Results: 20 epoch
# Adam: [ 0.9216  0.9214  0.9189  0.9232  0.9224] -> mean 0.9215
# GDO:  [ 0.9244  0.9243  0.9263  0.9243  0.9249] -> mean 0.9248

# Results: 200 epoch
# Adam accuracy: [ 0.9714  0.9709  0.9701  0.9725  0.9731]
# GDO:  [ 0.9703  0.968   0.9691  0.969   0.9694] ->  Mean accuracy: 0.9691

# Results: 20 epoch with gradient clipping (40 eval pass, 1 train, lr = 2e-3)
# Adam: [ 0.9753  0.9768  0.9771  0.977   0.9771] -> Mean 0.97666
# GDO:  0.9797  0.981   0.9825  0.9812  0.9828] -> Mean 0.98144

# Results: 20 epoch with gradient clipping AND extra cost term (40 eval pass, 1 train, lr = 3e-3)
# Adam:  0.9772  0.9768  0.9718  0.9794  0.978  -> Mean 0.97664
# GDO: 0.9778  0.9816  0.9798  0.9801  0.981 ] -> Mean 0.98006

# Results: 20 epoch with gradient clipping AND big extra cost term (40 eval pass, 1 train, lr = 3e-3)
# Adam:  [ 0.9767  0.9733  0.9737  0.9748  0.9766] -> 0.97502
# GDO: [ 0.9825  0.9835  0.9828  0.9838  0.9824] -> 0.983  ****

# Results: 200 epoch with gradient clipping AND big extra cost term (40 eval pass, 1 train, lr = 1e-3)
# Adam: [ 0.9796
# GDO:  [ 0.9831  0.982   0.9831  0.9835  0.9827] -> 0.98288

# Results: 200 epoch with gradient clipping AND big extra cost term (40 eval pass, 1 train, lr = 1e-3)
# Repeated to get likelihoods
# GDO accuracy: [ 0.9824  0.9834  0.9834  0.9853  0.9832] -> 0.983
# GDO likelihoods: [-0.06876133 -0.06876963 -0.06817838 -0.063859   -0.06839153] -> -0.0676

# Results: 200 epoch with gradient clipping AND NO big extra cost term (40 eval pass, 1 train, lr = 1e-3)
# GDO accuracy: [0.9787  0.9792  0.9781  0.9793  0.9767]
# GDO likelihoods: [-0.10307571 -0.09865605 -0.10417016 -0.09949302 -0.11244867] -> 0.1035

# Results: 200 epoch with new full extra cost term (40 eval pass, 1 train, lr = 1e-3)
# GDO Accuracy:
#  [0.9829  0.9836  0.9836  0.9821  0.9824  0.9828  0.9827  0.9831  0.9824      0.9836] -> 0.98292
#    likelihoods: [-0.0707785 - 0.06766419 - 0.06848358 - 0.07242996 - 0.0704145 - 0.07105065      - 0.07299981 - 0.07197466 - 0.069243 - 0.06973318]
# -> -0.070477


# Results: 200 epoch no extra cost term but new initialisation (40 eval pass, 1 train, lr = 1e-3)
# GDO [ 0.981   0.9814  0.9808  0.9801  0.9817] -> 0.981
    # Likeli -0.074672919459
# Adam: