# Performance comparison: Adam vs SGD
import tensorflow as tf
import numpy as np
import logging

from examples.run_mnist import run_mnist_test

logger = logging.getLogger('tipper')
logger.addHandler(logging.StreamHandler())
logging.basicConfig(level=logging.DEBUG)

FLAGS = tf.app.flags.FLAGS
N_CYCLES = 1
NOISE_AMPLITUDE = 50           # Rms noise relative to rms signal
TRAIN_PASSES = [1, 10]         # [1, 4, 64]
DEFAULT_EVAL_PASSES = [1, 10]  # [1, 4, 64]
TF_LOG_PATH = '/tmp/'
TRAIN_PATH = '/mnt/pika/Networks/'
QUICK_TEST = False


def run_mnist_tests(optimisation_method, eval_passes=DEFAULT_EVAL_PASSES):

    accuracy_list = []
    likeli_array = np.zeros(N_CYCLES)
    config = build_config(optimisation_method)

    for train_pass in TRAIN_PASSES:
        config['n_train_passes'] = train_pass
        eval_time, accuracy = run_mnist_test(config)
        accuracy_list.extend(accuracy)

    print(optimisation_method, 'accuracy:', accuracy_list)
    # print(optimisation_method, 'likelihoods:', likeli_array)
    accuracy_array = np.asarray(accuracy_list)
    print('Mean accuracy:', np.mean(accuracy_array))
    # print('Log likelihood:', np.mean(likeli_array))


def build_config(optimisation_method):
    config={}
    config["train_path"] = TRAIN_PATH
    config["tensorboard_log_path"] = TF_LOG_PATH
    config["optimisation_method"] = optimisation_method
    config["use_convolution"] = True
    config["quick_test"] = QUICK_TEST
    config["multi_eval_passes"] = DEFAULT_EVAL_PASSES
    config['noise_amplitude'] = NOISE_AMPLITUDE
    return config

opt_methods = ['GDO', 'Adam']  # GDO Adam

for method in opt_methods:
    run_mnist_tests(method)


# NOISE 50 RESULTS: GDO/ ADAM for (1 t 1 e, 1 t 10 e, 10 t 1 e, 10 t 10 e)
  #   Adam accuracy: [0.1135, 0.1135, 0.0974, 0.089700000000000002]
   # GDO accuracy: [0.126, 0.1298, 0.1464, 0.15049999999999999]
  # Repeat with 400 batch size:
  #  Adam accuracy: [0.1135, 0.1135, 0.095799999999999996, 0.085900000000000004] Mean accuracy: 0.102175
   # GDO accuracy: [0.12959999999999999, 0.1283, 0.21340000000000001, 0.23250000000000001] Mean accuracy: 0.17595


# NOISE 10 RESULTS: GDO/ ADAM for train / eval of 1 and 10 each
 #   GDO accuracy: [0.76029999999999998, 0.76090000000000002, 0.74570000000000003, 0.74260000000000004]
   # Mean accuracy: 0.752375
 #   Adam accuracy: [0.73939999999999995, 0.7399, 0.5978, 0.75339999999999996]
  # Mean accuracy: 0.707625
# 10 epoch test:
    # Adam accuracy: [0.80969999999999998, 0.79649999999999999, 0.65920000000000001, 0.67649999999999999]
#  Mean accuracy: 0.735475
    # GDO GDO accuracy: [0.98250000000000004, 0.98089999999999999, 0.9829, 0.98450000000000004]
   #    Mean accuracy: 0.9827. Best performer was 4 train ,4 eval


# Latest 10 epoch test results: cycled via TRAIN_PASSES = [1, 4, 64] and inside each triplet was  EVAL_PASSES = [1, 4, 64]
    # Adam
    # accuracy: [0.98319999999999996, 0.98099999999999998, 0.98229999999999995, 0.97860000000000003, 0.97919999999999996,
    #            0.97889999999999999, 0.98099999999999998, 0.98219999999999996, 0.98199999999999998]

# GDO accuracy: [0.97950000000000004, 0.97860000000000003, 0.97919999999999996, 0.9839, 0.98450000000000004, 0.98450000000000004, 0.98460000000000003, 0.98299999999999998, 0.98399999999999999]

# Now try adjusting prior for multipass:
# GDO accuracy: [0.98240000000000005, 0.98209999999999997, 0.98209999999999997, 0.97950000000000004, 0.98250000000000004,   0.98080000000000001, 0.97989999999999999, 0.98060000000000003, 0.98050000000000004]
# Mean accuracy: 0.981155555556
    # best performance: 1, 1

# Check reproducibililty:
# GDO accuracy: [0.97940000000000005, 0.97840000000000005, 0.97940000000000005, 0.97909999999999997, 0.98029999999999995, 0.97919999999999996, 0.98140000000000005, 0.97999999999999998, 0.98119999999999996]
# Mean accuracy: 0.979822222222
    # best performance: 64 train, 1 eval
# Likelihood comparison: -4.71,

#  Adam accuracy: [0.98380000000000001, 0.9829, 0.98370000000000002, 0.97929999999999995, 0.98080000000000001, 0.98060000000000003, 0.9819, 0.98129999999999995, 0.98260000000000003]
#  Mean accuracy: 0.981877777778,
    # best performance: 1 train, 1 eval, twice got 0.984, beating GDO. Min probability assigned to unsuccessful forecast: 5.10601864789e-07

# more passes: 0.9819, min prob 2.36592532019e-05
# bb alpha cost, 32 passes:  (train time: 484 )   accuracy: 0.9806

# Latest 100 epoch test results: cycled via TRAIN_PASSES = [1, 4, 64] and inside each triplet was  EVAL_PASSES = [1, 4, 64]
# KNOWN BUG 14/07/17: randomisation of different passes isn'ae working
#  GDO accuracy: [0.99219999999999997, 0.99219999999999997, 0.99219999999999997, 0.99270000000000003, 0.99270000000000003, 0.99270000000000003, 0.99219999999999997, 0.99219999999999997, 0.99219999999999997]
#  Mean accuracy: 0.992366666667
# Adam accuracy: [0.9929, 0.9929, 0.9929, 0.99299999999999999, 0.99299999999999999, 0.99299999999999999, 0.99309999999999998, 0.99309999999999998, 0.99309999999999998]
#  Mean accuracy: 0.993


#     OLDER TEST RESULTS (no batch normalisation etc)
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