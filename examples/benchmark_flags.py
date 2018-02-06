import tensorflow as tf

from alphai_cromulon_oracle import flags as fl
from examples.helpers import FLAGS

def set_benchmark_flags(config):

    fl.build_tensorflow_flags(config)

    # this flag is only used in benchmark.
    tf.app.flags.DEFINE_integer('n_training_samples_benchmark', 60000, """Number of samples for benchmarking.""")
    tf.app.flags.DEFINE_integer('n_prediction_sample', 10000, """Number of samples for benchmarking.""")

    FLAGS._parse_flags()
