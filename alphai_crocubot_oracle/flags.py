import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def default():

    default_config = load_default_config()
    set_training_flags(default_config)

def set_training_flags(config):
    """ Assigns flags based on entries in dictionary"""

    tf.app.flags.DEFINE_string('graph_save_path', config['graph_save_path'], """Path to save graph.""")
    tf.app.flags.DEFINE_string('d_type', config['D_TYPE'], """Data type for numpy.""")
    tf.app.flags.DEFINE_integer('TF_TYPE', config['TF_TYPE'], """Data type for tensorflow.""")
    tf.app.flags.DEFINE_integer('random_seed', 0, """Seed used to identify random noise realisiation.""")

    # Training specific
    tf.app.flags.DEFINE_integer('n_epochs', config['n_epochs'], """How many epochs to be used for training.""")
    tf.app.flags.DEFINE_integer('n_training_samples', config['n_training_samples'], """Total number of data samples to be used for training.""")
    tf.app.flags.DEFINE_float('learning_rate', config['learning_rate'], """Total number of data samples to be used for training.""")
    tf.app.flags.DEFINE_integer('batch_size', config['batch_size'], """Total number of data samples to be used for training.""")
    tf.app.flags.DEFINE_string('cost_type', config['cost_type'], """Total number of data samples to be used for training.""")
    tf.app.flags.DEFINE_integer('n_train_passes', 50, """Number of passes to average over during training.""")
    tf.app.flags.DEFINE_integer('n_eval_passes', 100, """Number of passes to average over during evaluation.""")

    # Initial conditions
    tf.app.flags.DEFINE_float('INITIAL_ALPHA', config['INITIAL_ALPHA'], """Prior on weights.""")
    tf.app.flags.DEFINE_float('INITIAL_WEIGHT_UNCERTAINTY', config['INITIAL_WEIGHT_UNCERTAINTY'], """Initial standard deviation on weights.""")
    tf.app.flags.DEFINE_float('INITIAL_BIAS_UNCERTAINTY', config['INITIAL_BIAS_UNCERTAINTY'], """Initial standard deviation on bias.""")
    tf.app.flags.DEFINE_float('INITIAL_WEIGHT_DISPLACEMENT', config['INITIAL_WEIGHT_DISPLACEMENT'], """Initial offset on weight distributions.""")
    tf.app.flags.DEFINE_float('INITIAL_BIAS_DISPLACEMENT', config['INITIAL_BIAS_DISPLACEMENT'], """Initial offset on bias distributions.""")
    tf.app.flags.DEFINE_boolean('USE_PERFECT_NOISE', config['USE_PERFECT_NOISE'],
                                """Whether to set noise such that its mean and std are exactly the desired values""")

    # Priors
    tf.app.flags.DEFINE_boolean('double_gaussian_weights_prior', config['double_gaussian_weights_prior'],
                                """Whether to impose a double Gaussian prior.""")
    tf.app.flags.DEFINE_float('wide_prior_std', config['wide_prior_std'], """Initial standard deviation on weights.""")
    tf.app.flags.DEFINE_float('narrow_prior_std', config['narrow_prior_std'], """Initial standard deviation on weights.""")
    tf.app.flags.DEFINE_float('spike_slab_weighting', config['spike_slab_weighting'], """Initial standard deviation on weights.""")
    FLAGS._parse_flags()


def load_default_config():

    config = {}
    config['graph_save_path'] = '/tmp/crocubot/'
    config['d_type'] =  'float32'
    config['TF_TYPE'] =  32
    config['random_seed']= 0

    # Training specific
    config['n_epochs'] = 1
    config['n_training_samples'] = 1000
    config['learning_rate'] = 2e-3
    config['batch_size'] = 100
    config['cost_type'] = 'bayes'
    config['n_train_passes'] = 30
    config['n_eval_passes'] = 100

    # Initial conditions
    config['INITIAL_ALPHA'] = 0.2
    config['INITIAL_WEIGHT_UNCERTAINTY'] = 0.4
    config['INITIAL_BIAS_UNCERTAINTY'] = 0.4
    config['INITIAL_WEIGHT_DISPLACEMENT'] = 0.1
    config['INITIAL_BIAS_DISPLACEMENT']= 0.4
    config['USE_PERFECT_NOISE'] = True,

    # Priors
    config['double_gaussian_weights_prior'] = False
    config['wide_prior_std'] = 1.2
    config['narrow_prior_std'] = 0.05
    config['spike_slab_weighting'] = 0.5

    return config
