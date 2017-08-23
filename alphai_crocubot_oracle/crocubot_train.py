from timeit import default_timer as timer

import tensorflow as tf

import alphai_crocubot_oracle.network as nt
import alphai_crocubot_oracle.bayesian_cost as cost
import alphai_crocubot_oracle.tensormaths as tm
import alphai_crocubot_oracle.iotools as io
import alphai_crocubot_oracle.classifier as cl

FLAGS = tf.app.flags.FLAGS

DEFAULT_COST = 'bayes'
DEFAULT_TRY_TO_RESTORE = 'True'
DEFAULT_EPOCHS = 100
DEFAULT_NUMBER_OF_PASSES = 50
LEARNING_RATE = 3e-3

PRINT_LOSS_INTERVAL = 20
N_TRAIN_SAMPLES = 1000
BATCH_SIZE = 100

tf.app.flags.DEFINE_integer('num_eval_passes', 50, """Number of passes to average over.""")


def train(topology, data_source, cost_type=DEFAULT_COST, do_load_model=DEFAULT_TRY_TO_RESTORE,
          n_epochs=DEFAULT_EPOCHS, save_file_name=None, dtype=tm.DEFAULT_TF_TYPE, bin_distribution=None,
          n_passes=DEFAULT_NUMBER_OF_PASSES):
    """ Train network on either MNIST or time series data

    :param Topology topology:
    :param str data_source:
    :param str cost_type:
    :param do_load_model:
    :param n_epochs:
    :param save_file_name:
    :param dtype:
    :param bin_distribution:
    :param n_passes:
    :return:
    """

    # Start from a clean graph
    nt.reset()
    nt.initialise_parameters(topology)

    # Placeholders for the inputs and outputs of neural networks
    x = tf.placeholder(dtype, shape=[None, topology.n_features_per_series, topology.n_series])
    y = tf.placeholder(dtype)
    global_step = tf.Variable(0, trainable=False, name='global_step')

    cost_operator = set_cost_operator(x, y, topology, cost_type, number_of_passes=n_passes)
    training_operator = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost_operator, global_step=global_step)
    model_initialiser = tf.global_variables_initializer()

    n_batches = int(N_TRAIN_SAMPLES / BATCH_SIZE)

    saver = tf.train.Saver()
    if save_file_name is None:
        save_file_name = io.load_file_name(data_source, topology)

    # Launch the graph
    print("Launching Graph.")
    with tf.Session() as sess:

        if do_load_model:
            try:
                saver.restore(sess, save_file_name)
                print("Model restored.")
            except:
                sess.run(model_initialiser)
        else:
            sess.run(model_initialiser)

        for epoch in range(n_epochs):

            epoch_loss = 0.
            epoch_loss_list = []
            start_time = timer()

            for b in range(n_batches):  # The randomly sampled weights are fixed within single batch
                epoch_x, epoch_y = io.load_training_batch(data_source, batch_number=b, batch_size=BATCH_SIZE, labels_per_series=topology.n_classification_bins, dtype=dtype)
                if bin_distribution is not None:
                    epoch_y = cl.classify_labels(bin_distribution["bin_edges"], epoch_y)

                _, batch_loss = sess.run([training_operator, cost_operator], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += batch_loss
                nt.increment_noise_seed()

            time_epoch = timer() - start_time
            epoch_loss_list.append(epoch_loss)

            if (epoch % PRINT_LOSS_INTERVAL) == 0:
                print('Epoch', epoch, "loss:", str.format('{0:.2e}', epoch_loss), "in", str.format('{0:.2f}', time_epoch), "seconds")

        save_path = saver.save(sess, save_file_name)
        print("Model saved in file:", save_path)

    return topology, epoch_loss_list


def set_cost_operator(x, labels, topology, cost_type='bayes', number_of_passes=DEFAULT_NUMBER_OF_PASSES):
    cost_object = cost.BayesianCost(topology, FLAGS.double_gaussian_weights_prior, FLAGS.wide_prior_std,
                                    FLAGS.narrow_prior_std, FLAGS.spike_slab_weighting)
    predictions = nt.average_multiple_passes(x, number_of_passes, topology)

    if cost_type == 'bayes':
        operator = cost_object.get_bayesian_cost(predictions, labels)
    elif cost_type == 'hellinger':
        operator = cost.get_hellinger_cost(predictions, labels)
    elif cost_type == 'softmax':
        operator = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels)
    else:
        raise NotImplementedError

    return tf.reduce_mean(operator)
