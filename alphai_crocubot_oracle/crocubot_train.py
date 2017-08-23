from timeit import default_timer as timer

import tensorflow as tf

import alphai_crocubot_oracle.crocubot_model as cr
import alphai_crocubot_oracle.bayesian_cost as cost
import alphai_crocubot_oracle.iotools as io

FLAGS = tf.app.flags.FLAGS
PRINT_LOSS_INTERVAL = 20


def train(topology, data_source, train_x=None, train_y=None, bin_edges=None, save_path=None):
    """ Train network on either MNIST or time series data

    :param Topology topology:
    :param str data_source:
    :return: epoch_loss_list
    """

    # Start from a clean graph
    cr.reset()
    cr.initialise_parameters(topology)

    if train_x is None:
        use_data_loader = True
    else:
        use_data_loader = False

    # Placeholders for the inputs and outputs of neural networks
    x = tf.placeholder(FLAGS.d_type, shape=[None, topology.n_features_per_series, topology.n_series])
    y = tf.placeholder(FLAGS.d_type)
    global_step = tf.Variable(0, trainable=False, name='global_step')

    cost_operator = _set_cost_operator(x, y, topology)
    training_operator = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cost_operator, global_step=global_step)
    model_initialiser = tf.global_variables_initializer()

    n_batches = int(N_TRAIN_SAMPLES / FLAGS.batch_size)
    if save_path is None:
        save_path = io.load_file_name(data_source, topology)
    saver = tf.train.Saver()

    # Launch the graph
    print("Launching Graph.")
    with tf.Session() as sess:

        if FLAGS.resume_training:
            try:
                saver.restore(sess, save_path)
                print("Model restored.")
            except:
                print("Previous save file not found. Training from scratch")
                sess.run(model_initialiser)
        else:
            sess.run(model_initialiser)

        for epoch in range(FLAGS.n_epochs):

            epoch_loss = 0.
            epoch_loss_list = []
            start_time = timer()

            for b in range(n_batches):  # The randomly sampled weights are fixed within single batch
                if use_data_loader:
                    batch_x, batch_y = io.load_training_batch(data_source, batch_number=b, batch_size=FLAGS.batch_size, labels_per_series=topology.n_classification_bins, bin_edges=bin_edges)
                else:
                    lo_index = b*FLAGS.batch_size
                    hi_index = lo_index + FLAGS.batch_size
                    batch_x = train_x[lo_index:hi_index, :]
                    batch_y = train_y[lo_index:hi_index, :]

                _, batch_loss = sess.run([training_operator, cost_operator], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += batch_loss
                cr.increment_noise_seed()

            time_epoch = timer() - start_time
            epoch_loss_list.append(epoch_loss)

            if (epoch % PRINT_LOSS_INTERVAL) == 0:
                print('Epoch', epoch, "loss:", str.format('{0:.2e}', epoch_loss), "in", str.format('{0:.2f}', time_epoch), "seconds")

        saver.save(sess, save_path)
        print("Model saved in file:", save_path)

    return epoch_loss_list


def set_cost_operator(x, labels, topology, cost_type='bayes', number_of_passes=DEFAULT_NUMBER_OF_PASSES):

    cost_object = cost.BayesianCost(topology, FLAGS.double_gaussian_weights_prior, FLAGS.wide_prior_std,
                                    FLAGS.narrow_prior_std, FLAGS.spike_slab_weighting)
    predictions = cr.average_multiple_passes(x, FLAGS.n_train_passes, topology)

    if FLAGS.cost_type == 'bayes':
        operator = cost_object.get_bayesian_cost(predictions, labels)
    elif FLAGS.cost_type == 'hellinger':
        operator = cost.get_hellinger_cost(predictions, labels)
    elif FLAGS.cost_type == 'softmax':
        operator = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels)
    else:
        raise NotImplementedError

    return tf.reduce_mean(operator)
