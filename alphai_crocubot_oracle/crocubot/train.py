# Trains the network
# Used by oracle.py


import logging
from timeit import default_timer as timer

import tensorflow as tf

import alphai_crocubot_oracle.bayesian_cost as cost
from alphai_crocubot_oracle.crocubot.model import CrocuBotModel, Estimator
import alphai_crocubot_oracle.iotools as io

FLAGS = tf.app.flags.FLAGS
PRINT_LOSS_INTERVAL = 1
PRINT_SUMMARY_INTERVAL = 5


def train(topology, data_source, train_x=None, train_y=None, bin_edges=None, save_path=None, restore_path=None):
    """ Train network on either MNIST or time series data

    :param Topology topology:
    :param str data_source:
    :return: epoch_loss_list
    """

    if topology.n_parameters > 1e7:
        logging.warning("Ambitious number of parameters: {}".format(topology.n_parameters))
    else:
        logging.info("Number of parameters: {}".format(topology.n_parameters))

    # Start from a clean graph
    tf.reset_default_graph()
    model = CrocuBotModel(topology, FLAGS)
    model.build_layers_variables()

    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    for var in tf.trainable_variables():   # Add histograms for trainable variables
        summaries.append(tf.summary.histogram(var.op.name, var))

    use_data_loader = True if train_x is None else False

    # Placeholders for the inputs and outputs of neural networks
    x = tf.placeholder(FLAGS.d_type, shape=[None, topology.n_features_per_series, topology.n_series])
    y = tf.placeholder(FLAGS.d_type)
    global_step = tf.Variable(0, trainable=False, name='global_step')

    n_batches = int(FLAGS.n_training_samples / FLAGS.batch_size)
    cost_operator = _set_cost_operator(model, x, y, n_batches)
    training_operator = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cost_operator, global_step=global_step)
    summary_op = tf.summary.merge(summaries)

    model_initialiser = tf.global_variables_initializer()

    if save_path is None:
        save_path = io.load_file_name(data_source, topology)
    saver = tf.train.Saver()

    # Launch the graph
    logging.info("Launching Graph.")
    with tf.Session() as sess:

        if restore_path is not None:
            try:
                logging.info("Attempting to load model from {}".format(restore_path))
                saver.restore(sess, restore_path)
                logging.info("Model restored.")
                n_epochs = FLAGS.n_retrain_epochs
            except:
                logging.warning("Restore file not recovered. Training from scratch")
                n_epochs = FLAGS.n_epochs
                sess.run(model_initialiser)
        else:
            logging.info("Initialising new model.")
            n_epochs = FLAGS.n_epochs
            sess.run(model_initialiser)

        summary_writer = tf.summary.FileWriter(
            FLAGS.tensorboard_log_path,
            graph=sess.graph)

        epoch_loss_list = []
        for epoch in range(n_epochs):

            epoch_loss = 0.
            start_time = timer()
            logging.info("Training epoch {} of {}".format(epoch, n_epochs))

            for batch_number in range(n_batches):  # The randomly sampled weights are fixed within single batch

                if use_data_loader:
                    batch_x, batch_y = io.load_training_batch(data_source, batch_number=batch_number,
                                                              batch_size=FLAGS.batch_size, bin_edges=bin_edges)
                else:
                    lo_index = batch_number * FLAGS.batch_size
                    hi_index = lo_index + FLAGS.batch_size
                    batch_x = train_x[lo_index:hi_index, :]
                    batch_y = train_y[lo_index:hi_index, :]

                if batch_number == 0 and epoch == 0:
                    logging.info("Training {} batches of size {} and {}".format(n_batches, batch_x.shape, batch_y.shape))

                _, batch_loss = sess.run([training_operator, cost_operator], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += batch_loss

            time_epoch = timer() - start_time
            epoch_loss_list.append(epoch_loss)

            if (epoch % PRINT_LOSS_INTERVAL) == 0:
                msg = 'Epoch ' + str(epoch) + " loss:" + str.format('{0:.2e}', epoch_loss) + " in " + str.format('{0:.2f}', time_epoch) + " seconds"
                logging.info(msg)

            if (epoch % PRINT_SUMMARY_INTERVAL) == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, epoch)

        out_path = saver.save(sess, save_path)
        logging.info("Model saved in file:{}".format(out_path))

    return epoch_loss_list


def _set_cost_operator(crocubot_model, x, labels, n_batches):
    """
    Set the cost operator

    :param CrocubotModel crocubot_model:
    :param data x:
    :param labels:
    :return:
    """

    cost_object = cost.BayesianCost(crocubot_model,
                                    FLAGS.double_gaussian_weights_prior,
                                    FLAGS.wide_prior_std,
                                    FLAGS.narrow_prior_std,
                                    FLAGS.spike_slab_weighting,
                                    n_batches
                                    )

    estimator = Estimator(crocubot_model, FLAGS)
    predictions, _ = estimator.average_multiple_passes(x, FLAGS.n_train_passes)

    if FLAGS.cost_type == 'bayes':
        operator = cost_object.get_bayesian_cost(predictions, labels)
    elif FLAGS.cost_type == 'hellinger':
        operator = cost.get_hellinger_cost(predictions, labels)
    elif FLAGS.cost_type == 'softmax':
        operator = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels)
    else:
        raise NotImplementedError

    return tf.reduce_mean(operator)
