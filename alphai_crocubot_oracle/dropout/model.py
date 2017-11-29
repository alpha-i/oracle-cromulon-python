import tensorflow as tf
import tensorflow.contrib.slim as slim


def dropout_net(inputs, is_training, scope='dropout_net'):
    with tf.variable_scope(scope, 'dropout_net'):
        # First Group: Convolution + Pooling 28x28x1 => 28x28x20 => 14x14x20
        net = slim.conv2d(inputs, 20, [5, 5], padding='SAME', scope='layer1-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')

        # Second Group: Convolution + Pooling 14x14x20 => 10x10x40 => 5x5x40
        net = slim.conv2d(net, 40, [5, 5], padding='VALID', scope='layer3-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer4-max-pool')

        # Reshape: 5x5x40 => 1000x1
        net = tf.reshape(net, [-1, 5*5*40])

        # Fully Connected Layer: 1000x1 => 1000x1
        net = slim.fully_connected(net, 1000, scope='layer5')
        net = slim.dropout(net, is_training=is_training, scope='layer5-dropout')

        # Second Fully Connected: 1000x1 => 1000x1
        net = slim.fully_connected(net, 1000, scope='layer6')
        net = slim.dropout(net, is_training=is_training, scope='layer6-dropout')

        # Output Layer: 1000x1 => 10x1
        net = slim.fully_connected(net, 10, scope='output')
        net = slim.dropout(net, is_training=is_training, scope='output-dropout')

        return net
