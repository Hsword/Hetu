import numpy as np
import tensorflow as tf


def tf_conv_pool(x, in_channel, out_channel):
    weight = tf.Variable(np.random.normal(scale=0.1, size=(
        out_channel, in_channel, 5, 5)).transpose([2, 3, 1, 0]).astype(np.float32))
    x = tf.nn.conv2d(x, weight, padding='SAME', strides=[1, 1, 1, 1])
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                       padding='VALID', strides=[1, 2, 2, 1])
    return x


def tf_fc(x, shape, with_relu=True):
    weight = tf.Variable(np.random.normal(
        scale=0.1, size=shape).astype(np.float32))
    bias = tf.Variable(np.random.normal(
        scale=0.1, size=shape[-1:]).astype(np.float32))
    x = tf.matmul(x, weight) + bias
    if with_relu:
        x = tf.nn.relu(x)
    return x


def tf_lenet(x, y_):
    '''
    LeNet model in TensorFlow, for MNIST dataset.

    Parameters:
        x: Variable(tensorflow.python.framework.ops.Tensor), shape (N, dims)
        y_: Variable(tensorflow.python.framework.ops.Tensor), shape (N, num_classes)
    Return:
        loss: Variable(tensorflow.python.framework.ops.Tensor), shape (1,)
        y: Variable(tensorflow.python.framework.ops.Tensor), shape (N, num_classes)
    '''

    print('Building LeNet model in tensorflow...')
    x = tf.reshape(x, [-1, 28, 28, 1])
    x = tf_conv_pool(x, 1,  6)
    x = tf_conv_pool(x, 6, 16)
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, (-1, 7*7*16))
    x = tf_fc(x, (7*7*16, 120), with_relu=True)
    x = tf_fc(x, (120, 84), with_relu=True)
    y = tf_fc(x, (84,  10), with_relu=False)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    loss = tf.reduce_mean(loss)
    return loss, y
