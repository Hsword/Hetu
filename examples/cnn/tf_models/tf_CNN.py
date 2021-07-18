import numpy as np
import tensorflow as tf


def tf_conv_relu_avg(x, shape):
    weight = tf.Variable(np.random.normal(
        scale=0.1, size=shape).transpose([2, 3, 1, 0]).astype(np.float32))
    x = tf.nn.conv2d(x, weight, padding='SAME', strides=[1, 1, 1, 1])
    x = tf.nn.relu(x)
    x = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                       padding='VALID', strides=[1, 2, 2, 1])
    return x


def tf_fc(x, shape):
    weight = tf.Variable(np.random.normal(
        scale=0.1, size=shape).astype(np.float32))
    bias = tf.Variable(np.random.normal(
        scale=0.1, size=shape[-1:]).astype(np.float32))
    x = tf.reshape(x, (-1, shape[0]))
    y = tf.matmul(x, weight) + bias
    return y


def tf_cnn_3_layers(x, y_):
    '''
    3-layer-CNN model in TensorFlow, for MNIST dataset.

    Parameters:
        x: Variable(tensorflow.python.framework.ops.Tensor), shape (N, dims)
        y_: Variable(tensorflow.python.framework.ops.Tensor), shape (N, num_classes)
    Return:
        loss: Variable(tensorflow.python.framework.ops.Tensor), shape (1,)
        y: Variable(tensorflow.python.framework.ops.Tensor), shape (N, num_classes)
    '''

    print('Building 3-layer-CNN model in tensorflow...')
    x = tf.reshape(x, [-1, 28, 28, 1])
    x = tf_conv_relu_avg(x, (32, 1, 5, 5))
    x = tf_conv_relu_avg(x, (64, 32, 5, 5))
    x = tf.transpose(x, [0, 3, 1, 2])
    y = tf_fc(x, (7 * 7 * 64, 10))
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    loss = tf.reduce_mean(loss)
    return loss, y
