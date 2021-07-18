import numpy as np
import tensorflow as tf


def tf_fc(x, shape, with_relu=True):
    weight = tf.Variable(np.random.normal(
        scale=0.1, size=shape).astype(np.float32))
    bias = tf.Variable(np.random.normal(
        scale=0.1, size=shape[-1:]).astype(np.float32))
    x = tf.matmul(x, weight) + bias
    if with_relu:
        x = tf.nn.relu(x)
    return x


def tf_mlp(x, y_, num_class=10):
    '''
    MLP model in TensorFlow, for CIFAR dataset.

    Parameters:
        x: Variable(tensorflow.python.framework.ops.Tensor), shape (N, dims)
        y_: Variable(tensorflow.python.framework.ops.Tensor), shape (N, num_classes)
    Return:
        loss: Variable(tensorflow.python.framework.ops.Tensor), shape (1,)
        y: Variable(tensorflow.python.framework.ops.Tensor), shape (N, num_classes)
    '''

    print("Building MLP model in tensorflow...")
    x = tf_fc(x, (3072, 256), with_relu=True)
    x = tf_fc(x, (256, 256), with_relu=True)
    y = tf_fc(x, (256, num_class), with_relu=False)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    loss = tf.reduce_mean(loss)
    return loss, y
