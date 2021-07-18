import numpy as np
import tensorflow as tf


def tf_logreg(x, y_):
    '''
    Logistic Regression model in TensorFlow, for MNIST dataset.

    Parameters:
        x: Variable(tensorflow.python.framework.ops.Tensor), shape (N, dims)
        y_: Variable(tensorflow.python.framework.ops.Tensor), shape (N, num_classes)
    Return:
        loss: Variable(tensorflow.python.framework.ops.Tensor), shape (1,)
        y: Variable(tensorflow.python.framework.ops.Tensor), shape (N, num_classes)
    '''

    print("Build logistic regression model in tensorflow...")
    weight = tf.Variable(np.zeros(shape=(784, 10)).astype(np.float32))
    bias = tf.Variable(np.zeros(shape=(10, )).astype(np.float32))
    y = tf.matmul(x, weight) + bias
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    loss = tf.reduce_mean(loss)
    return loss, y
