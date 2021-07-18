import numpy as np
import tensorflow as tf


def tf_conv2d(x, in_channel, out_channel, stride=1):
    weight = tf.Variable(np.random.normal(scale=0.1, size=(
        out_channel, in_channel, 3, 3)).transpose([2, 3, 1, 0]).astype(np.float32))
    x = tf.nn.conv2d(x, weight, strides=[1, stride, stride, 1], padding='SAME')
    return x


def tf_batch_norm_with_relu(x, hidden):
    scale = tf.Variable(np.random.normal(
        scale=0.1, size=(hidden,)).astype(np.float32))
    bias = tf.Variable(np.random.normal(
        scale=0.1, size=(hidden,)).astype(np.float32))
    axis = list(range(len(x.shape) - 1))
    a_mean, a_var = tf.nn.moments(x, axis)
    x = tf.nn.batch_normalization(
        x, mean=a_mean, variance=a_var, scale=scale, offset=bias, variance_epsilon=1e-2)
    x = tf.nn.relu(x)
    return x


def tf_resnet_block(x, in_channel, num_blocks, is_first=False):
    if is_first:
        out_channel = in_channel
        identity = x
        x = tf_conv2d(x, in_channel, out_channel, stride=1)
        x = tf_batch_norm_with_relu(x, out_channel)
        x = tf_conv2d(x, out_channel, out_channel, stride=1)
        x = x + identity
    else:
        out_channel = 2 * in_channel
        identity = x
        x = tf_batch_norm_with_relu(x, in_channel)
        x = tf_conv2d(x, in_channel, out_channel, stride=2)
        x = tf_batch_norm_with_relu(x, out_channel)
        x = tf_conv2d(x, out_channel, out_channel, stride=1)
        identity = tf.nn.avg_pool(identity, ksize=[1, 2, 2, 1], strides=[
                                  1, 2, 2, 1], padding='VALID')
        identity = tf.pad(identity, [[0, 0], [0, 0], [0, 0], [
                          in_channel // 2, in_channel // 2]])
        x = x + identity

    for i in range(1, num_blocks):
        identity = x
        x = tf_batch_norm_with_relu(x, out_channel)
        x = tf_conv2d(x, out_channel, out_channel, stride=1)
        x = tf_batch_norm_with_relu(x, out_channel)
        x = tf_conv2d(x, out_channel, out_channel, stride=1)
        x = x + identity

    return x


def tf_fc(x, shape):
    weight = tf.Variable(np.random.normal(
        scale=0.1, size=shape).astype(np.float32))
    bias = tf.Variable(np.random.normal(
        scale=0.1, size=shape[-1:]).astype(np.float32))
    x = tf.matmul(x, weight) + bias
    return x


def tf_resnet(x, y_, num_layers, num_class=10):
    '''
    ResNet model in TensorFlow, for CIFAR10 dataset.

    Parameters:
        x: Variable(tensorflow.python.framework.ops.Tensor), shape (N, H, W, C)
        y_: Variable(tensorflow.python.framework.ops.Tensor), shape (N, num_classes)
        num_layers: 18 or 34
    Return:
        loss: Variable(tensorflow.python.framework.ops.Tensor), shape (1,)
        y: Variable(tensorflow.python.framework.ops.Tensor), shape (N, num_classes)
    '''
    print("Number of Class: {}".format(num_class))
    base_size = 16

    x = tf_conv2d(x, 3, base_size, stride=1)
    x = tf_batch_norm_with_relu(x, base_size)

    if num_layers == 18:
        print("Building ResNet-18 model in tensorflow...")
        x = tf_resnet_block(x,     base_size, num_blocks=2, is_first=True)
        x = tf_resnet_block(x,     base_size, num_blocks=2)
        x = tf_resnet_block(x, 2 * base_size, num_blocks=2)
        x = tf_resnet_block(x, 4 * base_size, num_blocks=2)
    elif num_layers == 34:
        print("Building ResNet-34 model in tensorflow...")
        x = tf_resnet_block(x,     base_size, num_blocks=3, is_first=True)
        x = tf_resnet_block(x,     base_size, num_blocks=4)
        x = tf_resnet_block(x, 2 * base_size, num_blocks=6)
        x = tf_resnet_block(x, 4 * base_size, num_blocks=3)
    else:
        assert False, "Number of layers should be 18 or 34 !"

    x = tf_batch_norm_with_relu(x, 8 * base_size)
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, [-1, 128 * base_size])
    y = tf_fc(x, (128 * base_size, num_class))
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    loss = tf.reduce_mean(loss)
    return loss, y


def tf_resnet18(x, y_, num_class=10):
    return tf_resnet(x, y_, 18, num_class)


def tf_resnet34(x, y_, num_class=10):
    return tf_resnet(x, y_, 34, num_class)
