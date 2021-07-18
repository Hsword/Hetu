import numpy as np
import tensorflow as tf


def conv_bn_relu(x, in_channel, out_channel):
    weight = tf.Variable(np.random.normal(scale=0.1, size=(
        out_channel, in_channel, 3, 3)).transpose([2, 3, 1, 0]).astype(np.float32))
    scale = tf.Variable(np.random.normal(
        scale=0.1, size=(out_channel,)).astype(np.float32))
    bias = tf.Variable(np.random.normal(
        scale=0.1, size=(out_channel,)).astype(np.float32))
    x = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')
    axis = list(range(len(x.shape) - 1))
    a_mean, a_var = tf.nn.moments(x, axis)
    x = tf.nn.batch_normalization(
        x, mean=a_mean, variance=a_var, scale=scale, offset=bias, variance_epsilon=1e-2)
    x = tf.nn.relu(x)
    return x


def vgg_2block(x, in_channel, out_channel):
    x = conv_bn_relu(x, in_channel, out_channel)
    x = conv_bn_relu(x, out_channel, out_channel)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[
                       1, 2, 2, 1], padding='VALID')
    return x


def vgg_3block(x, in_channel, out_channel):
    x = conv_bn_relu(x, in_channel, out_channel)
    x = conv_bn_relu(x, out_channel, out_channel)
    x = conv_bn_relu(x, out_channel, out_channel)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[
                       1, 2, 2, 1], padding='VALID')
    return x


def vgg_4block(x, in_channel, out_channel):
    x = conv_bn_relu(x, in_channel, out_channel)
    x = conv_bn_relu(x, out_channel, out_channel)
    x = conv_bn_relu(x, out_channel, out_channel)
    x = conv_bn_relu(x, out_channel, out_channel)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[
                       1, 2, 2, 1], padding='VALID')
    return x


def tf_fc(x, in_feat, out_feat):
    weight = tf.Variable(np.random.normal(
        scale=0.1, size=(in_feat, out_feat)).astype(np.float32))
    bias = tf.Variable(np.random.normal(
        scale=0.1, size=(out_feat,)).astype(np.float32))
    x = tf.matmul(x, weight) + bias
    return x


def tf_vgg(x, y_, num_layers, num_class=10):
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
    if num_layers == 16:
        print('Building VGG-16 model in tensorflow')
        x = vgg_2block(x,   3,  64)
        x = vgg_2block(x,  64, 128)
        x = vgg_3block(x, 128, 256)
        x = vgg_3block(x, 256, 512)
        x = vgg_3block(x, 512, 512)

    elif num_layers == 19:
        print('Building VGG-19 model in tensorflow')
        x = vgg_2block(x,   3,  64)
        x = vgg_2block(x,  64, 128)
        x = vgg_4block(x, 128, 256)
        x = vgg_4block(x, 256, 512)
        x = vgg_4block(x, 512, 512)
    else:
        assert False, "Number of layers should be 18 or 34 !"

    x = tf.reshape(x, [-1, 512])
    x = tf_fc(x,  512, 4096)
    x = tf_fc(x, 4096, 4096)
    y = tf_fc(x, 4096, num_class)
    print("Number of Class: {}".format(num_class))

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    loss = tf.reduce_mean(loss)
    return loss, y


def tf_vgg16(x, y_, num_class=10):
    return tf_vgg(x, y_, 16, num_class)


def tf_vgg19(x, y_, num_class=10):
    return tf_vgg(x, y_, 34, num_class)
