import hetu as ht
from hetu import init


def conv_relu_avg(x, shape):
    weight = init.random_normal(shape=shape, stddev=0.1)
    x = ht.conv2d_op(x, weight, padding=2, stride=1)
    x = ht.relu_op(x)
    x = ht.avg_pool2d_op(x, kernel_H=2, kernel_W=2, padding=0, stride=2)
    return x


def fc(x, shape):
    weight = init.random_normal(shape=shape, stddev=0.1)
    bias = init.random_normal(shape=shape[-1:], stddev=0.1)
    x = ht.array_reshape_op(x, (-1, shape[0]))
    x = ht.matmul_op(x, weight)
    y = x + ht.broadcastto_op(bias, x)
    return y


def cnn_3_layers(x, y_):
    '''
    3-layer-CNN model, for MNIST dataset.

    Parameters:
        x: Variable(hetu.gpu_ops.Node.Node), shape (N, dims)
        y_: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
    Return:
        loss: Variable(hetu.gpu_ops.Node.Node), shape (1,)
        y: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
    '''

    print('Building 3-layer-CNN model...')
    x = ht.array_reshape_op(x, [-1, 1, 28, 28])
    x = conv_relu_avg(x, (32, 1, 5, 5))
    x = conv_relu_avg(x, (64, 32, 5, 5))
    y = fc(x, (7 * 7 * 64, 10))
    loss = ht.softmaxcrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    return loss, y
