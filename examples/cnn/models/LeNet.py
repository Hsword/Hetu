import hetu as ht
from hetu import init


def conv_pool(x, in_channel, out_channel, name):
    weight = init.random_normal(
        shape=(out_channel, in_channel, 5, 5), stddev=0.1, name=name+'_weight')
    x = ht.conv2d_op(x, weight, padding=2, stride=1)
    x = ht.relu_op(x)
    x = ht.max_pool2d_op(x, kernel_H=2, kernel_W=2, padding=0, stride=2)
    return x


def fc(x, shape, name, with_relu=True):
    weight = init.random_normal(shape=shape, stddev=0.1, name=name+'_weight')
    bias = init.random_normal(shape=shape[-1:], stddev=0.1, name=name+'_bias')
    x = ht.matmul_op(x, weight)
    x = x + ht.broadcastto_op(bias, x)
    if with_relu:
        x = ht.relu_op(x)
    return x


def lenet(x, y_):
    '''
    LeNet model, for MNIST dataset.

    Parameters:
        x: Variable(hetu.gpu_ops.Node.Node), shape (N, dims)
        y_: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
    Return:
        loss: Variable(hetu.gpu_ops.Node.Node), shape (1,)
        y: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
    '''

    print('Building LeNet model...')
    x = ht.array_reshape_op(x, (-1, 1, 28, 28))
    x = conv_pool(x, 1,  6, name='lenet_conv1')
    x = conv_pool(x, 6, 16, name='lenet_conv2')
    x = ht.array_reshape_op(x, (-1, 7*7*16))
    x = fc(x, (7*7*16, 120), name='lenet_fc1', with_relu=True)
    x = fc(x, (120, 84), name='lenet_fc2', with_relu=True)
    y = fc(x, (84,  10), name='lenet_fc3', with_relu=False)
    loss = ht.softmaxcrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    return loss, y
