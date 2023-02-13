import hetu as ht
from hetu import init


def fc(x, shape, name, with_relu=True):
    weight = init.random_normal(shape=shape, stddev=0.1, name=name+'_weight')
    bias = init.random_normal(shape=shape[-1:], stddev=0.1, name=name+'_bias')
    x = ht.matmul_op(x, weight)
    x = x + ht.broadcastto_op(bias, x)
    if with_relu:
        x = ht.relu_op(x)
    return x


def mlp(x, y_):
    '''
    MLP model, for MNIST dataset.

    Parameters:
        x: Variable(hetu.gpu_ops.Node.Node), shape (N, dims)
        y_: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
    Return:
        loss: Variable(hetu.gpu_ops.Node.Node), shape (1,)
        y: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
    '''

    print("Building MLP model...")
    x = fc(x, (784, 256), 'mlp_fc1', with_relu=True)
    x = fc(x, (256, 256), 'mlp_fc2', with_relu=True)
    y = fc(x, (256, 10), 'mlp_fc3', with_relu=False)
    loss = ht.softmaxcrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    return loss, y
