import hetu as ht
from hetu import init


def logreg(x, y_):
    '''
    Logistic Regression model, for MNIST dataset.

    Parameters:
        x: Variable(hetu.gpu_ops.Node.Node), shape (N, dims)
        y_: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
    Return:
        loss: Variable(hetu.gpu_ops.Node.Node), shape (1,)
        y: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
    '''

    print("Build logistic regression model...")
    weight = init.zeros((784, 10), name='logreg_weight')
    bias = init.zeros((10,), name='logreg_bias')
    x = ht.matmul_op(x, weight)
    y = x + ht.broadcastto_op(bias, x)
    loss = ht.softmaxcrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    return loss, y
