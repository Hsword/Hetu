import numpy as np
import scipy
import hetu as ht


def softmax_func(y):
    """Numerically stable softmax."""
    b = y - np.max(y, axis=1, keepdims=True)
    expb = np.exp(b)
    softmax = expb / np.sum(expb, axis=1, keepdims=True)
    return softmax


def test_csrmm_op(executor_ctx):
    X = ht.Variable(name="X")
    W = ht.Variable(name="W")
    Y = ht.csrmm_op(X, W)
    Y_ = ht.Variable(name="Y_")
    loss = ht.softmaxcrossentropy_op(Y, Y_)
    loss = ht.reduce_mean_op(loss, [0])
    grads = ht.gradients(loss, [W, Y])

    executor = ht.Executor(
        [loss, grads[0], grads[1]], ctx=executor_ctx)

    rand = np.random.RandomState(seed=123)

    W_val = rand.normal(scale=0.1, size=[70000, 2]).astype(np.float32)
    if ht.is_gpu_ctx(executor_ctx):
        W_val = ht.array(W_val, ctx=executor_ctx)

    X_val = scipy.sparse.rand(500, 70000, density=1e-5,
                              format='coo', dtype=np.float32)
    Y_val = np.random.uniform(0, 10, size=(500, 2)).astype(np.float32)

    loss_val = executor.run(feed_dict={X: X_val, Y_: Y_val, W: W_val})

    if ht.is_gpu_ctx(executor_ctx):
        W_val = W_val.asnumpy()
    loss_val = [val.asnumpy() for val in loss_val]

    y_groundtruth = X_val.dot(W_val)
    loss_groundtruth = np.mean(
        -np.sum(Y_val * np.log(softmax_func(y_groundtruth)), axis=1), keepdims=True)
    Y_grad_groundtruth = (softmax_func(y_groundtruth) + -
                          1 * Y_val) * np.ones(loss_groundtruth.shape) / 500
    W_grad_groundtruth = X_val.T.dot(Y_grad_groundtruth)

    np.testing.assert_allclose(loss_val[0], loss_groundtruth, rtol=1e-4)
    np.testing.assert_allclose(loss_val[1], W_grad_groundtruth, rtol=1e-4)
    np.testing.assert_allclose(loss_val[2], Y_grad_groundtruth, rtol=1e-4)


test_csrmm_op(ht.cpu(0))
test_csrmm_op(ht.gpu(1))


def test_csrmv_op(executor_ctx):
    X = ht.Variable(name="X")
    W = ht.Variable(name="W")
    Y = ht.csrmv_op(X, W)
    Y_ = ht.Variable(name="Y_")
    temp = Y + (-1) * Y_
    loss = temp * temp

    grads = ht.gradients(loss, [W, Y])

    executor = ht.Executor(
        [loss, grads[0], grads[1]], ctx=executor_ctx)

    rand = np.random.RandomState(seed=123)

    W_val = rand.normal(scale=0.1, size=[70000, ])
    if ht.is_gpu_ctx(executor_ctx):
        W_val = ht.array(W_val, ctx=executor_ctx)

    X_val = scipy.sparse.rand(500, 70000, density=1e-5,
                              format='coo', dtype=np.float32)
    Y_val = np.random.uniform(0, 10, size=(500, )).astype(np.float32)

    loss_val = executor.run(feed_dict={X: X_val, Y_: Y_val, W: W_val})

    if ht.is_gpu_ctx(executor_ctx):
        W_val = W_val.asnumpy()
    loss_val = [val.asnumpy() for val in loss_val]

    y_groundtruth = X_val.dot(W_val)
    loss_groundtruth = (y_groundtruth - Y_val) ** 2
    Y_grad_groundtruth = 2 * (y_groundtruth - Y_val) * \
        np.ones(loss_groundtruth.shape)
    W_grad_groundtruth = X_val.T.dot(Y_grad_groundtruth)

    np.testing.assert_allclose(loss_val[0], loss_groundtruth, rtol=1e-4)
    np.testing.assert_allclose(loss_val[1], W_grad_groundtruth, rtol=1e-4)
    np.testing.assert_allclose(loss_val[2], Y_grad_groundtruth, rtol=1e-4)


test_csrmv_op(ht.cpu(0))
test_csrmv_op(ht.gpu(1))
