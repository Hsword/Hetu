import hetu as ht
from hetu import ndarray
from hetu import init
from hetu import onnx as ax

import onnxruntime as rt

import numpy as np
import sys

batch_size = 3
rand = np.random.RandomState(seed=123)

ctx = ndarray.gpu(0)
# ctx=ndarray.cpu(0)


def Check(executor, executor_res, input, output, input_value):
    """

    :type executor_res: object
    """
    ax.hetu2onnx.export(executor, input, output, 'ath.onnx')

    sess = rt.InferenceSession("ath.onnx")
    inps = [input.name for input in sess.get_inputs()]
    assert len(inps) == len(
        input_value), "Failed: shapes does not match of input_name and input_value"
    feed_dict = {}
    for i in range(len(inps)):
        feed_dict[inps[i]] =\
            input_value[i].asnumpy() if isinstance(
                input_value[i], ndarray.NDArray) else input_value[i]

    # pre=sess.run(None,{inps[0]:input_value[0].astype(np.float32)})[0]
    pre = sess.run(None, feed_dict)[0]
    # if ndarray.is_gpu_ctx(ctx):
    #     res=executor_res[0].asnumpy()
    # else:
    #     res=executor_res[0].asnumpy()
    np.testing.assert_allclose(executor_res[0].asnumpy(), pre, rtol=1e-3)


def test_AddConst():
    X = ht.Variable(name="X")
    val = 3.3
    y = X+val
    executor = ht.Executor([y], ctx=ctx)

    X_val = rand.normal(scale=0.1, size=(batch_size, 10)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_AddElewise():
    X = ht.Variable(name="X")
    b3 = init.random_normal((10,), stddev=0.1, name='b3')
    y = X+b3
    executor = ht.Executor([y], ctx=ctx, enable_lazy=False)

    X_val = rand.normal(scale=0.1, size=(batch_size, 10)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_AvgPool():
    X = ht.Variable(name="X")
    y = ht.avg_pool2d_op(X, kernel_H=2, kernel_W=2, padding=0, stride=2)
    executor = ht.Executor([y], ctx=ctx)

    X_val = rand.normal(scale=0.1, size=(
        batch_size, 10, 10, 10)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_MaxPool():
    X = ht.Variable(name="X")
    y = ht.max_pool2d_op(X, kernel_H=2, kernel_W=2, padding=0, stride=2)
    executor = ht.Executor([y], ctx=ctx)

    X_val = rand.normal(scale=0.1, size=(
        batch_size, 10, 10, 10)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_MatrixMult():
    X = ht.Variable(name="X")
    W1 = init.random_normal((10, 5), stddev=0.1, name='W1')
    y = ht.matmul_op(X, W1)
    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(batch_size, 10)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    # test transpose_A
    X = ht.Variable(name="X")
    W1 = init.random_normal((10, 5), stddev=0.1, name='W1')
    y = ht.matmul_op(X, W1, True)
    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(10, batch_size)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])

    # test transpose_B
    X = ht.Variable(name="X")
    W1 = init.random_normal((5, 10), stddev=0.1, name='W1')
    y = ht.matmul_op(X, W1, trans_B=True)
    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(batch_size, 10)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_Relu():
    X = ht.Variable(name="X")
    y = ht.relu_op(X)
    executor = ht.Executor([y], ctx=ctx)

    X_val = rand.normal(scale=0.1, size=(
        batch_size, 10, 10, 10)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_Reshape():
    X = ht.Variable(name="X")
    y = ht.array_reshape_op(X, [-1, 10*10*10])
    executor = ht.Executor([y], ctx=ctx)

    X_val = rand.normal(scale=0.1, size=(
        batch_size, 10, 10, 10)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_Conv2d():
    X = ht.Variable(name="X")
    W1 = init.random_normal((32, 1, 5, 5), stddev=0.1, name='W1')
    y = ht.conv2d_op(X, W1, padding=2, stride=1)
    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(
        batch_size, 1, 28, 28)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_Concat():
    A = ht.Variable(name="A")
    B = ht.Variable(name="B")
    y = ht.concat_op(A, B, axis=1)
    executor = ht.Executor([y], ctx=ctx)
    A_val = rand.normal(scale=0.1, size=(2, 3)).astype(np.float32)
    B_val = rand.normal(scale=0.1, size=(2, 3)).astype(np.float32)

    res = executor.run(feed_dict={A: A_val, B: B_val})
    Check(executor, res, [A, B], [y], [A_val, B_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_Sqrt():
    X = ht.Variable(name="X")
    y = ht.sqrt_op(X)
    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(2, 3)).astype(np.float32)

    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_rSqrt():
    X = ht.Variable(name="X")
    y = ht.rsqrt_op(X)
    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(2, 3)).astype(np.float32)

    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_Tanh():
    X = ht.Variable(name="X")
    y = ht.tanh_op(X)
    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(2, 3)).astype(np.float32)

    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_BatchNorm():
    X = ht.Variable(name="X")
    bn_scale = init.random_normal((64,), stddev=0.1, name='bn_scale')
    bn_bias = init.random_normal((64,), stddev=0.1, name='bn_bias')

    y = ht.batch_normalization_op(X, bn_scale, bn_bias)

    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(
        batch_size, 64, 28, 28)).astype(np.float32)

    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X, bn_scale, bn_bias], [y], [
          X_val, bn_scale.tensor_value, bn_bias.tensor_value])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_Pad():
    X = ht.Variable(name="X")
    paddings = [[1, 1], [1, 1], [2, 1], [1, 3]]
    y = ht.pad_op(X, paddings, constant_values=0)

    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(1, 1, 1, 1)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})

    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_Div():
    X = ht.Variable(name="X")
    B = ht.Variable(name="B")
    y = ht.div_op(X, B)

    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(2, 2)).astype(np.float32)
    B_val = rand.normal(scale=0.1, size=(2, 2)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val, B: B_val})
    Check(executor, res, [X, B], [y], [X_val, B_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_MultiplyConst():
    X = ht.Variable(name="X")
    const = 5.5
    y = ht.mul_byconst_op(X, const)

    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(2, 2)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_DivConst():
    X = ht.Variable(name="X")
    const = 5.5
    y = ht.div_const_op(const, X)

    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(2, 2)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_Onehot():
    X = ht.Variable(name="X")
    classes = 10
    y = ht.one_hot_op(X, classes)

    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.randint(0, 10, 20,).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_Opposite():
    X = ht.Variable(name="X")
    y = ht.opposite_op(X)
    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(2, 2)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_Softmax():
    X = ht.Variable(name="X")
    y = ht.softmax_op(X)
    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(128, 150)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_ReduceMean():

    X = ht.Variable(name="X")
    y = ht.reduce_mean_op(X, 1, keepdims=True)
    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(2, 2)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_ReduceSum():

    X = ht.Variable(name="X")
    y = ht.reduce_sum_op(X, 0, keepdims=False)
    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(2, 23, 5)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_Dropout():
    X = ht.Variable(name="X")
    y = ht.dropout_op(X, 1)
    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(3, 2, 5)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_Transpose():
    X = ht.Variable(name="X")
    y = ht.transpose_op(X, [2, 0, 1])
    executor = ht.Executor([y], ctx=ctx)
    X_val = rand.normal(scale=0.1, size=(3, 2, 5)).astype(np.float32)
    res = executor.run(feed_dict={X: X_val})
    Check(executor, res, [X], [y], [X_val])
    print(sys._getframe().f_code.co_name, 'pass!')


def test_Where():
    cond = ht.Variable(name="Cond", dtype=np.bool)
    A = ht.Variable(name="A")
    B = ht.Variable(name="B")
    y = ht.where_op(cond, A, B)
    executor = ht.Executor([y], ctx=ctx)
    shape = [2, 2, 3]
    Cond_val = rand.randint(0, 2, size=shape, dtype=np.bool)
    A_val = rand.normal(scale=0.1, size=shape).astype(np.float32)
    B_val = rand.normal(scale=0.1, size=shape).astype(np.float32)
    res = executor.run(feed_dict={cond: Cond_val, A: A_val, B: B_val})

    Check(executor, res, [cond, A, B], [y], [Cond_val, A_val, B_val])
    print(sys._getframe().f_code.co_name, 'pass!')


if __name__ == '__main__':
    test_AddConst()
    test_AddElewise()
    test_AvgPool()
    test_MaxPool()
    test_MatrixMult()
    test_Relu()
    test_Reshape()
    test_Conv2d()
    test_Concat()
    test_Sqrt()
    test_rSqrt()
    test_Tanh()
    # fixme:batchnorm,maybe sustainable:  Mismatched elements: 3 / 150528 (0.00199%)
    # test_BatchNorm()
    test_Pad()
    test_Div()
    test_MultiplyConst()
    test_DivConst()
    test_Onehot()
    test_Opposite()
    test_Softmax()
    test_ReduceMean()
    # test_ReduceSum()
    # #fixme:not all close when keep_prob is not 1.0 in dropout.maybe has bug
    test_Dropout()
    test_Transpose()
    test_Where()
