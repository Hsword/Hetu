import numpy as np
import time
import ctypes
from hetu import cpu_links as cpu_op
from hetu import ndarray
from hetu.ndarray import numpyasdlarrayhandle


def save_to_file(data, file):
    f = open(file, 'a+')
    f.write(data)
    f.close()


#     0    1    2    3    4    5   6    7     8    9   10   11   12     13    14    15    16   17    18     19
ll = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900,
      1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]


def test_boradcast_to():
    for i in range(len(ll)):
        # ctx = ndarray.cpu(0)
        shape = (ll[i], ll[i])
        to_shape = (1000, ll[i], ll[i])
        x = np.random.uniform(-1, 1, shape).astype(np.float32)
        y = np.empty(to_shape, dtype=np.float32)
        arr_x = numpyasdlarrayhandle(x)
        arr_y = numpyasdlarrayhandle(y)
        # print(arr_x.asnumpy())
        start = time.time()
        for _ in range(10):
            cpu_op.broadcast_to(arr_x, arr_y)
        end = time.time()
        for _ in range(10):
            kkk = np.broadcast_to(x, to_shape)
        end1 = time.time()
        print(ll[i], " cpu:", end - start, "  ", "numpy:", end1 - end)
        np.testing.assert_allclose(kkk, y, rtol=1e-5)


# test_boradcast_to()

def test_reduce_sum_axis_zero():
    for i in range(len(ll)):
        a = 1
        #
        # shape = (ll[i], ll[i])
        # to_shape = (1000, ll[i], ll[i])
        # x = np.random.uniform(-1, 1, shape).astype(np.float32)
        # y=np.empty(to_shape,dtype=np.float32)
        # arr_x  = numpyasdlarrayhandle(x)
        # arr_y = numpyasdlarrayhandle(y)
        # # print(arr_x.asnumpy())
        # start = time.time()
        # for _ in range(10):
        #     cpu_op.broadcast_to(arr_x, arr_y)
        # end = time.time()
        # for _ in range(10):
        #     kkk = np.broadcast_to(x, to_shape)
        # end1 = time.time()
        # print(ll[i], " cpu:", end - start, "  ", "numpy:", end1 - end)
        # np.testing.assert_allclose(kkk, y, rtol=1e-5)
    shape = (2, 2, 2)
    to_shape = (2, 2)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    y = np.empty(to_shape, dtype=np.float32)
    arr_x = numpyasdlarrayhandle(x)
    arr_y = numpyasdlarrayhandle(y)
    cpu_op.reduce_sum_axis_zero(arr_x, arr_y)
    np_y = np.sum(x, axis=0)
    print('x:', x)
    print('np_y:', np_y)
    print('y:', y)

# test_reduce_sum_axis_zero()


def test_average_pooling():
    ctx = ndarray.cpu(0)

    def np_average_pooling(input, kernel_H, kernel_W, padding=0, stride=1):
        N, C, H, W = input.shape
        assert ((H + 2 * padding - kernel_H) % stride == 0)
        assert ((W + 2 * padding - kernel_W) % stride == 0)
        pooled_H = (H + 2 * padding - kernel_H) / stride + 1
        pooled_W = (W + 2 * padding - kernel_W) / stride + 1
        pooled_layer = np.zeros(
            shape=(N, C, pooled_H, pooled_W), dtype=np.float32)
        pooling_size = kernel_H * kernel_W
        for n in xrange(N):
            for c in xrange(C):
                for h in xrange(pooled_H):
                    for w in xrange(pooled_W):
                        hs = h * stride - padding
                        ws = w * stride - padding
                        hend = min(hs + kernel_H, H)
                        wend = min(ws + kernel_W, W)
                        hs = max(hs, 0)
                        ws = max(ws, 0)
                        for i in xrange(hs, hend):
                            for j in xrange(ws, wend):
                                pooled_layer[n][c][h][w] += input[n][c][i][j]
                        pooled_layer[n][c][h][w] /= pooling_size
        return pooled_layer

    def np_average_pooling_gradient(gradient_y, kernel_H, kernel_W, padding=0, stride=1):
        N, C, pooled_H, pooled_W = gradient_y.shape
        H = (pooled_H - 1) * stride + kernel_H - 2 * padding
        W = (pooled_W - 1) * stride + kernel_W - 2 * padding

        gradient_x = np.zeros(shape=(N, C, H, W), dtype=np.float32)
        pooling_size = kernel_H * kernel_W
        for n in xrange(N):
            for c in xrange(C):
                for h in xrange(pooled_H):
                    for w in xrange(pooled_W):
                        hs = h * stride - padding
                        ws = w * stride - padding
                        hend = min(hs + kernel_H, H)
                        wend = min(ws + kernel_W, W)
                        hs = max(hs, 0)
                        ws = max(ws, 0)
                        for i in xrange(hs, hend):
                            for j in xrange(ws, wend):
                                gradient_x[n][c][i][j] += gradient_y[n][c][h][w] / \
                                    pooling_size

        return gradient_x

    shapeX = (100, 3, 28, 28)
    # (1,1,5,5)
    shapeY = (100, 3, 24, 24)
    #  input : x , filter : f , output: y
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    gradient_y = np.random.uniform(0, 10, size=shapeY).astype(np.float32)

    arr_x = numpyasdlarrayhandle(x)
    arr_gradient_y = numpyasdlarrayhandle(gradient_y)
    pool_layer = np.empty(shapeY, dtype=np.float32)
    gradient_x = np.empty(shapeX, dtype=np.float32)
    arr_pool_layer = numpyasdlarrayhandle(pool_layer)
    arr_gradient_x = numpyasdlarrayhandle(gradient_x)

    cpu_op.avg_pool(arr_x, 5, 5, arr_pool_layer)
    cpu_op.avg_pool_gradient(arr_gradient_y, 5, 5, arr_gradient_x)

    np_pool_layer = np_average_pooling(x, 5, 5)
    np_gradient_x = np_average_pooling_gradient(gradient_y, 5, 5)

    np.testing.assert_allclose(np_pool_layer, pool_layer, rtol=1e-5)

    np.testing.assert_allclose(np_gradient_x, gradient_x, rtol=1e-5)
    # print(arr_gradient_x.asnumpy())
    # print("asdasdf:",np_gradient_x)


# test_average_pooling()


def test_max_pooling():
    ctx = ndarray.cpu(0)

    def np_max_pooling(input, kernel_H, kernel_W, padding=0, stride=1):
        N, C, H, W = input.shape
        assert ((H + 2 * padding - kernel_H) % stride == 0)
        assert ((W + 2 * padding - kernel_W) % stride == 0)
        pooled_H = (H + 2 * padding - kernel_H) / stride + 1
        pooled_W = (W + 2 * padding - kernel_W) / stride + 1

        pooled_layer = np.zeros(
            shape=(N, C, pooled_H, pooled_W), dtype=np.float32)
        pooling_size = kernel_H * kernel_W

        for n in range(N):
            for c in range(C):
                for h in range(pooled_H):
                    for w in range(pooled_W):
                        hs = h * stride - padding
                        ws = w * stride - padding
                        hend = min(hs + kernel_H, H)
                        wend = min(ws + kernel_W, W)
                        hs = max(hs, 0)
                        ws = max(ws, 0)

                        hargmax = hs
                        wargmax = ws
                        for i in range(hs, hend):
                            for j in range(ws, wend):
                                if input[n][c][i][j] > input[n][c][hargmax][wargmax]:
                                    hargmax = i
                                    wargmax = j
                        pooled_layer[n][c][h][w] = input[n][c][hargmax][wargmax]

        return pooled_layer

    def np_max_pooling_gradient(input, gradient_y, kernel_H, kernel_W, padding=0, stride=1):
        N, C, pooled_H, pooled_W = gradient_y.shape
        H = (pooled_H - 1) * stride + kernel_H - 2 * padding
        W = (pooled_W - 1) * stride + kernel_W - 2 * padding
        # print(N,C,H,W)
        gradient_x = np.zeros(shape=(N, C, H, W), dtype=np.float32)
        pooling_size = kernel_H * kernel_W

        for n in xrange(N):
            for c in xrange(C):
                for h in xrange(pooled_H):
                    for w in xrange(pooled_W):
                        hs = h * stride - padding
                        ws = w * stride - padding
                        hend = min(hs + kernel_H, H)
                        wend = min(ws + kernel_W, W)
                        hs = max(hs, 0)
                        ws = max(ws, 0)

                        hargmax = hs
                        wargmax = ws
                        for i in xrange(hs, hend):
                            for j in xrange(ws, wend):
                                # print(n,c,i,j)
                                if input[n][c][i][j] > input[n][c][hargmax][wargmax]:
                                    hargmax = i
                                    wargmax = j
                        gradient_x[n][c][hargmax][wargmax] += gradient_y[n][c][h][w]

        return gradient_x

    shapeX = (100, 3, 28, 28)
    shapeY = (100, 3, 14, 14)
    # shapeX=(1,1,2,2)
    # shapeY=(1,1,1,1)
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    # x = np.arange(1,37).reshape(shapeX)
    # print(x)
    # x = np.ones(shapeX).astype(np.float32)
    gradient_y = np.random.uniform(0, 10, size=shapeY).astype(np.float32)
    # gradient_y = np.ones(shapeY).astype(np.float32)
    arr_x = numpyasdlarrayhandle(x)
    arr_gradient_y = numpyasdlarrayhandle(gradient_y)
    pool_layer = np.empty(shapeY, dtype=np.float32)
    gradient_x = np.empty(shapeX, dtype=np.float32)
    arr_pool_layer = numpyasdlarrayhandle(pool_layer)
    arr_gradient_x = numpyasdlarrayhandle(gradient_x)

    pool_layer1 = np.empty(shapeY, dtype=np.float32)
    gradient_x1 = np.empty(shapeX, dtype=np.float32)
    arr_pool_layer1 = numpyasdlarrayhandle(pool_layer1)
    arr_gradient_x1 = numpyasdlarrayhandle(gradient_x1)

    np_pool_layer = np_max_pooling(x, 2, 2, 0, 2)
    cpu_op.max_pool(arr_x, 2, 2, arr_pool_layer, 0, 2)
    # print('poollayer:',np_pool_layer)
    # print(arr_pool_layer.asnumpy())

    np_gradient_x = np_max_pooling_gradient(x, np_pool_layer, 2, 2, 0, 2)
    cpu_op.max_pool_gradient(arr_x, arr_pool_layer, 2,
                             2, arr_gradient_x1, 0, 2)

    # print(arr_pool_layer.asnumpy())
    # print(np_pool_layer)

    np.testing.assert_allclose(np_pool_layer, pool_layer, rtol=1e-5)

    np.testing.assert_allclose(np_gradient_x, gradient_x1, rtol=1e-5)

    '''
   for i in range(len(ll)):
        ctx = ndarray.cpu(0)
        shape = (3, 3, ll[i], ll[i])
        to_shape = (3, 3, ll[i] / 2, ll[i] / 2)
        x = np.random.uniform(-1, 1, shape).astype(np.float32)
        arr_x = ndarray.array(x, ctx=ctx)
        arr_y = ndarray.empty(to_shape, ctx=ctx)
        # print(arr_x.asnumpy())
        start = time.time()
        for _ in range(10):
            cpu_op.max_pooling(arr_x, 2, 2, arr_y, 0, 2)
        end = time.time()
        for _ in range(10):
            out=np_max_pooling(x,2,2,0,2)
        end1=time.time()
        print(ll[i], " cpu:", end - start, "  ", "numpy:",end1-end )
        y = arr_y.asnumpy()
        np.testing.assert_allclose (out, y, rtol=1e-5)
    '''


# test_max_pooling()


def test_matrix_multiply():
    for i in range(len(ll)):
        # ctx = ndarray.cpu(0)
        shape = (ll[i], ll[i])
        x = np.random.uniform(-5, 5, size=shape).astype(np.float32)
        y = np.random.uniform(-5, 5, size=shape).astype(np.float32)
        z = np.zeros(shape, dtype=np.float32)

        start = time.time()
        # numpy
        for _ in range(10):
            c_np = np.dot(x, y)
        time1 = time.time()
        arr_x = numpyasdlarrayhandle(x)
        arr_y = numpyasdlarrayhandle(y)
        arr_z = numpyasdlarrayhandle(z)
        time2 = time.time()
        for _ in range(1):
            cpu_op.matrix_multiply(
                arr_x, False,
                arr_y, False,
                arr_z)
        time4 = time.time()
        print(ll[i], " cpu:", time4 - time2, "  ", "numpy:", time1 - start)
        np.testing.assert_allclose(c_np, z, rtol=1e-5)
#
#
# test_matrix_multiply()


def test_matrix_elementwise_multiply_by_const():
    for i in range(len(ll)):
        ctx = ndarray.cpu(0)
        shape = (ll[i], ll[i])
        x = np.random.uniform(-1, 1, shape).astype(np.float32)
        # y = np.random.uniform(-1, 1, shape).astype(np.float32)
        arr_x = numpyasdlarrayhandle(x)
        # arr_y = ndarray.array(y, ctx=ctx)
        val = 4.754545
        z = np.empty(shape, dtype=np.float32)
        arr_z = numpyasdlarrayhandle(z)

        start = time.time()
        for _ in range(10):
            cpu_op.matrix_elementwise_multiply_by_const(arr_x, val, arr_z)
        end = time.time()
        for _ in range(10):
            nu = x * val
        end1 = time.time()
        print(ll[i], " cpu:", end - start, "  ", "numpy:", end1 - end)
#         output=z.asnumpy()
        np.testing.assert_allclose(nu, z, rtol=1e-5)
#

# test_matrix_elementwise_multiply_by_const()


def test_matrix_elementwise_add_by_const():
    for i in range(len(ll)):
        ctx = ndarray.cpu(0)
        shape = (ll[i], ll[i])
        x = np.random.uniform(-1, 1, shape).astype(np.float32)

        # y = np.random.uniform(-1, 1, shape).astype(np.float32)
        arr_x = numpyasdlarrayhandle(x)
        # arr_y = ndarray.array(y, ctx=ctx)
        val = 4.754545
        z = np.empty(shape, dtype=np.float32)
        arr_z = numpyasdlarrayhandle(z)

        start = time.time()
        for _ in range(10):
            cpu_op.matrix_elementwise_add_by_const(arr_x, val, arr_z)
        end = time.time()
        for _ in range(10):
            nu = x + val
        end1 = time.time()
        print(ll[i], " cpu:", end - start, "  ", "numpy:", end1 - end)
        np.testing.assert_allclose(nu, z, rtol=1e-5)


# test_matrix_elementwise_add_by_const()


def test_matrix_elementwise_multiply_by_const():
    for i in range(len(ll)):
        ctx = ndarray.cpu(0)
        shape = (ll[i], ll[i])
        x = np.random.uniform(-1, 1, shape).astype(np.float32)
        # y = np.random.uniform(-1, 1, shape).astype(np.float32)
        arr_x = numpyasdlarrayhandle(x)
        # arr_y = ndarray.array(y, ctx=ctx)
        val = np.random.uniform(-5, 5)
        z = np.empty(shape, dtype=np.float32)
        arr_z = numpyasdlarrayhandle(z)

        start = time.time()
        for _ in range(10):
            cpu_op.matrix_elementwise_multiply_by_const(arr_x, val, arr_z)
        end = time.time()
        for _ in range(10):
            nu = x * val
        end1 = time.time()
        print(ll[i], " cpu:", end - start, "  ", "numpy:", end1 - end)
        np.testing.assert_allclose(nu, z, rtol=1e-5)


# test_matrix_elementwise_multiply_by_const()

def test_matrix_elementwise_multiply():
    for i in range(len(ll)):
        ctx = ndarray.cpu(0)
        shape = (ll[i], ll[i])
        x = np.random.uniform(-5, 5, size=shape).astype(np.float32)
        y = np.random.uniform(-5, 5, size=shape).astype(np.float32)
        start = time.time()
        # numpy
        for _ in range(10):
            c_np = x * y
        time1 = time.time()
        arr_x = numpyasdlarrayhandle(x)
        arr_y = numpyasdlarrayhandle(y)
        z = np.empty(shape, dtype=np.float32)
        arr_z = numpyasdlarrayhandle(z)
        time2 = time.time()
        for _ in range(10):
            cpu_op.matrix_elementwise_multiply(
                arr_x,
                arr_y,
                arr_z)
        time4 = time.time()
        print(ll[i], " cpu:", time4 - time2, "  ", "numpy:", time1 - start)
        np.testing.assert_allclose(c_np, z, rtol=1e-5)


# test_matrix_elementwise_multiply()

def test_matrix_elementwise_add():
    for i in range(len(ll)):
        ctx = ndarray.cpu(0)
        shape = (ll[i], ll[i])
        x = np.random.uniform(-5, 5, size=shape).astype(np.float32)
        y = np.random.uniform(-5, 5, size=shape).astype(np.float32)
        start = time.time()
        # numpy
        for _ in range(10):
            c_np = x + y
        time1 = time.time()
        arr_x = numpyasdlarrayhandle(x)
        arr_y = numpyasdlarrayhandle(y)
        z = np.empty(shape, dtype=np.float32)
        arr_z = numpyasdlarrayhandle(z)
        time2 = time.time()
        for _ in range(10):
            cpu_op.matrix_elementwise_add(
                arr_x,
                arr_y,
                arr_z)
        time4 = time.time()
        print(ll[i], " cpu:", time4 - time2, "  ", "numpy:", time1 - start)
        np.testing.assert_allclose(c_np, z, rtol=1e-5)


# test_matrix_elementwise_add()

def test_matrix_div_const():
    for i in range(len(ll)):
        ctx = ndarray.cpu(0)
        shape = (ll[i], ll[i])
        x = np.random.uniform(-1, 1, shape).astype(np.float32)
        # y = np.random.uniform(-1, 1, shape).astype(np.float32)
        # arr_y = ndarray.array(y, ctx=ctx)
        val = 4.754545
        arr_x = numpyasdlarrayhandle(x)
        z = np.empty(shape, dtype=np.float32)
        arr_z = numpyasdlarrayhandle(z)
        start = time.time()
        for _ in range(10):
            cpu_op.matrix_elementwise_divide_by_const(arr_x, val, arr_z)
        end = time.time()
        for _ in range(10):
            nu = x / val
        end1 = time.time()
        print(ll[i], " cpu:", end - start, "  ", "numpy:", end1 - end)
        np.testing.assert_allclose(nu, z, rtol=1e-5)

# test_matrix_div_const()


def test_divide_elewise():
    for i in range(len(ll)):
        ctx = ndarray.cpu(0)
        shape = (ll[i], ll[i])
        x = np.random.uniform(-5, 5, size=shape).astype(np.float32)
        y = np.random.uniform(-5, 5, size=shape).astype(np.float32)
        start = time.time()
        # numpy

        for _ in range(10):
            c_np = x / y
        time1 = time.time()
        arr_x = numpyasdlarrayhandle(x)
        arr_y = numpyasdlarrayhandle(y)
        z = np.empty(shape, dtype=np.float32)
        arr_z = numpyasdlarrayhandle(z)
        time2 = time.time()
        for _ in range(10):
            cpu_op.matrix_elementwise_divide(
                arr_x,
                arr_y,
                arr_z)
        time4 = time.time()
        print(ll[i], " cpu:", time4 - time2, "  ", "numpy:", time1 - start)
        np.testing.assert_allclose(c_np, z, rtol=1e-5)


# test_divide_elewise()


def test_arrayset_oneslike():
    for i in range(10000):  # len(ll)):
        ctx = ndarray.cpu(0)
        shape = (10000, 10000)  # ll[i], ll[i])
        #x = np.random.uniform(-1, 1, shape).astype(np.float32)
        # y = np.random.uniform(-1, 1, shape).astype(np.float32)
        #arr_x = ndarray.array(x, ctx=ctx)
        # arr_y = ndarray.array(y, ctx=ctx)
        val = 1
        z = np.zeros(shape, dtype=np.float32)
        arr_z = numpyasdlarrayhandle(z)
        # print(z)
        # print(x)
        # print(val)
        start = time.time()
        for _ in range(10):
            cpu_op.array_set(arr_z, val)
        end = time.time()
        for _ in range(10):
            output_val = np.ones(shape)
        end1 = time.time()
        print(ll[i], " cpu:", end - start, "  ", "numpy:", end1 - end)
        # print(out)
        # print(x*val)
        # print(z)
        np.testing.assert_allclose(output_val, z, rtol=1e-5)

# test_arrayset_oneslike()


def test_arrayset_zeroslike():
    for i in range(10000):
        ctx = ndarray.cpu(0)
        shape = (10, 100)
        #x = np.random.uniform(-1, 1, shape).astype(np.float32)
        # y = np.random.uniform(-1, 1, shape).astype(np.float32)
        #arr_x = ndarray.array(x, ctx=ctx)
        # arr_y = ndarray.array(y, ctx=ctx)
        val = 0
        z = np.empty(shape, dtype=np.float32)
        arr_z = numpyasdlarrayhandle(z)
        # print(x)
        # print(val)
        start = time.time()
        for _ in range(10):
            cpu_op.array_set(arr_z, val)
        end = time.time()
        for _ in range(10):
            output_val = np.zeros(shape)
        end1 = time.time()
        # print(ll[i], " cpu:", end - start, "  ", "numpy:", end1 - end)
        # print(out)
        # print(x*val)
        # print(z)
        np.testing.assert_allclose(output_val, z, rtol=1e-5)

# test_arrayset_zeroslike()


def test_softmax():
    for i in range(len(ll)):
        ctx = ndarray.cpu(0)
        shape = (ll[i], ll[i])
        x = np.random.uniform(-1, 1, shape).astype(np.float32)
        arr_x = ndarray.array(x, ctx=ctx)
        z = ndarray.empty(shape, ctx=ctx)
        start = time.time()
        for _ in range(10):
            cpu_op.softmax(arr_x, z)
        end = time.time()
        for _ in range(10):
            b = x - np.max(x, axis=1, keepdims=True)
            expb = np.exp(b)
            softmax = expb / np.sum(expb, axis=1, keepdims=True)
        end1 = time.time()
        print(ll[i], " cpu:", end - start, "  ", "numpy:", end1 - end)
        out = z.asnumpy()
        np.testing.assert_allclose(softmax, out, rtol=1e-5)


# test_softmax()

def test_softmax_crossentropy():
    for i in range(len(ll)):
        ctx = ndarray.cpu(0)
        shape = (ll[i], 10)
        y = np.random.uniform(-5, 5, shape).astype(np.float32)
        y_ = np.random.uniform(-5, 5, shape).astype(np.float32)
        arr_y = ndarray.array(y, ctx=ctx)
        arr_y_ = ndarray.array(y_, ctx=ctx)
        arr_out = ndarray.empty((1,), ctx=ctx)
        start = time.time()
        for _ in range(10):
            cpu_op.softmax_crossentropy(arr_y, arr_y_, arr_out)
        end = time.time()
        for _ in range(10):
            b = y - np.max(y, axis=1, keepdims=True)
            expb = np.exp(b)
            softmax = expb / np.sum(expb, axis=1, keepdims=True)
            cross_entropy = np.mean(
                -np.sum(y_ * np.log(softmax), axis=1), keepdims=True)
        end1 = time.time()
        print(ll[i], " cpu:", end - start, "  ", "numpy:", end1 - end)
        out = arr_out.asnumpy()
        # print(out)
        # print(cross_entropy)
        np.testing.assert_allclose(cross_entropy, out, rtol=1e-3)


# test_softmax_crossentropy()

def test_sqrt():
    for i in range(len(ll)):
        ctx = ndarray.cpu(0)
        shape = (ll[i], ll[i])
        x = np.random.uniform(0, 10, shape).astype(np.float32)
        arr_x = ndarray.array(x, ctx=ctx)
        arr_y = ndarray.empty(shape, ctx=ctx)
        start = time.time()
        for _ in range(10):
            cpu_op.sqrt(arr_x, arr_y)
        end = time.time()
        for _ in range(10):
            out = np.sqrt(x)
        end1 = time.time()
        print(ll[i], " cpu:", end - start, "  ", "numpy:", end1 - end)
        y = arr_y.asnumpy()
        np.testing.assert_allclose(out, y, rtol=1e-3)

# test_sqrt()


def test_tanh():
    # TODO
    raise NotImplementedError


# test_tanh()

def test_sigmoid():
    # TODO
    raise NotImplementedError


# test_sigmoid()


def test_opposite():
    for i in range(len(ll)):
        ctx = ndarray.cpu(0)
        shape = (ll[i], ll[i])
        x = np.random.uniform(-1, 1, shape).astype(np.float32)
        arr_x = ndarray.array(x, ctx=ctx)
        arr_y = ndarray.empty(shape, ctx=ctx)
        start = time.time()
        for _ in range(10):
            cpu_op.opposite(arr_x, arr_y)
        end = time.time()
        for _ in range(10):
            out = -x
        end1 = time.time()
        print(ll[i], " cpu:", end - start, "  ", "numpy:", end1 - end)
        y = arr_y.asnumpy()
        np.testing.assert_allclose(out, y, rtol=1e-5)


# test_opposite()

def test_relu():
    for i in range(len(ll)):
        ctx = ndarray.cpu(0)
        shape = (ll[i], ll[i])
        x = np.random.uniform(-1, 1, shape).astype(np.float32)
        arr_x = ndarray.array(x, ctx=ctx)
        arr_y = ndarray.empty(shape, ctx=ctx)
        start = time.time()
        for _ in range(10):
            cpu_op.relu(arr_x, arr_y)
        end = time.time()
        for _ in range(10):
            out = np.maximum(x, 0).astype(np.float32)
        end1 = time.time()
        print(ll[i], " cpu:", end - start, "  ", "numpy:", end1 - end)
        y = arr_y.asnumpy()
        np.testing.assert_allclose(out, y, rtol=1e-5)
# test_relu()


def test_relu_gradient():
    shape = (2, 2)
    ctx = ndarray.cpu(0)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    print("x:", x)
    grad_x = np.random.uniform(-5, 5, shape).astype(np.float32)
    print("g:", grad_x)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_grad_x = ndarray.array(grad_x, ctx=ctx)
    arr_y = ndarray.empty(shape, ctx=ctx)
    cpu_op.relu_gradient(arr_x, arr_grad_x, arr_y)
    y = arr_y.asnumpy()
    print(y)
    np.testing.assert_allclose(((x > 0) * grad_x).astype(np.float32), y)

# test_relu_gradient()


def test_conv2d():
    ctx = ndarray.cpu(0)

    # im2col and np_conv2d are helper functions
    def im2col(X, filter_H, filter_W, padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = int((H + 2 * padding - filter_H) / stride + 1)
        out_W = int((W + 2 * padding - filter_W) / stride + 1)

        y_row_size = int(C * filter_H * filter_W)
        y_col_size = int(out_H * out_W)
        y_shape = (N, y_row_size, y_col_size)
        Y = np.empty(y_shape, dtype=X.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index / out_W
                out_x = col_index % out_W
                in_y = int(out_y * stride - padding)
                in_x = int(out_x * stride - padding)
                row_idx = 0
                for c in range(0, C):
                    for y in range(in_y, in_y + filter_H):
                        for x in range(in_x, in_x + filter_W):
                            if (x < 0 or x >= W or y < 0 or y >= H):
                                Y[batch_index, row_idx, col_index] = 0
                            else:
                                Y[batch_index, row_idx,
                                    col_index] = X[batch_index, c, y, x]
                            row_idx += 1
        return Y

    def np_conv2d(X, Filter, padding=0, stride=1):
        """Implement a conv2d as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = int((H + 2 * padding - filter_H) / stride + 1)
        out_W = int((W + 2 * padding - filter_W) / stride + 1)

        im2col_matrix = im2col(X, filter_H, filter_W, padding, stride)
        filter_matrix = Filter.reshape(filter_outChannel, -1)
        #print("shape", im2col_matrix.shape)
        #print("shape", filter_matrix.shape)
        #print("shape", np.matmul(filter_matrix, im2col_matrix).shape)
        return np.matmul(filter_matrix, im2col_matrix).reshape(N, filter_outChannel, out_H, out_W)
        # return im2col_matrix

    shapeX = (100, 3, 28, 28)
    shapeF = (10, 3, 5, 5)
    shapeY = (100, 10, 24, 24)
    # shapeX=(1,1,2,2)
    # shapeF=(1,1,1,1)
    # shapeY=(1,1,2,2)
    #shapeW = (100, 3 * 5 * 5, 24 * 24)
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    f = np.random.uniform(0, 10, size=shapeF).astype(np.float32)
    # print("x:",x)
    # print("f:",f)
    # y = np.zeros(shapeY).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_f = ndarray.array(f, ctx=ctx)
    arr_y = ndarray.empty(shapeY, ctx=ctx)
    #arr_workspace = ndarray.empty(shapeW, ctx=ctx)

    cpu_op.conv2d(arr_x, arr_f, arr_y, 0, 1)
    y = arr_y.asnumpy()
    # print("y:",y)
    z = np_conv2d(x, f)
    # print("z:",z)
    np.testing.assert_allclose(z, y, rtol=1e-5)

# test_conv2d()


def test_conv2d_Gradient():
    ctx = ndarray.cpu(0)

    def im2col(X, filter_H, filter_W, padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) / stride + 1
        out_W = (W + 2 * padding - filter_W) / stride + 1

        y_row_size = C * filter_H * filter_W
        y_col_size = out_H * out_W
        y_shape = (N, y_row_size, y_col_size)
        Y = np.empty(y_shape, dtype=X.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index / out_W
                out_x = col_index % out_W
                in_y = out_y * stride - padding
                in_x = out_x * stride - padding
                row_idx = 0
                for c in range(0, C):
                    for y in range(in_y, in_y + filter_H):
                        for x in range(in_x, in_x + filter_W):
                            if (x < 0 or x >= W or y < 0 or y >= H):
                                Y[batch_index, row_idx, col_index] = 0
                            else:
                                Y[batch_index, row_idx,
                                    col_index] = X[batch_index, c, y, x]
                            row_idx += 1
        return Y

    def np_conv2d(X, Filter, padding=0, stride=1):
        """Implement a conv2d as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) / stride + 1
        out_W = (W + 2 * padding - filter_W) / stride + 1

        im2col_matrix = im2col(X, filter_H, filter_W, padding, stride)
        filter_matrix = Filter.reshape(filter_outChannel, -1)
        # print("shape", im2col_matrix.shape)
        # print("shape", filter_matrix.shape)
        # print("shape", np.matmul(filter_matrix, im2col_matrix).shape)
        return np.matmul(filter_matrix, im2col_matrix).reshape(N, filter_outChannel, out_H, out_W)
        # return im2col_matrix

    def im2col_transpose(X, filter_H, filter_W, Y, padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) / stride + 1
        out_W = (W + 2 * padding - filter_W) / stride + 1
        _, y_row_size, y_col_size = Y.shape

        der_X_shape = (N, C, H, W)
        der_X = np.zeros(der_X_shape, dtype=X.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index / out_W
                out_x = col_index % out_W
                in_y = out_y * stride - padding
                in_x = out_x * stride - padding
                row_idx = 0
                for c in range(0, C):
                    for y in range(in_y, in_y + filter_H):
                        for x in range(in_x, in_x + filter_W):
                            if (x < 0 or x >= W or y < 0 or y >= H):
                                Y[batch_index, row_idx, col_index] = 0
                            else:
                                der_X[batch_index, c, y,
                                      x] += Y[batch_index, row_idx, col_index]
                            row_idx += 1
        return der_X

    def np_conv2d_transpose(X, Filter, Y, padding=0, stride=1):
        """Implement a conv2d_transpose as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        X_N, X_C, X_H, X_W = X.shape
        Y_N, Y_C, Y_H, Y_W = Y.shape
        YY = Y.reshape((Y_N, Y_C, Y_H * Y_W))  # transformed to im2col Y
        # XX = X.reshape((X_N, X_C, X_W * X_H))   # transformed to im2col X
        F_filter = Filter.reshape((filter_outChannel, -1))
        gradient_im2col_XX = np.matmul(F_filter.T, YY)

        gradient_X = im2col_transpose(
            X, filter_H, filter_W, gradient_im2col_XX, padding, stride)  # gradient of x
        im2col_XX = im2col(X, filter_H, filter_W, padding, stride)
        gradient_filter = np.zeros(shape=F_filter.shape, dtype=X.dtype)

        for i in range(X_N):
            gradient_filter += np.matmul(YY[i], im2col_XX[i].T)
        gradient_filter = gradient_filter.reshape(Filter.shape)

        return gradient_X, gradient_filter

    # shapeX = (100, 3, 28, 28)
    # shapeF = (10, 3, 5, 5)
    # shapeY = (100, 10, 24, 24)
    shapeX = (1, 1, 2, 2)
    shapeF = (1, 1, 1, 1)
    shapeY = (1, 1, 2, 2)
    shapeW = (100, 3 * 5 * 5, 24 * 24)
    shapeFF = (100, 10, 3, 5, 5)
    #  input : x , filter : f , output: y
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    f = np.random.uniform(0, 10, size=shapeF).astype(np.float32)
    # print('x:',x)
    # print('f:',f)

    der_y = np.ones(shape=shapeY)
    gradient_x, gradient_f = np_conv2d_transpose(x, f, der_y)

    # print('gredent_f:',gradient_f)
    print(gradient_f.shape)
    # print('der_y:',der_y)

    arr_x = ndarray.array(x, ctx=ctx)
    arr_f = ndarray.array(f, ctx=ctx)
    gradient_y = ndarray.array(der_y, ctx=ctx)
    gradient_xx = ndarray.array(x, ctx=ctx)
    gradient_ff = ndarray.array(f, ctx=ctx)
    cpu_op.conv2d_gradient_of_filter(arr_x, gradient_y, gradient_ff)
    cpu_op.conv2d_gradient_of_data(arr_f, gradient_y, gradient_xx)

    np.testing.assert_allclose(gradient_x, gradient_xx.asnumpy(), rtol=1e-5)
    np.testing.assert_allclose(gradient_f, gradient_ff.asnumpy(), rtol=1e-5)


# test_conv2d_Gradient()


def test_concat():
    ctx = ndarray.cpu(0)
    shape1 = (1, 2)
    shape2 = (1, 2)
    x1 = np.array([1, 2, 2, 2]).reshape((2, 2))
    x2 = np.array([3, 4, 4, 4, 4, 4]).reshape((2, 3))
    arr_x1 = ndarray.array(x1, ctx=ctx)
    arr_x2 = ndarray.array(x2, ctx=ctx)
    to_shape = (2, 5)

    arr_y = ndarray.empty(to_shape, ctx=ctx)
    cpu_op.concat(arr_x1, arr_x2, arr_y, axis=1)

    print(arr_y.asnumpy())
    #
    # grad_x1 = ndarray.empty(shape1, ctx=ctx)
    # grad_x2 = ndarray.empty(shape2, ctx=ctx)
    # grad_y = np.array([1, 2, 3, 4]).reshape((1, 4))
    # grad_y_arr = ndarray.array(grad_y, ctx=ctx)
    #
    # gpu_op.concat_gradient(grad_y_arr, grad_x1, axis=1, idx=0)
    # gpu_op.concat_gradient(grad_y_arr, grad_x2, axis=1, idx=1)
    # print(grad_x1.asnumpy())
    # print(grad_x2.asnumpy())
# test_concat()


def test_Batch_Normalization():
    ctx = ndarray.cpu(0)
    shape = (1, 1, 2, 2)
    shape2 = [2]
    np.random.seed(111)
    x = np.random.uniform(-5, 5, size=shape).astype(np.float32)
    scale = np.random.uniform(0, 1, size=shape2).astype(np.float32)
    bias = np.random.uniform(0, 1, size=shape2).astype(np.float32)
    arr_x = ndarray.array(x, ctx=ctx)
    arr_scale = ndarray.array(scale, ctx=ctx)
    arr_bias = ndarray.array(bias, ctx=ctx)

    arr_y = ndarray.empty(shape, ctx=ctx)
    print('x:', x)
    print('scale:', scale)
    print('bias:', bias)

    cpu_op.Batch_Normalization(arr_x, arr_scale, arr_bias, arr_y, 0.99, 0.01)

    print(arr_y.asnumpy())

    # gradient
    arr_gradient_x = ndarray.empty(shape, ctx=ctx)
    arr_gradient_scale = ndarray.empty(shape2, ctx=ctx)
    arr_gradient_bias = ndarray.empty(shape2, ctx=ctx)

    cpu_op.Batch_Normalization_gradient(arr_y, arr_x, arr_scale, arr_bias,
                                        arr_gradient_x, arr_gradient_scale,
                                        arr_gradient_bias, 0.01)
    print('arr_gradient_x:', arr_gradient_x.asnumpy())
    print('arr_gradient_scale:', arr_gradient_scale.asnumpy())
    print('arr_gradient_bias:', arr_gradient_bias.asnumpy())

#
    # #   tf
    # import tensorflow as tf
    #
    # tf_x=tf.placeholder(tf.float32,[1,1,2,2])
    # train_flag=tf.placeholder(tf.bool)
    # tf_scale=tf.Variable(scale,dtype=tf.float32)
    # tf_bias=tf.Variable(bias,dtype=tf.float32)
    #
    # def batch_norm(input, scale, shift):
    #     axis = list(range(len(input.get_shape()) - 1))
    #     a_mean, a_var = tf.nn.moments(input, axis)
    #     return tf.nn.batch_normalization(input, mean=a_mean, variance=a_var,
    #                 offset=shift, scale=scale, variance_epsilon=1e-2, name=None)
    #
    # out=batch_norm(tf_x,tf_scale,tf_bias)
    #
    # with tf.Session as sess:
    #     sess.run(tf.global_variables_initializer())
    #     oo=sess.run([out],feed_dict={tf_x:x,train_flag:True})
    #     print(oo)


# test_Batch_Normalization()


'''

for i in range(1):
    ll=[500,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000]

    transA='N'
    transB='T'


    M = ll[i]
    N = ll[i]
    K = ll[i]
    A = np.random.randn(M, K)
    B = np.random.randn(K, N)
    C = np.zeros((M, N))


    start = time.time()
    ###numpy
    for i in range(100):
        C_np = np.dot(A, B)
    #print(C_np)
    time1 = time.time()
    print("numpy:total cost time:%.4f " % (time1 - start))
    M, K = A.shape
    K1, N = B.shape
    #print(B)
    linear_A = np.reshape(A, (-1))
    linear_B = np.reshape(B, (-1),order='F')
    linear_C = np.reshape(C, (-1))
    #print(linear_B)
    linear_A = np.ascontiguousarray(linear_A, dtype=np.float32)
    linear_B = np.ascontiguousarray(linear_B, dtype=np.float32)
    linear_C = np.ascontiguousarray(linear_C, dtype=np.float32)
    #print(linear_B)
    #print(linear_A)
    time4 = time.time()
    for i in range(100):
        #cpu_op.matmul(A,B,C)
        xx=cpu_op.matmul2(M,N,K,linear_A,linear_B,linear_C)
        #xx=cpu_op.sgemm(transA,transB,M,N,K,linear_A,linear_B,linear_C)
    time2 = time.time()
    print("dnnl:total cost time:%.4f " % (time2 - time4))
    #print(xx)

    #save_to_file("%d,%.4f,%.4f\n" % (M, time1 - start, time2 - time4), 'time.txt')

'''

'''
##c++
linear_A = np.reshape(A, (-1))
linear_B = np.reshape(B, (-1))
linear_C = np.reshape(C, (-1))

time4 = time.time()
print("total cost time:%.4f " % (time4 - time1))

so = ctypes.cdll.LoadLibrary
lib = so("../../build/lib/lib_dnnl_op.so")
star = time.time()
print("total cost time:%.4f " % (star - time4))

#if not linear_A.flags['C_CONTIGUOUS']:
linear_A = np.ascontiguousarray(linear_A, dtype=np.float32)
#if not linear_B.flags['C_CONTIGUOUS']:
linear_B = np.ascontiguousarray(linear_B, dtype=np.float32)
#if not linear_C.flags['C_CONTIGUOUS']:
linear_C = np.ascontiguousarray(linear_C, dtype=np.float32)

start2 = time.time()
print("total cost time:%.4f " % (start2 - star))
for i in range(100):
    lib.test(M, N, K, linear_A.ctypes.data_as(ctypes.c_void_p), linear_B.ctypes.data_as(ctypes.c_void_p),
         linear_C.ctypes.data_as(ctypes.c_void_p))
time2 = time.time()
print("last,total cost time:%.4f " % ((time2 - start2)/100))

print(linear_C)
'''


def test_transpose():
    shape = (4321, 1234)
    ctx = ndarray.cpu(0)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    y = np.empty((shape[1], shape[0]), dtype=np.float32)

    arr_x = ndarray.numpyasdlarrayhandle(x)
    arr_y = ndarray.numpyasdlarrayhandle(y)
    cpu_op.transpose(arr_x, arr_y, [1, 0])
    np.testing.assert_allclose(np.transpose(x), y)

    shape = (21, 43, 65, 11)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    y = np.empty((65, 11, 43, 21), dtype=np.float32)

    arr_x = ndarray.numpyasdlarrayhandle(x)
    arr_y = ndarray.numpyasdlarrayhandle(y)
    cpu_op.transpose(arr_x, arr_y,  perm=[2, 3, 1, 0])
    np.testing.assert_allclose(np.transpose(x, [2, 3, 1, 0]), y)

# test_transpose()


def test_embedding_lookup():
    emb = np.random.rand(5, 5)
    ctx = ndarray.cpu(0)
    print(emb)
    emb = ndarray.array(emb, ctx=ctx)
    ids = [[0, 1], [0, 1]]
    ids = np.array(ids)
    print(ids)
    ids = ndarray.array(ids, ctx=ctx)

    output = ndarray.empty((2, 2, 5), ctx=ctx)
    cpu_op.embedding_lookup(emb, ids, output)
    print(output.asnumpy())


test_embedding_lookup()
