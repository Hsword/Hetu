import time
import random
import numpy as np
import hetu as ht
from hetu import gpu_links as gpu_opl
from hetu import gpu_ops as gpu_op
from hetu.stream import create_stream_handle
from hetu.gpu_links.Argmax import argmax


def test_common(op, num_args, input_shapes, has_const=False, need_output=True, *args):
    ctx = ht.gpu(0)
    stream = create_stream_handle(ctx)
    TOTAL_ROUNDS = 5
    for shape in input_shapes:
        total_time = 0
        input_vals = [ht.array(np.random.normal(size=shape), ctx=ctx) for _ in range(num_args)]
        if has_const:
            input_vals.append(random.randint(-1000, 1000))
        if need_output:
            output = ht.array(np.random.normal(size=shape), ctx=ctx)
        for i in range(TOTAL_ROUNDS):
            start = time.time()
            if need_output:
                op(*input_vals, output, *args, stream=stream)
            else:
                op(*input_vals, *args, stream=stream)
            stream.sync()
            if i > 0:
                total_time += time.time() - start

        total_time *= 1000 / (TOTAL_ROUNDS - 1)
        print('Op %s test with shape: %s takes %fms' %
            (op, shape, total_time))


def test_pooling(op, num_args, input_shapes, kernel_H, kernel_W, padding=0, stride=1, has_const=False, need_output=True):
    ctx = ht.gpu(0)
    stream = create_stream_handle(ctx)
    TOTAL_ROUNDS = 5
    for shape in input_shapes:
        total_time = 0
        input_vals = [ht.array(np.random.normal(size=shape), ctx=ctx) for _ in range(num_args)]
        if has_const:
            input_vals.append(random.randint(-1000, 1000))
        if need_output:
            output = ht.array(np.random.normal(size=shape), ctx=ctx)
        for i in range(TOTAL_ROUNDS):
            start = time.time()
            if need_output:
                op(*input_vals, kernel_H, kernel_W, output, padding, stride, stream=stream)
            else:
                op(*input_vals, kernel_H, kernel_W, padding, stride, stream=stream)
            stream.sync()
            if i > 0:
                total_time += time.time() - start

        total_time *= 1000 / (TOTAL_ROUNDS - 1)
        print('Op %s test with shape: %s takes %fms' %
            (op, shape, total_time))


def test_argmax(shapes):
    ctx = ht.gpu(0)
    stream = create_stream_handle(ctx)
    TOTAL_ROUNDS = 5
    for shape in shapes:
        total_time = 0
        input_val = ht.array(np.random.normal(size=shape), ctx=ctx)
        output = ht.array(np.random.normal(size=(shape[0], shape[1])), ctx=ctx)
        for i in range(TOTAL_ROUNDS):
            start = time.time()
            gpu_opl.argmax(input_val, output, dim=2, stream=stream)
            stream.sync()
            if i > 0:
                total_time += time.time() - start

        total_time *= 1000 / (TOTAL_ROUNDS - 1)
        print('Op %s test with shape: %s takes %fms' %
            ('argmax', shape, total_time))


def test_concat(shapes):
    ctx = ht.gpu(0)
    stream = create_stream_handle(ctx)
    TOTAL_ROUNDS = 5
    for shape in shapes:
        total_time = 0
        input_a = ht.array(np.random.normal(size=shape), ctx=ctx)
        input_b = ht.array(np.random.normal(size=shape), ctx=ctx)
        output = ht.array(np.random.normal(size=(shape[0], shape[1], shape[2]*2)), ctx=ctx)
        for i in range(TOTAL_ROUNDS):
            start = time.time()
            gpu_opl.concat(input_a, input_b, output, axis=2, stream=stream)
            stream.sync()
            if i > 0:
                total_time += time.time() - start

        total_time *= 1000 / (TOTAL_ROUNDS - 1)
        print('Op %s test with shape: %s takes %fms' %
            ('concat', shape, total_time))


def test_matmul(shapes):
    ctx = ht.gpu(0)
    stream = create_stream_handle(ctx)
    TOTAL_ROUNDS = 5
    for shape in shapes:
        total_time = 0
        input_vals = [ht.array(np.random.normal(size=shape), ctx=ctx) for _ in range(2)]
        output = ht.array(np.random.normal(size=shape), ctx=ctx)
        for i in range(TOTAL_ROUNDS):
            start = time.time()
            gpu_opl.batch_matrix_multiply(input_vals[0], False, input_vals[1], True, output, stream=stream)
            stream.sync()
            if i > 0:
                total_time += time.time() - start

        total_time *= 1000 / (TOTAL_ROUNDS - 1)
        print('Op %s test with shape: %s takes %fms' %
            ('matmul', shape, total_time))


def test_broadcast_to(shapes):
    ctx = ht.gpu(0)
    stream = create_stream_handle(ctx)
    TOTAL_ROUNDS = 5
    for shape in shapes:
        total_time = 0
        input_vals = [ht.array(np.random.normal(size=(shape[1], shape[2])), ctx=ctx) for _ in range(2)]
        output = ht.array(np.random.normal(size=shape), ctx=ctx)
        for i in range(TOTAL_ROUNDS):
            start = time.time()
            gpu_opl.broadcast_to(input_vals[0], output, stream=stream)
            stream.sync()
            if i > 0:
                total_time += time.time() - start

        total_time *= 1000 / (TOTAL_ROUNDS - 1)
        print('Op %s test with shape: %s takes %fms' %
            ('broadcast_to', shape, total_time))


def test_conv2d(shapes):
    ctx = ht.gpu(0)
    stream = create_stream_handle(ctx)
    TOTAL_ROUNDS = 5
    for shape in shapes:
        shapeX = shape
        shapeF = (shape[1]*2, shape[1], 3, 3)
        shapeY = (shape[0], shape[1]*2, shape[2], shape[3])
        x = np.random.uniform(0, 1, size=shapeX).astype(np.float32)
        f = np.random.uniform(0, 1, size=shapeF).astype(np.float32)
        arr_x = ht.array(x, ctx=ctx)
        arr_f = ht.array(f, ctx=ctx)
        arr_y = ht.empty(shapeY, ctx=ctx)
        total_time = 0
        for i in range(TOTAL_ROUNDS):
            start = time.time()
            gpu_opl.CuDNN_conv2d(arr_x, arr_f, arr_y, (1, 1), (1, 1), stream=stream)
            stream.sync()
            if i > 0:
                total_time += time.time() - start

        total_time *= 1000 / (TOTAL_ROUNDS - 1)
        print('Op %s test with shape: %s takes %fms' %
            ('conv2d', shape, total_time))


def test_reduce_sum(shapes, axes):
    ctx = ht.gpu(0)
    stream = create_stream_handle(ctx)
    TOTAL_ROUNDS = 2
    for shape in shapes:
        input = ht.Variable(name='input', ctx=ctx)
        conv = gpu_op.reduce_sum_op(input, axes, ctx=ctx)
        executor = ht.Executor([conv], ctx=ctx)
        total_time = 0
        for i in range(TOTAL_ROUNDS):
            x = ht.array(np.random.uniform(-1, 1, size=shape).astype(np.float32), ctx=ctx)
            start = time.time()
            executor.run(feed_dict={input: x})
            stream.sync()
            if i > 0:
                total_time += time.time() - start

        total_time *= 1000 / (TOTAL_ROUNDS - 1)
        print('Op %s test with shape: %s takes %fms' %
            ('reduce_sum', shape, total_time))


def test_dropout(shapes):
    ctx = ht.gpu(0)
    stream = create_stream_handle(ctx)
    TOTAL_ROUNDS = 2
    for shape in shapes:
        input = ht.Variable(name='dropout_input', ctx=ctx)
        dropout_op = gpu_op.dropout_op(input, 0.3, inplace=True, ctx=ctx)
        executor = ht.Executor([dropout_op], ctx=ctx)
        total_time = 0
        for i in range(TOTAL_ROUNDS):
            x = ht.array(np.random.uniform(-1, 1, size=shape).astype(np.float32), ctx=ctx)
            start = time.time()
            executor.run(feed_dict={input: x})
            stream.sync()
            if i > 0:
                total_time += time.time() - start

        total_time *= 1000 / (TOTAL_ROUNDS - 1)
        print('Op %s test with shape: %s takes %fms' %
            ('dropout', shape, total_time))


def test_layernorm(shapes):
    ctx = ht.gpu(0)
    stream = create_stream_handle(ctx)
    TOTAL_ROUNDS = 5
    for shape in shapes:
        input = ht.Variable(name='layer_norm_input', ctx=ctx)
        scale = ht.init.ones(name='layer_norm_scale', shape=(shape[-1], ), ctx=ctx)
        bias = ht.init.zeros(name='layer_norm_biad', shape=(shape[-1], ), ctx=ctx)
        layernorm = gpu_op.layer_normalization_op(input, scale, bias, ctx=ctx)
        executor = ht.Executor([layernorm], ctx=ctx)
        total_time = 0
        for i in range(TOTAL_ROUNDS):
            x = ht.array(np.random.uniform(-1, 1, size=shape).astype(np.float32), ctx=ctx)
            start = time.time()
            ret = executor.run(feed_dict={input: x})
            stream.sync()
            if i > 0:
                total_time += time.time() - start

        total_time *= 1000 / (TOTAL_ROUNDS - 1)
        print('Op %s test with shape: %s takes %fms' %
            ('layernorm', shape, total_time))


def test_cross_entropy(shapes):
    ctx = ht.gpu(0)
    stream = create_stream_handle(ctx)
    TOTAL_ROUNDS = 2
    for shape in shapes:
        arr_x = ht.array(np.random.uniform(-1, 1, size=shape).astype(np.float32), ctx=ctx)
        arr_y = ht.array(np.random.randint(0, shape[1], size=(shape[0], 1)).astype(np.int32), ctx=ctx)
        out = ht.array(np.random.uniform(-1, 1, size=(shape[0])), ctx=ctx)
        total_time = 0
        for i in range(TOTAL_ROUNDS):
            start = time.time()
            gpu_opl.cross_entropy(arr_x, arr_y, out, stream=stream)
            stream.sync()
            if i > 0:
                total_time += time.time() - start

        total_time *= 1000 / (TOTAL_ROUNDS - 1)
        print('Op %s test with shape: %s takes %fms' %
            ('cross_entropy', shape, total_time))


def test_linear(shapes):
    ctx = ht.gpu(0)
    stream = create_stream_handle(ctx)
    TOTAL_ROUNDS = 5
    for shape in shapes:
        nodeA = ht.Variable(name='linear_node_A', ctx=ctx)
        nodeB = ht.Variable(name='linear_node_B', ctx=ctx)
        bias = ht.Variable(name='linear_bias', ctx=ctx)
        linear = gpu_op.linear_op(nodeA, nodeB, bias, ctx=ctx)
        executor = ht.Executor([linear], ctx=ctx)
        total_time = 0
        for i in range(TOTAL_ROUNDS):
            x = ht.array(np.random.uniform(-1, 1, size=shape).astype(np.float32), ctx=ctx)
            y = ht.array(np.random.uniform(-1, 1, size=(shape[1], shape[1]*2)).astype(np.float32), ctx=ctx)
            b = ht.array(np.random.uniform(-1, 1, size=(shape[1]*2)).astype(np.float32), ctx=ctx)
            start = time.time()
            executor.run(feed_dict={nodeA: x, nodeB: y, bias: b})
            stream.sync()
            if i > 0:
                total_time += time.time() - start

        total_time *= 1000 / (TOTAL_ROUNDS - 1)
        print('Op %s test with shape: %s takes %fms' %
            ('linear', shape, total_time))


def test_slice(shapes, slices):
    ctx = ht.gpu(0)
    stream = create_stream_handle(ctx)
    TOTAL_ROUNDS = 5
    for shape in shapes:
        total_time = 0
        input = ht.array(np.random.normal(size=shape), ctx=ctx)
        output_shape = []
        for dim, slice in zip(shape, slices):
            output_shape.append(dim - slice)
        output = ht.array(np.random.normal(size=output_shape), ctx=ctx)
        for i in range(TOTAL_ROUNDS):
            start = time.time()
            gpu_opl.matrix_slice(input, output, slices, stream=stream)
            stream.sync()
            if i > 0:
                total_time += time.time() - start

        total_time *= 1000 / (TOTAL_ROUNDS - 1)
        print('Op %s test with shape: %s takes %fms' %
            ('slice', shape, total_time))


def test_transpose(shapes, perm):
    ctx = ht.gpu(0)
    stream = create_stream_handle(ctx)
    TOTAL_ROUNDS = 5
    for shape in shapes:
        total_time = 0
        input = ht.array(np.random.normal(size=shape), ctx=ctx)
        output_shape = []
        for i in range(len(shape)):
            output_shape.append(shape[i])
        output = ht.array(np.random.normal(size=output_shape), ctx=ctx)
        for i in range(TOTAL_ROUNDS):
            start = time.time()
            gpu_opl.matrix_transpose(input, output, perm, stream=stream)
            stream.sync()
            if i > 0:
                total_time += time.time() - start

        total_time *= 1000 / (TOTAL_ROUNDS - 1)
        print('Op %s test with shape: %s takes %fms' %
            ('transpose', shape, total_time))


shapes = [(64, 1024, 392), (256, 1024, 392), (1024, 1024, 392), (1024, 1024, 784)]
shapes_conv = [(64, 392, 32, 32), (256, 392, 32, 32), (1024, 392, 32, 32), (1024, 784, 32, 32)]
shapes_1d = [(64, 1024*392), (256, 1024*392), (1024, 1024*392), (1024, 1024*784)]
shapes_linear = [(64*1024, 392), (256*1024, 392), (1024*1024, 392), (1024*1024, 784)]

# test_common(gpu_opl.matrix_elementwise_add_by_const, 1, shapes, has_const=True)
# test_common(gpu_opl.matrix_elementwise_add, 2, shapes)
# test_argmax(shapes)
# test_common(gpu_opl.array_set, 1, shapes, False, False, 0.5)
# test_pooling(gpu_opl.average_pooling2d, 1, shapes_conv, 3, 3, 1, 1)
# test_matmul(shapes)
# test_broadcast_to(shapes)
# test_common(gpu_opl.clone, 1, shapes)
# test_concat(shapes)
# test_conv2d(shapes_conv)
# test_reduce_sum(shapes_conv, [0, 2, 3])
# test_cross_entropy(shapes_linear)
# test_common(gpu_opl.matrix_dot, 2, shapes)
# test_dropout(shapes)
# test_common(gpu_opl.gelu, 1, shapes)
# test_layernorm(shapes)
# test_linear(shapes_linear)
# test_common(gpu_opl.log_link, 1, shapes)
# test_common(gpu_opl.CuDNN_softmax, 1, shapes_linear)
# test_common(gpu_opl.sigmoid, 1, shapes)
# test_slice(shapes, slices=[0, 512, 0])
# test_transpose(shapes, [0, 2, 1])
# test_common(gpu_opl.where, 3, shapes)
