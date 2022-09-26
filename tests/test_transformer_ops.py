import numpy as np
import hetu as ht
from hetu import gpu_links as gpu_op


def test_batch_matmul(shape1=(7, 4, 6), shape2=(7, 6, 5), transA=False, transB=False):
    executor_ctx = ht.gpu(1)

    if transA:
        shape1 = tuple(list(shape1)[:-2] + [shape1[-1], shape1[-2]])
    if transB:
        shape2 = tuple(list(shape2)[:-2] + [shape2[-1], shape2[-2]])

    data = np.random.normal(0.0, 0.2, shape1).astype(np.float32)
    weights = np.random.normal(0.0, 0.1, shape2).astype(np.float32)

    ath_data = ht.Variable(name='data')
    ath_weights = ht.Variable(name='weights')
    ath_output = ht.batch_matmul_op(
        ath_data, ath_weights, trans_A=transA, trans_B=transB)

    ath_grads = ht.gradients(ath_output, [ath_data, ath_weights])

    executor = ht.Executor(
        [ath_output] + ath_grads,
        ctx=executor_ctx
    )

    ath_results = executor.run(
        feed_dict={ath_data: data, ath_weights: weights})
    ath_results = [res.asnumpy() for res in ath_results]

    import tensorflow as tf
    tf_data = tf.placeholder(name='data', dtype=tf.float32)
    tf_weights = tf.placeholder(name='weights', dtype=tf.float32)
    tf_output = tf.matmul(tf_data, tf_weights,
                          transpose_a=transA, transpose_b=transB)
    tf_grads = tf.gradients(tf_output, [tf_data, tf_weights])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf_results = sess.run([tf_output] + tf_grads,
                              feed_dict={tf_data: data, tf_weights: weights})

    np.testing.assert_allclose(ath_results[0], tf_results[0], atol=1e-6)
    np.testing.assert_allclose(ath_results[1], tf_results[1], atol=1e-6)
    np.testing.assert_allclose(ath_results[2], tf_results[2], atol=1e-6)
    print('Pass batch matmul op test with shape ', shape1, shape2)


test_batch_matmul()
test_batch_matmul(transA=True)
test_batch_matmul(transB=True)
test_batch_matmul(transA=True, transB=True)

test_batch_matmul(shape1=(11, 3, 23, 17), shape2=(11, 3, 17, 13))
test_batch_matmul(shape1=(11, 3, 23, 17), shape2=(11, 3, 17, 13), transA=True)
test_batch_matmul(shape1=(11, 3, 23, 17), shape2=(11, 3, 17, 13), transB=True)
test_batch_matmul(shape1=(11, 3, 23, 17), shape2=(
    11, 3, 17, 13), transA=True, transB=True)


def test_broadcast(shape1=(3, 1), shape2=(2, 3, 4)):
    ctx = ht.gpu(1)
    x = np.random.random(shape1).astype(np.float32)
    ath_x = ht.Variable(name='x', value=x)
    ath_y = ht.broadcast_shape_op(ath_x, shape2)
    ath_grad = ht.gradients(ath_y, [ath_x])[0]
    executor = ht.Executor([ath_y, ath_grad], ctx=ctx)
    ath_results = [var.asnumpy() for var in executor.run()]

    import tensorflow as tf
    tf_x = tf.convert_to_tensor(x)
    tf_y = tf.broadcast_to(tf_x, shape2)
    tf_grad = tf.gradients(tf_y, tf_x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf_results = sess.run([tf_y, tf_grad])

    np.testing.assert_allclose(ath_results[0], tf_results[0])
    np.testing.assert_allclose(ath_results[1], np.reshape(
        tf_results[1], ath_results[1].shape))
    print('Passed broadcast shape op test with shape ', shape1, shape2)


test_broadcast()
test_broadcast((1,), (2, 3, 4, 5))
test_broadcast((1, 1, 3, 1), (9, 8, 3, 7))


def test_reduce_sum(shape=(2, 3, 4), axes=[2]):
    ctx = ht.gpu(1)
    x = np.random.random(shape).astype(np.float32)
    ath_x = ht.Variable(name='x', value=x)
    ath_y = ht.reduce_sum_op(ath_x, axes, keepdims=False)
    ath_grad = ht.gradients(ath_y, [ath_x])[0]
    executor = ht.Executor([ath_y, ath_grad], ctx=ctx)
    ath_results = [var.asnumpy() for var in executor.run()]

    import tensorflow as tf
    tf_x = tf.convert_to_tensor(x)
    tf_y = tf.reduce_sum(tf_x, axes)
    tf_grad = tf.gradients(tf_y, tf_x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf_results = sess.run([tf_y, tf_grad])

    np.testing.assert_allclose(ath_results[0], np.reshape(
        tf_results[0], ath_results[0].shape), rtol=1e-6)
    np.testing.assert_allclose(ath_results[1], np.reshape(
        tf_results[1], ath_results[1].shape), rtol=1e-6)
    print('Passed reduce sum op test with shape and axes ', shape, axes)


test_reduce_sum()
test_reduce_sum((2, 3, 4), [2, 1])
test_reduce_sum((2, 3, 4), [2, 1, 0])
test_reduce_sum((2, 3, 1, 5, 6), [1, 2, 4])


def test_reduce_mean(shape=(2, 3, 4), axes=[2]):
    ctx = ht.gpu(1)
    x = np.random.random(shape).astype(np.float32)
    ath_x = ht.Variable(name='x', value=x)
    ath_y = ht.reduce_mean_op(ath_x, axes, keepdims=False)
    ath_grad = ht.gradients(ath_y, [ath_x])[0]
    executor = ht.Executor([ath_y, ath_grad], ctx=ctx)
    ath_results = [var.asnumpy() for var in executor.run()]

    import tensorflow as tf
    tf_x = tf.convert_to_tensor(x)
    tf_y = tf.reduce_mean(tf_x, axes)
    tf_grad = tf.gradients(tf_y, tf_x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf_results = sess.run([tf_y, tf_grad])

    np.testing.assert_allclose(ath_results[0], np.reshape(
        tf_results[0], ath_results[0].shape), rtol=1e-6)
    np.testing.assert_allclose(ath_results[1], np.reshape(
        tf_results[1], ath_results[1].shape), rtol=1e-6)
    print('Passed reduce mean op test with shape and axes ', shape, axes)


test_reduce_mean()
test_reduce_mean((2, 3, 4), [2, 1])
test_reduce_mean((2, 3, 4), [2, 1, 0])
test_reduce_mean((2, 3, 1, 5, 6), [1, 2, 4])


def test_layernorm_forward(shape=(5, 3)):
    ctx = ht.gpu(1)
    # shape = (5, 3)
    last_dim = shape[-1]
    x = np.random.random(shape).astype(np.float32)
    scale = np.random.random((last_dim,)).astype(np.float32)
    bias = np.random.random((last_dim,)).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_scale = ht.array(scale, ctx=ctx)
    arr_bias = ht.array(bias, ctx=ctx)
    arr_mean = ht.empty(list(shape[:-1]) + [1], ctx=ctx)
    arr_var = ht.empty(list(shape[:-1]) + [1], ctx=ctx)
    arr_y = ht.empty((shape), ctx=ctx)
    gpu_op.layer_normalization(
        arr_x, arr_scale, arr_bias, arr_mean, arr_var, arr_y, 0.01)

    y = arr_y.asnumpy()

    np_means = x.mean(axis=-1, dtype=np.float32, keepdims=True)
    np_vars = x.var(axis=-1, dtype=np.float32, keepdims=True)
    std = np.sqrt(np_vars + 0.01, dtype=np.float32)
    centered_input = x - np_means
    normed_input = centered_input / std

    bc_shape = [1] * len(x.shape)
    bc_shape[-1] = x.shape[-1]

    y_ = scale.reshape(bc_shape) * normed_input + \
        bias.reshape(bc_shape)

    np.testing.assert_allclose(np_means, arr_mean.asnumpy(), atol=1e-6)
    np.testing.assert_allclose(np_vars, arr_var.asnumpy(), atol=1e-6)
    np.testing.assert_allclose(y_, y, atol=1e-6)
    print('Pass forward test with shape ', shape)

# test_layernorm_forward()
# test_layernorm_forward(shape=(4, 500, 67))
# test_layernorm_forward(shape=(2, 3, 5, 7, 11))


def test_layernorm_backward(shape=(5, 3)):
    ctx = ht.gpu(1)
    # shape = (5, 3)
    last_dim = shape[-1]
    grads = np.random.random(shape).astype(np.float32)
    x = np.random.random(shape).astype(np.float32)
    scale = np.random.random((last_dim,)).astype(np.float32)
    mean = np.random.random(list(shape[:-1])+[1]).astype(np.float32)
    var = np.random.random(list(shape[:-1])+[1]).astype(np.float32)

    arr_grads = ht.array(grads, ctx=ctx)
    arr_x = ht.array(x, ctx=ctx)
    arr_scale = ht.array(scale, ctx=ctx)
    arr_mean = ht.array(mean, ctx=ctx)
    arr_var = ht.array(var, ctx=ctx)

    grad_inarr = ht.empty(shape, ctx=ctx)
    grad_scale = ht.empty((last_dim,), ctx=ctx)
    grad_bias = ht.empty((last_dim,), ctx=ctx)
    gpu_op.layer_normalization_gradient(arr_grads, arr_x, arr_scale,
                                        grad_inarr, grad_scale, grad_bias, arr_mean, arr_var, 0.01)

    # numpy calculate phase
    red_axis = tuple(range(grads.ndim-1))
    np_grad_bias = grads.sum(red_axis)  # (X,)

    std = np.sqrt(var + 0.01)  # (N, 1)
    x_centered = x - mean  # (N, X)
    x_norm = x_centered / std  # (N, X)
    np_grad_scale = (grads * x_norm).sum(red_axis)  # (X,)

    last_dim = x.shape[-1]
    dx_norm = grads * scale.reshape([1] * (grads.ndim - 1) + [-1])  # (N, X)
    dvar = (dx_norm * x_centered).sum(axis=-1, keepdims=True) * - \
        0.5 / (var + 0.01) / std  # (N, 1)
    dx_mu_1 = dx_norm / std  # (N, X)
    dx_mu_2 = dvar * 2 * x_centered / last_dim  # (N, X)
    dx_1 = dx_mu_1 + dx_mu_2  # (N, X)
    dx_2 = -1 * dx_1.sum(axis=-1, keepdims=True) / last_dim  # (N, 1)
    np_grad_inarr = dx_1 + dx_2  # (N, X)

    np.testing.assert_allclose(
        np_grad_bias, grad_bias.asnumpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        np_grad_scale, grad_scale.asnumpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        np_grad_inarr, grad_inarr.asnumpy(), rtol=1e-4, atol=1e-4)
    print('Pass backward test with shape ', shape)

# test_layernorm_backward()
# test_layernorm_backward(shape=(4, 500, 67))
# test_layernorm_backward(shape=(2, 3, 5, 7, 11))


def test_layer_norm_op(shape=(5, 3)):
    # scale = np.random.random((shape[-1],)).astype(np.float32)
    # bias = np.random.random((shape[-1],)).astype(np.float32)
    scale = np.ones((shape[-1], )).astype(np.float32)
    bias = np.zeros((shape[-1], )).astype(np.float32)

    scale_data = ht.Variable(name='layer_norm_scale', value=scale)
    bias_data = ht.Variable(name='layer_norm_biad', value=bias)
    input_data = ht.Variable(name='input')
    output = ht.layer_normalization_op(
        input_data, scale_data, bias_data, 1e-12)
    grads = ht.gradients(output, [scale_data, bias_data, input_data])

    executor_ctx = ht.gpu(1)

    executor = ht.Executor(
        [output]+grads,
        ctx=executor_ctx)

    x = np.random.normal(loc=0.0, scale=1, size=shape).astype(np.float32)

    results = executor.run(feed_dict={input_data: x})
    y = results[0].asnumpy()
    grad_scale = results[1].asnumpy()
    grad_bias = results[2].asnumpy()
    grad_input = results[3].asnumpy()
    # print(y)

    np_means = x.mean(axis=-1, dtype=np.float32, keepdims=True)
    np_vars = x.var(axis=-1, dtype=np.float32, keepdims=True)
    std = np.sqrt(np_vars + 1e-12, dtype=np.float32)
    centered_input = x - np_means
    normed_input = centered_input / std

    bc_shape = [1] * len(x.shape)
    bc_shape[-1] = x.shape[-1]

    y_ = scale.reshape(bc_shape) * normed_input + \
        bias.reshape(bc_shape)

    np.testing.assert_allclose(y_, y, rtol=1e-6, atol=1e-6)
    # print(y_)

    prev_grad = np.ones(y_.shape).astype(np.float32)

    red_axis = tuple(range(prev_grad.ndim-1))
    np_grad_bias = prev_grad.sum(red_axis)  # (X,)

    std = np.sqrt(np_vars + 1e-12)  # (N, 1)
    x_centered = x - np_means  # (N, X)
    x_norm = x_centered / std  # (N, X)
    np_grad_scale = (prev_grad * x_norm).sum(red_axis)  # (X,)

    last_dim = x.shape[-1]
    dx_norm = prev_grad * \
        scale.reshape([1] * (prev_grad.ndim - 1) + [-1])  # (N, X)
    dvar = (dx_norm * x_centered).sum(axis=-1, keepdims=True) * - \
        0.5 / (np_vars + 1e-12) / std  # (N, 1)
    dx_mu_1 = dx_norm / std  # (N, X)
    dx_mu_2 = dvar * 2 * x_centered / last_dim  # (N, X)
    dx_1 = dx_mu_1 + dx_mu_2  # (N, X)
    dx_2 = -1 * dx_1.sum(axis=-1, keepdims=True) / last_dim  # (N, 1)
    np_grad_inarr = dx_1 + dx_2  # (N, X)

    np.testing.assert_allclose(grad_bias, np_grad_bias, rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(grad_scale, np_grad_scale, rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(grad_input, np_grad_inarr, rtol=1e-6, atol=1e-4)

    import tensorflow as tf
    tf_input = tf.convert_to_tensor(x)
    tf_result = tf.contrib.layers.layer_norm(
        inputs=tf_input, begin_norm_axis=-1, begin_params_axis=-1)
    tf_gamma = tf.global_variables()[-1]
    tf_beta = tf.global_variables()[-2]
    tf_grads = tf.gradients(tf_result, [tf_gamma, tf_beta, tf_input])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf_all_results = sess.run([tf_result]+tf_grads)
    y_tf = tf_all_results[0]
    tf_scale_grad = tf_all_results[1]
    tf_bias_grad = tf_all_results[2]
    tf_input_grad = tf_all_results[3]
    # print(y_tf)
    np.testing.assert_allclose(y_tf, y, rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(grad_bias, tf_bias_grad, rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(grad_scale, tf_scale_grad, rtol=1e-6, atol=1e-4)
    if shape[-1] > 100:
        atol = 1e-4
    else:
        atol = 1e-5
    np.testing.assert_allclose(grad_input, tf_input_grad, atol=1e-4)

    print('Pass op test with shape ', shape)


test_layer_norm_op()
test_layer_norm_op(shape=(4, 5, 6))
test_layer_norm_op(shape=(2, 256, 768))
test_layer_norm_op(shape=(2, 64, 3072))
