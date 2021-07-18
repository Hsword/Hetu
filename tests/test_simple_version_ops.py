import numpy as np
import tensorflow as tf
import hetu as ht


def test_broadcast_shape(shape1=(3, 1), shape2=(2, 3, 4)):
    ctx = ht.gpu(1)
    x = np.random.random(shape1).astype(np.float32)
    ath_x = ht.Variable(name='x', value=x)
    ath_y = ht.broadcast_shape_op(ath_x, shape2)
    ath_grad = ht.gradients(ath_y, [ath_x])[0]
    executor = ht.Executor([ath_y, ath_grad], ctx=ctx, enable_lazy=False)
    ath_results = [var.asnumpy() for var in executor.run()]

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


test_broadcast_shape()
test_broadcast_shape((1,), (2, 3, 4, 5))
test_broadcast_shape((1, 1, 3, 1), (9, 8, 3, 7))


def test_broadcast(shape1=(3, 1), shape2=(2, 3, 4)):
    ctx = ht.gpu(1)
    x = np.random.random(shape1).astype(np.float32)
    y = np.random.random(shape2).astype(np.float32)
    ath_x = ht.Variable(name='x', value=x)
    ath_z = ht.Variable(name='y', value=y)
    ath_y = ht.broadcastto_op(ath_x, ath_z)
    ath_grad = ht.gradients(ath_y, [ath_x])[0]
    executor = ht.Executor([ath_y, ath_grad], ctx=ctx, enable_lazy=False)
    ath_results = [var.asnumpy() for var in executor.run()]

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


def test_transpose(shape=(2, 3, 4, 5), perm=None):
    ctx = ht.gpu(1)
    x = np.random.random(shape).astype(np.float32)
    ath_x = ht.Variable(name='x', value=x)
    ath_y = ht.transpose_op(ath_x, perm)
    ath_grad = ht.gradients(ath_y, [ath_x])[0]
    executor = ht.Executor([ath_y, ath_grad], ctx=ctx, enable_lazy=False)
    ath_results = [var.asnumpy() for var in executor.run()]

    tf_x = tf.convert_to_tensor(x)
    tf_y = tf.transpose(tf_x, perm)
    tf_grad = tf.gradients(tf_y, tf_x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf_results = sess.run([tf_y, tf_grad])

    np.testing.assert_allclose(ath_results[0], tf_results[0])
    np.testing.assert_allclose(ath_results[1], np.reshape(
        tf_results[1], ath_results[1].shape))
    print('Passed transpose shape op test with shape ', shape, ' and perm ', perm)


test_transpose()
test_transpose(perm=(1, 0, 3, 2))
test_transpose((5, 6, 7, 8, 9), (4, 2, 0, 3, 1))


def test_slice(shape1=(7, 11, 13), shape2=(2, 3, 4), begin_pos=(0, 0, 0)):
    ctx = ht.gpu(1)
    x = np.random.random(shape1).astype(np.float32)
    ath_x = ht.Variable(name='x', value=x)
    ath_y = ht.slice_op(ath_x, begin_pos, shape2)
    ath_grad = ht.gradients(ath_y, [ath_x])[0]
    executor = ht.Executor([ath_y, ath_grad], ctx=ctx, enable_lazy=False)
    ath_results = [var.asnumpy() for var in executor.run()]

    tf_x = tf.convert_to_tensor(x)
    tf_y = tf.slice(tf_x, begin_pos, shape2)
    tf_grad = tf.gradients(tf_y, tf_x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf_results = sess.run([tf_y, tf_grad])

    np.testing.assert_allclose(ath_results[0], tf_results[0])
    np.testing.assert_allclose(ath_results[1], np.reshape(
        tf_results[1], ath_results[1].shape))
    print('Passed slice op test with shape ', shape1,
          shape2, ' and begin pos ', begin_pos)


test_slice()
test_slice(shape1=(5,), shape2=(2,), begin_pos=(1,))
test_slice(shape1=(2, 3, 4, 5), shape2=(1, 2, 3, 4), begin_pos=(0, 0, 0, 1))
test_slice(shape1=(2, 3, 4, 5, 6), shape2=(
    1, 2, 3, 4, 5), begin_pos=(0, 1, 0, 1, 0))


def test_add(shape=(2, 3, 4, 5), ctx=ht.gpu(1)):
    x = np.random.random(shape).astype(np.float32)
    z = np.random.random(shape).astype(np.float32)
    ath_x = ht.Variable(name='x', value=x)
    ath_z = ht.Variable(name='z', value=z)
    ath_y = ht.add_op(ath_x, ath_z)
    executor = ht.Executor([ath_y], ctx=ctx, enable_lazy=False)
    ath_results = [var.asnumpy() for var in executor.run()]

    tf_x = tf.convert_to_tensor(x)
    tf_z = tf.convert_to_tensor(z)
    tf_y = tf_x + tf_z
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf_results = sess.run([tf_y])

    np.testing.assert_allclose(ath_results[0], tf_results[0])
    print('Passed add op test with shape ', shape)


test_add()
test_add((7, 9))
test_add((4, 5, 6, 7, 8))
test_add(ctx=ht.cpu(0))
test_add((7, 9), ctx=ht.cpu(0))
test_add((4, 5, 6, 7, 8), ctx=ht.cpu(0))


def test_add_broadcast(shape1=(2, 3, 4, 5), shape2=(1, 4, 1), ctx=ht.gpu(1)):
    x = np.random.random(shape1).astype(np.float32)
    z = np.random.random(shape2).astype(np.float32)
    ath_x = ht.Variable(name='x', value=x)
    ath_z = ht.Variable(name='z', value=z)
    ath_y = ht.add_op(ath_x, ath_z)
    executor = ht.Executor([ath_y], ctx=ctx, enable_lazy=False)
    ath_results = [var.asnumpy() for var in executor.run()]

    tf_x = tf.convert_to_tensor(x)
    tf_z = tf.convert_to_tensor(z)
    tf_y = tf_x + tf_z
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf_results = sess.run([tf_y])

    np.testing.assert_allclose(ath_results[0], tf_results[0])
    print('Passed add op test with shape ', shape1, shape2)


test_add_broadcast()
test_add_broadcast((7, 9), (9,))
test_add_broadcast((1, 1), (4, 5, 6, 7, 8))
test_add_broadcast(ctx=ht.cpu(0))
test_add_broadcast((9,), (7, 9), ctx=ht.cpu(0))
test_add_broadcast((4, 5, 6, 7, 8), (1, 7, 1), ctx=ht.cpu(0))


def test_add_lazy(shape1=(1, 4, 1), shape2=(2, 3, 4, 5), ctx=ht.gpu(1)):
    x = np.random.random(shape1).astype(np.float32)
    z = np.random.random(shape2).astype(np.float32)
    ath_x = ht.Variable(name='x', value=x)
    ath_z = ht.Variable(name='z', value=z)
    ath_y = ht.add_op(ht.broadcast_shape_op(ath_x, shape2), ath_z)
    executor = ht.Executor([ath_y], ctx=ctx)
    ath_results = [var.asnumpy() for var in executor.run()]

    tf_x = tf.convert_to_tensor(x)
    tf_z = tf.convert_to_tensor(z)
    tf_y = tf_x + tf_z
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf_results = sess.run([tf_y])

    np.testing.assert_allclose(ath_results[0], tf_results[0])
    print('Passed add op test with shape ', shape1, shape2)


test_add_lazy()
test_add_lazy((9,), (7, 9))
test_add_lazy((1, 1), (4, 5, 6, 7, 8))
test_add_lazy(ctx=ht.cpu(0))
test_add_lazy((9,), (7, 9), ctx=ht.cpu(0))
test_add_lazy((1, 7, 1), (4, 5, 6, 7, 8), ctx=ht.cpu(0))
