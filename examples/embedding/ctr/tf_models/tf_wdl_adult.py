import tensorflow as tf
import numpy as np


def wdl_adult(X_deep, X_wide, y_, cluster=None, task_id=None):
    lr_ = 5 / 128
    dim_wide = 809
    dim_deep = 68
    use_ps = cluster is not None

    if use_ps:
        device = tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d/gpu:0" % (task_id),
            cluster=cluster))
    else:
        device = tf.device('/gpu:0')
        global_step = tf.Variable(0, name="global_step", trainable=False)
    with device:
        if use_ps:
            global_step = tf.Variable(0, name="global_step", trainable=False)

        rand = np.random.RandomState(seed=123)
        W = tf.Variable(rand.normal(scale=0.1, size=[
                        dim_wide+20, 2]), dtype=tf.float32)
        W1 = tf.Variable(rand.normal(scale=0.1, size=[
                         dim_deep, 50]), dtype=tf.float32)
        b1 = tf.Variable(rand.normal(scale=0.1, size=[50]), dtype=tf.float32)
        W2 = tf.Variable(rand.normal(
            scale=0.1, size=[50, 20]), dtype=tf.float32)
        b2 = tf.Variable(rand.normal(scale=0.1, size=[20]), dtype=tf.float32)

        Embedding = []

        for i in range(8):
            Embedding.append(tf.Variable(rand.normal(
                scale=0.1, size=[20, 8]), dtype=tf.float32))

        # deep
        X_deep_input = None
        for i in range(8):
            now = tf.nn.embedding_lookup(Embedding[i], X_deep[i])
            now = tf.reshape(now, (-1, 8))
            if X_deep_input is None:
                X_deep_input = now
            else:
                X_deep_input = tf.concat([X_deep_input, now], 1)

        for i in range(4):
            now = tf.reshape(X_deep[i + 8], (-1, 1))
            X_deep_input = tf.concat([X_deep_input, now], 1)

        mat1 = tf.matmul(X_deep_input, W1)
        add1 = tf.add(mat1, b1)
        relu1 = tf.nn.relu(add1)
        dropout1 = relu1
        mat2 = tf.matmul(dropout1, W2)
        add2 = tf.add(mat2, b2)
        relu2 = tf.nn.relu(add2)
        dropout2 = relu2
        dmodel = dropout2

        # wide
        wmodel = tf.concat([X_wide, dmodel], 1)
        wmodel = tf.matmul(wmodel, W)

        y = wmodel
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        )

        optimizer = tf.train.GradientDescentOptimizer(lr_)
        train_op = optimizer.minimize(loss, global_step=global_step)

        if use_ps:
            return loss, y, train_op, global_step
        else:
            return loss, y, train_op
