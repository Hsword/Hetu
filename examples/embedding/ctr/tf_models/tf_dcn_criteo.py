import tensorflow as tf


def cross_layer(x0, x1, device):
    # x0: input embedding feature (batch_size, 26 * embedding_size + 13)
    # x1: the output of last layer (batch_size, 26 * embedding_size + 13)

    embed_dim = x1.shape[-1]
    with tf.device(device):
        w = tf.compat.v1.get_variable(name='w', shape=(embed_dim,))
        b = tf.compat.v1.get_variable(name='b', shape=(embed_dim,))
    x_1w = tf.tensordot(tf.reshape(x1, [-1, 1, embed_dim]), w, axes=1)
    cross = x0 * x_1w
    return cross + x1 + b


def build_cross_layer(x0, num_layers=3, device=tf.device('/gpu:0')):
    x1 = x0
    for i in range(num_layers):
        with tf.compat.v1.variable_scope('layer%d' % i):
            x1 = cross_layer(x0, x1, device)
    return x1


def dcn_criteo(dense_input, sparse_input, y_, partitioner=None, part_all=True, param_on_gpu=True):
    feature_dimension = 33762577
    embedding_size = 128
    learning_rate = 0.003 / 8  # here to comply with HETU
    all_partitioner, embed_partitioner = (
        partitioner, None) if part_all else (None, partitioner)
    with tf.compat.v1.variable_scope('dcn', dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01), partitioner=all_partitioner):
        with tf.device('/cpu:0'):
            Embedding = tf.compat.v1.get_variable(name="Embedding", shape=(
                feature_dimension, embedding_size), partitioner=embed_partitioner)
            sparse_input_embedding = tf.nn.embedding_lookup(
                Embedding, sparse_input)

        device = '/gpu:0' if param_on_gpu else '/cpu:0'
        with tf.device(device):
            W1 = tf.compat.v1.get_variable(
                name='W1', shape=[26*embedding_size + 13, 256])
            W2 = tf.compat.v1.get_variable(name='W2', shape=[256, 256])
            W3 = tf.compat.v1.get_variable(name='W3', shape=[256, 256])
            W4 = tf.compat.v1.get_variable(
                name='W4', shape=[256 + 26 * embedding_size + 13, 1])

        with tf.device('/gpu:0'):
            flatten = tf.reshape(sparse_input_embedding,
                                 (-1, 26*embedding_size))
            x = tf.concat((flatten, dense_input), 1)
            # CrossNet
            cross_output = build_cross_layer(x, num_layers=3, device=device)
            # DNN
            flatten = x

            fc1 = tf.matmul(flatten, W1)
            relu1 = tf.nn.relu(fc1)
            fc2 = tf.matmul(relu1, W2)
            relu2 = tf.nn.relu(fc2)
            y3 = tf.matmul(relu2, W3)

            y4 = tf.concat((cross_output, y3), 1)
            y = tf.matmul(y4, W4)
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))

            optimizer = tf.compat.v1.train.GradientDescentOptimizer(
                learning_rate)
            return loss, y, optimizer
