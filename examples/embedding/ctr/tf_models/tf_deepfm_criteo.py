import tensorflow as tf


def dfm_criteo(dense_input, sparse_input, y_, partitioner=None, part_all=True, param_on_gpu=True):
    feature_dimension = 33762577
    embedding_size = 128
    learning_rate = 0.01 / 8  # here to comply with HETU
    all_partitioner, embed_partitioner = (
        partitioner, None) if part_all else (None, partitioner)
    with tf.compat.v1.variable_scope('dfm', dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01), partitioner=all_partitioner):
        with tf.device('/cpu:0'):
            Embedding1 = tf.compat.v1.get_variable(name="Embedding1", shape=(
                feature_dimension, 1), partitioner=embed_partitioner)
            Embedding2 = tf.compat.v1.get_variable(name="embeddings", shape=(
                feature_dimension, embedding_size), partitioner=embed_partitioner)
            sparse_1dim_input = tf.nn.embedding_lookup(
                Embedding1, sparse_input)
            sparse_2dim_input = tf.nn.embedding_lookup(
                Embedding2, sparse_input)

        device = '/gpu:0' if param_on_gpu else '/cpu:0'
        with tf.device(device):
            FM_W = tf.compat.v1.get_variable(name='FM_W', shape=[13, 1])
            W1 = tf.compat.v1.get_variable(
                name='W1', shape=[26*embedding_size, 256])
            W2 = tf.compat.v1.get_variable(name='W2', shape=[256, 256])
            W3 = tf.compat.v1.get_variable(name='W3', shape=[256, 1])

        with tf.device('/gpu:0'):
            fm_dense_part = tf.matmul(dense_input, FM_W)
            fm_sparse_part = tf.reduce_sum(sparse_1dim_input, 1)
            # fst order output
            y1 = fm_dense_part + fm_sparse_part

            sparse_2dim_sum = tf.reduce_sum(sparse_2dim_input, 1)
            sparse_2dim_sum_square = tf.multiply(
                sparse_2dim_sum, sparse_2dim_sum)

            sparse_2dim_square = tf.multiply(
                sparse_2dim_input, sparse_2dim_input)
            sparse_2dim_square_sum = tf.reduce_sum(sparse_2dim_square, 1)
            sparse_2dim = sparse_2dim_sum_square + -1 * sparse_2dim_square_sum
            sparse_2dim_half = sparse_2dim * 0.5
            # snd order output
            y2 = tf.reduce_sum(sparse_2dim_half, 1, keepdims=True)

            # DNN
            flatten = tf.reshape(sparse_2dim_input, (-1, 26*embedding_size))
            fc1 = tf.matmul(flatten, W1)
            relu1 = tf.nn.relu(fc1)
            fc2 = tf.matmul(relu1, W2)
            relu2 = tf.nn.relu(fc2)
            y3 = tf.matmul(relu2, W3)

            y4 = y1 + y2
            y = y4 + y3
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))

            optimizer = tf.compat.v1.train.GradientDescentOptimizer(
                learning_rate)
            return loss, y, optimizer
