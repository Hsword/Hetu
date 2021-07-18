import tensorflow as tf


def neural_mf(user_input, item_input, y_, num_users, num_items, embed_partitioner=None):
    embed_dim = 8
    layers = [64, 32, 16, 8]
    learning_rate = 0.01
    with tf.compat.v1.variable_scope('nmf', dtype=tf.float32):
        with tf.device('/cpu:0'):
            User_Embedding = tf.compat.v1.get_variable(name="user_embed", shape=(
                num_users, embed_dim + layers[0] // 2), initializer=tf.random_normal_initializer(stddev=0.01), partitioner=embed_partitioner)
            Item_Embedding = tf.compat.v1.get_variable(name="item_embed", shape=(
                num_items, embed_dim + layers[0] // 2), initializer=tf.random_normal_initializer(stddev=0.01), partitioner=embed_partitioner)

            user_latent = tf.nn.embedding_lookup(User_Embedding, user_input)
            item_latent = tf.nn.embedding_lookup(Item_Embedding, item_input)

            W1 = tf.compat.v1.get_variable(name='W1', shape=(
                layers[0], layers[1]), initializer=tf.random_normal_initializer(stddev=0.1))
            W2 = tf.compat.v1.get_variable(name='W2', shape=(
                layers[1], layers[2]), initializer=tf.random_normal_initializer(stddev=0.1))
            W3 = tf.compat.v1.get_variable(name='W3', shape=(
                layers[2], layers[3]), initializer=tf.random_normal_initializer(stddev=0.1))
            W4 = tf.compat.v1.get_variable(name='W4', shape=(
                embed_dim + layers[3], 1), initializer=tf.random_normal_initializer(stddev=0.1))

        with tf.device('/gpu:0'):
            mf_user_latent, mlp_user_latent = tf.split(
                user_latent, [embed_dim, layers[0] // 2], 1)
            mf_item_latent, mlp_item_latent = tf.split(
                item_latent, [embed_dim, layers[0] // 2], 1)
            mf_vector = tf.multiply(mf_user_latent, mf_item_latent)
            mlp_vector = tf.concat((mlp_user_latent, mlp_item_latent), 1)
            fc1 = tf.matmul(mlp_vector, W1)
            relu1 = tf.nn.relu(fc1)
            fc2 = tf.matmul(relu1, W2)
            relu2 = tf.nn.relu(fc2)
            fc3 = tf.matmul(relu2, W3)
            relu3 = tf.nn.relu(fc3)
            concat_vector = tf.concat((mf_vector, relu3), 1)
            y = tf.reshape(tf.matmul(concat_vector, W4), (-1,))
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_)
            loss = tf.reduce_mean(loss)
            y = tf.sigmoid(y)
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(
                learning_rate)
    return loss, y, optimizer
