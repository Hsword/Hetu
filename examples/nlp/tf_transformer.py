import numpy as np
import tensorflow as tf

from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)


def ln(inputs, epsilon=1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape,
                               initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape,
                                initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def get_token_embeddings(vocab_size, num_units, initializer=tf.contrib.layers.xavier_initializer(), zero_pad=True):
    '''Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.

    Returns
    weight variable: (V, E)
    '''
    with tf.variable_scope("shared_weight_matrix"):
        embeddings = tf.get_variable('weight_mat',
                                     dtype=tf.float32,
                                     shape=(vocab_size, num_units),
                                     initializer=initializer)
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings


def multihead_attention(
        queries, keys, values,
        batch_size, hidden_size,
        num_attention_heads=8,
        query_act=None, key_act=None, value_act=None,
        attention_mask=None,
        attention_probs_dropout_prob=0.0,
        training=True, causality=False,
        scope="multihead_attention"):

    def transpose_for_scores(input_tensor):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, -1, num_attention_heads, hidden_size // num_attention_heads])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    size_per_head = hidden_size // num_attention_heads
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # linear transformation
        query_layer = tf.layers.dense(
            queries, hidden_size, activation=query_act)  # (N, T_q, d_model)
        key_layer = tf.layers.dense(
            keys, hidden_size, activation=key_act)  # (N, T_k, d_model)
        value_layer = tf.layers.dense(
            values, hidden_size, activation=value_act)  # (N, T_k, d_model)

        # transpose
        query_layer = transpose_for_scores(
            query_layer)  # (N, h, T_q, d_model/h)
        key_layer = transpose_for_scores(key_layer)  # (N, h, T_k, d_model/h)
        value_layer = transpose_for_scores(
            value_layer)  # (N, h, T_k, d_model/h)

        # score
        attention_scores = tf.matmul(
            query_layer, key_layer, transpose_b=True)  # (N, h, T_q, T_k)
        attention_scores /= size_per_head ** 0.5

        # mask
        if attention_mask is not None:
            attention_mask = tf.to_float(attention_mask)  # (N, T_k)
            attention_mask = tf.reshape(
                attention_mask, [batch_size, 1, 1, -1])  # (N, 1, 1, T_k)
            attention_scores = attention_scores + \
                attention_mask * (-2**32+1)  # (N, h, T_q, T_k)
        if causality:
            diag_vals = tf.ones_like(
                attention_scores[0, 0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(
                diag_vals).to_dense()  # (T_q, T_k)
            future_masks = tf.broadcast_to(
                tril, [batch_size, num_attention_heads, tril.shape[0], tril.shape[1]])  # (N, h, T_q, T_k)
            paddings = tf.ones_like(future_masks) * (-2**32+1)
            attention_scores = tf.where(
                tf.equal(future_masks, 0), paddings, attention_scores)

        # probs
        attention_probs = tf.nn.softmax(attention_scores)  # (N, h, T_q, T_k)
        attention_probs = tf.layers.dropout(
            attention_probs, rate=attention_probs_dropout_prob, training=training)
        # (N, h, T_q, d_model/h)
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(
            context_layer, [0, 2, 1, 3])  # (N, T_q, h, d_model/h)
        outputs = tf.reshape(context_layer, [
                             batch_size, -1, num_attention_heads * size_per_head])  # (N, T_q, d_model)

        # Residual connection
        outputs += queries  # (N, T_q, d_model)

        # Normalize
        outputs = ln(outputs)  # (N, T_q, d_model)

    return outputs


def ff(inputs, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3

    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])
        # Residual connection
        outputs += inputs
        # Normalize
        outputs = ln(outputs)
    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    V = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1-epsilon) * inputs) + (epsilon / V)


def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1]  # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(
            tf.range(T), 0), [N, 1])  # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i % 2)/E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(
            position_enc, tf.float32)  # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)

# def noam_scheme(init_lr, global_step, warmup_steps=4000.):
#     '''Noam scheme learning rate decay
#     init_lr: initial learning rate. scalar.
#     global_step: scalar.
#     warmup_steps: scalar. During warmup_steps, learning rate increases
#         until it reaches init_lr.
#     '''
#     step = tf.cast(global_step + 1, dtype=tf.float32)
#     return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


class Transformer(object):
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''

    def __init__(self, hp):
        self.hp = hp
        # self.token2idx, self.idx2token = load_vocab(hp.vocab)
        self.embeddings = get_token_embeddings(
            self.hp.vocab_size, self.hp.d_model, zero_pad=True)

    def encode(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x = xs

            # src_masks
            src_masks = tf.math.equal(x, 0)  # (N, T1)

            # embedding
            enc = tf.nn.embedding_lookup(
                self.embeddings, x)  # (N, T1, d_model)
            enc *= self.hp.d_model**0.5  # scale

            enc += positional_encoding(enc, self.hp.maxlen1)
            enc = tf.layers.dropout(
                enc, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(
                        queries=enc, keys=enc, values=enc,
                        batch_size=self.hp.batch_size, hidden_size=self.hp.d_model,
                        num_attention_heads=self.hp.num_heads,
                        attention_mask=src_masks,
                        attention_probs_dropout_prob=self.hp.dropout_rate,
                        training=training,
                        causality=False
                    )
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory, src_masks

    def decode(self, ys, memory, src_masks, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)
        src_masks: (N, T1)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            decoder_inputs = ys

            # tgt_masks
            tgt_masks = tf.math.equal(decoder_inputs, 0)  # (N, T2)

            # embedding
            dec = tf.nn.embedding_lookup(
                self.embeddings, decoder_inputs)  # (N, T2, d_model)
            dec *= self.hp.d_model ** 0.5  # scale

            dec += positional_encoding(dec, self.hp.maxlen2)
            dec = tf.layers.dropout(
                dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(
                        queries=dec, keys=dec, values=dec,
                        batch_size=self.hp.batch_size, hidden_size=self.hp.d_model,
                        num_attention_heads=self.hp.num_heads,
                        attention_mask=tgt_masks,
                        attention_probs_dropout_prob=self.hp.dropout_rate,
                        training=training,
                        causality=True,
                        scope="self_attention"
                    )
                    # Vanilla attention
                    dec = multihead_attention(
                        queries=dec, keys=memory, values=memory,
                        batch_size=self.hp.batch_size, hidden_size=self.hp.d_model,
                        num_attention_heads=self.hp.num_heads,
                        attention_mask=src_masks,
                        attention_probs_dropout_prob=self.hp.dropout_rate,
                        training=training,
                        causality=False,
                        scope="vanilla_attention"
                    )
                    # Feed Forward
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings)  # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', dec, weights)  # (N, T2, vocab_size)
        # y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits

    def train(self, xs, ys):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        memory, src_masks = self.encode(xs)
        logits = self.decode(ys[0], memory, src_masks)

        # train scheme
        y = ys[1]
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.vocab_size))
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=y_)

        return loss

    # def eval(self, xs, ys):
    #     '''Predicts autoregressively
    #     At inference, input ys is ignored.
    #     Returns
    #     y_hat: (N, T2)
    #     '''
    #     decoder_inputs, y, y_seqlen, sents2 = ys

    #     decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<s>"]
    #     ys = (decoder_inputs, y, y_seqlen, sents2)

    #     memory, sents1, src_masks = self.encode(xs, False)

    #     logging.info("Inference graph is being built. Please be patient.")
    #     for _ in tqdm(range(self.hp.maxlen2)):
    #         logits, y_hat, y, sents2 = self.decode(ys, memory, src_masks, False)
    #         if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break

    #         _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
    #         ys = (_decoder_inputs, y, y_seqlen, sents2)

    #     # monitor a random sample
    #     n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
    #     sent1 = sents1[n]
    #     pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
    #     sent2 = sents2[n]

    #     tf.summary.text("sent1", sent1)
    #     tf.summary.text("pred", pred)
    #     tf.summary.text("sent2", sent2)
    #     summaries = tf.summary.merge_all()

    #     return y_hat, summaries


# def convert_idx_to_token_tensor(inputs, idx2token):
#     '''Converts int32 tensor to string tensor.
#     inputs: 1d int32 tensor. indices.
#     idx2token: dictionary

#     Returns
#     1d string tensor.
#     '''
#     def my_func(inputs):
#         return " ".join(idx2token[elem] for elem in inputs)

#     return tf.py_func(my_func, [inputs], tf.string)

# def load_vocab(vocab_fpath):
#     '''Loads vocabulary file and returns idx<->token maps
#     vocab_fpath: string. vocabulary file path.
#     Note that these are reserved
#     0: <pad>, 1: <unk>, 2: <s>, 3: </s>

#     Returns
#     two dictionaries.
#     '''
#     vocab = [line.split()[0] for line in open(vocab_fpath, 'r', encoding='utf-8').read().splitlines()]
#     token2idx = {token: idx for idx, token in enumerate(vocab)}
#     idx2token = {idx: token for idx, token in enumerate(vocab)}
#     return token2idx, idx2token
