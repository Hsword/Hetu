import hetu as ht
from hetu import init
import numpy as np


def layer_norm(
    input_tensor,
    feature_size,
    eps=1e-8
):
    scale = init.ones(name='layer_norm_scale', shape=(feature_size, ))
    bias = init.zeros(name='layer_norm_biad', shape=(feature_size, ))
    return ht.layer_normalization_op(input_tensor, scale, bias, eps=eps)


def dense(
    input_tensor,
    fan_in,
    fan_out,
    activation=None,
    kernel_initializer=init.xavier_normal,
    bias_initializer=init.zeros
):
    weights = kernel_initializer(name='dense_weights', shape=(fan_in, fan_out))
    bias = bias_initializer(name='dense_bias', shape=(fan_out,))
    outputs = ht.matmul_op(input_tensor, weights)
    outputs = outputs + ht.broadcastto_op(bias, outputs)
    if activation is not None:
        outputs = activation(outputs)
    return outputs


def dropout(
    input_tensor,
    dropout_prob
):
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor
    output = ht.dropout_op(input_tensor, 1.0 - dropout_prob)
    return output


def get_token_embeddings(vocab_size, num_units, initializer=init.xavier_normal, zero_pad=True):
    if zero_pad:
        embedding_part = initializer(
            name='embedding_table', shape=(vocab_size-1, num_units))
        padding_zero = init.zeros(
            name='padding_zero', shape=(1, num_units), trainable=False)
        embeddings = ht.concat_op(padding_zero, embedding_part)
    else:
        embeddings = initializer(
            name='embedding_table', shape=(vocab_size, num_units))
    return embeddings


def multihead_attention(
        queries, keys, values,
        config,
        query_act=None, key_act=None, value_act=None,
        attention_mask=None,
        causality=False):

    def transpose_for_scores(input_tensor):
        output_tensor = ht.array_reshape_op(
            input_tensor, [config.batch_size, -1, config.num_heads, config.d_model // config.num_heads])

        output_tensor = ht.transpose_op(output_tensor, [0, 2, 1, 3])
        return output_tensor

    batch_size = config.batch_size
    hidden_size = config.d_model
    num_attention_heads = config.num_heads
    caus_len = config.maxlen2 - 1
    attention_probs_dropout_prob = config.dropout_rate

    size_per_head = hidden_size // num_attention_heads

    # reshape to 2d
    queries2d = ht.array_reshape_op(
        queries, [-1, hidden_size])  # (N * T_q, d_model)
    keys2d = ht.array_reshape_op(keys, [-1, hidden_size])  # (N * T_k, d_model)
    values2d = ht.array_reshape_op(
        values, [-1, hidden_size])  # (N * T_k, d_model)

    # linear transformation
    query_layer = dense(queries2d, hidden_size, hidden_size,
                        query_act)  # (N * T_k, d_model)
    key_layer = dense(keys2d, hidden_size, hidden_size,
                      key_act)  # (N * T_k, d_model)
    value_layer = dense(values2d, hidden_size, hidden_size,
                        value_act)  # (N * T_k, d_model)

    # transpose
    query_layer = transpose_for_scores(query_layer)  # (N, h, T_q, d_model/h)
    key_layer = transpose_for_scores(key_layer)  # (N, h, T_k, d_model/h)
    value_layer = transpose_for_scores(value_layer)  # (N, h, T_k, d_model/h)

    # score
    attention_scores = ht.batch_matmul_op(
        query_layer, key_layer, trans_B=True)  # (N, h, T_q, T_k)
    attention_scores = attention_scores * (1.0 / np.sqrt(float(size_per_head)))

    # mask
    if attention_mask is not None:
        zeros = ht.Variable('no_mask', value=np.array(
            (0,), dtype=np.float32), trainable=False)
        adder = ht.Variable('attention_mask', value=np.array(
            (-2**32+1,), dtype=np.float32), trainable=False)
        zeros = ht.broadcastto_op(zeros, attention_mask)
        adder = ht.broadcastto_op(adder, attention_mask)
        attention_mask = ht.where_op(attention_mask, zeros, adder)  # (N, T)
        attention_mask = ht.array_reshape_op(
            attention_mask, [batch_size, 1, 1, -1])
        attention_scores = attention_scores + \
            ht.broadcastto_op(attention_mask, attention_scores)
    if causality:
        tril = ht.Variable(name='tril', value=np.tril(
            np.ones((caus_len, caus_len))), trainable=False)  # (T, T)
        future_masks = ht.broadcast_shape_op(
            tril, [batch_size, num_attention_heads, caus_len, caus_len])
        adder = ht.Variable('future_mask', value=np.array(
            (-2**32+1,), dtype=np.float32), trainable=False)
        adder = ht.broadcastto_op(adder, future_masks)
        attention_scores = ht.where_op(
            future_masks, attention_scores, adder)  # (N, h, T, T)

    # probs
    attention_probs = ht.softmax_op(attention_scores)
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)
    context_layer = ht.batch_matmul_op(attention_probs, value_layer)
    context_layer = ht.transpose_op(context_layer, [0, 2, 1, 3])
    outputs = ht.array_reshape_op(
        context_layer,
        [batch_size, -1, num_attention_heads * size_per_head])

    # Residual connection
    outputs = outputs + queries  # (N, T_q, d_model)

    # Normalize
    outputs = layer_norm(outputs, hidden_size)  # (N, T_q, d_model)
    return outputs


def ff(inputs, config):
    outputs = ht.array_reshape_op(inputs, [-1, config.d_model])
    outputs = dense(outputs, config.d_model,
                    config.d_ff, activation=ht.relu_op)
    outputs = dense(outputs, config.d_ff, config.d_model)
    outputs = ht.array_reshape_op(
        outputs, [config.batch_size, -1, config.d_model])
    outputs = outputs + inputs
    outputs = layer_norm(outputs, config.d_model)
    return outputs


def label_smoothing(inputs, V, epsilon=0.1):
    # V = inputs.shape[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / V)


def positional_encoding(
    inputs,
    inputs_shape,
    maxlen,
    masking=True
):
    N, T, E = tuple(inputs_shape)
    position_enc = np.array([
        [pos / np.power(10000, (i & -2)/E) for i in range(E)]
        for pos in range(maxlen)])
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

    position_enc = position_enc[:T, :]
    outputs = ht.Variable(name='position_enc', value=np.tile(
        position_enc, [N, 1, 1]), trainable=False)
    zeros = ht.Variable(name='zeros', value=np.zeros(
        inputs_shape), trainable=False)

    if masking:
        outputs = ht.where_op(inputs, outputs, zeros)

    return outputs


class Transformer(object):
    def __init__(self, hp):
        self.hp = hp
        self.embeddings = get_token_embeddings(
            self.hp.vocab_size, self.hp.d_model, zero_pad=True)

    def encode(self, xs):
        x = xs

        # embedding
        enc = ht.embedding_lookup_op(self.embeddings, x)  # (N, T1, d_model)
        enc = enc * self.hp.d_model**0.5  # scale

        enc += positional_encoding(enc, (self.hp.batch_size,
                                         self.hp.maxlen1, self.hp.d_model), self.hp.maxlen1)
        enc = dropout(enc, self.hp.dropout_rate)

        # Blocks
        for i in range(self.hp.num_blocks):
            # self-attention
            enc = multihead_attention(
                queries=enc, keys=enc, values=enc,
                config=self.hp,
                attention_mask=x,
                causality=False
            )
            # feed forward
            enc = ff(enc, config=self.hp)
        memory = enc
        return memory

    def decode(self, ys, memory, src_masks):
        decoder_inputs = ys

        # embedding
        dec = ht.embedding_lookup_op(
            self.embeddings, decoder_inputs)  # (N, T2, d_model)
        dec = dec * self.hp.d_model ** 0.5  # scale

        dec += positional_encoding(dec, (self.hp.batch_size,
                                         self.hp.maxlen2-1, self.hp.d_model), self.hp.maxlen2)
        dec = dropout(dec, self.hp.dropout_rate)

        # Blocks
        for i in range(self.hp.num_blocks):
            # Masked self-attention (Note that causality is True at this time)
            dec = multihead_attention(
                queries=dec, keys=dec, values=dec,
                config=self.hp,
                attention_mask=decoder_inputs,
                causality=True,
            )
            # Vanilla attention
            dec = multihead_attention(
                queries=dec, keys=memory, values=memory,
                config=self.hp,
                attention_mask=src_masks,
                causality=False,
            )
            # Feed Forward
            dec = ff(dec, config=self.hp)

        dec = ht.array_reshape_op(
            dec, [-1, self.hp.d_model])  # (N * T, d_model)
        logits = ht.array_reshape_op(ht.matmul_op(dec, self.embeddings, trans_B=True), [
                                     self.hp.batch_size, -1, self.hp.vocab_size])  # (N, T, vocab)

        return logits

    def train(self, xs, ys):
        # forward
        memory = self.encode(xs)
        logits = self.decode(ys[0], memory, xs)

        # train scheme
        y = ys[1]
        y_ = label_smoothing(ht.one_hot_op(
            y, self.hp.vocab_size), self.hp.vocab_size)  # (N, T, vocab)
        loss = ht.softmaxcrossentropy_op(logits, y_)

        return loss
