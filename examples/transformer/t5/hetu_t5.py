import copy
import math
import os
import warnings
from typing import Optional, Tuple, Union
import numpy as np

import hetu as ht


class T5LayerNorm(object):
    def __init__(self, hidden_size, eps=1e-6, name='T5LayerNorm'):
        super().__init__()
        self.weight = ht.init.ones(shape=(hidden_size, ), name=name+'.weight')
        self.variance_epsilon = eps

    def __call__(self, hidden_states):
        variance = ht.pow_op(hidden_states, 2)
        variance = ht.reduce_mean_op(variance, -1, keepdims=True)
        variance = ht.rsqrt_op(variance + self.variance_epsilon)
        variance = ht.broadcastto_op(variance, hidden_states)
        hidden_states = hidden_states * variance
        self.weight = ht.broadcastto_op(self.weight, hidden_states)
        return self.weight * hidden_states


class T5DenseActDense(object):
    def __init__(self, config, name='T5DenseActDense'):
        super().__init__()
        self.wi = ht.layers.Linear(
            config.d_model, config.d_ff, bias=False, weight_transpose=True, name=name+'.wi')
        self.wo = ht.layers.Linear(
            config.d_ff, config.d_model, bias=False, weight_transpose=True, name=name+'.wo')
        self.dim = config.d_model
        self.dropout = ht.layers.DropOut(config.dropout_rate)
        if config.dense_act_fn == "relu":
            self.act = ht.relu_op
        elif config.dense_act_fn == "gelu":
            self.act = ht.gelu_op

    def __call__(self, hidden_states, input_shape):
        hidden_states = ht.array_reshape_op(
            hidden_states, [-1, input_shape[-1]])
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        hidden_states = ht.array_reshape_op(
            hidden_states, input_shape[:-1]+(self.dim, ))
        return hidden_states


class T5DenseGatedActDense(object):
    def __init__(self, config, name='T5DenseGatedActDense'):
        super().__init__()
        self.wi_0 = ht.layers.Linear(
            config.d_model, config.d_ff, bias=False, weight_transpose=True, name=name+'.wi_0')
        self.wi_1 = ht.layers.Linear(
            config.d_model, config.d_ff, bias=False, weight_transpose=True, name=name+'.wi_1')
        self.wo = ht.layers.Linear(
            config.d_ff, config.d_model, bias=False, weight_transpose=True, name=name+'.wo')
        self.dim = config.d_model
        self.dropout = ht.layers.DropOut(config.dropout_rate)
        if config.dense_act_fn == "relu":
            self.act = ht.relu_op
        elif config.dense_act_fn == "gelu":
            self.act = ht.gelu_op

    def __call__(self, hidden_states, input_shape):
        hidden_states = ht.array_reshape_op(
            hidden_states, [-1, input_shape[-1]])
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        hidden_states = ht.array_reshape_op(
            hidden_states, input_shape[:-1]+(self.dim, ))
        return hidden_states


class T5LayerFF(object):
    def __init__(self, config, name='T5LayerFF'):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(
                config, name=name+'.DenseReluDense')
        else:
            self.DenseReluDense = T5DenseActDense(
                config, name=name+'.DenseReluDense')

        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon, name=name+'.layer_norm')
        self.dropout = ht.layers.DropOut(config.dropout_rate)

    def __call__(self, hidden_states, input_shape):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states, input_shape)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(object):
    def __init__(self, config, has_relative_attention_bias=False, name='T5Attention'):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.q = ht.layers.Linear(
            self.d_model, self.inner_dim, bias=False, weight_transpose=True, name=name+'.q')
        self.k = ht.layers.Linear(
            self.d_model, self.inner_dim, bias=False, weight_transpose=True, name=name+'.k')
        self.v = ht.layers.Linear(
            self.d_model, self.inner_dim, bias=False, weight_transpose=True, name=name+'.v')
        self.o = ht.layers.Linear(
            self.inner_dim, self.d_model, bias=False, weight_transpose=True, name=name+'.o')

        if self.has_relative_attention_bias:
            self.relative_attention_bias = ht.layers.Embedding(
                self.relative_attention_num_buckets, self.n_heads, name=name+'.relative_attention_bias')

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets = ht.bool_op(relative_position) * num_buckets
            relative_position = ht.abs_op(relative_position)
        else:
            relative_position = (-1) * ht.min_op(relative_position,
                                                 ht.zeroslike_op(relative_position))

        max_exact = num_buckets // 2

        is_small = ht.bool_op(relative_position, max_exact, 1)

        relative_position_if_large = max_exact + (
            ht.log_op(relative_position * (1/max_exact)) * (1 /
                                                            math.log(max_distance / max_exact)) * (num_buckets - max_exact)
        )
        relative_position_if_large = ht.min_op(
            relative_position_if_large, ht.full_like_op(
                relative_position_if_large, num_buckets - 1)
        )
        if (relative_buckets):
            relative_buckets = relative_buckets + \
                ht.where_op(is_small, relative_position,
                            relative_position_if_large)
        else:
            relative_buckets = ht.where_op(
                is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, batch_size, query_length, key_length):
        context_position = ht.arange_op(0, query_length)
        context_position = ht.broadcast_shape_op(
            context_position, (query_length, key_length))
        memory_position = ht.arange_op(0, key_length)
        memory_position = ht.broadcast_shape_op(
            memory_position, (query_length, key_length))
        relative_position = ht.minus_op(memory_position, context_position)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        values = ht.transpose_op(values, [2, 0, 1])
        values = ht.broadcast_shape_op(
            values, (batch_size, self.n_heads, query_length, key_length), add_axes=(0))

        return values

    def __call__(
        self,
        hidden_states,
        input_shape,
        mask=None,
        key_value_states=None,
        key_value_states_shape=None,
        position_bias=None,
        past_key_value=None,
        past_key_value_shape=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        batch_size, seq_length = input_shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value_shape[0][2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states_shape[
            1]

        def shape(states):
            states = ht.array_reshape_op(
                states, (batch_size, -1, self.n_heads, self.key_value_proj_dim))
            states = ht.transpose_op(states, (0, 2, 1, 3))
            return states

        def unshape(states):
            states = ht.transpose_op(states, (0, 2, 1, 3))
            states = ht.array_reshape_op(states, (-1, self.inner_dim))
            return states

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            if key_value_states is None:
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    hidden_states = ht.concat_op(
                        past_key_value, hidden_states, axis=2)
                else:
                    hidden_states = past_key_value
            return hidden_states

        hidden_states = ht.array_reshape_op(
            hidden_states, (-1, input_shape[-1]))
        query_states = shape(self.q(hidden_states))

        if (key_value_states):
            key_value_states = ht.array_reshape_op(
                key_value_states, (-1, input_shape[-1]))

        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[
                0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[
                1] if past_key_value is not None else None
        )
        key_states = ht.transpose_op(key_states, (0, 1, 3, 2))
        scores = ht.batch_matmul_op(query_states, key_states)

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = ht.init.zeros(
                    (batch_size, self.n_heads, real_seq_length, key_length), trainable=False)
            else:
                position_bias = self.compute_bias(
                    batch_size, real_seq_length, key_length)

            if past_key_value is not None:
                position_bias = ht.slice_op(
                    position_bias, (0, 0, -input_shape[1], 0), (-1, -1, -1, -1))
            if mask is not None:
                mask = ht.broadcastto_op(mask, position_bias)
                position_bias = position_bias + mask

        scores += position_bias
        attn_weights = ht.softmax_op(scores)
        attn_weights = ht.dropout_op(attn_weights, 1-self.dropout)

        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(ht.batch_matmul_op(attn_weights, value_states))
        attn_output = self.o(attn_output)
        attn_output = ht.array_reshape_op(attn_output, input_shape)

        present_key_value_state = (key_states, value_states) if (
            self.is_decoder and use_cache) else None
        outputs = (attn_output,) + \
            (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerSelfAttention(object):
    def __init__(self, config, has_relative_attention_bias=False, name='T5LayerSelfAttention'):
        super().__init__()
        self.SelfAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias, name=name+'.SelfAttention')
        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon, name=name+'.layer_norm')
        self.dropout = ht.layers.DropOut(config.dropout_rate)

    def __call__(
        self,
        hidden_states,
        input_shape,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            input_shape,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class T5LayerCrossAttention(object):
    def __init__(self, config, name='T5LayerCrossAttention'):
        super().__init__()
        self.EncDecAttention = T5Attention(
            config, has_relative_attention_bias=False, name=name+'.EncDecAttention')
        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon, name=name+'.layer_norm')
        self.dropout = ht.layers.DropOut(config.dropout_rate)

    def __call__(
        self,
        hidden_states,
        input_shape,
        key_value_states,
        key_value_states_shape,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            input_shape,
            mask=attention_mask,
            key_value_states=key_value_states,
            key_value_states_shape=key_value_states_shape,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]
        return outputs


class T5Block(object):
    def __init__(self, config, has_relative_attention_bias=False, name='T5Block'):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = [T5LayerSelfAttention(
            config, has_relative_attention_bias=has_relative_attention_bias, name=name+'.layer.0')]
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(
                config, name=name+'.layer.1'))
            self.layer.append(T5LayerFF(config, name=name+'.layer.2'))
        else:
            self.layer.append(T5LayerFF(config, name=name+'.layer.1'))

    def __call__(
        self,
        hidden_states,
        input_shape,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_hidden_states_shape=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):

        if past_key_value is not None:
            if not self.is_decoder:
                print(
                    "`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            input_shape,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                input_shape,
                key_value_states=encoder_hidden_states,
                key_value_states_shape=encoder_hidden_states_shape,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + \
                    cross_attention_outputs[1]

            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        hidden_states = self.layer[-1](hidden_states, input_shape)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs


class T5Stack(object):
    def __init__(self, config, embed_tokens=None, name='T5Stack'):
        super().__init__()

        self.config = config
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.dim = config.d_model

        self.block = [T5Block(config, has_relative_attention_bias=bool(
            i == 0), name=name+'.block.'+str(i)) for i in range(config.num_layers)]

        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon, name=name+'.final_layer_norm')
        self.dropout = ht.layers.DropOut(config.dropout_rate)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def create_extended_attention_mask_for_decoder(self, input_shape, attention_mask, attention_mask_shape):
        batch_size, seq_length = input_shape
        seq_ids = ht.arange_op(0, seq_length)
        seq_ids_a = ht.broadcast_shape_op(
            seq_ids, (batch_size, seq_length, seq_length), add_axes=(0, 1))
        seq_ids_b = ht.broadcast_shape_op(
            seq_ids, (batch_size, seq_length, seq_length), add_axes=(0, 2))
        causal_mask = ht.bool_op(seq_ids_a, seq_ids_b, 3)

        if seq_length < attention_mask_shape[1]:
            prefix_seq_len = attention_mask_shape[1] - seq_length
            causal_mask = ht.concat_op(ht.init.ones(
                (batch_size, seq_length, prefix_seq_len)), causal_mask, axis=-1)

        causal_mask = ht.array_reshape_op(
            causal_mask, (batch_size, 1, seq_length, seq_length))
        attention_mask = ht.broadcast_shape_op(
            attention_mask, (batch_size, 1, seq_length, seq_length), add_axes=(1, 2))

        extended_attention_mask = causal_mask * attention_mask
        return extended_attention_mask

    def get_extended_attention_mask(self, attention_mask, attention_mask_shape, input_shape):

        if len(attention_mask_shape) == 3:
            b, s1, s2 = attention_mask_shape
            extended_attention_mask = ht.array_reshape_op(
                attention_mask, (b, 1, s1, s2))

        elif len(attention_mask_shape) == 2:
            b, s = attention_mask_shape
            if self.config.is_decoder:
                extended_attention_mask = self.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, attention_mask_shape)
            else:
                extended_attention_mask = ht.array_reshape_op(
                    attention_mask, (b, 1, 1, s))
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask_shape})"
            )
        extended_attention_mask = ht.minus_byconst_op(
            extended_attention_mask, 1.0)
        extended_attention_mask = extended_attention_mask * \
            np.finfo(np.float32).min
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask, encoder_attention_mask_shape):
        if len(encoder_attention_mask_shape) == 3:
            b, s1, s2 = encoder_attention_mask_shape
            encoder_extended_attention_mask = ht.array_reshape_op(
                encoder_attention_mask, (b, 1, s1, s2))
        if len(encoder_attention_mask_shape) == 2:
            b, s = encoder_attention_mask_shape
            encoder_extended_attention_mask = ht.array_reshape_op(
                encoder_attention_mask, (b, 1, 1, s))

        encoder_extended_attention_mask = ht.minus_byconst_op(
            encoder_extended_attention_mask, 1.0)
        encoder_extended_attention_mask = encoder_extended_attention_mask * \
            np.finfo(np.float32).min
        return encoder_extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            assert False
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def __call__(
        self,
        input_ids=None,
        input_shape=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_hidden_states_shape=None,
        encoder_attention_mask=None,
        encoder_attention_mask_shape=None,
        inputs_embeds=None,
        inputs_embeds_shape=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        past_key_values_shape=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_ids = ht.array_reshape_op(input_ids, [-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds_shape[:-1]
            hidden_states_shape = inputs_embeds_shape
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)
            hidden_states_shape = input_shape + (self.dim, )

        batch_size, seq_length = input_shape

        mask_seq_length = past_key_values_shape[0][0][2] + \
            seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = ht.init.ones((batch_size, mask_seq_length))
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states_shape[1]
            encoder_attention_mask = ht.init.ones(
                (batch_size, encoder_seq_length))

        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, (batch_size, mask_seq_length), input_shape)

        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states_shape
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ht.init.ones((encoder_hidden_shape))
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask, encoder_attention_mask_shape)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(
            cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                hidden_states_shape,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_shape=encoder_hidden_states_shape,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            if use_cache:
                present_key_value_states = present_key_value_states + \
                    (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + \
                        (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        else:
            assert False


class T5Model(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.d_model
        self.shared = ht.layers.Embedding(
            config.vocab_size, config.d_model, name='shared')

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared, name='encoder')

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared, name='decoder')

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def __call__(
        self,
        input_ids=None,
        input_shape=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_input_shape=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        inputs_embeds_shape=None,
        decoder_inputs_embeds=None,
        decoder_inputs_embeds_shape=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                input_shape=input_shape,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                inputs_embeds_shape=inputs_embeds_shape,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            input_shape=decoder_input_shape,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            inputs_embeds_shape=decoder_inputs_embeds_shape,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_hidden_states_shape=input_shape+(self.dim, ),
            encoder_attention_mask=attention_mask,
            encoder_attention_mask_shape=input_shape,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return decoder_outputs + encoder_outputs
