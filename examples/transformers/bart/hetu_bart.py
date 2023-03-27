import copy
import math
import random
import warnings
from typing import List, Optional, Tuple, Union

import hetu as ht
import numpy as np

def shift_tokens_right(input_ids, input_shape, pad_token_id, decoder_start_token_id):
    shifted_input_ids = ht.init.zeros(shape=input_shape, trainable=False)
    shifted_input_ids = ht.slice_assign_matrix_op(shifted_input_ids, input_ids, (0,1), (-1,-1), (0,0), (-1, input_shape[1]-1))
    shifted_input_ids = ht.slice_assign_op(shifted_input_ids, (0,0), (-1,1), decoder_start_token_id)
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    bool_matrix = ht.bool_op(shifted_input_ids, -100)
    shifted_input_ids = ht.masked_fill_op(shifted_input_ids, bool_matrix, pad_token_id)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape, past_key_values_length=0):
    bsz, tgt_len = input_ids_shape
    mask = ht.full_op((tgt_len, tgt_len), np.finfo(np.float32).min)
    mask_cond = ht.arange_op(start=0, end=tgt_len)
    mask_cond_ = mask_cond + 1
    mask_cond_ = ht.array_reshape_op(mask_cond_, (tgt_len, 1))
    bool_matrix = ht.bool_op(mask_cond, mask_cond_, 1)
    mask = ht.masked_fill_op(mask, bool_matrix, 0)

    if past_key_values_length > 0:
        mask = ht.concat_op(ht.init.zeros(shape=(tgt_len, past_key_values_length), trainable=False), mask, axis=-1)

    mask = ht.broadcast_shape_op(mask, (bsz, 1, tgt_len, tgt_len + past_key_values_length))
    return mask


def _expand_mask(mask, input_shape, tgt_len=None):
    bsz, src_len = input_shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = ht.broadcast_shape_op(mask, (bsz, 1, tgt_len, src_len), add_axes=(1, 2))
    inverted_mask = ht.minus_byconst_op(expanded_mask, 1.0)
    inverted_mask = ht.masked_fill_op(inverted_mask, inverted_mask, np.finfo(np.float32).min)

    return inverted_mask

class BartLearnedPositionalEmbedding(object):
    def __init__(self, num_embeddings, embedding_dim, name='BartLearnedPositionalEmbedding'):
        self.offset = 2
        self.embeddings = ht.layers.Embedding(num_embeddings + self.offset, embedding_dim, name=name)

    def __call__(self, input_ids, input_shape, past_key_values_length=0):
        bsz, seq_len = input_shape[:2]
        positions = ht.arange_op(past_key_values_length, past_key_values_length + seq_len)
        positions = ht.broadcast_shape_op(positions, (bsz, -1))
        return self.embeddings(positions + self.offset)


class BartAttention(object):
    def __init__(self, embed_dim, num_heads, dropout=0.0, is_decoder=False, bias=True, name='BartAttention'):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = ht.layers.Linear(embed_dim, embed_dim, bias=bias, weight_transpose=True, name=name+'.k_proj')
        self.v_proj = ht.layers.Linear(embed_dim, embed_dim, bias=bias, weight_transpose=True, name=name+'.v_proj')
        self.q_proj = ht.layers.Linear(embed_dim, embed_dim, bias=bias, weight_transpose=True, name=name+'.q_proj')
        self.out_proj = ht.layers.Linear(embed_dim, embed_dim, bias=bias, weight_transpose=True, name=name+'.out_proj')

    def _shape(self, tensor, seq_len, bsz):
        tensor = ht.array_reshape_op(tensor, (bsz, seq_len, self.num_heads, self.head_dim))
        tensor = ht.transpose_op(tensor, (0, 2, 1, 3))
        return tensor

    def __call__(self, hidden_states, input_shape, key_value_states=None, past_key_value=None,
        attention_mask=None, layer_head_mask=None, output_attentions=False):

        bsz, tgt_len, dim = input_shape
        hidden_states = ht.array_reshape_op(hidden_states, (-1, dim))
        is_cross_attention = key_value_states is not None
        
        query_states = self.q_proj(hidden_states) * self.scaling

        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_value_states = ht.array_reshape_op(key_value_states, (-1, dim))
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = ht.concat_op(past_key_value[0], key_states, axis=2)
            value_states = ht.concat_op(past_key_value[1], value_states, axis=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)
        query_states = ht.array_reshape_op(query_states, proj_shape)
        key_states = ht.array_reshape_op(key_states, proj_shape)
        value_states = ht.array_reshape_op(value_states, proj_shape)

        src_len = tgt_len
        k = ht.transpose_op(key_states, (0, 2, 1))
        attn_weights = ht.batch_matmul_op(query_states, k)

        if attention_mask is not None:
            attn_weights = ht.array_reshape_op(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + attention_mask
            attn_weights = ht.array_reshape_op(attn_weights, (bsz * self.num_heads, tgt_len, src_len))
    
        attn_weights = ht.softmax_op(attn_weights)        

        if layer_head_mask is not None:
            layer_head_mask = ht.array_reshape_op(layer_head_mask, (1, -1, 1, 1)) 
            attn_weights = ht.array_reshape_op(attn_weights, (bsz, self.num_heads, tgt_len, src_len))

            attn_weights = layer_head_mask * attn_weights
            attn_weights = ht.array_reshape_op(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        if output_attentions:
            attn_weights = ht.array_reshape_op(attn_weights, (bsz * self.num_heads, tgt_len, src_len))
        else:
            attn_weights_reshaped = None

        attn_probs = ht.dropout_op(attn_weights, 1-self.dropout)
        attn_output = ht.batch_matmul_op(attn_probs, value_states)

        attn_output = ht.array_reshape_op(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim))
        attn_output = ht.transpose_op(attn_output, (0, 2, 1, 3))

        attn_output = ht.array_reshape_op(attn_output, (-1, self.embed_dim))

        attn_output = self.out_proj(attn_output)
        attn_output = ht.array_reshape_op(attn_output, (bsz, tgt_len, -1))


        return attn_output, attn_weights_reshaped, past_key_value


class BartEncoderLayer(object):
    def __init__(self, config, name='BartEncoderLayer'):
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            name=name+'.self_attn'
        )
        self.self_attn_layer_norm = ht.layers.LayerNorm(self.embed_dim, name=name+'.self_attn_layer_norm')
        self.dropout = config.dropout
        if config.activation_function == "relu":
            self.activation_fn = ht.relu_op
        elif config.activation_function == "gelu":
            self.activation_fn = ht.gelu_op
        self.activation_dropout = config.activation_dropout

        self.fc1 = ht.layers.Linear(self.embed_dim, config.encoder_ffn_dim, weight_transpose=True, name=name+'.fc1')
        self.fc2 = ht.layers.Linear(config.encoder_ffn_dim, self.embed_dim, weight_transpose=True, name=name+'.fc2')
        self.final_layer_norm = ht.layers.LayerNorm(self.embed_dim, name=name+'.final_layer_norm')

    def __call__(self, hidden_states, input_shape, attention_mask, layer_head_mask, output_attentions=False):
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            input_shape=input_shape,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = ht.dropout_op(hidden_states, 1-self.dropout)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states = ht.array_reshape_op(hidden_states, (-1, self.embed_dim))
        residual = hidden_states
        
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = ht.dropout_op(hidden_states,1-self.activation_dropout)
        hidden_states = self.fc2(hidden_states)
        hidden_states = ht.dropout_op(hidden_states, 1-self.dropout)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = ht.array_reshape_op(hidden_states, input_shape)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BartDecoderLayer(object):
    def __init__(self, config, name='BartDecoderLayer'):
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            name=name+'.self_attn'
        )
        self.dropout = config.dropout
        if config.activation_function == "relu":
            self.activation_fn = ht.relu_op
        elif config.activation_function == "gelu":
            self.activation_fn = ht.gelu_op
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = ht.layers.LayerNorm(self.embed_dim, name=name+'.self_attn_layer_norm')
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            name=name+'.encoder_attn'
        )
        self.encoder_attn_layer_norm = ht.layers.LayerNorm(self.embed_dim, name=name+'.encoder_attn_layer_norm')

        self.fc1 = ht.layers.Linear(self.embed_dim, config.decoder_ffn_dim, weight_transpose=True, name=name+'.fc1')
        self.fc2 = ht.layers.Linear(config.decoder_ffn_dim, self.embed_dim, weight_transpose=True, name=name+'.fc2')
        self.final_layer_norm = ht.layers.LayerNorm(self.embed_dim, name=name+'.final_layer_norm')

    def __call__(self, hidden_states, input_shape,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=True,
    ) :
        residual = hidden_states

        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            input_shape=input_shape,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = ht.dropout_op(hidden_states, 1-self.dropout)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                input_shape=input_shape,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = ht.dropout_op(hidden_states, 1-self.dropout)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            present_key_value = present_key_value + cross_attn_present_key_value

        hidden_states = ht.array_reshape_op(hidden_states, (-1, self.embed_dim))
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = ht.dropout_op(hidden_states,1-self.activation_dropout)
        hidden_states = self.fc2(hidden_states)
        hidden_states = ht.dropout_op(hidden_states, 1-self.dropout)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = ht.array_reshape_op(hidden_states, input_shape)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartClassificationHead(object):
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout, name='BartClassificationHead'):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dense = ht.layers.Linear(input_dim, inner_dim, weight_transpose=True, name=name+'.dense')
        self.dropout = ht.layers.DropOut(pooler_dropout)
        self.out_proj = ht.layers.Linear(inner_dim, num_classes, weight_transpose=True, name=name+'.out_proj')

    def __call__(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = ht.array_reshape_op(hidden_states, (-1, self.input_dim))
        hidden_states = self.dense(hidden_states)
        hidden_states = ht.tanh_op(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class BartEncoder(object):
    def __init__(self, config, embed_tokens=None, name='BartEncoder'):
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.embed_dim = embed_dim
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = ht.layers.Embedding(config.vocab_size, embed_dim, name=name+'.embed_tokens')

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            name=name+'.embed_positions')

        self.layers = [BartEncoderLayer(config, name=name+'.layers.'+str(i)) for i in range(config.encoder_layers)]
        self.layernorm_embedding = ht.layers.LayerNorm(embed_dim, name=name+'.layernorm_embedding')
        

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def __call__(
        self,
        input_ids=None,
        input_shape=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        inputs_embeds_shape=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_shape = None
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = ht.array_reshape_op(input_ids, [-1, input_shape[-1]])
            encoder_shape = input_shape + (self.embed_dim, )
        elif inputs_embeds is not None:
            encoder_shape = inputs_embeds_shape[:-1] + (inputs_embeds_shape[2] - 1, )
            input = ht.slice_op(inputs_embeds, (0, 0, 0), (-1, -1, inputs_embeds_shape[2] - 1))
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input, input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = ht.dropout_op(hidden_states, 1-self.dropout)

        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, input_shape)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if head_mask is not None:
            raise NotImplementedError
        
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if (dropout_probability < self.layerdrop): 
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    encoder_shape,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        else:
            return False


class BartDecoder(object):
    def __init__(self, config, embed_tokens=None, name='BartDecoder'):
        
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_dim = config.d_model
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = ht.layers.Embedding(config.vocab_size, config.d_model, name=name+'.embed_tokens')

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model, 
            name=name+'.embed_positions'
        )
        self.layers = [BartDecoderLayer(config, name=name+'.layers.'+str(i)) for i in range(config.decoder_layers)]
        self.layernorm_embedding = ht.layers.LayerNorm(config.d_model, name=name+'.layernorm_embedding')
        
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, past_key_values_length=past_key_values_length
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, input_shape, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def __call__(
        self,
        input_ids=None,
        input_shape=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        inputs_embeds_shape=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        decoder_shape = None
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            decoder_shape = input_shape + (self.embed_dim, )
            input_ids = ht.array_reshape_op(input_ids, [-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds_shape[:-1]
            decoder_shape = inputs_embeds_shape[:-1] + (inputs_embeds_shape[2] - 1, )
            input = ht.slice_op(inputs_embeds, (0, 0, 0), (-1, -1, inputs_embeds_shape[2]-1))
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _expand_mask(encoder_attention_mask, input_shape, tgt_len=input_shape[-1])

        positions = self.embed_positions(input, input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = ht.dropout_op(hidden_states, 1-self.dropout)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if (dropout_probability < self.layerdrop):
                continue
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                decoder_shape,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                ),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        else:
            assert False


class BartModel(object):
    def __init__(self, config):
        self.config = config
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = ht.layers.Embedding(vocab_size, config.d_model, name='shared')
        self.encoder = BartEncoder(config, self.shared, name='encoder')
        self.decoder = BartDecoder(config, self.shared, name='decoder')

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

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
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, input_shape, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                input_shape=input_shape,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
    
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            input_shape=input_shape,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return decoder_outputs + encoder_outputs

class BartForCausalLM(object):
    def __init__(self, config):
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        self.config = config
        self.model = BartDecoder(config)
        self.d_model = config.d_model
        self.lm_head = ht.layers.Linear(config.d_model, config.vocab_size, weight_transpose=True, bias=False)
        
    def __call__(self, input_ids, input_shape, attention_mask=None, labels=None):
    
        hidden_states = self.model(input_ids, input_shape, attention_mask=attention_mask)
        hidden_states = ht.array_reshape_op(hidden_states[0], (-1, self.d_model))
        lm_logits = self.lm_head(hidden_states)
        lm_logits = ht.array_reshape_op(lm_logits, input_shape + (-1,))
        loss = None

        if labels is not None:
            loss = ht.crossentropy_sparse_op(ht.softmax_op(lm_logits), labels)
            return loss, lm_logits
        else:
            return lm_logits



class BartForSequenceClassification(object):
    def __init__(self, config):
        self.config = config
        self.model = BartModel(config)
        self.hidden_size = config.d_model
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
            'classification_head'
        )        
        
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
        
    def __call__(self, input_ids, input_ids_shape, attention_mask=None, labels=None):
        outputs = self.model(input_ids, input_ids_shape, attention_mask)
        hidden_states = outputs[0]
        index1 = ht.arange_op(0, input_ids_shape[0])
        index2 = ht.ne_op(input_ids, self.config.pad_token_id)
        index2 = ht.reduce_sum_op(index2, 1)
        
        hidden_states = ht.slice_by_matrix_op(hidden_states, index1, index2) 
        hidden_states = ht.array_reshape_op(hidden_states, (-1, self.hidden_size))
        logits = self.classification_head(hidden_states)

        if labels is not None:
            # loss = ht.softmaxcrossentropy_sparse_op(logits, labels, ignored_index = -1)
            loss = ht.crossentropy_sparse_op(
                ht.softmax_op(logits), labels, ignored_index=-1)
            return loss, logits
        else:
            return logits


