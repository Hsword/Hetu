import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import hetu as ht

def create_position_ids_from_input_ids(input_ids, padding_idx):
    mask = ht.ne_op(input_ids, padding_idx)
    incremental_indices = ht.cumsum_with_bias_op(mask, bias=0, dim=1) * mask
    return incremental_indices + padding_idx


class LongformerEmbeddings(object):
    def __init__(self, config, name='LongformerEmbeddings'):
        self.word_embeddings = ht.layers.Embedding(config.vocab_size, config.hidden_size, name=name+".word_embeddings")
        self.position_embeddings = ht.layers.Embedding(config.max_position_embeddings, config.hidden_size, name=name+".position_embeddings")
        self.token_type_embeddings = ht.layers.Embedding(config.type_vocab_size, config.hidden_size, name=name+".token_type_embeddings")

        self.LayerNorm = ht.layers.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, name=name+".LayerNorm")
        self.dropout = ht.layers.DropOut(config.hidden_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.padding_idx = config.pad_token_id
        self.position_embeddings = ht.layers.Embedding(
            config.max_position_embeddings, config.hidden_size, name=name+'.position_embeddings'
        )

    def __call__(self, input_ids=None, input_ids_shape=None, token_type_ids=None, position_ids=None, inputs_embeds=None, inputs_embeds_shape=None):
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, inputs_embeds_shape)

        if input_ids is not None:
            input_shape = input_ids_shape
        else:
            input_shape = inputs_embeds_shape[:-1]

        if token_type_ids is None:
            token_type_ids = ht.init.zeros(input_shape, trainable=False)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds, inputs_embeds_shape):
        input_shape = inputs_embeds_shape[:-1]
        sequence_length = input_shape[1]

        position_ids = ht.arange_op(self.padding_idx + 1, sequence_length + self.padding_idx + 1)
        position_ids = ht.unsqueeze_op(position_ids, 0)
        position_ids = ht.broadcast_shape_op(position_ids, input_shape)
        return position_ids


class LongformerSelfAttention(object):
    def __init__(self, config, layer_id, name='LongformerSelfAttention'):
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = ht.layers.Linear(config.hidden_size, self.embed_dim, weight_transpose=True, name=name+'.query')
        self.key = ht.layers.Linear(config.hidden_size, self.embed_dim, weight_transpose=True, name=name+'.key')
        self.value = ht.layers.Linear(config.hidden_size, self.embed_dim, weight_transpose=True, name=name+'.value')

        self.query_global = ht.layers.Linear(config.hidden_size, self.embed_dim, weight_transpose=True, name=name+'.query_global')
        self.key_global = ht.layers.Linear(config.hidden_size, self.embed_dim, weight_transpose=True, name=name+'.key_global')
        self.value_global = ht.layers.Linear(config.hidden_size, self.embed_dim, weight_transpose=True, name=name+'.value_global')

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2

        self.config = config

    def __call__(
        self,
        hidden_states,
        hidden_states_shape,
        attention_mask=None,
        attention_mask_shape=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        hidden_states = ht.transpose_op(hidden_states, (1, 0, 2))
        hidden_states = ht.array_reshape_op(hidden_states, (-1, hidden_states_shape[-1]))
        
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        batch_size, seq_len, embed_dim = hidden_states_shape
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        query_vectors = query_vectors * (1/math.sqrt(self.head_dim))
        query_vectors = ht.array_reshape_op(query_vectors, (seq_len, batch_size, self.num_heads, self.head_dim))
        query_vectors = ht.transpose_op(query_vectors, (1, 0, 2, 3))
        query_shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        
        key_vectors = ht.array_reshape_op(key_vectors, (seq_len, batch_size, self.num_heads, self.head_dim))
        key_vectors =ht.transpose_op(key_vectors, (1, 0, 2, 3))

        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, query_shape, key_vectors, query_shape, self.one_sided_attn_window_size
        )
        
        remove_from_windowed_attention_mask = ht.bool_op(attention_mask, 0, 5)
        remove_from_windowed_attention_mask = ht.unsqueeze_op(remove_from_windowed_attention_mask, (2, 3))

        float_mask = ht.masked_fill_op(remove_from_windowed_attention_mask, remove_from_windowed_attention_mask, np.finfo(np.float32).min)

        float_mask_shape = [attention_mask_shape[0], attention_mask_shape[1], 1, 1]
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            ht.init.ones(float_mask_shape, trainable=False), float_mask_shape, float_mask, float_mask_shape, self.one_sided_attn_window_size
        )

        attn_scores += diagonal_mask

        if is_global_attn:
            raise NotImplementedError

        attn_probs = ht.softmax_op(attn_scores)

        if layer_head_mask is not None:
            attn_probs = ht.array_reshape_op(layer_head_mask, (1, 1, -1, 1)) * attn_probs

        attn_probs_shape = (batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + 1)
        is_index_masked = ht.unsqueeze_op(is_index_masked, (2, 3))
        is_index_masked = ht.broadcast_shape_op(is_index_masked, attn_probs_shape)
        attn_probs = ht.masked_fill_op(attn_probs, is_index_masked, 0.0)
        attn_probs = ht.dropout_op(attn_probs, 1-self.dropout)

        value_vectors = ht.array_reshape_op(value_vectors, (seq_len, batch_size, self.num_heads, self.head_dim))
        value_vectors = ht.transpose_op(value_vectors, (1, 0, 2, 3))
        value_shape = (batch_size, seq_len, self.num_heads, self.head_dim)

        if is_global_attn:
            NotImplementedError
        else:
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, attn_probs_shape, value_vectors, value_shape, self.one_sided_attn_window_size
            )

        attn_output = ht.transpose_op(attn_output, (1, 0, 2, 3))
        attn_output = ht.array_reshape_op(attn_output, (seq_len, batch_size, embed_dim))

        if is_global_attn:
            NotImplementedError

        outputs = (ht.transpose_op(attn_output, (1, 0, 2)), )

        if output_attentions:
            outputs += (attn_probs,)

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states, chunked_hidden_states_shape):
        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states_shape
        pad_size = (total_num_heads, num_chunks, window_overlap, window_overlap+1)
        chunked_hidden_states = ht.concat_op(chunked_hidden_states, ht.full_op(pad_size, 0), axis=-1)
        chunked_hidden_states = ht.array_reshape_op(chunked_hidden_states, (total_num_heads, num_chunks, -1))
        chunked_hidden_states = ht.slice_op(chunked_hidden_states, (0, 0, 0), (-1, -1, window_overlap * (window_overlap + hidden_dim)))
        chunked_hidden_states = ht.array_reshape_op(chunked_hidden_states, (total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim))
        chunked_hidden_states = ht.slice_op(chunked_hidden_states, (0, 0, 0, 0), (-1, -1, -1, window_overlap + hidden_dim - 1))
        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, hidden_states_shape, window_overlap):

        hidden_states = ht.array_reshape_op(hidden_states, (hidden_states_shape[0], -1, window_overlap * 2, hidden_states_shape[2]))
        chunk_size = [hidden_states_shape[0], hidden_states_shape[1]//(window_overlap * 2), window_overlap * 2, hidden_states_shape[2]]
        chunk_stride = [chunk_size[1]*chunk_size[2]*chunk_size[3], chunk_size[2]*chunk_size[3], chunk_size[3], 1]
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride[1] = chunk_stride[1] // 2
        hidden_states = ht.as_strided_op(hidden_states, chunk_size, chunk_stride)
        return hidden_states

    @staticmethod
    def _mask_invalid_locations(input_tensor, input_tensor_shape, affected_seq_len):
        beginning_mask_2d = ht.init.ones((affected_seq_len, affected_seq_len + 1), trainable=False)
        beginning_mask_2d = ht.tril_op(beginning_mask_2d)
        beginning_mask_2d = ht.flip_op(beginning_mask_2d, dims=[0])
        ending_mask = ht.flip_op(beginning_mask_2d, dims=(0, 1))
        
        beginning_input = ht.slice_op(input_tensor, (0, 0, 0, 0), (-1, affected_seq_len, -1, affected_seq_len + 1))
        beginning_mask = ht.broadcast_shape_op(beginning_mask_2d, (input_tensor_shape[0], -1, input_tensor_shape[2], -1), add_axes=(0, 2))

        input_shape = (input_tensor_shape[0], affected_seq_len, input_tensor_shape[2], affected_seq_len + 1)
        full_val = ht.full_op(input_shape, -float("inf"))
        full_val = ht.where_op(beginning_mask, full_val, beginning_input)
        input_tensor = ht.slice_assign_matrix_op(input_tensor, full_val, (0, 0, 0, 0), (-1, affected_seq_len, -1, affected_seq_len + 1), (0, 0, 0, 0), (-1, -1, -1, -1))

        ending_input = ht.slice_op(input_tensor, (0, -affected_seq_len, 0, -(affected_seq_len + 1)), (-1, affected_seq_len, -1, affected_seq_len + 1))
        ending_mask = ht.broadcast_shape_op(ending_mask, (input_tensor_shape[0], -1, input_tensor_shape[2], -1), add_axes=(0, 2))

        full_val = ht.full_op(input_shape, -float("inf"))
        full_val = ht.where_op(ending_mask, full_val, ending_input)

        input_tensor = ht.slice_assign_matrix_op(input_tensor, full_val, (0, -affected_seq_len, 0, -(affected_seq_len + 1)), (-1, affected_seq_len, -1, affected_seq_len + 1), (0, 0, 0, 0), (-1, -1, -1, -1))
        return input_tensor

    def _sliding_chunks_query_key_matmul(self, query, query_shape, key, key_shape, window_overlap):
        batch_size, seq_len, num_heads, head_dim = query_shape
        assert (
            seq_len % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
        assert query_shape == key_shape

        chunks_count = math.trunc(seq_len/window_overlap) - 1

        shape = (batch_size * num_heads, seq_len, head_dim)
        query = ht.transpose_op(query, (0, 2, 1, 3))
        query = ht.array_reshape_op(query, shape)
        key = ht.transpose_op(key, (0, 2, 1, 3))
        key = ht.array_reshape_op(key, shape)        

        query = self._chunk(query, shape, window_overlap)
        key = self._chunk(key, shape, window_overlap)

        key = ht.transpose_op(key, (0, 1, 3, 2))
        diagonal_chunked_attention_scores = ht.batch_matmul_op(query, key)
        diagonal_chunked_attention_scores_shape = (batch_size * num_heads, chunks_count, 2*window_overlap, 2*window_overlap)

        diagonal_chunked_attention_scores = ht.concat_op(diagonal_chunked_attention_scores, ht.full_op(diagonal_chunked_attention_scores_shape[:-2]+(1, diagonal_chunked_attention_scores_shape[-1]), 0), axis=2)
        shape = [diagonal_chunked_attention_scores_shape[0], diagonal_chunked_attention_scores_shape[1], diagonal_chunked_attention_scores_shape[3], diagonal_chunked_attention_scores_shape[2]+1]
        diagonal_chunked_attention_scores = ht.array_reshape_op(diagonal_chunked_attention_scores, shape)

        diagonal_attention_scores = ht.init.zeros((batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1), trainable=False)

        diagonal_attention_scores = ht.slice_assign_matrix_op(diagonal_attention_scores, diagonal_chunked_attention_scores,
        (0, 0, 0, window_overlap), (-1, chunks_count, -1, -1), (0, 0,0,0),(-1, -1, window_overlap, window_overlap + 1))

        diagonal_attention_scores = ht.slice_assign_matrix_op(diagonal_attention_scores, diagonal_chunked_attention_scores,
        (0, -1, 0, window_overlap), (-1, 1, -1, -1), (0, -1,window_overlap,0),(-1, 1, -1, window_overlap + 1))

        diagonal_attention_scores = ht.slice_assign_matrix_op(diagonal_attention_scores, diagonal_chunked_attention_scores,
        (0, 1, 0, 0), (-1, -1, -1, window_overlap), (0, 0,-(window_overlap + 1),window_overlap + 1),(-1, -1, window_overlap, -1))

        diagonal_attention_scores = ht.slice_assign_matrix_op(diagonal_attention_scores, diagonal_chunked_attention_scores,
        (0, 0, 1, 1), (-1, 1, window_overlap-1, window_overlap-1), (0, 0,0,1 - window_overlap),(-1, 1,  window_overlap - 1, -1))

        diagonal_attention_scores = ht.array_reshape_op(diagonal_attention_scores, (
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ))
        diagonal_attention_scores = ht.transpose_op(diagonal_attention_scores, (0, 2, 1, 3))

        diagonal_attention_scores = self._mask_invalid_locations(diagonal_attention_scores, (batch_size, seq_len, num_heads, 2 * window_overlap + 1), window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs, attn_probs_shape, value, value_shape, window_overlap
    ):

        batch_size, seq_len, num_heads, head_dim = value_shape

        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs_shape[:3] == value_shape[:3]
        assert attn_probs_shape[3] == 2 * window_overlap + 1

        chunks_count = math.trunc(seq_len/window_overlap) - 1

        shape = (batch_size * num_heads, chunks_count + 1, window_overlap, 2 * window_overlap + 1)

        chunked_attn_probs = ht.transpose_op(attn_probs, (0, 2, 1, 3))
        chunked_attn_probs = ht.array_reshape_op(chunked_attn_probs, shape)

        value = ht.transpose_op(value, (0, 2, 1, 3))
        value = ht.array_reshape_op(value, (batch_size * num_heads, seq_len, head_dim))
        
        tmp = ht.full_op((batch_size * num_heads, window_overlap, head_dim), -1)
        padded_value = ht.concatenate_op([tmp, value, tmp], axis=1)

        padded_value_size =  (batch_size * num_heads, seq_len + 2 * window_overlap, head_dim)
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)

        chunked_value_stride = (padded_value_size[1] * padded_value_size[2], padded_value_size[2], 1)
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = ht.as_strided_op(padded_value, chunked_value_size, chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs, shape)
        
        context = ht.batch_matmul_op(chunked_attn_probs, chunked_value)
        context = ht.array_reshape_op(context, (batch_size, num_heads, seq_len, head_dim))
        context = ht.transpose_op(context, (0, 2, 1, 3))
        return context

class LongformerSelfOutput(object):
    def __init__(self, config, name='LongformerSelfOutput'):
        self.hidden_size = config.hidden_size
        self.dense = ht.layers.Linear(config.hidden_size, config.hidden_size, weight_transpose=True, name=name+'.dense')
        self.LayerNorm = ht.layers.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, name=name+'.LayerNorm')
        self.dropout = ht.layers.DropOut(config.hidden_dropout_prob)

    def __call__(self, hidden_states, hidden_states_shape, input_tensor):
        hidden_states = ht.array_reshape_op(hidden_states, [-1, hidden_states_shape[-1]])
        hidden_states = self.dense(hidden_states)
        hidden_states = ht.array_reshape_op(hidden_states, hidden_states_shape)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)       
        return hidden_states


class LongformerAttention(object):
    def __init__(self, config, layer_id=0, name='LongformerAttention'):
        self.self = LongformerSelfAttention(config, layer_id, name=name+'.self')
        self.output = LongformerSelfOutput(config, name=name+'.output')

    def __call__(
        self,
        hidden_states,
        hidden_states_shape,
        attention_mask=None,
        attention_mask_shape=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            hidden_states_shape,
            attention_mask=attention_mask,
            attention_mask_shape=attention_mask_shape,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self.output(self_outputs[0], hidden_states_shape, hidden_states)
        outputs = (attn_output,) + self_outputs[1:]
        return outputs

class LongformerIntermediate(object):
    def __init__(self, config, name='LongformerIntermediate'):
        self.intermediate_size = config.intermediate_size
        self.dense = ht.layers.Linear(config.hidden_size, config.intermediate_size, weight_transpose=True, name=name+'.dense')
        if config.hidden_act == "relu":
            self.intermediate_act_fn = ht.relu_op
        elif config.hidden_act == "gelu":
            self.intermediate_act_fn = ht.gelu_op

    def __call__(self, hidden_states, hidden_states_shape):
        hidden_states = ht.array_reshape_op(hidden_states, [-1, hidden_states_shape[-1]])
        hidden_states = self.dense(hidden_states)
        hidden_states = ht.array_reshape_op(hidden_states, hidden_states_shape[:-1] + [self.intermediate_size])       
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class LongformerOutput(object):
    def __init__(self, config, name='LongformerOutput'):
        self.hidden_size = config.hidden_size
        self.dense = ht.layers.Linear(config.intermediate_size, config.hidden_size, weight_transpose=True, name=name+'.dense')
        self.LayerNorm = ht.layers.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, name=name+'.LayerNorm')
        self.dropout = ht.layers.DropOut(config.hidden_dropout_prob)

    def __call__(self, hidden_states, hidden_states_shape, input_tensor):
        hidden_states = ht.array_reshape_op(hidden_states, [-1, hidden_states_shape[-1]])
        hidden_states = self.dense(hidden_states)
        hidden_states = ht.array_reshape_op(hidden_states, hidden_states_shape[:-1] + [self.hidden_size])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LongformerLayer(object):
    def __init__(self, config, layer_id=0, name='LongformerLayer'):
        super().__init__()
        self.attention = LongformerAttention(config, layer_id, name=name+'.attention')
        self.intermediate = LongformerIntermediate(config, name=name+'.intermediate')
        self.intermediate_size = config.intermediate_size
        self.output = LongformerOutput(config, name=name+'.output')
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

    def __call__(
        self,
        hidden_states,
        hidden_states_shape,
        attention_mask=None,
        attention_mask_shape=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        self_attn_outputs = self.attention(
            hidden_states,
            hidden_states_shape,
            attention_mask=attention_mask,
            attention_mask_shape=attention_mask_shape,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self_attn_outputs[0]
        outputs = self_attn_outputs[1:]

        intermediate_output = self.intermediate(attn_output, hidden_states_shape)
        layer_output = self.output(intermediate_output, hidden_states_shape[:-1]+[self.intermediate_size], attn_output)
        
        outputs = (layer_output,) + outputs
        return outputs



class LongformerEncoder(object):
    def __init__(self, config, name='LongformerEncoder'):

        self.config = config
        self.layer = [LongformerLayer(config, layer_id=i, name=name+'.layer.'+str(i)) for i in range(config.num_hidden_layers)]

    def __call__(
        self,
        hidden_states,
        hidden_states_shape,
        attention_mask=None,
        attention_mask_shape=None,
        head_mask=None,
        head_mask_shape=None,
        padding_len=0,
        output_attentions=False,
        output_hidden_states=False,
    ):

        is_index_masked = ht.bool_op(attention_mask, 0, 1)
        is_index_global_attn = ht.bool_op(attention_mask, 0, 2)
        is_global_attn = None

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None  # All local attentions.
        all_global_attentions = () if (output_attentions and is_global_attn) else None

        if head_mask is not None:
            assert head_mask_shape[0] == (
                len(self.layer)
            ), f"The head_mask should be specified for {len(self.layer)} layers, but it is for {head_mask.size()[0]}."
        for idx, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                hidden_states_shape,
                attention_mask=attention_mask,
                attention_mask_shape=attention_mask_shape,
                layer_head_mask=head_mask[idx] if head_mask is not None else None,
                is_index_masked=is_index_masked,
                is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (ht.transpose_op(layer_outputs[1], (0, 2, 1)), )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if padding_len > 0:
            hidden_states = ht.slice_op(hidden_states, (0, 0), (-1, hidden_states_shape[1]-padding_len))

        return [hidden_states]
 

class LongformerPooler(object):
    def __init__(self, config, name='LongformerPooler'):
        super().__init__()
        self.dense = ht.layers.Linear(config.hidden_size, config.hidden_size, weight_transpose=True, name=name+'.dense')
        self.activation = ht.tanh_op

    def __call__(self, hidden_states, hidden_states_shape):
        first_token_tensor = ht.slice_op(hidden_states, (0, 0), (-1, 1))
        first_token_tensor = ht.array_reshape_op(first_token_tensor, (-1, hidden_states_shape[-1]))
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LongformerModel(object):
    def __init__(self, config, add_pooling_layer=True):
        self.config = config
        self.hidden_size = config.hidden_size

        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        self.embeddings = LongformerEmbeddings(config, name='embeddings')
        self.encoder = LongformerEncoder(config, name='encoder')
        self.pooler = LongformerPooler(config, name='pooler') if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
        
    def _pad_to_window_size(
        self,
        input_ids,
        input_ids_shape,
        attention_mask,
        token_type_ids,
        position_ids,
        inputs_embeds,
        inputs_embeds_shape,
        pad_token_id: int,
    ):
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids_shape if input_ids is not None else inputs_embeds_shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            print(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )
            if input_ids is not None:
                input_ids = ht.concat_op(input_ids, ht.full_op((batch_size, padding_len), pad_token_id), axis=1)
            if position_ids is not None:
                position_ids = ht.concat_op(position_ids, ht.full_op((batch_size, padding_len), pad_token_id), axis=1)
            if inputs_embeds is not None:
                input_ids_padding = ht.full_op((batch_size, padding_len), self.config.pad_token_id)
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = ht.concat_op(inputs_embeds, inputs_embeds_padding, axis=-2)

            attention_mask = ht.concat_op(attention_mask, ht.full_op((batch_size, padding_len), 0), axis=1)
            token_type_ids = ht.concat_op(token_type_ids, ht.full_op((batch_size, padding_len), 0), axis=1)

        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds

    def _merge_to_attention_mask(self, attention_mask, global_attention_mask):
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            attention_mask = global_attention_mask + 1
        return attention_mask

    def get_extended_attention_mask(self, attention_mask, attention_mask_shape, input_shape):
        
        if len(attention_mask_shape) == 3:
            b, s1, s2 = attention_mask_shape
            extended_attention_mask = ht.array_reshape_op(attention_mask,(b, 1, s1, s2))

        elif len(attention_mask_shape) == 2:
            b, s = attention_mask_shape
            if self.config.is_decoder:
                extended_attention_mask = self.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, attention_mask_shape)
            else:
                extended_attention_mask = ht.array_reshape_op(attention_mask,(b, 1, 1, s))
        else:            
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask_shape})"
            )
        extended_attention_mask = ht.minus_byconst_op(extended_attention_mask, 1.0)
        extended_attention_mask = extended_attention_mask * np.finfo(np.float32).min
        return extended_attention_mask

    def __call__(
        self,
        input_ids=None,
        input_ids_shape=None,
        attention_mask=None,
        attention_mask_shape=None,
        global_attention_mask=None,
        head_mask=None,
        head_mask_shape=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        inputs_embeds_shape=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = list(input_ids_shape)
        elif inputs_embeds is not None:
            input_shape = list(inputs_embeds_shape[:-1])
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = ht.init.ones(input_shape, trainable=False)
            attention_mask_shape = input_shape
        if token_type_ids is None:
            token_type_ids = ht.init.zeros(input_shape, trainable=False)

        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)

        padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = self._pad_to_window_size(
            input_ids=input_ids,
            input_ids_shape=input_ids_shape,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            inputs_embeds_shape=inputs_embeds_shape,
            pad_token_id=self.config.pad_token_id,
        )
        input_shape [1] += padding_len
        input_ids_shape = input_shape
        if inputs_embeds_shape is not None:
            inputs_embeds_shape = input_shape + [inputs_embeds_shape[-1]]
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, attention_mask_shape, input_shape)
        extended_attention_mask = ht.slice_op(extended_attention_mask, (0, 0, 0, 0), (-1, 1, 1, -1))
        extended_attention_mask = ht.array_reshape_op(extended_attention_mask, input_ids_shape)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            input_ids_shape=input_ids_shape,
            position_ids=position_ids, 
            token_type_ids=token_type_ids, 
            inputs_embeds=inputs_embeds,
            inputs_embeds_shape=inputs_embeds_shape,
        )

        embedding_output_shape = input_ids_shape + [self.hidden_size]

        encoder_outputs = self.encoder(
            embedding_output,
            embedding_output_shape,
            attention_mask=extended_attention_mask,
            attention_mask_shape=input_ids_shape,
            head_mask=head_mask,
            head_mask_shape=head_mask_shape,
            padding_len=padding_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output, embedding_output_shape) if self.pooler is not None else None

        return [sequence_output, pooled_output]

class LongformerLMHead(object):
    def __init__(self, config, name='LongformerLMHead'):
        self.dense = ht.layers.Linear(config.hidden_size, config.hidden_size, weight_transpose=True, name=name+'.dense')
        self.layer_norm = ht.layers.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, name=name+".layer_norm")
        self.decoder = ht.layers.Linear(config.hidden_size, config.vocab_size, weight_transpose=True, name=name+'.decoder')
        self.bias = ht.init.zeros((config.vocab_size,) ,name=name+'.bias')
        self.decoder.bias = self.bias

    def __call__(self, features):
        x = self.dense(features)
        x = ht.gelu_op(x)
        x = self.layer_norm(x)
        x = self.decoder(x)

        return x


class LongformerForMaskedLM(object):
    def __init__(self, config):
        self.d_model = config.hidden_size
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.lm_head = LongformerLMHead(config, name='lm_head')

    def __call__(self, input_ids=None, input_ids_shape=None, attention_mask=None, attention_mask_shape=None, labels=None):
        outputs = self.longformer(input_ids,
                                  input_ids_shape,
                                  attention_mask=attention_mask,
                                  attention_mask_shape=attention_mask_shape)
        sequence_output = outputs[0]
        sequence_output = ht.array_reshape_op(sequence_output, (-1, self.d_model))
        prediction_scores = self.lm_head(sequence_output)
        prediction_scores = ht.array_reshape_op(prediction_scores, input_ids_shape + (-1,))
        
        if labels is not None:
            loss = ht.crossentropy_sparse_op(ht.softmax_op(prediction_scores), labels, ignored_index=-100)
            return loss, prediction_scores
        else:
            return prediction_scores


class LongformerClassificationHead(object):
    def __init__(self, config, name='LongformerClassificationHead'):
        self.hidden_size = config.hidden_size
        self.dense = ht.layers.Linear(config.hidden_size, config.hidden_size, weight_transpose=True, name=name+'.dense')
        self.dropout = ht.layers.DropOut(config.hidden_dropout_prob)
        self.out_proj = ht.layers.Linear(config.hidden_size, config.num_labels, weight_transpose=True, name=name+'.out_proj')

    def __call__(self, hidden_states, input_shape):
        hidden_states = ht.slice_op(hidden_states, (0, 0, 0), (-1, 1, -1))
        hidden_states = ht.array_reshape_op(hidden_states, (-1, self.hidden_size))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = ht.tanh_op(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output
        
        
class LongformerForSequenceClassification(object):
    def __init__(self, config):
        self.config = config
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.classifier = LongformerClassificationHead(config, name='classifier') 
        
    def get_input_embeddings(self):
        return self.longformer.get_input_embeddings()
        
    def __call__(self, input_ids, input_ids_shape, attention_mask=None, attention_mask_shape=None, labels=None):
        outputs = self.longformer(input_ids, input_ids_shape, attention_mask=attention_mask, attention_mask_shape=attention_mask_shape)
        hidden_states = outputs[0]
        logits = self.classifier(hidden_states, input_ids_shape)

        if labels is not None:
            # loss = ht.softmaxcrossentropy_sparse_op(logits, labels, ignored_index = -1)
            loss = ht.crossentropy_sparse_op(
                ht.softmax_op(logits), labels, ignored_index=-1)
            return loss, logits
        else:
            return logits