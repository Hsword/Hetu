import sys
import math
from collections import namedtuple
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple, Union


import numpy as np
import hetu as ht


def _stable_argsort(vector, vector_shape, dim):
    scale_offset = ht.arange_op(0, vector_shape[dim])
    scale_offset = ht.array_reshape_op(scale_offset, (1, 1, -1))
    scale_offset = ht.broadcast_shape_op(scale_offset, vector_shape)

    scaled_vector = vector_shape[dim] * vector + \
        ht.fmod_op(scale_offset, vector_shape[dim])
    return ht.argsort_op(scaled_vector, dim=dim)


def _get_least_common_mult_chunk_len(config):
    attn_types = config.attn_layers
    attn_types_set = set(attn_types)
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":
        return config.lsh_attn_chunk_length
    elif len(attn_types_set) == 1 and attn_types[0] == "local":
        return config.local_attn_chunk_length
    elif len(attn_types_set) == 2 and attn_types_set == set(["lsh", "local"]):
        return np.lcm(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    else:
        raise NotImplementedError(
            f"Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {config.attn_layers}. Select "
            "attn layer types from ['lsh', 'local'] only."
        )


def _get_min_chunk_len(config):
    attn_types = config.attn_layers
    attn_types_set = set(attn_types)
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":
        return config.lsh_attn_chunk_length
    elif len(attn_types_set) == 1 and attn_types[0] == "local":
        return config.local_attn_chunk_length
    elif len(attn_types_set) == 2 and attn_types_set == set(["lsh", "local"]):
        return min(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    else:
        raise NotImplementedError(
            f"Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {config.attn_layers}. Select "
            "attn layer types from ['lsh', 'local'] only."
        )


class AxialPositionEmbeddings(object):
    def __init__(self, config, name='AxialPositionEmbeddings'):

        self.axial_pos_shape = config.axial_pos_shape
        self.axial_pos_embds_dim = config.axial_pos_embds_dim
        self.dropout = config.hidden_dropout_prob

        self.least_common_mult_chunk_length = _get_least_common_mult_chunk_len(
            config)
        self.weights = []
        self.weights_shape = []

        if sum(self.axial_pos_embds_dim) != config.hidden_size:
            raise ValueError(
                f"Make sure that config.axial_pos_embds factors: {self.axial_pos_embds_dim} sum to "
                f"config.hidden_size: {config.hidden_size}"
            )
        i = 0
        for axis, axial_pos_embd_dim in enumerate(self.axial_pos_embds_dim):
            ax_shape = [1] * len(self.axial_pos_shape)
            ax_shape[axis] = self.axial_pos_shape[axis]
            ax_shape = tuple(ax_shape) + (axial_pos_embd_dim,)
            self.weights.append(ht.init.ones(ax_shape, name=name+'.weights.'+str(i)))
            self.weights_shape.append(ax_shape)
            i += 1

    def __call__(self, position_ids, position_ids_shape):
        batch_size=position_ids_shape[0]
        sequence_length=position_ids_shape[1]

        broadcasted_weights=[]
        for i in range(len(self.weights)):
            weight=self.weights[i]
            weight_shape=self.weights_shape[i]
            broadcasted_weights.append(ht.broadcast_shape_op(
                weight, (batch_size,) + self.axial_pos_shape + weight_shape[-1:]))


        if reduce(mul, self.axial_pos_shape) != sequence_length:
            raise ValueError(
                f"If training, make sure that config.axial_pos_shape factors: {self.axial_pos_shape} multiply to "
                f"sequence length. Got prod({self.axial_pos_shape}) != sequence_length: {sequence_length}. "
                f"You might want to consider padding your sequence length to {reduce(mul, self.axial_pos_shape)} "
                "or changing config.axial_pos_shape."
            )

        if self.dropout > 0:
            weights=ht.concatenate_op(broadcasted_weights, axis=3)
            transposed_weights=ht.transpose_op(weights, (0, 2, 1, 3))
            dropped_transposed_weights=ht.dropout2d_op(
                transposed_weights, 1 - self.dropout)
            dropped_weights=ht.transpose_op(
                dropped_transposed_weights, (0, 2, 1, 3))
            position_encodings=ht.array_reshape_op(
                dropped_weights, (batch_size, sequence_length, -1))

        else:
            position_encodings=ht.concatenate_op([ht.array_reshape_op(
                weight, (batch_size, sequence_length, -1)) for weight in broadcasted_weights], axis=2)

        return position_encodings


class PositionEmbeddings(object):
    def __init__(self, config, name='PositionEmbeddings'):
        self.dropout=config.hidden_dropout_prob
        self.embedding=ht.layers.Embedding(
            config.max_position_embeddings, config.hidden_size, name=name+'.embedding')

    def __call__(self, position_ids, position_ids_shape):
        position_embeddings=self.embedding(position_ids)
        position_embeddings=ht.dropout_op(
            position_embeddings, 1 - self.dropout)
        return position_embeddings


class ReformerEmbeddings(object):
    def __init__(self, config, name='ReformerEmbeddings'):

        self.max_position_embeddings=config.max_position_embeddings
        self.dropout=config.hidden_dropout_prob

        self.word_embeddings=ht.layers.Embedding(
            config.vocab_size, config.hidden_size, name=name+'.word_embeddings')
        self.position_embeddings=(
            AxialPositionEmbeddings(config, name=name+'.position_embeddings') if config.axial_pos_embds
            else PositionEmbeddings(config, name=name+'.position_embeddings')
        )

    def __call__(self, input_ids=None, input_ids_shape=None, position_ids=None, position_ids_shape=None, inputs_embeds=None, inputs_embeds_shape=None, start_idx_pos_encodings=0):
        if input_ids is not None:
            input_shape=input_ids_shape
        else:
            input_shape=inputs_embeds_shape[:-1]

        seq_length=input_shape[1]
        if position_ids is None:
            position_ids=ht.arange_op(
                start_idx_pos_encodings, start_idx_pos_encodings + seq_length)
            position_ids=ht.unsqueeze_op(position_ids, 0)
            position_ids=ht.broadcast_shape_op(position_ids, input_shape)
            position_ids_shape=input_shape

        if inputs_embeds is None:
            inputs_embeds=self.word_embeddings(input_ids)

        if position_ids_shape[-1] > self.max_position_embeddings:
            raise ValueError(
                f"Sequence Length: {position_ids_shape[-1]} has to be less or equal than "
                f"config.max_position_embeddings {self.max_position_embeddings}."
            )

        embeddings=ht.dropout_op(inputs_embeds, 1 - self.dropout)

        position_embeddings=self.position_embeddings(
            position_ids, position_ids_shape)
        embeddings=embeddings + position_embeddings
        return embeddings


class EfficientAttentionMixin:
    def _look_adjacent(self, vectors, num_chunks_before, num_chunks_after):
        if num_chunks_before == 0 and num_chunks_after == 0:
            return vectors

        slices=[]
        for i in range(-num_chunks_before, num_chunks_after + 1):
            if i == 0:
                slices.append(vectors)
            else:
                vector_1=ht.slice_op(vectors, (0, 0, i), (-1, -1, -1))
                vector_2=ht.slice_op(vectors, (0, 0, 0), (-1, -1, i))

                slices.append(ht.concatenate_op([vector_1, vector_2], axis=2))
        return ht.concatenate_op(slices, axis=3)

    def _split_hidden_size_dim(self, x, x_shape, num_attn_heads, attn_head_size):
        new_x_shape=x_shape[:-1] + (num_attn_heads, attn_head_size)
        x=ht.array_reshape_op(x, new_x_shape)
        return ht.transpose_op(x, (0, 2, 1, 3))

    def _merge_hidden_size_dims(self, x, x_shape, num_attn_heads, attn_head_size):
        x=ht.transpose_op(x, (0, 2, 1, 3))
        return ht.array_reshape_op(x, (x_shape[0], -1, num_attn_heads * attn_head_size))

    def _split_seq_length_dim_to(self, vectors, vectors_shape, dim_factor_1, dim_factor_2, num_attn_heads, attn_head_size=None):
        batch_size=vectors_shape[0]
        split_dim_shape=(batch_size, num_attn_heads,
                         dim_factor_1, dim_factor_2)

        if len(vectors_shape) == 4:
            return ht.array_reshape_op(vectors, split_dim_shape + (attn_head_size,))
        elif len(vectors_shape) == 3:
            return ht.array_reshape_op(vectors, split_dim_shape)
        else:
            raise ValueError(
                f"Input vector rank should be one of [3, 4], but is: {len(vectors_shape)}")


class LSHSelfAttention(EfficientAttentionMixin):
    def __init__(self, config, name='LSHSelfAttention'):

        self.config=config
        self.chunk_length=config.lsh_attn_chunk_length
        self.num_hashes=config.num_hashes
        self.num_buckets=config.num_buckets
        self.num_chunks_before=config.lsh_num_chunks_before
        self.num_chunks_after=config.lsh_num_chunks_after
        self.hash_seed=config.hash_seed
        self.is_decoder=config.is_decoder
        self.max_position_embeddings=config.max_position_embeddings

        self.dropout=config.lsh_attention_probs_dropout_prob

        self.num_attention_heads=config.num_attention_heads
        self.attention_head_size=config.attention_head_size
        self.all_head_size=self.num_attention_heads * self.attention_head_size
        self.hidden_size=config.hidden_size

        self.query_key=ht.layers.Linear(
            self.hidden_size, self.all_head_size, bias=False, weight_transpose=True, name=name+'.query_key')
        self.value=ht.layers.Linear(
            self.hidden_size, self.all_head_size, bias=False, weight_transpose=True, name=name+'.value')

        self.self_mask_value_float32=-1e5
        self.mask_value_float32=-1e9

    def __call__(
        self,
        hidden_states,
        hidden_states_shape,
        attention_mask=None,
        attention_mask_shape=None,
        head_mask=None,
        head_mask_shape=None,
        num_hashes=None,
        buckets=None,
        past_buckets_states=None,
        past_buckets_states_shape=None,
        use_cache=False,
        output_attentions=False,
        do_cached_attention=False,
        **kwargs,
    ):
        sequence_length=hidden_states_shape[1]
        batch_size=hidden_states_shape[0]

        num_hashes=num_hashes if num_hashes is not None else self.num_hashes
    
        query_vectors=None
        query_key_vectors=self.query_key(ht.array_reshape_op(
            hidden_states, (-1, hidden_states_shape[-1])))
        value_vectors=self.value(ht.array_reshape_op(
            hidden_states, (-1, hidden_states_shape[-1])))

        query_key_vectors_shape=value_vectors_shape=hidden_states_shape[:-1] + (self.all_head_size, )

        query_key_vectors=self._split_hidden_size_dim(
            query_key_vectors, query_key_vectors_shape, self.num_attention_heads, self.attention_head_size
        )
        value_vectors=self._split_hidden_size_dim(
            value_vectors, value_vectors_shape, self.num_attention_heads, self.attention_head_size
        )

        do_standard_self_attention=(sequence_length <= self.chunk_length)

        if not do_standard_self_attention:
            raise NotImplementedError
        else:
            sorted_bucket_idx_per_hash=ht.arange_op(0, sequence_length)
            sorted_bucket_idx_per_hash=ht.repeat_op(sorted_bucket_idx_per_hash, (batch_size, self.num_attention_heads, 1))

        key_vectors=self._len_and_dim_norm(query_key_vectors)

        query_vectors=query_vectors if query_vectors is not None else query_key_vectors
        query_vectors_shape = (batch_size, self.num_attention_heads, sequence_length, self.attention_head_size)
        out_vectors, logits, attention_probs=self._attend(
            query_vectors=query_vectors,
            query_vectors_shape=query_vectors_shape,
            key_vectors=key_vectors,
            value_vectors=value_vectors,
            sorted_bucket_idx_per_hash=sorted_bucket_idx_per_hash,
            attention_mask=attention_mask,
            attention_mask_shape=attention_mask_shape,
            head_mask=head_mask,
            head_mask_shape=head_mask_shape,
            do_standard_self_attention=do_standard_self_attention,
            do_cached_attention=do_cached_attention,
        )


        if not do_standard_self_attention:
            out_vectors, logits=ReverseSort.apply(
                out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx)

        if not do_standard_self_attention or (do_cached_attention and past_buckets is not None):
            if num_hashes > 1:
                out_vectors=self._split_seq_length_dim_to(
                    out_vectors,
                    num_hashes,
                    sequence_length,
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                logits=self._split_seq_length_dim_to(
                    logits,
                    num_hashes,
                    sequence_length,
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                logits=ht.unsqueeze_op(logits, -1)

                log_sum_exp=ht.exp_op(logits)
                log_sum_exp=ht.reduce_sum_op(log_sum_exp, axes=2, keepdims=True)
                log_sum_exp=ht.log_op(log_sum_exp)     
                log_sum_exp=ht.broadcastto_op(log_sum_exp, logits)
                probs_vectors=ht.exp_op(logits + (-1)*log_sum_exp)

                out_vectors=ht.reduce_sum_op(out_vectors * probs_vectors, axes=2)

        out_vectors_shape = (
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        )
        out_vectors=self._merge_hidden_size_dims(
            out_vectors, out_vectors_shape, self.num_attention_heads, self.attention_head_size)

        return out_vectors

    def _set_num_buckets(self, sequence_length):

        num_buckets_pow_2=(
            2 * (sequence_length // self.chunk_length)).bit_length() - 1
        num_buckets=2**num_buckets_pow_2

        num_buckets_limit=2 * max(
            int((self.max_position_embeddings // self.chunk_length) ** (0.5)),
            self.chunk_length,
        )
        if num_buckets > num_buckets_limit:
            num_buckets=[2 ** (num_buckets_pow_2 // 2), 2 **
                               (num_buckets_pow_2 - num_buckets_pow_2 // 2)]

        logger.warning(
            f"config.num_buckets is not set. Setting config.num_buckets to {num_buckets}...")

        self.config.num_buckets=num_buckets
        self.num_buckets=num_buckets

    def _attend(
        self,
        query_vectors,
        query_vectors_shape,
        key_vectors,
        value_vectors,
        sorted_bucket_idx_per_hash,
        attention_mask,
        attention_mask_shape,
        head_mask,
        head_mask_shape,
        do_standard_self_attention,
        do_cached_attention,
    ):
        if not do_standard_self_attention:
            key_vectors=self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
            value_vectors=self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)

        batch_size, sequence_length = query_vectors_shape[0], query_vectors_shape[2]

        key_vectors = ht.transpose_op(key_vectors, (0, 1, 3, 2))
        query_key_dots=ht.batch_matmul_op(query_vectors, key_vectors)
        query_key_dots_shape= (batch_size, self.num_attention_heads, sequence_length, sequence_length)

        if not do_standard_self_attention:
            query_bucket_idx=self._split_seq_length_dim_to(sorted_bucket_idx_per_hash, -1, self.chunk_length, self.num_attention_heads)
            key_value_bucket_idx=self._look_adjacent(
                query_bucket_idx, self.num_chunks_before, self.num_chunks_after)
        elif do_cached_attention:
            #need
            query_bucket_idx=(query_key_dots_shape[-1] - 1) * ht.ones_like_op(query_key_dots)[:, :, :, -1]
            key_value_bucket_idx=ht.arange_op(0,query_key_dots_shape[-1])
            key_value_bucket_idx=ht.broadcast_shape_op(key_value_bucket_idx, query_key_dots_shape[:2]+(-1,),add_axes=(0,1))
        else:
            query_bucket_idx=key_value_bucket_idx=sorted_bucket_idx_per_hash

        self_mask_value=self.self_mask_value_float32
        mask_value=self.mask_value_float32

        if not do_cached_attention:
            mask=self._compute_attn_mask(
                query_bucket_idx,
                key_value_bucket_idx,
                attention_mask,
                attention_mask_shape,
                query_key_dots_shape,
                do_standard_self_attention,
            )

            if mask is not None:
                query_key_dots=ht.where_const_op(mask, query_key_dots, mask_value)

        query_bucket_idx=ht.broadcast_shape_op(query_bucket_idx, (-1, -1, -1, sequence_length), add_axes=(3, ))
        key_value_bucket_idx=ht.broadcast_shape_op(key_value_bucket_idx, (-1, -1, sequence_length, -1), add_axes=(2, ))

        self_mask=ht.ne_op(query_bucket_idx, key_value_bucket_idx)
        query_key_dots=ht.where_const_op(self_mask, query_key_dots, self_mask_value)

  
        #need
        logits=ht.exp_op(query_key_dots)
        logits=ht.reduce_sum_op(logits, axes=-1, keepdims=True)
        logits=ht.log_op(logits)   
        logits=ht.broadcastto_op(logits, query_key_dots)  
        attention_probs=ht.exp_op(query_key_dots + (-1)*logits)

        attention_probs=ht.dropout_op(attention_probs, 1-self.dropout)

        if head_mask is not None:
            attention_probs=attention_probs * head_mask

        out_vectors=ht.batch_matmul_op(attention_probs, value_vectors)
        return out_vectors, logits, attention_probs

    def _compute_attn_mask(
        self, query_indices, key_indices, attention_mask, attention_mask_shape, query_key_dots_shape, do_standard_self_attention
    ):
        if attention_mask is not None:
            attention_mask=ht.unsqueeze_op(attention_mask, 1)
            attention_mask_shape = (attention_mask_shape[0], 1, attention_mask_shape[1])
            if not do_standard_self_attention:
                attention_mask=self._split_seq_length_dim_to(
                    attention_mask, attention_mask_shape, -1, self.chunk_length, 1)
                attention_mask=self._look_adjacent(
                    attention_mask, self.num_chunks_before, self.num_chunks_after)
            attention_mask=ht.unsqueeze_op(attention_mask,-2)
            attention_mask = ht.broadcast_shape_op(attention_mask, query_key_dots_shape)

        if self.is_decoder is True:
            causal_mask=ht.ge_op(ht.unsqueeze_op(query_indices, -1),
                                 ht.unsqueeze_op(key_indices, -1))

            if attention_mask is not None:
                attention_mask=causal_mask * attention_mask
            else:
                attention_mask=causal_mask

        return attention_mask


    def _expand_to_indices_in_relevant_chunk(self, indices, indices_shape, sequence_length):

        start_indices_chunk=(
            (ht.slice_op(indices,(0,-1),(-1,1))// self.chunk_length) - self.num_chunks_before) * self.chunk_length
        total_chunk_size=self.chunk_length * \
            (1 + self.num_chunks_before + self.num_chunks_after)


        expanded_start_indices=ht.unsqueeze_op(start_indices_chunk,-1)
        expanded_start_indices=ht.broadcast_shape_op(expanded_start_indices, (indices_shape[0], total_chunk_size))
        tmp_indices = ht.arange_op(0, 
            total_chunk_size)
        tmp_indices=ht.unsqueeze_op(start_indices_chunk,0)
        tmp_indices=ht.broadcast_shape_op(tmp_indices, (indices_shape[0], total_chunk_size))
        chunk_sequence_indices=expanded_start_indices + tmp_indices

        chunk_sequence_indices = ht.array_reshape_op(chunk_sequence_indices, (-1, ))
        chunk_sequence_indices = ht.fmod_op(chunk_sequence_indices, sequence_length)

        indices=ht.unsqueeze_op(indices,-1)
        indices=ht.broadcast_shape_op(indices, (indices_shape[0], total_chunk_size, -1))
        indices=ht.flatten(chunk_sequence_indices, 0, 1)

        indices=ht.slice_assign_matrix_op(indices, chunk_sequence_indices,(0,-1),(-1,1),(0,0),(-1,-1))
        return indices

    def _len_and_dim_norm(self, vectors):
        vectors=self._len_norm(vectors)
        vectors=vectors *( 1 / math.sqrt(self.attention_head_size))
        return vectors

    def _len_norm(self, x, epsilon=1e-6):
        variance = ht.pow_op(x, 2)
        variance = ht.reduce_mean_op(variance, -1, keepdims=True)
        variance = ht.rsqrt_op(variance + epsilon)
        variance = ht.broadcastto_op(variance, x)
        norm_x = x * variance
        return norm_x

    def _gather_by_expansion(self, vectors, idxs, num_hashes):
        expanded_idxs=ht.unsqueeze_op(idxs, -1)
        expanded_idxs=ht.broadcast_shape_op(expanded_idxs, (-1, -1, -1, self.attention_head_size))
        vectors=ht.repeat_op(vectors, (1, 1, num_hashes, 1))
        vectors=ht.gather_op(vectors, 2, expanded_idxs)
        return vectors


class ReverseSort:
    @ staticmethod
    def __call__(ctx, out_vectors, out_vectors_shape, logits, sorted_bucket_idx, undo_sorted_bucket_idx):
        expanded_undo_sort_indices=ht.unsqueeze_op(undo_sorted_bucket_idx, -1)
        expanded_undo_sort_indices=ht.broadcast_shape_op(expanded_undo_sort_indices, out_vectors_shape)

        out_vectors = ht.gather_op(out_vectors, 2, expanded_undo_sort_indices)
        logits = ht.gather_op(logits, 2, undo_sorted_bucket_idx)
        return out_vectors, logits

class LocalSelfAttention(EfficientAttentionMixin):
    def __init__(self, config, name='LocalSelfAttention'):
        super().__init__()

        self.num_attention_heads=config.num_attention_heads
        self.chunk_length=config.local_attn_chunk_length
        self.num_chunks_before=config.local_num_chunks_before
        self.num_chunks_after=config.local_num_chunks_after
        self.is_decoder=config.is_decoder
        self.pad_token_id=config.pad_token_id

        self.attention_head_size=config.attention_head_size
        self.all_head_size=self.num_attention_heads * self.attention_head_size
        self.hidden_size=config.hidden_size


        self.query=ht.layers.Linear(
            self.hidden_size, self.all_head_size, bias=False, weight_transpose=True, name=name+".query")
        self.key=ht.layers.Linear(
            self.hidden_size, self.all_head_size, bias=False, weight_transpose=True, name=name+".key")
        self.value=ht.layers.Linear(
            self.hidden_size, self.all_head_size, bias=False, weight_transpose=True, name=name+".value")

        self.dropout=config.local_attention_probs_dropout_prob
        self.mask_value_float32=-1e9

    def __call__(
        self,
        hidden_states,
        hidden_states_shape,
        attention_mask=None,
        attention_mask_shape=None,
        head_mask=None,
        head_mask_shape=None,
        past_buckets_states=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        sequence_length=hidden_states_shape[1]
        batch_size=hidden_states_shape[0]

        hidden_states=ht.array_reshape_op(
            hidden_states, (-1, hidden_states_shape[-1]))
        query_vectors=self.query(hidden_states)
        key_vectors=self.key(hidden_states)
        value_vectors=self.value(hidden_states)

        vector_size=hidden_states_shape[:-1] + (self.all_head_size, )
        query_vectors=self._split_hidden_size_dim(
            query_vectors, vector_size, self.num_attention_heads, self.attention_head_size)
        key_vectors=self._split_hidden_size_dim(
            key_vectors, vector_size, self.num_attention_heads, self.attention_head_size)
        value_vectors=self._split_hidden_size_dim(
            value_vectors, vector_size, self.num_attention_heads, self.attention_head_size)
        vector_size=hidden_states_shape[:-1] + (self.num_attention_heads, self.attention_head_size)

        if self.chunk_length is None:
            assert self.num_chunks_before == 0 and self.num_chunks_after == 0, (
                "If `config.chunk_length` is `None`, make sure `config.num_chunks_after` and"
                " `config.num_chunks_before` are set to 0."
            )
        key_vectors=key_vectors / math.sqrt(self.attention_head_size)

        indices=ht.arange_op(0, sequence_length)
        indices=ht.repeat_op(indices, (batch_size, self.num_attention_heads, 1))
        indices_shape = (batch_size, self.num_attention_heads, sequence_length)

        do_standard_self_attention=sequence_length <= self.chunk_length

        if not do_standard_self_attention:
            query_vectors=self._split_seq_length_dim_to(
                query_vectors,
                vector_size,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )
            key_vectors=self._split_seq_length_dim_to(
                key_vectors,
                vector_size,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )
            value_vectors=self._split_seq_length_dim_to(
                value_vectors,
                vector_size,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )

            query_indices=self._split_seq_length_dim_to(
                indices, indices_shape, -1, self.chunk_length, self.num_attention_heads)
            key_indices=self._split_seq_length_dim_to(
                indices, indices_shape, -1, self.chunk_length, self.num_attention_heads)

            key_vectors=self._look_adjacent(
                key_vectors, self.num_chunks_before, self.num_chunks_after)
            value_vectors=self._look_adjacent(
                value_vectors, self.num_chunks_before, self.num_chunks_after)
            key_indices=self._look_adjacent(
                key_indices, self.num_chunks_before, self.num_chunks_after)
        else:
            query_indices=key_indices=indices

        key_vectors = ht.transpose_op(key_vectors, (0, 1, 3, 2))
        query_key_dots = ht.batch_matmul_op(query_vectors, key_vectors)
        query_key_dots_shape = (batch_size, self.num_attention_heads, sequence_length, sequence_length)

        mask = self._compute_attn_mask(
            query_indices, key_indices, attention_mask, attention_mask_shape, query_key_dots_shape, do_standard_self_attention
        )

        if mask is not None:
            mask_value=self.mask_value_float32
            query_key_dots=ht.where_const_op(mask, query_key_dots, mask_value)


        logits=ht.exp_op(query_key_dots)
        logits=ht.reduce_sum_op(logits, axes=-1, keepdims=True)
        logits=ht.log_op(logits)     
        logits=ht.broadcastto_op(logits, query_key_dots)
        attention_probs=ht.exp_op(query_key_dots + (-1)*logits)

        attention_probs=ht.dropout_op(attention_probs, 1-self.dropout)

        if head_mask is not None:
            attention_probs=attention_probs * head_mask

        out_vectors = ht.batch_matmul_op(attention_probs, value_vectors)
        if not do_standard_self_attention:
            out_vectors = ht.flatten_op(out_vectors, start_dim=2, end_dim=3)

        out_vectors_shape = (
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        )

        out_vectors=self._merge_hidden_size_dims(
            out_vectors,out_vectors_shape, self.num_attention_heads, self.attention_head_size)

        return out_vectors

    def _compute_attn_mask(
        self, query_indices, key_indices, attention_mask, attention_mask_shape, query_key_dots_shape, do_standard_self_attention
    ):
        if attention_mask is not None:
            attention_mask=ht.unsqueeze_op(attention_mask, 1)
            attention_mask_shape = (attention_mask_shape[0], 1, attention_mask_shape[1])
            if not do_standard_self_attention:
                attention_mask=self._split_seq_length_dim_to(
                    attention_mask, attention_mask_shape, -1, self.chunk_length, 1)
                attention_mask=self._look_adjacent(
                    attention_mask, self.num_chunks_before, self.num_chunks_after)
            attention_mask=ht.unsqueeze_op(attention_mask,-2)
            attention_mask = ht.broadcast_shape_op(attention_mask, query_key_dots_shape)

        if self.is_decoder is True:
            causal_mask=ht.ge_op(ht.unsqueeze_op(query_indices, -1),
                                 ht.unsqueeze_op(key_indices, -1))

            if attention_mask is not None:
                attention_mask=causal_mask * attention_mask
            else:
                attention_mask=causal_mask

        return attention_mask


class ReformerSelfOutput(object):
    def __init__(self, config, name='ReformerSelfOutput'):
        all_head_size=config.num_attention_heads * config.attention_head_size
        self.all_head_size = all_head_size
        self.dropout=config.hidden_dropout_prob
        self.dense=ht.layers.Linear(all_head_size, config.hidden_size, bias=False, weight_transpose=True, name=name+'.dense')

    def __call__(self, hidden_states, hidden_states_shape):
        hidden_states=ht.array_reshape_op(hidden_states, (-1, self.all_head_size))
        hidden_states=self.dense(hidden_states)
        hidden_states=ht.array_reshape_op(hidden_states, hidden_states_shape[:-1]+(-1, ))
        hidden_states=ht.dropout_op(hidden_states, 1 - self.dropout)
        return hidden_states


class ReformerAttention(object):
    def __init__(self, config, layer_id=0, name='ReformerAttention'):
        super().__init__()
        self.layer_id=layer_id
        self.attn_layers=config.attn_layers

        self.layer_norm=ht.layers.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, name=name+".layer_norm")

        if len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "lsh":
            self.self_attention=LSHSelfAttention(config, name=name+".self_attention")
        elif len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "local":
            self.self_attention=LocalSelfAttention(config, name=name+".self_attention")
        elif len(set(self.attn_layers)) == 2 and set(self.attn_layers) == set(["lsh", "local"]):
            if self.attn_layers[self.layer_id] == "lsh":
                self.self_attention=LSHSelfAttention(config, name=name+".self_attention")
            else:
                self.self_attention=LocalSelfAttention(config, name=name+".self_attention")
        else:
            raise NotImplementedError(
                f"Only attn layer types 'lsh' and 'local' exist, but got `config.attn_layers`: {self.attn_layers}. "
                "Select attn layer types from ['lsh', 'local'] only."
            )
        self.output=ReformerSelfOutput(config, name=name+'.output')

    def __call__(
        self,
        hidden_states,
        hidden_states_shape,
        attention_mask=None,
        attention_mask_shape=None,
        head_mask=None,
        head_mask_shape=None,
        num_hashes=None,
        past_buckets_states=None,
        use_cache=False,
        orig_sequence_length=None,
        output_attentions=False,
        buckets=None,
    ):
        hidden_states=self.layer_norm(hidden_states)

        if past_buckets_states is not None:
            past_buckets_states_layer=past_buckets_states[self.layer_id]
        else:
            past_buckets_states_layer=None

        hidden_states=self.self_attention(
            hidden_states=hidden_states,
            hidden_states_shape=hidden_states_shape,
            head_mask=head_mask,
            head_mask_shape=head_mask_shape,
            attention_mask=attention_mask,
            attention_mask_shape=attention_mask_shape,
            num_hashes=num_hashes,
            past_buckets_states=past_buckets_states_layer,
            use_cache=use_cache,
            output_attentions=output_attentions,
            buckets=buckets,
        )

        attention_output=self.output(hidden_states, hidden_states_shape)

        return [attention_output]


class ReformerFeedForwardDense(object):
    def __init__(self, config, name='ReformerFeedForwardDense'):
        self.dropout=config.hidden_dropout_prob
        self.hidden_size = config.hidden_size
        self.dense=ht.layers.Linear(config.hidden_size, config.feed_forward_size, weight_transpose=True, name=name+".dense")
        if config.hidden_act == 'relu':
            self.act_fn=ht.relu_op
        else:
            self.act_fn=ht.gelu_op

    def __call__(self, hidden_states, hidden_states_shape):
        hidden_states=ht.array_reshape_op(hidden_states, (-1, self.hidden_size))
        hidden_states=self.dense(hidden_states)
        hidden_states=ht.array_reshape_op(hidden_states, hidden_states_shape[:-1]+(-1, ))
        hidden_states=ht.dropout_op(hidden_states, 1-self.dropout)
        hidden_states=self.act_fn(hidden_states)
        return hidden_states


class ReformerFeedForwardOutput(object):
    def __init__(self, config, name='ReformerFeedForwardOutput'):
        self.dropout=config.hidden_dropout_prob
        self.feed_forward_size = config.feed_forward_size
        self.dense=ht.layers.Linear(config.feed_forward_size, config.hidden_size, weight_transpose=True, name=name+'.dense')

    def __call__(self, hidden_states, hidden_states_shape):
        hidden_states=ht.array_reshape_op(hidden_states, (-1, self.feed_forward_size))
        hidden_states=self.dense(hidden_states)
        hidden_states=ht.array_reshape_op(hidden_states, hidden_states_shape[:-1]+(-1, ))
        hidden_states=ht.dropout_op(hidden_states, 1 - self.dropout)
        return hidden_states


class ChunkReformerFeedForward(object):
    def __init__(self, config, name='ChunkReformerFeedForward'):
        super().__init__()
        self.layer_norm=ht.layers.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, name=name+'.layer_norm')
        self.dense=ReformerFeedForwardDense(config, name=name+'.dense')
        self.feed_forward_size=config.feed_forward_size
        self.output=ReformerFeedForwardOutput(config, name=name+'.output')

    def __call__(self, hidden_states, hidden_states_shape):
        hidden_states=self.layer_norm(hidden_states)
        hidden_states=self.dense(hidden_states, hidden_states_shape)
        return self.output(hidden_states, hidden_states_shape[:-1]+(self.feed_forward_size, ))


class ReformerLayer(object):
    def __init__(self, config, layer_id=0, name='ReformerLayer'):
        self.attention=ReformerAttention(config, layer_id, name=name+".attention")
        self.feed_forward=ChunkReformerFeedForward(config, name=name+".feed_forward")

    def __call__(
        self,
        prev_attn_output,
        hidden_states,
        hidden_states_shape,
        attention_mask=None,
        attention_mask_shape=None,
        head_mask=None,
        head_mask_shape=None,
        num_hashes=None,
        past_buckets_states=None,
        use_cache=False,
        orig_sequence_length=None,
        output_attentions=False,
    ):
        attn_outputs=self.attention(
            hidden_states=hidden_states,
            hidden_states_shape=hidden_states_shape,
            head_mask=head_mask,
            head_mask_shape=head_mask_shape,
            attention_mask=attention_mask,
            attention_mask_shape=attention_mask_shape,
            num_hashes=num_hashes,
            past_buckets_states=past_buckets_states,
            use_cache=use_cache,
            orig_sequence_length=orig_sequence_length,
            output_attentions=output_attentions,
        )
        attn_output=attn_outputs[0]
        attn_output=prev_attn_output + attn_output

        hidden_states=hidden_states + self.feed_forward(attn_output, hidden_states_shape)

        return [attn_output, hidden_states]




class ReformerEncoder(object):
    def __init__(self, config, name='ReformerEncoder'):
        self.dropout=config.hidden_dropout_prob

        self.layers=[ReformerLayer(config, i, name=name+'.layers.'+str(i))
                                   for i in range(config.num_hidden_layers)]
        self.layer_norm=ht.layers.LayerNorm(
            2 * config.hidden_size, eps=config.layer_norm_eps, name=name+'.layer_norm')

    def __call__(
        self,
        hidden_states,
        hidden_states_shape,
        attention_mask=None,
        attention_mask_shape=None,
        head_mask=None,
        head_mask_shape=None,
        num_hashes=None,
        use_cache=False,
        orig_sequence_length=None,
        output_hidden_states=False,
        output_attentions=False,
    ):

        past_buckets_states=[((None), (None)) for i in range(len(self.layers))]
        attn_output=hidden_states
        for layer_id, (layer, layer_head_mask) in enumerate(zip(self.layers, head_mask)):
            layer_outputs=layer(
                prev_attn_output=attn_output,
                hidden_states=hidden_states,
                hidden_states_shape=hidden_states_shape,
                attention_mask=attention_mask,
                attention_mask_shape=attention_mask_shape,
                head_mask=layer_head_mask,
                head_mask_shape=None if head_mask_shape is None else head_mask_shape[layer_id],
                num_hashes=num_hashes,
                past_buckets_states=past_buckets_states,
                use_cache=use_cache,
                orig_sequence_length=orig_sequence_length,
                output_attentions=output_attentions,
            )
            attn_output=layer_outputs[0]
            hidden_states=layer_outputs[1]

        hidden_states=ht.concat_op(attn_output, hidden_states, axis=-1)
        hidden_states=self.layer_norm(hidden_states)
        hidden_states=ht.dropout_op(hidden_states, 1 - self.dropout)

        return hidden_states


class ReformerModel(object):
    def __init__(self, config):
        self.config=config
        assert (
            self.config.num_hidden_layers > 0
        ), "`config.attn_layers` is empty. Select at least one attn layer form ['lsh', 'local']"

        self.hidden_size=config.hidden_size
        self.embeddings=ReformerEmbeddings(config, name='embeddings')
        self.encoder=ReformerEncoder(config, name='encoder')

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask
        
    def __call__(
        self,
        input_ids=None,
        input_ids_shape=None,
        attention_mask=None,
        attention_mask_shape=None,
        position_ids=None,
        head_mask=None,
        head_mask_shape=None,
        inputs_embeds=None,
        inputs_embeds_shape=None,
        num_hashes=None,
        use_cache=None,
        output_hidden_states=None,
        output_attentions=None
    ):
        use_cache=use_cache if use_cache is not None else self.config.use_cache
        output_attentions=output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states=(
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape=input_ids_shape
        elif inputs_embeds is not None:
            input_shape=inputs_embeds_shape[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        assert (
            len(input_shape) == 2
        ), f"`input_ids` have be of shape `[batch_size, sequence_length]`, but got shape: {input_shape}"

        # get head mask
        head_mask=self.get_head_mask(
            head_mask, self.config.num_hidden_layers, is_attention_chunked=True)

        orig_sequence_length=input_shape[-1]

        least_common_mult_chunk_length=_get_least_common_mult_chunk_len(
            self.config)
        min_chunk_length=_get_min_chunk_len(self.config)

        must_pad_to_match_chunk_length=(
            input_shape[-1] % least_common_mult_chunk_length != 0
            and input_shape[-1] > min_chunk_length
        )

        if must_pad_to_match_chunk_length:
            padding_length=least_common_mult_chunk_length - \
                input_shape[-1] % least_common_mult_chunk_length
            raise ValueError(
                f"If training, sequence length {input_shape[-1]} has to be a multiple of least common multiple "
                f"chunk_length {least_common_mult_chunk_length}. Please consider padding the input to a length "
                f"of {input_shape[-1] + padding_length}."
            )

        start_idx_pos_encodings=0

        embedding_output=self.embeddings(
            input_ids=input_ids,
            input_ids_shape=input_ids_shape,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            inputs_embeds_shape=inputs_embeds_shape,
            start_idx_pos_encodings=start_idx_pos_encodings,
        )

        hidden_states_shape=input_shape + (self.hidden_size, )
        sequence_output=self.encoder(
            hidden_states=embedding_output,
            hidden_states_shape=hidden_states_shape,
            head_mask=head_mask,
            head_mask_shape=head_mask_shape,
            attention_mask=attention_mask,
            attention_mask_shape=attention_mask_shape,
            num_hashes=num_hashes,
            use_cache=use_cache,
            orig_sequence_length=orig_sequence_length,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        return [sequence_output]

class ReformerForPretraining(object):
    def __init__(self, config):
        self.transformer = ReformerModel(config)
        self.n_embd = config.hidden_size
        self.lm_head = ht.layers.Linear(config.hidden_size, config.vocab_size, weight_transpose=True, bias=False)

    def __call__(self, input_ids, input_shape, attention_mask=None, labels=None):
        hidden_states = self.transformer(input_ids, input_shape, attention_mask, input_shape)[0]
        hidden_states = ht.array_reshape_op(hidden_states, (-1, self.n_embd))
        lm_logits = self.lm_head(hidden_states)
        lm_logits = ht.array_reshape_op(lm_logits, input_shape + (-1,))
        loss = ht.crossentropy_sparse_op(ht.softmax_op(lm_logits), labels)
        return loss, lm_logits


