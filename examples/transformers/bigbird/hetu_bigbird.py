import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import logging
import numpy as np
import hetu as ht

class BigBirdEmbeddings(object):
    def __init__(self, config, name='BigBirdEmbeddings'):
        self.word_embeddings = ht.layers.Embedding(config.vocab_size, config.hidden_size, name=name+'.word_embeddings')
        self.position_embeddings = ht.layers.Embedding(config.max_position_embeddings, config.hidden_size, name=name+'.position_embeddings')
        self.token_type_embeddings = ht.layers.Embedding(config.type_vocab_size, config.hidden_size, name=name+'.token_type_embeddings')

        self.LayerNorm = ht.layers.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, name=name+'.LayerNorm')
        self.dropout = ht.layers.DropOut(config.hidden_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.position_ids = ht.array_reshape_op(ht.arange_op(0, config.max_position_embeddings), (1, -1))

        self.token_type_ids = ht.init.zeros((1, config.max_position_embeddings), trainable=False)

        self.rescale_embeddings = config.rescale_embeddings
        self.hidden_size = config.hidden_size

    def __call__(
        self, input_ids=None, input_ids_shape=None, token_type_ids=None, position_ids=None, inputs_embeds=None, inputs_embeds_shape=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids_shape
        else:
            input_shape = inputs_embeds_shape

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = ht.slice_op(self.position_ids, (0, past_key_values_length), (-1, seq_length))


        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = ht.slice_op(self.token_type_ids, (0, 0), (-1, seq_length))
                buffered_token_type_ids_expanded = ht.broadcast_shape_op(buffered_token_type_ids, (input_shape[0], seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ht.init.zeros(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.rescale_embeddings:
            inputs_embeds = inputs_embeds * (self.hidden_size**0.5)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.dropout(embeddings)
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class BigBirdSelfAttention(object):
    def __init__(self, config, name='BigBirdSelfAttention'):

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = ht.layers.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias, weight_transpose=True, name=name+'.query')
        self.key = ht.layers.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias, weight_transpose=True, name=name+'.key')
        self.value = ht.layers.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias, weight_transpose=True, name=name+'.value')

        self.dropout = ht.layers.DropOut(config.attention_probs_dropout_prob)
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x, shape):
        new_x_shape = shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = ht.array_reshape_op(x, new_x_shape)
        x = ht.transpose_op(x, (0, 2, 1, 3))
        return x

    def __call__(
        self,
        hidden_states,
        hidden_states_shape=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        hidden_states = ht.array_reshape_op(hidden_states, (-1, hidden_states_shape[-1]))
        mixed_query_layer = self.query(hidden_states)


        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = ht.concatenate_op(past_key_value[0], key_layer, axis=2)
            value_layer = ht.concatenate_op(past_key_value[1], value_layer, axis=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        key_layer = ht.transpose_op(key_layer, (0, 1, 3, 2))

        attention_scores = ht.batch_matmul_op(query_layer, key_layer)
        attention_scores = attention_scores * (1 / math.sqrt(self.attention_head_size))

        if attention_mask is not None:    
            attention_scores = attention_scores + attention_mask

        attention_probs = ht.softmax_op(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ht.batch_matmul_op(attention_probs, value_layer)

        shape = input_shape[:-1] + (self.all_head_size, )
        context_layer =  ht.transpose_op(context_layer, (0, 2, 1, 3))
        context_layer = ht.array_reshape_op(context_layer, shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BigBirdBlockSparseAttention(object):
    def __init__(self, config, seed=None, name='BigBirdBlockSparseAttention'):
        self.max_seqlen = config.max_position_embeddings
        self.seed = seed

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.num_random_blocks = config.num_random_blocks
        self.block_size = config.block_size

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = ht.layers.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias, weight_transpose=True, name=name+'.query')
        self.key = ht.layers.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias, weight_transpose=True, name=name+'.key')
        self.value = ht.layers.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias, weight_transpose=True, name=name+'.value')

    def transpose_for_scores(self, x, shape):
        new_x_shape = shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = ht.array_reshape_op(x, new_x_shape)
        x = ht.transpose_op(x, (0, 2, 1, 3))
        return x

    def __call__(
        self,
        hidden_states,
        hidden_states_shape,
        band_mask=None,
        band_mask_shape=None,
        from_mask=None,
        from_mask_shape=None,
        to_mask=None,
        to_mask_shape=None,
        from_blocked_mask=None,
        from_blocked_mask_shape=None,
        to_blocked_mask=None,
        to_blocked_mask_shape=None,
        output_attentions=None,
    ):
        batch_size, seqlen, _ = hidden_states_shape
        to_seq_length = from_seq_length = seqlen
        from_block_size = to_block_size = self.block_size

        if from_seq_length % from_block_size != 0:
            raise ValueError("Query sided sequence length must be multiple of block size")

        if to_seq_length % to_block_size != 0:
            raise ValueError("Key/Value sided sequence length must be multiple of block size")

        hidden_states = ht.array_reshape_op(hidden_states, (-1, hidden_states_shape[-1]))
        query_layer = self.transpose_for_scores(self.query(hidden_states), hidden_states_shape)
        key_layer = self.transpose_for_scores(self.key(hidden_states), hidden_states_shape)
        value_layer = self.transpose_for_scores(self.value(hidden_states), hidden_states_shape)

        layer_shape = (batch_size, self.num_attention_heads, seqlen, self.attention_head_size)
        context_layer, attention_probs = self.bigbird_block_sparse_attention(
            query_layer,
            layer_shape,
            key_layer,
            layer_shape,
            value_layer,
            layer_shape,
            band_mask,
            band_mask_shape,
            from_mask,
            from_mask_shape,
            to_mask,
            to_mask_shape,
            from_blocked_mask,
            from_blocked_mask_shape,
            to_blocked_mask,
            to_blocked_mask_shape,
            self.num_attention_heads,
            self.num_random_blocks,
            self.attention_head_size,
            from_block_size,
            to_block_size,
            batch_size,
            from_seq_length,
            to_seq_length,
            seed=self.seed,
            plan_from_length=None,
            plan_num_rand_blocks=None,
            output_attentions=output_attentions,
        )

        context_layer = ht.array_reshape_op(context_layer, (batch_size, from_seq_length, -1))
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    @staticmethod
    def torch_bmm_nd(inp_1, inp_2, inp_1_shape, inp_2_shape, ndim=None):
        inp_1 = ht.array_reshape_op(inp_1, ((-1,) + inp_1_shape[-2:]))
        inp_2 = ht.array_reshape_op(inp_2, ((-1,) + inp_2_shape[-2:]))     
        inp = ht.batch_matmul_op(inp_1, inp_2)
        inp = ht.array_reshape_op(inp, (inp_1_shape[: ndim - 2] + (inp_1_shape[ndim - 2], inp_2_shape[ndim - 1])))
        return inp

    @staticmethod
    def torch_bmm_nd_transpose(inp_1, inp_2, inp_1_shape, inp_2_shape, ndim=None):
        inp_1 = ht.array_reshape_op(inp_1, ((-1,) + inp_1_shape[-2:]))
        inp_2 = ht.array_reshape_op(inp_2, ((-1,) + inp_2_shape[-2:])) 
        inp_2 = ht.transpose_op(inp_2, (0, 2, 1))
        inp = ht.batch_matmul_op(inp_1, inp_2)
        inp = ht.array_reshape_op(inp, (inp_1_shape[: ndim - 2] + (inp_1_shape[ndim - 2], inp_2_shape[ndim - 2])))
        return inp

    def bigbird_block_sparse_attention(
        self,
        query_layer,
        query_layer_shape,
        key_layer,
        key_layer_shape,
        value_layer,
        value_layer_shape,
        band_mask,
        band_mask_shape,
        from_mask,
        from_mask_shape,
        to_mask,
        to_mask_shape,
        from_blocked_mask,
        from_blocked_mask_shape,
        to_blocked_mask,
        to_blocked_mask_shape,
        n_heads,
        n_rand_blocks,
        attention_head_size,
        from_block_size,
        to_block_size,
        batch_size,
        from_seq_len,
        to_seq_len,
        seed,
        plan_from_length,
        plan_num_rand_blocks,
        output_attentions,
    ):
        if from_seq_len // from_block_size != to_seq_len // to_block_size:
            raise ValueError("Error the number of blocks needs to be same!")

        rsqrt_d = 1 / math.sqrt(attention_head_size)
        bsz = batch_size
        attn_mask_penalty = -10000.0


        np.random.seed(seed)
        if from_seq_len in [1024, 3072, 4096]:  
            rand_attn = [
                self._bigbird_block_rand_mask(
                    self.max_seqlen, self.max_seqlen, from_block_size, to_block_size, n_rand_blocks, last_idx=1024
                )[: (from_seq_len // from_block_size - 2)]
                for _ in range(n_heads)
            ]
        else:
            if plan_from_length is None:
                plan_from_length, plan_num_rand_blocks = self._get_rand_attn_plan(
                    from_seq_len, from_block_size, n_rand_blocks
                )

            rand_attn = self._bigbird_block_rand_mask_with_head(
                from_seq_length=from_seq_len,
                to_seq_length=to_seq_len,
                from_block_size=from_block_size,
                to_block_size=to_block_size,
                num_heads=n_heads,
                plan_from_length=plan_from_length,
                plan_num_rand_blocks=plan_num_rand_blocks,
            )

        rand_attn = np.stack(rand_attn, axis=0)
        rand_attn_shape = rand_attn.shape
        rand_attn = ht.Variable(name='rand_attn', value=rand_attn, trainable=False)
        rand_attn = ht.unsqueeze_op(rand_attn, 0)
        rand_attn = ht.concatenate_op([rand_attn for _ in range(batch_size)], axis=0)

        rand_mask = self._create_rand_mask_from_inputs(
            from_blocked_mask, from_blocked_mask_shape, to_blocked_mask, to_blocked_mask_shape, rand_attn, (bsz, )+rand_attn_shape, n_heads, n_rand_blocks, bsz, from_seq_len, from_block_size
        )

        blocked_query_matrix = ht.array_reshape_op(query_layer, (bsz, n_heads, from_seq_len // from_block_size, from_block_size, -1))
        blocked_key_matrix = ht.array_reshape_op(key_layer, (bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1))
        blocked_value_matrix = ht.array_reshape_op(value_layer, (bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1))

        blocked_key_matrix_shape = (bsz, n_heads, to_seq_len // to_block_size, to_block_size, self.attention_head_size)
        rand_attn_shape = (bsz, ) + rand_attn_shape
        gathered_key = self.hetu_gather_b2(blocked_key_matrix, rand_attn, blocked_key_matrix_shape, rand_attn_shape)
        gathered_key = ht.array_reshape_op(gathered_key, (
            bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1
        ))
        gathered_value = self.hetu_gather_b2(blocked_value_matrix, rand_attn, blocked_key_matrix_shape, rand_attn_shape)
        gathered_value = ht.array_reshape_op(gathered_value, (
            bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1
        ))
        
        first_product_query = ht.slice_op(blocked_query_matrix, (0, 0, 0, 0, 0), (-1, -1, 1, -1, -1))
        first_product_query_shape = (bsz, n_heads, from_block_size, self.attention_head_size)        
        first_product_query = ht.array_reshape_op(first_product_query, first_product_query_shape)
        first_product = self.torch_bmm_nd_transpose(first_product_query, key_layer, first_product_query_shape, key_layer_shape, ndim=4)

        first_product = first_product * rsqrt_d
        first_product += (1.0 - to_mask) * attn_mask_penalty
        first_attn_weights = ht.softmax_op(first_product)
        first_attn_weights_shape = (bsz, n_heads, from_block_size, to_seq_len)

        first_context_layer = self.torch_bmm_nd(first_attn_weights, value_layer, first_attn_weights_shape, value_layer_shape, ndim=4)
        first_context_layer = ht.unsqueeze_op(first_context_layer, 2)

        second_key_mat = ht.concatenate_op(
            [
                ht.slice_op(blocked_key_matrix, (0, 0, 0, 0, 0), (-1, -1, 1, -1, -1)),
                ht.slice_op(blocked_key_matrix, (0, 0, 1, 0, 0), (-1, -1, 1, -1, -1)),
                ht.slice_op(blocked_key_matrix, (0, 0, 2, 0, 0), (-1, -1, 1, -1, -1)),
                ht.slice_op(blocked_key_matrix, (0, 0, -1, 0, 0), (-1, -1, 1, -1, -1)),
                ht.slice_op(gathered_key, (0, 0, 0, 0, 0), (-1, -1, 1, -1, -1)),
            ],
            axis=3,
        ) 

        second_mat_shape = (bsz, n_heads, (4+n_rand_blocks)*to_block_size, self.attention_head_size)
        second_key_mat = ht.array_reshape_op(second_key_mat, second_mat_shape)
        second_value_mat = ht.concatenate_op(
            [
                ht.slice_op(blocked_value_matrix, (0, 0, 0, 0, 0), (-1, -1, 1, -1, -1)),
                ht.slice_op(blocked_value_matrix, (0, 0, 1, 0, 0), (-1, -1, 1, -1, -1)),
                ht.slice_op(blocked_value_matrix, (0, 0, 2, 0, 0), (-1, -1, 1, -1, -1)),
                ht.slice_op(blocked_value_matrix, (0, 0, -1, 0, 0), (-1, -1, 1, -1, -1)),
                ht.slice_op(gathered_value, (0, 0, 0, 0, 0), (-1, -1, 1, -1, -1)),
            ],
            axis=3,
        )  
        second_value_mat = ht.array_reshape_op(second_value_mat, second_mat_shape)

        second_product_query = ht.slice_op(blocked_query_matrix, (0, 0, 1, 0, 0), (-1, -1, 1, -1, -1))
        second_product_query_shape = (bsz, n_heads, from_block_size, self.attention_head_size)        
        second_product_query = ht.array_reshape_op(second_product_query, second_product_query_shape)
        second_product = self.torch_bmm_nd_transpose(second_product_query, second_key_mat, second_product_query_shape, second_mat_shape, ndim=4)
        
        second_seq_pad = ht.concatenate_op(
            [
                ht.slice_op(to_mask, (0, 0, 0, 0), (-1, -1, -1, 3 * to_block_size)),
                ht.slice_op(to_mask, (0, 0, 0, -to_block_size), (-1, -1, -1, -1)),             
                ht.init.zeros([bsz, 1, 1, n_rand_blocks * to_block_size], trainable=False),
            ],
            axis=3,
        )
        
        second_rand_pad = ht.concatenate_op(
            [
                ht.init.zeros([bsz, n_heads, from_block_size, 4 * to_block_size], trainable=False),
                ht.array_reshape_op(ht.slice_op(rand_mask, (0, 0, 0, 0, 0), (-1, -1, 1, -1, -1)), [bsz, n_heads, from_block_size, -1])     
            ],
            axis=3,
        )
        second_product = second_product * rsqrt_d
        second_product += ht.minus_byconst_op(ht.min_op(second_seq_pad, second_rand_pad), 1.0) * attn_mask_penalty
        second_attn_weights = ht.softmax_op(second_product)
        second_attn_weights_shape = (bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size)

        second_context_layer = self.torch_bmm_nd(second_attn_weights, second_value_mat, second_attn_weights_shape, second_mat_shape, ndim=4)
        second_context_layer = ht.unsqueeze_op(second_context_layer, 2)

        exp_blocked_key_matrix = ht.concatenate_op(
            [
                ht.slice_op(blocked_key_matrix, (0, 0, 1, 0, 0), (-1, -1, to_seq_len // to_block_size - 4, -1, -1)),
                ht.slice_op(blocked_key_matrix, (0, 0, 2, 0, 0), (-1, -1, to_seq_len // to_block_size - 4, -1, -1)),
                ht.slice_op(blocked_key_matrix, (0, 0, 3, 0, 0), (-1, -1, to_seq_len // to_block_size - 4, -1, -1)),
            ],
            axis=3,
        ) 

        exp_blocked_value_matrix = ht.concatenate_op(
            [
                ht.slice_op(blocked_value_matrix, (0, 0, 1, 0, 0), (-1, -1, to_seq_len // to_block_size - 4, -1, -1)),
                ht.slice_op(blocked_value_matrix, (0, 0, 2, 0, 0), (-1, -1, to_seq_len // to_block_size - 4, -1, -1)),
                ht.slice_op(blocked_value_matrix, (0, 0, 3, 0, 0), (-1, -1, to_seq_len // to_block_size - 4, -1, -1)),
            ],
            axis=3,
        ) 
        middle_query_matrix = ht.slice_op(blocked_query_matrix, (0, 0, 2, 0, 0), (-1, -1, from_seq_len // from_block_size - 4, -1, -1))
        
        middle_query_matrix_shape = (bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, self.attention_head_size)
        exp_blocked_key_matrix_shape = (bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, self.attention_head_size)
        inner_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, exp_blocked_key_matrix, middle_query_matrix_shape, exp_blocked_key_matrix_shape, ndim=5)
        inner_band_product = inner_band_product * rsqrt_d

        gathered_key_shape = (bsz, n_heads, from_seq_len//from_block_size-4, n_rand_blocks*to_block_size, self.attention_head_size)
        rand_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, 
                                                        ht.slice_op(gathered_key, (0, 0, 1, 0, 0), (-1, -1, to_seq_len // to_block_size - 4, -1, -1)), 
                                                        middle_query_matrix_shape,    
                                                        gathered_key_shape,                       
                                                        ndim=5)
        rand_band_product = rand_band_product * rsqrt_d
        
        #meidui
        blocked_key_matrix_first = ht.slice_op(blocked_key_matrix, (0, 0, 0, 0, 0), (-1, -1, 1, -1, -1))
        blocked_key_matrix_first = ht.array_reshape_op(blocked_key_matrix_first, (bsz, n_heads, to_block_size, -1))
        blocked_key_matrix_first = ht.transpose_op(blocked_key_matrix_first, (0, 1, 3, 2))        
        middle_query_matrix = ht.array_reshape_op(middle_query_matrix, (bsz, n_heads, (from_seq_len//from_block_size-4) * from_block_size, -1))
        first_band_product = ht.batch_matmul_op(middle_query_matrix, blocked_key_matrix_first)
        first_band_product = ht.array_reshape_op(first_band_product, (bsz, n_heads, from_seq_len//from_block_size-4 , from_block_size, -1))
        first_band_product = first_band_product * rsqrt_d

        blocked_key_matrix_last = ht.slice_op(blocked_key_matrix, (0, 0, -1, 0, 0), (-1, -1, 1, -1, -1))
        blocked_key_matrix_last = ht.array_reshape_op(blocked_key_matrix_last, (bsz, n_heads, to_block_size, -1))
        blocked_key_matrix_last = ht.transpose_op(blocked_key_matrix_last, (0, 1, 3, 2))      
        last_band_product = ht.batch_matmul_op(middle_query_matrix, blocked_key_matrix_last)
        last_band_product = ht.array_reshape_op(last_band_product, (bsz, n_heads, from_seq_len//from_block_size-4 , from_block_size, -1))
        last_band_product = last_band_product * rsqrt_d

        inner_band_product += (1.0 - band_mask) * attn_mask_penalty
        to_mask_first = ht.slice_op(to_mask, (0, 0, 0, 0), (-1, -1, -1, to_block_size))
        to_mask_first = ht.unsqueeze_op(to_mask_first, 3)
        first_band_product += (1.0 - to_mask_first) * attn_mask_penalty

        to_mask_last = ht.slice_op(to_mask, (0, 0, 0, -to_block_size), (-1, -1, -1, -1))
        to_mask_last = ht.unsqueeze_op(to_mask_last, 3)
        last_band_product += (1.0 - to_mask_last) * attn_mask_penalty

        rand_mask_product = ht.slice_op(rand_mask, (0, 0, 1, 0, 0), (-1, -1, from_blocked_mask_shape[1]-4, -1, -1))
        rand_band_product += (1.0 - rand_mask_product) * attn_mask_penalty

        band_product = ht.concatenate_op(
            [first_band_product, inner_band_product, rand_band_product, last_band_product], axis=-1
        ) 

        attn_weights = ht.softmax_op(band_product)

        attn_weights_0 = ht.slice_op(attn_weights, (0, 0, 0, 0, to_block_size), (-1, -1, -1, -1, 3*to_block_size))
        context_layer = self.torch_bmm_nd(attn_weights_0, 
                                          exp_blocked_value_matrix, 
                                          (bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, 3*to_block_size),
                                          (bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, self.attention_head_size),
                                          ndim=5)    

        attn_weights_1 = ht.slice_op(attn_weights, (0, 0, 0, 0, 4 * to_block_size), (-1, -1, -1, -1, n_rand_blocks*to_block_size))
        gathered_value_1 = ht.slice_op(gathered_value, (0, 0, 1, 0, 0), (-1, -1, from_seq_len//from_block_size-4, -1, -1))
        context_layer += self.torch_bmm_nd(attn_weights_1, 
                                           gathered_value_1, 
                                           (bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, n_rand_blocks*to_block_size),
                                           (bsz, n_heads, from_seq_len//from_block_size-4, n_rand_blocks*to_block_size, self.attention_head_size),
                                           ndim=5)         
        
        #mei dui
        attn_weights_2 = ht.slice_op(attn_weights, (0, 0, 0, 0, 0), (-1, -1, -1, -1, to_block_size))
        attn_weights_2 = ht.array_reshape_op(attn_weights_2, (bsz, n_heads, (from_seq_len//from_block_size-4) * from_block_size, -1))
        blocked_value_matrix_2 = ht.slice_op(blocked_value_matrix, (0, 0, 0, 0, 0), (-1, -1, 1, -1, -1))
        blocked_value_matrix_2 = ht.array_reshape_op(blocked_value_matrix_2, (bsz, n_heads, to_block_size, -1))
        context_layer_2 = ht.batch_matmul_op(attn_weights_2, blocked_value_matrix_2)
        context_layer_2 = ht.array_reshape_op(context_layer_2, (bsz, n_heads, from_seq_len//from_block_size-4 , from_block_size, -1))
        context_layer += context_layer_2

        attn_weights_3 = ht.slice_op(attn_weights, (0, 0, 0, 0, -to_block_size), (-1, -1, -1, -1, -1))
        attn_weights_3 = ht.array_reshape_op(attn_weights_3, (bsz, n_heads, (from_seq_len//from_block_size-4) * from_block_size, -1))
        blocked_value_matrix_3 = ht.slice_op(blocked_value_matrix, (0, 0, -1, 0, 0), (-1, -1, 1, -1, -1))
        blocked_value_matrix_3 = ht.array_reshape_op(blocked_value_matrix_3, (bsz, n_heads, to_block_size, -1))
        context_layer_3 = ht.batch_matmul_op(attn_weights_3, blocked_value_matrix_3)
        context_layer_3 = ht.array_reshape_op(context_layer_3, (bsz, n_heads, from_seq_len//from_block_size-4 , from_block_size, -1))
        context_layer += context_layer_3


        second_last_key_mat = ht.concatenate_op(
            [
                ht.slice_op(blocked_key_matrix, (0, 0, 0, 0, 0), (-1, -1, 1, -1, -1)),
                ht.slice_op(blocked_key_matrix, (0, 0, -3, 0, 0), (-1, -1, 1, -1, -1)),
                ht.slice_op(blocked_key_matrix, (0, 0, -2, 0, 0), (-1, -1, 1, -1, -1)),
                ht.slice_op(blocked_key_matrix, (0, 0, -1, 0, 0), (-1, -1, 1, -1, -1)),
                ht.slice_op(gathered_key, (0, 0, -1, 0, 0), (-1, -1, 1, -1, -1)),          
            ],
            axis=3,
        ) 
        second_last_key_mat = ht.array_reshape_op(second_last_key_mat, (bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1))

        second_last_value_mat = ht.concatenate_op(
            [
                ht.slice_op(blocked_value_matrix, (0, 0, 0, 0, 0), (-1, -1, 1, -1, -1)),
                ht.slice_op(blocked_value_matrix, (0, 0, -3, 0, 0), (-1, -1, 1, -1, -1)),
                ht.slice_op(blocked_value_matrix, (0, 0, -2, 0, 0), (-1, -1, 1, -1, -1)),
                ht.slice_op(blocked_value_matrix, (0, 0, -1, 0, 0), (-1, -1, 1, -1, -1)),
                ht.slice_op(gathered_value, (0, 0, -1, 0, 0), (-1, -1, 1, -1, -1)),        
            ],
            axis=3,
        ) 
        second_last_value_mat = ht.array_reshape_op(second_last_value_mat, (bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1))



        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]

        blocked_query_matrix_second_last = ht.slice_op(blocked_query_matrix, (0, 0, -2, 0, 0), (-1, -1, 1, -1, -1))
        blocked_query_matrix_second_last = ht.array_reshape_op(blocked_query_matrix_second_last, (bsz, n_heads, from_block_size, -1))
        second_last_product = self.torch_bmm_nd_transpose(blocked_query_matrix_second_last, 
                                                          second_last_key_mat,
                                                          (bsz, n_heads, from_block_size, self.attention_head_size),
                                                          (bsz, n_heads, (4+n_rand_blocks)*to_block_size, self.attention_head_size), 
                                                          ndim=4)

        second_last_seq_pad = ht.concatenate_op(
            [
                ht.slice_op(to_mask, (0, 0, 0, 0), (-1, -1, -1, to_block_size)),
                ht.slice_op(to_mask, (0, 0, 0, -3 * to_block_size), (-1, -1, -1, -1)),
                ht.init.ones((bsz, 1, 1, n_rand_blocks * to_block_size), trainable=False)
            ],
            axis=3,
        )         

        second_last_rand_pad = ht.concatenate_op(
            [
                ht.init.ones((bsz, n_heads, from_block_size, 4 * to_block_size), trainable=False),
                ht.array_reshape_op(ht.slice_op(rand_mask, (0, 0, -1, 0, 0), (-1, -1, 1, -1, -1)), (bsz, n_heads, from_block_size, n_rand_blocks*to_block_size))  
            ],
            axis=3,
        )         

        second_last_product = second_last_product * rsqrt_d
        second_last_product += (1.0 - ht.min_op(second_last_seq_pad, second_last_rand_pad)) * attn_mask_penalty
        second_last_attn_weights = ht.softmax_op(second_last_product)

        second_last_context_layer = self.torch_bmm_nd(second_last_attn_weights, 
                                                      second_last_value_mat, 
                                                      (bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size),
                                                      (bsz, n_heads, (4+n_rand_blocks)*to_block_size, self.attention_head_size),
                                                      ndim=4)
        second_last_context_layer = ht.unsqueeze_op(second_last_context_layer, 2)

        blocked_query_matrix_last = ht.slice_op(blocked_query_matrix, (0, 0, -1, 0, 0), (-1, -1, 1, -1, -1))
        blocked_query_matrix_last = ht.array_reshape_op(blocked_query_matrix_last, (bsz, n_heads, from_block_size, -1))
        last_product = self.torch_bmm_nd_transpose(blocked_query_matrix_last, 
                                                   key_layer, 
                                                   (bsz, n_heads, from_block_size, self.attention_head_size),
                                                   (bsz, n_heads, to_seq_len, self.attention_head_size),
                                                   ndim=4)

        last_product = last_product * rsqrt_d
        last_product += (1.0 - to_mask) * attn_mask_penalty
        last_attn_weights = ht.softmax_op(last_product)

        last_context_layer = self.torch_bmm_nd(last_attn_weights, 
                                               value_layer, 
                                               (bsz, n_heads, from_block_size, to_seq_len),
                                               (bsz, n_heads, to_seq_len, self.attention_head_size),
                                               ndim=4)
        last_context_layer = ht.unsqueeze_op(last_context_layer, 2)


        context_layer = ht.concatenate_op(
            [first_context_layer, second_context_layer, context_layer, second_last_context_layer, last_context_layer],
            axis=2,
        )
        context_layer = ht.array_reshape_op(context_layer, (bsz, n_heads, from_seq_len, -1))
        from_mask = ht.broadcastto_op(from_mask, context_layer)
        context_layer = context_layer * from_mask
        context_layer = ht.transpose_op(context_layer, (0, 2, 1, 3))

        attention_probs = None

        return context_layer, attention_probs

    @staticmethod
    def hetu_gather_b2(params, indices, params_shape, indices_shape):

        if params_shape[:2] != indices_shape[:2]:
            raise ValueError(
                "Make sure that the first two dimensions of params and indices are identical,                 but"
                f" they are params: {params_shape[:2]} vs. indices: {indices_shape[:2]}"
            )
        num_indices_to_gather = indices_shape[-2] * indices_shape[-1]
        num_indices_to_pick_from = params_shape[2]

        indices_shift = ht.arange_op(0, indices_shape[0] * indices_shape[1] * num_indices_to_gather)
        indices_shift /= num_indices_to_gather
        indices_shift = ht.floor_op(indices_shift)
        indices_shift *= num_indices_to_pick_from

        flattened_indices = ht.array_reshape_op(indices, (-1,)) + indices_shift
        flattened_params = ht.array_reshape_op(params, (-1, params_shape[-2], params_shape[-1]))

        out_flattened = ht.index_select_op(flattened_params, flattened_indices, 0)
        out = ht.array_reshape_op(out_flattened, (params_shape[:2] + (num_indices_to_gather,) + params_shape[3:]))
        return out

    @staticmethod
    def _create_rand_mask_from_inputs(
            from_blocked_mask,
            from_blocked_mask_shape,
            to_blocked_mask,
            to_blocked_mask_shape,
            rand_attn,
            rand_attn_shape,
            num_attention_heads,
            num_rand_blocks,
            batch_size,
            from_seq_length,
            from_block_size,
        ):
            num_windows = from_seq_length // from_block_size - 2

            rand_mask = []
            for i in range(rand_attn_shape[0]):
                p1 = ht.slice_op(to_blocked_mask, (i, 0, 0), (1, -1, -1))
                p1 = ht.array_reshape_op(p1, to_blocked_mask_shape[1:])
                i1 = ht.slice_op(rand_attn, (i, 0, 0, 0), (1, -1, -1, -1))
                p1 = ht.index_select_op(p1, ht.array_reshape_op(i1, (-1, )), 0)
                rand_mask.append(p1)
            rand_mask = ht.concatenate_op(rand_mask, axis=0)
            rand_mask = ht.array_reshape_op(rand_mask, (batch_size, num_attention_heads, num_windows, num_rand_blocks * from_block_size))
            
            from_blocked_mask = ht.slice_op(from_blocked_mask, (0, 1, 0), (-1, from_blocked_mask_shape[1] - 2, -1))
            from_blocked_mask = ht.broadcast_shape_op(from_blocked_mask, (-1, num_attention_heads, -1, -1, num_rand_blocks * from_block_size), add_axes=(1, 4))
            rand_mask = ht.broadcast_shape_op(rand_mask, (-1, -1, -1, from_blocked_mask_shape[2], -1), add_axes=(3, ))
            rand_mask = from_blocked_mask * rand_mask
            return rand_mask

    @staticmethod
    def _get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
        plan_from_length = []
        plan_num_rand_blocks = []
        if (2 * num_rand_blocks + 5) < (from_seq_length // from_block_size):
            plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(0)
        elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
            plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks // 2)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks // 2))
        else:
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks)

        return plan_from_length, plan_num_rand_blocks

    @staticmethod
    def _bigbird_block_rand_mask(
        from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1
    ):
        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError("Error the number of blocks needs to be same!")

        rand_attn = np.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
        middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
        last = to_seq_length // to_block_size - 1
        if last_idx > (2 * to_block_size):
            last = (last_idx // to_block_size) - 1

        r = num_rand_blocks 
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            if i == 1:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
            elif i == 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
            elif i == from_seq_length // from_block_size - 3:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            elif i == from_seq_length // from_block_size - 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            else:
                if start > last:
                    start = last
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                elif (end + 1) == last:
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                else:
                    rand_attn[i - 1, :] = np.random.permutation(
                        np.concatenate((middle_seq[:start], middle_seq[end + 1 : last]))
                    )[:r]
        return rand_attn

    def _bigbird_block_rand_mask_with_head(
        self,
        from_seq_length,
        to_seq_length,
        from_block_size,
        to_block_size,
        num_heads,
        plan_from_length,
        plan_num_rand_blocks,
        window_block_left=1,
        window_block_right=1,
        global_block_top=1,
        global_block_bottom=1,
        global_block_left=1,
        global_block_right=1,
    ):
        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError("Error the number of blocks needs to be same!")

        if from_seq_length not in plan_from_length:
            raise ValueError("Error from sequence length not in plan!")

        num_blocks = from_seq_length // from_block_size
        plan_block_length = np.array(plan_from_length) // from_block_size
        max_plan_idx = plan_from_length.index(from_seq_length)
        rand_attn = [
            np.zeros((num_blocks, np.sum(plan_num_rand_blocks[: max_plan_idx + 1])), dtype=np.int32)
            for i in range(num_heads)
        ]


        for plan_idx in range(max_plan_idx + 1):
            rnd_r_cnt = 0
            if plan_idx > 0:
                if plan_num_rand_blocks[plan_idx] > 0:
                    rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                    curr_r_cnt = int(np.sum(plan_num_rand_blocks[: plan_idx + 1]))
                    for blk_rw_idx in range(global_block_top, plan_block_length[plan_idx - 1]):
                        for h in range(num_heads):
                            rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(
                                block_id=blk_rw_idx,
                                to_start_block_id=plan_block_length[plan_idx - 1],
                                to_end_block_id=plan_block_length[plan_idx],
                                num_rand_blocks=plan_num_rand_blocks[plan_idx],
                                window_block_left=window_block_left,
                                window_block_right=window_block_right,
                                global_block_left=global_block_left,
                                global_block_right=global_block_right,
                            )

                for pl_id in range(plan_idx):
                    if plan_num_rand_blocks[pl_id] == 0:
                        continue
                    for blk_rw_idx in range(plan_block_length[plan_idx - 1], plan_block_length[plan_idx]):
                        rnd_r_cnt = 0
                        to_start_block_id = 0
                        if pl_id > 0:
                            rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id]))
                            to_start_block_id = plan_block_length[pl_id - 1]
                        curr_r_cnt = int(np.sum(plan_num_rand_blocks[: pl_id + 1]))
                        for h in range(num_heads):
                            rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(
                                block_id=blk_rw_idx,
                                to_start_block_id=to_start_block_id,
                                to_end_block_id=plan_block_length[pl_id],
                                num_rand_blocks=plan_num_rand_blocks[pl_id],
                                window_block_left=window_block_left,
                                window_block_right=window_block_right,
                                global_block_left=global_block_left,
                                global_block_right=global_block_right,
                            )

            if plan_num_rand_blocks[plan_idx] == 0:
                continue
            curr_r_cnt = int(np.sum(plan_num_rand_blocks[: plan_idx + 1]))
            from_start_block_id = global_block_top
            to_start_block_id = 0
            if plan_idx > 0:
                rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                from_start_block_id = plan_block_length[plan_idx - 1]
                to_start_block_id = plan_block_length[plan_idx - 1]

            for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
                for h in range(num_heads):
                    rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(
                        block_id=blk_rw_idx,
                        to_start_block_id=to_start_block_id,
                        to_end_block_id=plan_block_length[plan_idx],
                        num_rand_blocks=plan_num_rand_blocks[plan_idx],
                        window_block_left=window_block_left,
                        window_block_right=window_block_right,
                        global_block_left=global_block_left,
                        global_block_right=global_block_right,
                    )

        for nh in range(num_heads):
            rand_attn[nh] = rand_attn[nh][global_block_top : num_blocks - global_block_bottom, :]

        return rand_attn

    @staticmethod
    def _get_single_block_row_attention(
        block_id,
        to_start_block_id,
        to_end_block_id,
        num_rand_blocks,
        window_block_left=1,
        window_block_right=1,
        global_block_left=1,
        global_block_right=1,
    ):
        to_block_list = np.arange(to_start_block_id, to_end_block_id, dtype=np.int32)
        perm_block = np.random.permutation(to_block_list)

        illegal_blocks = list(range(block_id - window_block_left, block_id + window_block_right + 1))

        illegal_blocks.extend(list(range(global_block_left)))
        illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))

        if block_id == 1:
            illegal_blocks.append(to_end_block_id - 2)

        if block_id == to_end_block_id - 2:
            illegal_blocks.append(1)

        selected_random_blokcs = []

        for i in range(to_end_block_id - to_start_block_id):
            if perm_block[i] not in illegal_blocks:
                selected_random_blokcs.append(perm_block[i])
            if len(selected_random_blokcs) == num_rand_blocks:
                break
        return np.array(selected_random_blokcs, dtype=np.int32)


class BigBirdSelfOutput(object):
    def __init__(self, config, name='BigBirdSelfOutput'):
        self.hidden_size = config.hidden_size
        self.dense = ht.layers.Linear(config.hidden_size, config.hidden_size, weight_transpose=True, name=name+'.dense')
        self.LayerNorm = ht.layers.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, name=name+'.LayerNorm')
        self.dropout = ht.layers.DropOut(config.hidden_dropout_prob)

    def __call__(self, hidden_states, hidden_states_shape, input_tensor):
        hidden_states = ht.array_reshape_op(hidden_states, [-1, hidden_states_shape[-1]])
        hidden_states = self.dense(hidden_states)
        hidden_states = ht.array_reshape_op(hidden_states, hidden_states_shape[:-1] + (self.hidden_size, ))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)       
        return hidden_states


class BigBirdAttention(object):
    def __init__(self, config, seed=None, name='BigBirdAttention'):
        self.name = name
        self.attention_type = config.attention_type
        self.config = config
        self.seed = seed

        if self.config.attention_type == "original_full":
            self.self = BigBirdSelfAttention(config, name=name+'.self')
        elif self.config.attention_type == "block_sparse":
            self.self = BigBirdBlockSparseAttention(config, seed, name=name+'.self')
        else:
            raise ValueError(
                f"attention_type can either be original_full or block_sparse, but is {self.config.attention_type}"
            )

        self.output = BigBirdSelfOutput(config, name=name+'.output')

    def set_attention_type(self, value: str):
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        if value == self.attention_type:
            return

        self.attention_type = value
        if value == "original_full":
            attn_weights = BigBirdSelfAttention(self.config, name=self.name+'.self')
        else:
            attn_weights = BigBirdBlockSparseAttention(self.config, self.seed, name=self.name+'.self')

        attn_weights.query = self.self.query
        attn_weights.value = self.self.value
        attn_weights.key = self.self.key
        self.self = attn_weights
        self.attention_type = value

    def __call__(
        self,
        hidden_states,
        hidden_states_shape,
        attention_mask=None,
        attention_mask_shape=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        # block_sparse config
        band_mask=None,
        band_mask_shape=None,
        from_mask=None,
        from_mask_shape=None,
        to_mask=None,
        to_mask_shape=None,
        from_blocked_mask=None,
        from_blocked_mask_shape=None,
        to_blocked_mask=None,
        to_blocked_mask_shape=None,
    ):

        if self.attention_type == "original_full":
            self_outputs = (
                hidden_states,
                hidden_states_shape,
                attention_mask,
                attention_mask_shape,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            
        else:
            if encoder_hidden_states is not None:
                raise ValueError("BigBird cannot be used as a decoder when config.attention_type != 'original_full'")
            self_outputs = self.self(
                hidden_states, hidden_states_shape, band_mask, band_mask_shape, from_mask, from_mask_shape, to_mask, to_mask_shape, from_blocked_mask, from_blocked_mask_shape, to_blocked_mask, to_blocked_mask_shape, output_attentions
            )

        attention_output = self.output(self_outputs[0], hidden_states_shape, hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BigBirdIntermediate(object):
    def __init__(self, config, name='BigBirdIntermediate'):
        self.intermediate_size = config.intermediate_size
        self.dense = ht.layers.Linear(config.hidden_size, config.intermediate_size, weight_transpose=True, name=name+'.dense')
        if config.hidden_act == "relu":
            self.intermediate_act_fn = ht.relu_op
        elif config.hidden_act == "gelu":
            self.intermediate_act_fn = ht.gelu_op

    def __call__(self, hidden_states, hidden_states_shape):
        hidden_states = ht.array_reshape_op(hidden_states, [-1, hidden_states_shape[-1]])
        hidden_states = self.dense(hidden_states)
        hidden_states = ht.array_reshape_op(hidden_states, hidden_states_shape[:-1] + (self.intermediate_size, ))       
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BigBirdOutput(object):
    def __init__(self, config, name='BigBirdOutput'):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dense = ht.layers.Linear(config.intermediate_size, config.hidden_size, weight_transpose=True, name=name+'.dense')
        self.LayerNorm = ht.layers.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, name=name+'.LayerNorm')
        self.dropout = ht.layers.DropOut(config.hidden_dropout_prob)

    def __call__(self, hidden_states, hidden_states_shape, input_tensor):
        hidden_states = ht.array_reshape_op(hidden_states, [-1, hidden_states_shape[-1]])
        hidden_states = self.dense(hidden_states)
        hidden_states = ht.array_reshape_op(hidden_states, hidden_states_shape[:-1] + (self.hidden_size, ))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BigBirdLayer(object):
    def __init__(self, config, seed=None, name='BigBirdLayer'):
        self.config = config
        self.attention_type = config.attention_type
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.intermediate_size = config.intermediate_size
        self.attention = BigBirdAttention(config, seed=seed, name=name+'.attention')
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise TypeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BigBirdAttention(config, name=name+'.crossattention')
        self.intermediate = BigBirdIntermediate(config, name=name+'.intermediate')
        self.output = BigBirdOutput(config, name=name+'.output')

    def set_attention_type(self, value: str):
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        if value == self.attention_type:
            return
        self.attention_type = value
        self.attention.set_attention_type(value)

        if self.add_cross_attention:
            self.crossattention.set_attention_type(value)

    def __call__(
        self,
        hidden_states,
        hidden_states_shape,
        attention_mask=None,
        attention_mask_shape=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        band_mask=None,
        band_mask_shape=None,
        from_mask=None,
        from_mask_shape=None,
        to_mask=None,
        to_mask_shape=None,
        blocked_encoder_mask=None,
        blocked_encoder_mask_shape=None,
        past_key_value=None,
        output_attentions=False,
    ):

        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            hidden_states_shape,
            attention_mask,
            attention_mask_shape,
            head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
            band_mask=band_mask,
            band_mask_shape=band_mask_shape,
            from_mask=from_mask,
            from_mask_shape=from_mask_shape,
            to_mask=to_mask,
            to_mask_shape=to_mask_shape,
            from_blocked_mask=blocked_encoder_mask,
            from_blocked_mask_shape=blocked_encoder_mask_shape,
            to_blocked_mask=blocked_encoder_mask,
            to_blocked_mask_shape=blocked_encoder_mask_shape,
        )
        attention_output = self_attention_outputs[0]

        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with                    "
                    " cross-attention layers by setting `config.add_cross_attention=True`"
                )

            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                hidden_states_shape,
                attention_mask,
                attention_mask_shape,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1] 

            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = self.feed_forward_chunk(attention_output, hidden_states_shape)

        outputs = (layer_output,) + outputs

        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output, attention_output_shape):
        intermediate_output = self.intermediate(attention_output, attention_output_shape)
        layer_output = self.output(intermediate_output, attention_output_shape[:-1] + (self.intermediate_size, ), attention_output)
        return layer_output


class BigBirdEncoder(object):
    def __init__(self, config, name='BigBirdEncoder'):
        self.config = config
        self.attention_type = config.attention_type
        self.layer = [BigBirdLayer(config, seed=layer_idx, name=name+'.layer.'+str(layer_idx)) for layer_idx in range(config.num_hidden_layers)]

    def set_attention_type(self, value: str):
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )

        if value == self.attention_type:
            return
        self.attention_type = value
        for layer in self.layer:
            layer.set_attention_type(value)

    def __call__(
        self,
        hidden_states,
        hidden_states_shape,
        attention_mask=None,
        attention_mask_shape=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        band_mask=None,
        band_mask_shape=None,
        from_mask=None,
        from_mask_shape=None,
        to_mask=None,
        to_mask_shape=None,
        blocked_encoder_mask=None,
        blocked_encoder_mask_shape=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                hidden_states_shape,
                attention_mask,
                attention_mask_shape,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                band_mask,
                band_mask_shape,
                from_mask,
                from_mask_shape,
                to_mask,
                to_mask_shape,
                blocked_encoder_mask,
                blocked_encoder_mask_shape,
                past_key_value,
                output_attentions,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )


class BigBirdModel(object):
    def __init__(self, config, add_pooling_layer=True):
        self.config = config
        self.attention_type = self.config.attention_type

        self.block_size = self.config.block_size

        self.embeddings = BigBirdEmbeddings(config, name='embeddings')
        self.encoder = BigBirdEncoder(config, name='encoder')

        if add_pooling_layer:
            self.pooler = ht.layers.Linear(config.hidden_size, config.hidden_size, weight_transpose=True, name='pooler')
            self.activation = ht.tanh_op
        else:
            self.pooler = None
            self.activation = None

        if self.attention_type != "original_full" and config.add_cross_attention:
            logging.warning(
                "When using `BigBirdForCausalLM` as decoder, then `attention_type` must be `original_full`. Setting"
                " `attention_type=original_full`"
            )
            self.set_attention_type("original_full")


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def set_attention_type(self, value: str):
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )

        if value == self.attention_type:
            return
        self.attention_type = value
        self.encoder.set_attention_type(value)

    def __call__(
        self,
        input_ids=None,
        input_ids_shape=None,
        attention_mask=None,
        attention_mask_shape=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        inputs_embeds_shape=None,
        encoder_hidden_states=None,
        encoder_hidden_states_shape=None,
        encoder_attention_mask=None,
        encoder_attention_mask_shape=None,
        past_key_values=None,
        past_key_values_shape=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds_shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        past_key_values_length = past_key_values_shape[0][0][2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask_shape = (batch_size, seq_length + past_key_values_length)
            attention_mask = ht.init.ones(attention_mask_shape, trainable=False)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = ht.slice_op(self.embeddings.token_type_ids, (0, 0), (-1, seq_length))
                buffered_token_type_ids_expanded = ht.broadcast_shape_op(buffered_token_type_ids, (batch_size, seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ht.init.zeros(input_shape, trainable=False)

        max_tokens_to_attend = (5 + 2 * self.config.num_random_blocks) * self.config.block_size
        if self.attention_type == "block_sparse" and seq_length <= max_tokens_to_attend:
            sequence_length = input_ids_shape[1] if input_ids is not None else inputs_embeds_shape[1]
            logging.warning(
                "Attention type 'block_sparse' is not possible if sequence_length: "
                f"{sequence_length} <= num global tokens: 2 * config.block_size "
                "+ min. num sliding tokens: 3 * config.block_size "
                "+ config.num_random_blocks * config.block_size "
                "+ additional buffer: config.num_random_blocks * config.block_size "
                f"= {max_tokens_to_attend} with config.block_size "
                f"= {self.config.block_size}, config.num_random_blocks "
                f"= {self.config.num_random_blocks}. "
                "Changing attention type to 'original_full'..."
            )
            self.set_attention_type("original_full")

        if self.attention_type == "block_sparse":
            (
                padding_len,
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                inputs_embeds,
            ) = self._pad_to_block_size(
                input_ids=input_ids,
                input_ids_shape=input_shape,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                inputs_embeds_shape=inputs_embeds_shape,
                pad_token_id=self.config.pad_token_id,
            )
        else:
            padding_len = 0

        if self.attention_type == "block_sparse":
            blocked_encoder_mask, blocked_encoder_mask_shape, band_mask, band_mask_shape, from_mask, from_mask_shape, to_mask, to_mask_shape = self.create_masks_for_block_sparse_attn(
                attention_mask, (batch_size, attention_mask_shape[1] + padding_len), self.block_size
            )
            extended_attention_mask = None
            extended_attention_mask_shape = None

        elif self.attention_type == "original_full":
            blocked_encoder_mask, blocked_encoder_mask_shape = None, None
            band_mask, band_mask_shape = None, None
            from_mask, from_mask_shape = None, None
            to_mask, to_mask_shape = None, None
            extended_attention_mask, extended_attention_mask_shape = self.get_extended_attention_mask(attention_mask, attention_mask_shape, input_shape)
        else:
            raise ValueError(
                f"attention_type can either be original_full or block_sparse, but is {self.attention_type}"
            )

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states_shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ht.init.ones(encoder_hidden_shape, trainable=False)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask, encoder_attention_mask_shape)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            input_ids_shape=(batch_size, seq_length + padding_len),
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            (batch_size, seq_length + padding_len, self.config.hidden_size),
            attention_mask=extended_attention_mask,
            attention_mask_shape=extended_attention_mask_shape,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            band_mask=band_mask,
            band_mask_shape=band_mask_shape,
            from_mask=from_mask,
            from_mask_shape=from_mask_shape,
            to_mask=to_mask,
            to_mask_shape=to_mask_shape,
            blocked_encoder_mask=blocked_encoder_mask,
            blocked_encoder_mask_shape=blocked_encoder_mask_shape
        )
        sequence_output = encoder_outputs[0]

        if self.pooler is not None:
            pooler_output = ht.slice_op(sequence_output, (0,0,0), (-1, 1, -1))
            pooler_output = ht.array_reshape_op(pooler_output, (-1, self.config.hidden_size))
            pooler_output = self.activation(self.pooler(pooler_output))
        else:
            pooler_output = None

        if padding_len > 0:
            sequence_output = ht.slice_op(sequence_output, (0, 0, 0), (-1, seq_length, -1))

        return [sequence_output, pooler_output]


    @staticmethod
    def create_masks_for_block_sparse_attn(attention_mask, attention_mask_shape, block_size: int):

        batch_size, seq_length = attention_mask_shape
        if seq_length % block_size != 0:
            raise ValueError(
                f"Sequence length must be multiple of block size, but sequence length is {seq_length}, while block"
                f" size is {block_size}."
            )

        def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask, from_blocked_mask_shape, to_blocked_mask_shape):

            to_blocked_mask_0 = ht.slice_op(to_blocked_mask, (0, 1, 0), (-1, to_blocked_mask_shape[1]-4, -1))
            to_blocked_mask_1 = ht.slice_op(to_blocked_mask, (0, 2, 0), (-1, to_blocked_mask_shape[1]-4, -1))
            to_blocked_mask_2 = ht.slice_op(to_blocked_mask, (0, 3, 0), (-1, to_blocked_mask_shape[1]-4, -1))
            exp_blocked_to_pad = ht.concatenate_op([to_blocked_mask_0, to_blocked_mask_1, to_blocked_mask_2], axis=2)

            from_blocked_mask = ht.slice_op(from_blocked_mask, (0, 2, 0), (-1, from_blocked_mask_shape[1]-4, -1))

            from_blocked_mask = ht.broadcast_shape_op(from_blocked_mask, (-1, -1, -1, to_blocked_mask_shape[2] * 3), add_axes=(3,))
            exp_blocked_to_pad = ht.broadcast_shape_op(exp_blocked_to_pad, (-1, -1, from_blocked_mask_shape[2], -1), add_axes=(2,))

            band_mask = from_blocked_mask * exp_blocked_to_pad
            band_mask = ht.unsqueeze_op(band_mask, 1)
            b, s, z = from_blocked_mask_shape
            band_mask_shape = (b, 1, s-4, z, 3*z)
            return band_mask, band_mask_shape

        blocked_encoder_mask_shape = (batch_size, seq_length // block_size, block_size)
        blocked_encoder_mask = ht.array_reshape_op(attention_mask, blocked_encoder_mask_shape)
        band_mask, band_mask_shape = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask, blocked_encoder_mask_shape, blocked_encoder_mask_shape)

        from_mask_shape = (batch_size, 1, seq_length, 1)
        to_mask_shape = (batch_size, 1, 1, seq_length)
        from_mask = ht.array_reshape_op(attention_mask, from_mask_shape)
        to_mask = ht.array_reshape_op(attention_mask, to_mask_shape)
        return blocked_encoder_mask, blocked_encoder_mask_shape, band_mask, band_mask_shape, from_mask, from_mask_shape, to_mask, to_mask_shape

    def _pad_to_block_size(
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

        block_size = self.config.block_size

        input_shape = input_ids_shape if input_ids is not None else inputs_embeds_shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (block_size - seq_len % block_size) % block_size
        if padding_len > 0:
            logging.info(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.block_size`: {block_size}"
            )
            if input_ids is not None:
                input_ids = ht.pad_op(input_ids, (0, padding_len), constant_values=pad_token_id)
            if position_ids is not None:
                position_ids = ht.pad_op(position_ids, (0, padding_len), constant_values=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = ht.full_op((batch_size, padding_len), self.config.pad_token_id)
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = ht.concat(inputs_embeds, inputs_embeds_padding, axis=-2)

            attention_mask = ht.pad_op(attention_mask, (0, padding_len), constant_values=0)
            token_type_ids = ht.pad_op(token_type_ids, (0, padding_len), constant_values=0)
        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds

    def create_extended_attention_mask_for_decoder(input_shape, attention_mask, attention_mask_shape):
        batch_size, seq_length = input_shape
        seq_ids = ht.arange_op(0, seq_length)
        seq_ids_left = ht.array_reshape_op(seq_ids, (1, 1, -1))
        seq_ids_left = ht.repeat_op(seq_ids_left, (batch_size, seq_length, 1))

        seq_ids_right = ht.array_reshape_op(seq_ids, (1, -1, 1))
        seq_ids_right = ht.repeat_op(seq_ids_right, (batch_size, 1, seq_length))
        
        causal_mask = ht.bool_op(seq_ids_left, seq_ids_right, 3)

        if seq_length < attention_mask_shape[1]:
            prefix_seq_len = attention_mask_shape[1] - seq_length
            causal_mask = ht.concat_op(ht.init.ones((batch_size, seq_length, prefix_seq_len), trainable=False),
                                       causal_mask,
                                       axis=-1)
        
        causal_mask = ht.unsqueeze_op(causal_mask, 1)
        attention_mask = ht.array_reshape_op(attention_mask, (batch_size, 1, 1, -1))
        attention_mask = ht.broadcastto_op(attention_mask, causal_mask)
        extended_attention_mask = causal_mask * attention_mask
        return extended_attention_mask
    
    def get_extended_attention_mask(self, attention_mask, attention_mask_shape, input_shape):
        
        if len(attention_mask_shape) == 3:
            b, s1, s2 = attention_mask_shape
            extended_attention_mask_shape = (b, 1, s1, s2)
            extended_attention_mask = ht.array_reshape_op(attention_mask, extended_attention_mask_shape)

        elif len(attention_mask_shape) == 2:
            b, s = attention_mask_shape
            if self.config.is_decoder:
                extended_attention_mask = self.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, attention_mask_shape)
                extended_attention_mask_shape = (b, 1, input_shape[1], s)
            else:
                extended_attention_mask_shape = (b, 1, 1, s)
                extended_attention_mask = ht.array_reshape_op(attention_mask, extended_attention_mask_shape)
        else:            
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask_shape})"
            )
        extended_attention_mask = ht.minus_byconst_op(extended_attention_mask, 1.0)
        extended_attention_mask = extended_attention_mask * np.finfo(np.float32).min
        return extended_attention_mask, extended_attention_mask_shape


    def invert_attention_mask(self, encoder_attention_mask, encoder_attention_mask_shape):
        if len(encoder_attention_mask_shape) == 3:
            b, s1, s2 = encoder_attention_mask_shape
            encoder_extended_attention_mask = ht.array_reshape_op(encoder_attention_mask,(b, 1, s1, s2))
        if len(encoder_attention_mask_shape) == 2:
            b, s = encoder_attention_mask_shape
            encoder_extended_attention_mask = ht.array_reshape_op(encoder_attention_mask,(b, 1, 1, s))

        encoder_extended_attention_mask = ht.minus_byconst_op(encoder_extended_attention_mask, 1.0)
        encoder_extended_attention_mask = encoder_extended_attention_mask * np.finfo(np.float32).min
        return encoder_extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            assert False
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask


class BigBirdPredictionHeadTransform(object):
    def __init__(self, config, name='BigBirdPredictionHeadTransform'):
        self.dense = ht.layers.Linear(config.hidden_size, config.hidden_size, weight_transpose=True, name=name+'.dense')
        if config.hidden_act == "relu":
            self.transform_act_fn = ht.relu_op
        elif config.hidden_act == "gelu":
            self.transform_act_fn = ht.gelu_op
        self.LayerNorm = ht.layers.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, name=name+'.LayerNorm')

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
        

class BigBirdLMPredictionHead(object):
    def __init__(self, config, name='BigBirdLMPredictionHead'):
        self.transform = BigBirdPredictionHeadTransform(config)
        self.decoder = ht.layers.Linear(config.hidden_size, config.vocab_size, bias=True, weight_transpose=True, name=name+'.decoder')

    def __call__(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
                
class BigBirdPreTrainingHeads(object):
    def __init__(self, config, name='BigBirdPreTrainingHeads'):
        self.dim = config.hidden_size
        self.predictions = BigBirdLMPredictionHead(config, name=name+'.predictions')
        self.seq_relationship = ht.layers.Linear(config.hidden_size, 2, weight_transpose=True, name=name+'.seq_relationship')

    def __call__(self, sequence_output, pooled_output, input_shape):
        sequence_output = ht.array_reshape_op(sequence_output, (-1, self.dim))
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        prediction_scores = ht.array_reshape_op(prediction_scores, input_shape[:-1] + (-1, ))
        return prediction_scores, seq_relationship_score
        
                
class BigBirdForPreTraining(object):
    def __init__(self, config):
        self.config = config
        self.bert = BigBirdModel(config, add_pooling_layer=True)
        self.cls = BigBirdPreTrainingHeads(config)


    def __call__(self, input_ids, input_ids_shape, token_type_ids, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        outputs = self.bert(
            input_ids,
            input_ids_shape,
            attention_mask=attention_mask,
            attention_mask_shape=input_ids_shape,
            token_type_ids=token_type_ids
        )

        sequence_output, pooled_output = outputs[:2]
        input_shape = input_ids_shape + (self.config.hidden_size, )
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output, input_shape)

        return_op = [prediction_scores, seq_relationship_score]
        if masked_lm_labels is not None and next_sentence_label is not None:
            masked_lm_loss = ht.crossentropy_sparse_op(ht.softmax_op(
                prediction_scores), masked_lm_labels, ignored_index=-1)
            next_sentence_loss = ht.crossentropy_sparse_op(ht.softmax_op(
                seq_relationship_score), next_sentence_label, ignored_index=-1)

            return_op += [masked_lm_loss, next_sentence_loss]
        return return_op

class BigBirdClassificationHead(object):
    def __init__(self, config, name='BigBirdClassificationHead'):
        self.hidden_size = config.hidden_size
        self.dense = ht.layers.Linear(config.hidden_size, config.hidden_size, weight_transpose=True, name=name+'.dense')
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = ht.layers.DropOut(classifier_dropout)
        if config.hidden_act == "relu":
            self.hidden_act = ht.relu_op
        elif config.hidden_act == "gelu":
            self.hidden_act = ht.gelu_op
            
        self.out_proj = ht.layers.Linear(config.hidden_size, config.num_labels, weight_transpose=True, name=name+'.out_proj')
        self.config = config

    def __call__(self, features):
        x = ht.slice_op(features, (0, 0, 0), (-1, 1, -1))
        x = ht.array_reshape_op(x, (-1, self.hidden_size))
        x = self.dropout(x)
        x = self.dense(x)
        x = self.hidden_act(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class BigBirdForSequenceClassification(object):
    def __init__(self, config):
        self.config = config
        self.bert = BigBirdModel(config)
        self.classifier = BigBirdClassificationHead(config, name='classifier')

    def get_input_embeddings(self):
        return self.bert.get_input_embeddings()
                        
    def __call__(self, input_ids, input_ids_shape, attention_mask=None, labels=None):
        hidden_states = self.bert(input_ids, input_ids_shape, attention_mask=attention_mask)[0]
        logits = self.classifier(hidden_states)

        if labels is not None:
            # loss = ht.softmaxcrossentropy_sparse_op(logits, labels, ignored_index = -1)
            loss = ht.crossentropy_sparse_op(
                ht.softmax_op(logits), labels, ignored_index=-1)
            return loss, logits
        else:
            return logits
