import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import hetu as ht


class XLNetRelativeAttention(object):
    def __init__(self, config, name='XLNetRelativeAttention'):
        if config.d_model % config.n_head != 0:
            raise ValueError(
                f"The hidden size ({config.d_model}) is not a multiple of the number of attention "
                f"heads ({config.n_head}"
            )

        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / (config.d_head**0.5)

        self.q = ht.init.random_normal(
            (config.d_model, self.n_head, self.d_head), name=name+'.q')
        self.k = ht.init.random_normal(
            (config.d_model, self.n_head, self.d_head), name=name+'.k')
        self.v = ht.init.random_normal(
            (config.d_model, self.n_head, self.d_head), name=name+'.v')
        self.o = ht.init.random_normal(
            (config.d_model, self.n_head, self.d_head), name=name+'.o')
        self.r = ht.init.random_normal(
            (config.d_model, self.n_head, self.d_head), name=name+'.r')

        self.r_r_bias = ht.init.random_normal(
            (self.n_head, self.d_head), name=name+'.r_r_bias')
        self.r_s_bias = ht.init.random_normal(
            (self.n_head, self.d_head), name=name+'.r_s_bias')
        self.r_w_bias = ht.init.random_normal(
            (self.n_head, self.d_head), name=name+'.r_w_bias')
        self.seg_embed = ht.init.random_normal(
            (2, self.n_head, self.d_head), name=name+'.seg_embed')

        self.layer_norm = ht.layers.LayerNorm(
            config.d_model, eps=config.layer_norm_eps, name=name+'.layer_norm')
        self.dropout = ht.layers.DropOut(config.dropout)

    @staticmethod
    def rel_shift(x, x_shape, klen=-1):
        x_size = x_shape
        x = ht.array_reshape_op(
            x, (x_size[1], x_size[0], x_size[2], x_size[3]))
        x = ht.slice_op(x, (1, 0, 0, 0), (-1, -1, -1, -1))
        x = ht.array_reshape_op(
            x, (x_size[0], x_size[1] - 1, x_size[2], x_size[3]))
        x = ht.slice_op(x, (0, 0, 0, 0), (-1, klen, -1, -1))
        return x

    @staticmethod
    def rel_shift_bnij(x, x_shape, klen=-1):
        x_size = x_shape

        x = ht.array_reshape_op(
            x, (x_size[0], x_size[1], x_size[3], x_size[2]))
        x = ht.slice_op(x, (0, 0, 1, 0), (-1, -1, -1, -1))
        x = ht.array_reshape_op(
            x, (x_size[0], x_size[1], x_size[2], x_size[3] - 1))
        x = ht.slice_op(x, (0, 0, 0, 0), (-1, -1, -1, klen))
        return x

    def rel_attn_core(
        self,
        q_head,
        k_head_h,
        v_head_h,
        k_head_r,
        q_head_shape,
        k_head_h_shape,
        k_head_r_shape,
        seg_mat=None,
        attn_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        q_head_w = q_head + ht.broadcastto_op(self.r_w_bias, q_head)
        q_head_w = ht.transpose_op(q_head_w, (1, 2, 0, 3))
        k_head_h = ht.transpose_op(k_head_h, (1, 2, 3, 0))
        ac = ht.batch_matmul_op(q_head_w, k_head_h)

        q_head_r = q_head + ht.broadcastto_op(self.r_r_bias, q_head)
        q_head_r = ht.transpose_op(q_head_r, (1, 2, 0, 3))
        k_head_r = ht.transpose_op(k_head_r, (1, 2, 3, 0))
        bd = ht.batch_matmul_op(q_head_r, k_head_r)

        bd_shape = (q_head_shape[1], q_head_shape[2],
                    q_head_shape[0], k_head_r_shape[0])

        bd = self.rel_shift_bnij(bd, bd_shape, klen=k_head_h_shape[0])

        if seg_mat is None:
            ef = 0
        else:
            bsz = q_head_shape[1]
            q_head_s = q_head + self.r_s_bias
            q_head_s = ht.transpose_op(q_head_s, (1, 2, 0, 3))
            seg_embed = ht.transpose_op(self.seg_embed, (1, 2, 0))
            seg_embed = ht.broadcast_shape(seg_embed, (bsz, -1, -1, -1))
            ef = ht.batch_matmul_op(q_head_s, seg_embed)
            ef = ht.transpose_op(ef, (0, 2, 1, 3))
            seg_mat = ht.transpose_op(seg_mat, (2, 0, 3, 1))
            ef = ht.batch_matmul_op(ef, seg_mat)
            seg_mat = ht.transpose_op(seg_mat, (0, 2, 1, 3))

        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            attn_mask = ht.transpose_op(attn_mask, (2, 3, 0, 1))
            attn_score = attn_score - 1e30 * attn_mask

        attn_prob = ht.softmax_op(attn_score)
        attn_prob = self.dropout(attn_prob)

        if head_mask is not None:
            attn_prob = attn_prob * ht.transpose_op(head_mask, (2, 3, 0, 1))

        v_head_h = ht.transpose_op(v_head_h, (1, 2, 0, 3))
        attn_vec = ht.batch_matmul_op(attn_prob, v_head_h)
        attn_vec = ht.transpose_op(attn_vec, (2, 0, 1, 3))

        if output_attentions:
            return attn_vec, ht.transpose_op(attn_prob, (2, 3, 0, 1))

        return attn_vec

    def post_attention(self, h, attn_vec, attn_vec_shape, residual=True):
        o = ht.transpose_op(self.o, (1, 2, 0))
        o = ht.array_reshape_op(o, (attn_vec_shape[2] * attn_vec_shape[3], -1))
        attn_vec = ht.array_reshape_op(
            attn_vec, (-1, attn_vec_shape[2] * attn_vec_shape[3]))
        attn_out = ht.matmul_op(attn_vec, o)
        attn_out = ht.array_reshape_op(
            attn_out, (attn_vec_shape[0], attn_vec_shape[1], -1))

        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)

        return output

    def __call__(
        self,
        h,
        h_shape,
        g,
        g_shape,
        attn_mask_h,
        attn_mask_g,
        r,
        r_shape,
        seg_mat,
        mems=None,
        mems_shape=None,
        target_mapping=None,
        target_mapping_shape=None,
        head_mask=None,
        output_attentions=False,
    ):
        if g is not None:
            if mems is not None and len(mems_shape) > 1:
                cat = ht.concat_op(mems, h, axis=0)
                h_shape[0] += mems_shape[0]
            else:
                cat = h

            cat = ht.array_reshape_op(cat, (-1, h_shape[-1]))

            k_head_h = ht.matmul_op(
                cat, ht.array_reshape_op(self.k, (h_shape[-1], -1)))
            k_head_h = ht.array_reshape_op(
                k_head_h, (h_shape[0], h_shape[1], self.n_head, self.d_head))

            v_head_h = ht.matmul_op(
                cat, ht.array_reshape_op(self.v, (h_shape[-1], -1)))
            v_head_h = ht.array_reshape_op(
                v_head_h, (h_shape[0], h_shape[1], self.n_head, self.d_head))

            r = ht.array_reshape_op(r, (-1, r_shape[-1]))
            sr = ht.array_reshape_op(self.r, (r_shape[-1], -1))
            k_head_r = ht.matmul_op(r, sr)
            k_head_r = ht.array_reshape_op(
                k_head_r, (r_shape[0], r_shape[1], self.n_head, self.d_head))

            q_head_h = ht.matmul_op(ht.array_reshape_op(
                h, (-1, h_shape[-1])), ht.array_reshape_op(self.q, (h_shape[-1], -1)))
            q_head_h = ht.array_reshape_op(
                q_head_h, (h_shape[0], h_shape[1], self.n_head, self.d_head))

            attn_vec_h = self.rel_attn_core(
                q_head_h,
                k_head_h,
                v_head_h,
                k_head_r,
                (h_shape[0], h_shape[1], self.n_head, self.d_head),
                (h_shape[0], h_shape[1], self.n_head, self.d_head),
                (r_shape[0], r_shape[1], self.n_head, self.d_head),
                seg_mat=seg_mat,
                attn_mask=attn_mask_h,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )

            if output_attentions:
                attn_vec_h, attn_prob_h = attn_vec_h

            output_h = self.post_attention(h, attn_vec_h)

            g = ht.array_reshape_op(g, (-1, g_shape[-1]))
            q_head_g = ht.matmul_op(g, self.q)
            q_head_g = ht.array_reshape_op(
                q_head_g, (g_shape[0], g_shape[1], self.n_head, self.d_head))

            if target_mapping is not None:
                q_head_g = ht.transpose_op(q_head_g, (1, 3, 2, 0))
                target_mapping = ht.transpose_op(target_mapping, (2, 0, 1))
                target_mapping = ht.broadcast_shape_op(
                    target_mapping, (-1, self.d_head, -1, -1), add_axes=(1, ))
                q_head_g = ht.batch_matmul_op(q_head_g, target_mapping)
                q_head_g = ht.transpose_op(q_head_g, (3, 0, 2, 1))
                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    (target_mapping_shape[1], g_shape[1],
                     self.n_head, self.d_head),
                    (h_shape[0], h_shape[1], self.n_head, self.d_head),
                    (r_shape[0], r_shape[1], self.n_head, self.d_head),
                    seg_mat=seg_mat,
                    attn_mask=attn_mask_g,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                )

                if output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

                attn_vec_g = ht.transpose_op(attn_vec_g, (1, 2, 3, 0))
                target_mapping = ht.transpose_op(target_mapping, (2, 1, 0))
                target_mapping = ht.broadcast_shape_op(
                    target_mapping, (-1, self.n_head, -1, -1), add_axes=(1, ))
                attn_vec_g = ht.batch_matmul_op(attn_vec_g, target_mapping)
                attn_vec_g = ht.transpose_op(attn_vec_g, (3, 0, 1, 2))
            else:
                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    (g_shape[0], g_shape[1], self.n_head, self.d_head),
                    (h_shape[0], h_shape[1], self.n_head, self.d_head),
                    (r_shape[0], r_shape[1], self.n_head, self.d_head),
                    seg_mat=seg_mat,
                    attn_mask=attn_mask_g,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                )

                if output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

            output_g = self.post_attention(g, attn_vec_g)

            if output_attentions:
                attn_prob = attn_prob_h, attn_prob_g

        else:
            if mems is not None and len(mems_shape) > 1:
                cat = ht.concat_op(mems, h, axis=0)
                h_shape[0] += mems_shape[0]
            else:
                cat = h

            cat = ht.array_reshape_op(cat, (-1, h_shape[-1]))

            k_head_h = ht.matmul_op(
                cat, ht.array_reshape_op(self.k, (h_shape[-1], -1)))
            k_head_h = ht.array_reshape_op(
                k_head_h, (h_shape[0], h_shape[1], self.n_head, self.d_head))

            v_head_h = ht.matmul_op(
                cat, ht.array_reshape_op(self.v, (h_shape[-1], -1)))
            v_head_h = ht.array_reshape_op(
                v_head_h, (h_shape[0], h_shape[1], self.n_head, self.d_head))

            r = ht.array_reshape_op(r, (-1, r_shape[-1]))
            sr = ht.array_reshape_op(self.r, (r_shape[-1], -1))
            k_head_r = ht.matmul_op(r, sr)
            k_head_r = ht.array_reshape_op(
                k_head_r, (r_shape[0], r_shape[1], self.n_head, self.d_head))

            q_head_h = ht.matmul_op(ht.array_reshape_op(
                h, (-1, h_shape[-1])), ht.array_reshape_op(self.q, (h_shape[-1], -1)))
            q_head_h = ht.array_reshape_op(
                q_head_h, (h_shape[0], h_shape[1], self.n_head, self.d_head))

            attn_vec = self.rel_attn_core(
                q_head_h,
                k_head_h,
                v_head_h,
                k_head_r,
                (h_shape[0], h_shape[1], self.n_head, self.d_head),
                (h_shape[0], h_shape[1], self.n_head, self.d_head),
                (r_shape[0], r_shape[1], self.n_head, self.d_head),
                seg_mat=seg_mat,
                attn_mask=attn_mask_h,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )

            if output_attentions:
                attn_vec, attn_prob = attn_vec

            output_h = self.post_attention(
                h, attn_vec, (h_shape[0], h_shape[1], self.n_head, self.d_head))
            output_g = None

        outputs = (output_h, output_g)
        if output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs


class XLNetFeedForward(object):
    def __init__(self, config, name='XLNetFeedForward'):

        self.layer_norm = ht.layers.LayerNorm(
            config.d_model, eps=config.layer_norm_eps, name=name+'.layer_norm')
        self.layer_1 = ht.layers.Linear(
            config.d_model, config.d_inner, weight_transpose=True, name=name+'.layer_1')
        self.layer_2 = ht.layers.Linear(
            config.d_inner, config.d_model, weight_transpose=True, name=name+'.layer_2')
        self.dropout = ht.layers.DropOut(config.dropout)

        if config.ff_activation == "relu":
            self.activation_function = ht.relu_op
        elif config.ff_activation == "gelu":
            self.activation_function = ht.gelu_op

    def __call__(self, inp, inp_shape):
        output = inp
        output = ht.array_reshape_op(output, (-1, inp_shape[-1]))
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = ht.array_reshape_op(output, (inp_shape[0], inp_shape[1], -1))
        output = self.layer_norm(output + inp)
        return output


class XLNetLayer(object):
    def __init__(self, config, name='XLNetLayer'):
        self.rel_attn = XLNetRelativeAttention(config, name=name+'.rel_attn')
        self.ff = XLNetFeedForward(config, name=name+'.ff')
        self.dropout = ht.layers.DropOut(config.dropout)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

    def __call__(
        self,
        output_h,
        output_h_shape,
        output_g,
        output_g_shape,
        attn_mask_h,
        attn_mask_g,
        r,
        r_shape,
        seg_mat,
        mems=None,
        mems_shape=None,
        target_mapping=None,
        head_mask=None,
        output_attentions=False,
    ):
        outputs = self.rel_attn(
            output_h,
            output_h_shape,
            output_g,
            output_g_shape,
            attn_mask_h,
            attn_mask_g,
            r,
            r_shape,
            seg_mat,
            mems=mems,
            mems_shape=mems_shape,
            target_mapping=target_mapping,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        output_h, output_g = outputs[:2]

        if output_g is not None:
            output_g = self.ff(output_g, output_g_shape)
        output_h = self.ff(output_h, output_h_shape)
        outputs = (output_h, output_g) + outputs[2:]
        return outputs


class XLNetModel(object):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer

        self.word_embedding = ht.layers.Embedding(
            config.vocab_size, config.d_model, name='word_embedding')
        self.mask_emb = ht.init.random_normal(
            (1, 1, config.d_model), name='mask_emb')
        self.layer = [XLNetLayer(config, name='layer.'+str(i))
                      for i in range(config.n_layer)]
        self.dropout = ht.layers.DropOut(config.dropout)

    def create_mask(self, qlen, mlen):
        attn_mask = ht.init.ones([qlen, qlen], trainable=False)
        mask_up = ht.triu_op(attn_mask, diagonal=1)
        attn_mask_pad = ht.init.ones([qlen, mlen], trainable=False)
        ret = ht.concat_op(attn_mask_pad, mask_up, axis=1)
        if self.same_length:
            mask_lo = ht.tril_op(attn_mask, diagonal=-1)
            ret_0 = ht.slice_op(ret, (0, 0), (-1, qlen)) + mask_lo
            ret_1 = ht.slice_op(ret, (0, qlen), (-1, -1))
            ret = ht.concat_op(ret_0, ret_1, axis=1)
        return ret

    def cache_mem(self, curr_out, prev_mem):

        if self.reuse_len is not None and self.reuse_len > 0:
            curr_out = curr_out[: self.reuse_len]
            curr_out = ht.slice_op(curr_out, (0, 0), (-1, self.reuse_len))

        if self.mem_len is None or self.mem_len == 0:
            cutoff = 0
        else:
            cutoff = -self.mem_len

        if prev_mem is None:
            new_mem = ht.slice_op(curr_out, (cutoff, 0), (-1, -1))
        else:
            new_mem = ht.concat_op(prev_mem, curr_out, axis=0)
            new_mem = ht.slice_op(new_mem, (cutoff, 0), (-1, -1))

        return new_mem

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, pos_seq_len, bsz=None):
        sinusoid_inp = ht.outer_op(pos_seq, inv_freq)
        pos_emb = ht.concat_op(ht.sin_op(sinusoid_inp),
                               ht.cos_op(sinusoid_inp), axis=-1)

        if bsz is not None:
            pos_emb = ht.broadcast_shape_op(
                pos_emb, (-1, bsz, -1), add_axes=(1, ))
        else:
            pos_emb = ht.array_reshape_op(pos_emb, (pos_seq_len, 1, -1))

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        freq_seq = ht.arange_op(0, self.d_model, 2.0)
        freq = ht.const_pow_op(freq_seq * (1 / self.d_model), 10000)
        inv_freq = ht.div_const_op(1, freq)

        if self.attn_type == "bi":
            beg, end = klen, -qlen
        elif self.attn_type == "uni":
            beg, end = klen, -1
        else:
            raise ValueError(f"Unknown `attn_type` {self.attn_type}.")

        if self.bi_data:
            fwd_pos_seq = ht.arange_op(beg, end, -1.0)
            bwd_pos_seq = ht.arange_op(-beg, -end, 1.0)
            fwd_pos_seq_len = int((end-beg)/(-1.0))
            bwd_pos_seq_len = int((-end+beg)/(1.0))

            if self.clamp_len > 0:
                fwd_pos_seq = ht.clamp_op(
                    fwd_pos_seq, min=-self.clamp_len, max=self.clamp_len)
                bwd_pos_seq = ht.clamp_op(
                    bwd_pos_seq, min=-self.clamp_len, max=self.clamp_len)

            if bsz is not None:
                fwd_pos_emb = self.positional_embedding(
                    fwd_pos_seq, inv_freq, fwd_pos_seq_len, bsz // 2)
                bwd_pos_emb = self.positional_embedding(
                    bwd_pos_seq, inv_freq, bwd_pos_seq_len, bsz // 2)
                pos_emb_shape = (fwd_pos_seq_len, bsz, self.d_model)
            else:
                fwd_pos_emb = self.positional_embedding(
                    fwd_pos_seq, inv_freq, fwd_pos_seq_len)
                bwd_pos_emb = self.positional_embedding(
                    bwd_pos_seq, inv_freq, bwd_pos_seq_len)
                pos_emb_shape = (fwd_pos_seq_len, 2, self.d_model)

            pos_emb = ht.concat_op(fwd_pos_emb, bwd_pos_emb, axis=1)
        else:
            fwd_pos_seq = ht.arange_op(beg, end, -1.0)
            pos_seq_len = int((end-beg)/(-1.0))
            if self.clamp_len > 0:
                fwd_pos_seq = ht.clamp_op(
                    fwd_pos_seq, min=-self.clamp_len, max=self.clamp_len)
            pos_emb = self.positional_embedding(
                fwd_pos_seq, inv_freq, pos_seq_len, bsz)
            pos_emb_shape = (
                pos_seq_len, bsz if bsz is not None else 2, self.d_model)

        return pos_emb, pos_emb_shape

    def __call__(
        self,
        input_ids=None,
        input_ids_shape=None,
        attention_mask=None,
        attention_mask_shape=None,
        mems=None,
        mems_shape=None,
        perm_mask=None,
        perm_mask_shape=None,
        target_mapping=None,
        target_mapping_shape=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        head_mask_shape=None,
        inputs_embeds=None,
        inputs_embeds_shape=None,
        use_mems=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if "use_cache" in kwargs:
            warnings.warn(
                "The `use_cache` argument is deprecated and will be removed in a future version, use `use_mems`"
                " instead.",
                FutureWarning,
            )
            use_mems = kwargs["use_cache"]

        use_mems = use_mems if use_mems is not None else self.config.use_mems_train

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = ht.transpose_op(input_ids, (1, 0))
            bsz, qlen = input_ids_shape[0], input_ids_shape[1]
        elif inputs_embeds is not None:
            inputs_embeds = ht.transpose_op(inputs_embeds, (1, 0, 2))
            bsz, qlen = inputs_embeds[0], inputs_embeds[1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        token_type_ids = ht.transpose_op(
            token_type_ids, (1, 0)) if token_type_ids is not None else None
        input_mask = ht.transpose_op(
            input_mask, (1, 0)) if input_mask is not None else None
        attention_mask = ht.transpose_op(
            attention_mask, (1, 0)) if attention_mask is not None else None
        perm_mask = ht.transpose_op(
            perm_mask, (1, 2, 0)) if perm_mask is not None else None
        target_mapping = ht.transpose_op(
            target_mapping, (1, 2, 0)) if target_mapping is not None else None

        mlen = mems_shape[0][0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen

        if self.attn_type == "uni":
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = ht.broadcast_shape_op(
                attn_mask, (-1, -1, 1, 1), add_axes=(2, 3))
        elif self.attn_type == "bi":
            attn_mask = None
        else:
            raise ValueError(f"Unsupported attention type: {self.attn_type}")

        assert input_mask is None or attention_mask is None, "You can only use one of input_mask (uses 1 for padding) "
        "or attention_mask (uses 0 for padding, added for compatibility with BERT). Please choose one."
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask
        if input_mask is not None and perm_mask is not None:
            input_mask = ht.array_reshape_op(input_mask, (1, bsz, qlen))
            data_mask = input_mask + perm_mask
            data_mask_shape = perm_mask_shape
        elif input_mask is not None and perm_mask is None:
            data_mask = ht.array_reshape_op(input_mask, (1, bsz, qlen))
            data_mask_shape = (1, bsz, qlen)
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
            data_mask_shape = perm_mask_shape
        else:
            data_mask = None

        if data_mask is not None:
            if mlen > 0:
                mems_mask = ht.init.zeros(
                    [data_mask_shape[0], mlen, bsz], trainable=False)
                data_mask = ht.concat_op(mems_mask, data_mask, axis=1)
            if attn_mask is None:
                attn_mask = ht.broadcast_shape_op(
                    data_mask, (-1, -1, -1, 1), add_axes=(3, ))
            else:
                data_mask = ht.broadcast_shape_op(
                    data_mask, (-1, -1, -1, 1), add_axes=(3, ))
                attn_mask += data_mask

        if attn_mask is not None:
            attn_mask = ht.bool_op(attn_mask)

        if attn_mask is not None:
            non_tgt_mask = -ht.eye_op(qlen)
            if mlen > 0:
                non_tgt_mask = ht.concat_op(ht.init.zeros(
                    [qlen, mlen], trainable=False), non_tgt_mask, axis=-1)
            non_tgt_mask = ht.broadcast_shape_op(
                data_mask, (-1, -1, 1, 1), add_axes=(2, 3))
            non_tgt_mask = ht.bool_op(attn_mask + non_tgt_mask)
        else:
            non_tgt_mask = None

        if inputs_embeds is not None:
            word_emb_k = inputs_embeds
            output_h_shape = (qlen, bsz, inputs_embeds_shape[-1])
        else:
            word_emb_k = self.word_embedding(input_ids)
            output_h_shape = (qlen, bsz, self.d_model)
        output_h = self.dropout(word_emb_k)

        if target_mapping is not None:
            word_emb_q = ht.broadcast_shape_op(
                self.mask_emb, (target_mapping_shape[0], bsz, -1))
            output_g = self.dropout(word_emb_q)
            output_g_shape = (target_mapping_shape[0], bsz, self.d_model)
        else:
            output_g = None
            output_g_shape = None

        if token_type_ids is not None:
            raise NotImplementedError
        else:
            seg_mat = None

        pos_emb, pos_emb_shape = self.relative_positional_encoding(
            qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        if head_mask is not None:
            if len(head_mask_shape) == 1:
                head_mask = ht.broadcast_shape(
                    head_mask, (self.n_layer, 1, 1, 1, -1), add_axes=(0, 1, 2, 3))

            elif len(head_mask_shape) == 2:
                head_mask = ht.broadcast_shape(
                    head_mask, (-1, 1, 1, 1, -1), add_axes=(1, 2, 3))
        else:
            head_mask = [None] * self.n_layer

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)
            mems_shape = [None] * len(self.layer)

        attentions = [] if output_attentions else None
        hidden_states = [] if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if use_mems:
                new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if output_hidden_states:
                hidden_states.append((output_h, output_g)
                                     if output_g is not None else output_h)

            outputs = layer_module(
                output_h,
                output_h_shape,
                output_g,
                output_g_shape,
                attn_mask_h=non_tgt_mask,
                attn_mask_g=attn_mask,
                r=pos_emb,
                r_shape=pos_emb_shape,
                seg_mat=seg_mat,
                mems=mems[i],
                mems_shape=mems_shape[i],
                target_mapping=target_mapping,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            output_h, output_g = outputs[:2]
            if output_attentions:
                attentions.append(outputs[2])

        if output_hidden_states:
            hidden_states.append((output_h, output_g)
                                 if output_g is not None else output_h)

        output = self.dropout(output_g if output_g is not None else output_h)
        output = ht.transpose_op(output, (1, 0, 2))

        return [output]
