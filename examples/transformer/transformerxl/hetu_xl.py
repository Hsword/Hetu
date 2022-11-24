import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import math
import hetu as ht
import numpy as np


class PositionalEmbedding(object):
    def __init__(self, demb, name='PositionalEmbedding'):
        self.demb = demb
        self.inv_len = math.ceil(demb/2.0)
        inv_freq = ht.arange_op(0.0, demb, 2.0) *(1/demb)
        inv_freq = ht.const_pow_op(inv_freq, 10000)
        self.inv_freq = ht.div_const_op(1, inv_freq)

    def __call__(self, pos_seq, seq_len, bsz=None):
        sinusoid_inp = ht.outer_op(pos_seq, self.inv_freq)
        sinusoid_inp_sin = ht.sin_op(sinusoid_inp)
        sinusoid_inp_cos = ht.cos_op(sinusoid_inp)   
        pos_emb = ht.concat_op(sinusoid_inp_sin, sinusoid_inp_cos, axis=1)     

        if bsz is not None:
            pos_emb = ht.broadcast_shape_op(pos_emb, (seq_len, bsz, 2*self.inv_len), add_axes=(1))
            return pos_emb, (seq_len, bsz, 2*self.inv_len)
        else:
            pos_emb = ht.array_reshape_op(pos_emb, (seq_len, 1, 2*self.inv_len))
            return pos_emb, (seq_len, 1, 2*self.inv_len)
        

class MLP(object):
    def __init__(self, d_model, d_inner, dropout, name='MLP'):
        self.wi = ht.layers.Linear(d_model, d_inner, weight_transpose=True, name=name+'.0')
        self.wo = ht.layers.Linear(d_inner, d_model, weight_transpose=True, name=name+'.3')
        self.dropout = ht.layers.DropOut(dropout)
        self.dim = d_model

    def __call__(self, inp, input_shape):
        inp = ht.array_reshape_op(inp, [-1, input_shape[-1]])
        inp = self.wi(inp)
        inp = ht.relu_op(inp)
        inp = self.dropout(inp)
        inp = self.wo(inp)
        inp = self.dropout(inp)
        inp = ht.array_reshape_op(inp, input_shape[:-1]+(self.dim, ))
        return inp

class PositionwiseFF(object):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, layer_norm_epsilon=1e-5, name='PositionwiseFF'):
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.CoreNet = MLP(d_model, d_inner, dropout, name=name+'.CoreNet')
        self.layer_norm = ht.layers.LayerNorm(d_model, eps=layer_norm_epsilon, name=name+'.layer_norm')
        self.pre_lnorm = pre_lnorm

    def __call__(self, inp, inp_shape):
        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp), inp_shape)
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp, inp_shape)
            output = self.layer_norm(inp + core_out)
        return output


class RelPartialLearnableMultiHeadAttn(object):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=0,
        pre_lnorm=False,
        r_r_bias=None,
        r_w_bias=None,
        layer_norm_epsilon=1e-5,
        name='RelPartialLearnableMultiHeadAttn',
    ):
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.dim = n_head * d_head

        self.qkv_net = ht.layers.Linear(d_model, 3 * n_head * d_head, bias=False, weight_transpose=True, name=name+'.qkv_net')

        self.drop = ht.layers.DropOut(dropout)
        self.dropatt = ht.layers.DropOut(dropatt)
        self.o_net = ht.layers.Linear(n_head * d_head, d_model, bias=False, weight_transpose=True, name=name+'.o_net')

        self.layer_norm = ht.layers.LayerNorm(d_model, eps=layer_norm_epsilon, name=name+'.layer_norm')

        self.scale = 1 / (d_head**0.5)

        self.pre_lnorm = pre_lnorm

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = ht.init.random_normal((self.n_head, self.d_head), name=name+'.r_r_bias')
            self.r_w_bias = ht.init.random_normal((self.n_head, self.d_head), name=name+'.r_w_bias')
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

        self.r_net = ht.layers.Linear(self.d_model, self.n_head * self.d_head, bias=False, weight_transpose=True, name=name+'.r_net')

    def _rel_shift(self, x, x_shape):
        zero_pad_shape = (x_shape[0], 1) + x_shape[2:]
        zero_pad = ht.init.zeros(zero_pad_shape, trainable=False)
        x_padded = ht.concat_op(zero_pad, x, axis=1)

        x_padded_shape = (x_shape[1] + 1, x_shape[0]) + x_shape[2:]
        x_padded = ht.array_reshape_op(x_padded, x_padded_shape)

        x = ht.slice_op(x_padded, (1, 0, 0, 0), (-1, -1, -1, -1))
        x = ht.array_reshape_op(x, x_shape)

        return x

    def __call__(self, w, r, w_shape, r_shape, attn_mask=None, attn_mask_shape=None, mems=None, mems_shape=None, head_mask=None, output_attentions=False):
        qlen, rlen, bsz = w_shape[0], r_shape[0], w_shape[1]

        if mems is not None:
            klen = mems_shape[0] + w_shape[0]
            cat = ht.concat_op(mems, w, 0)
            cat = ht.array_reshape_op(cat, (-1, w_shape[-1]))
            r = ht.array_reshape_op(r, (-1, r_shape[-1]))
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)

            r_head_k = self.r_net(r)
            w_heads = ht.array_reshape_op(w_heads, (-1, bsz, 3 * self.dim))
            w_head_q = ht.slice_op(w_heads, (-qlen, 0, 0), (-1, -1, self.dim))
            w_head_k = ht.slice_op(w_heads, (0, 0, self.dim), (-1, -1, self.dim))
            w_head_v = ht.slice_op(w_heads, (0, 0, self.dim * 2), (-1, -1, self.dim))
        else:
            klen = w_shape[0]
            w = ht.array_reshape_op(w, (-1, w_shape[-1]))
            r = ht.array_reshape_op(r, (-1, r_shape[-1]))            
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)
            w_heads = ht.array_reshape_op(w_heads, (-1, bsz, 3 * self.dim))
            w_head_q = ht.slice_op(w_heads, (0, 0, 0), (-1, -1, self.dim))
            w_head_k = ht.slice_op(w_heads, (0, 0, self.dim), (-1, -1, self.dim))
            w_head_v = ht.slice_op(w_heads, (0, 0, self.dim * 2), (-1, -1, self.dim))
        


        w_head_q = ht.array_reshape_op(w_head_q, (qlen, bsz, self.n_head, self.d_head))
        w_head_k = ht.array_reshape_op(w_head_k, (klen, bsz, self.n_head, self.d_head))    
        w_head_v = ht.array_reshape_op(w_head_v, (klen, bsz, self.n_head, self.d_head))                
        r_head_k = ht.array_reshape_op(r_head_k, (rlen, self.n_head, self.d_head))              

        self.r_w_bias = ht.broadcast_shape_op(self.r_w_bias, (qlen, bsz, self.n_head, self.d_head))
        self.r_r_bias = ht.broadcast_shape_op(self.r_r_bias, (qlen, bsz, self.n_head, self.d_head))
        
        rw_head_q = w_head_q + self.r_w_bias 

        rw_head_q_AC = ht.transpose_op(rw_head_q, (1, 2, 0, 3))
        w_head_k_AC = ht.transpose_op(w_head_k, (1, 2, 3, 0))
        AC = ht.batch_matmul_op(rw_head_q_AC, w_head_k_AC)
        AC = ht.transpose_op(AC, (2, 3, 0, 1))

        rr_head_q = w_head_q + self.r_r_bias

        rr_head_q_BD = ht.transpose_op(rr_head_q, (1, 2, 0, 3))
        r_head_k_BD = ht.transpose_op(r_head_k, (1, 2, 0)) 
        r_head_k_BD = ht.broadcast_shape_op(r_head_k_BD, (bsz, self.n_head, self.d_head, rlen), add_axes=(0))
        
        BD = ht.batch_matmul_op(rr_head_q_BD, r_head_k_BD)
        BD = ht.transpose_op(BD, (2, 3, 0, 1))
        BD_shape = (qlen, rlen, bsz, self.n_head)
        BD = self._rel_shift(BD, BD_shape)

        attn_score = AC + BD
        attn_score *= self.scale

        mask_value = np.finfo(np.float32).min

        if attn_mask is not None:
            attn_mask = ht.bool_op(attn_mask, 1)

            if len(attn_mask_shape) == 2:
                a, b = attn_mask_shape
                attn_mask = ht.array_reshape_op(attn_mask, (1, a, b, 1))
                attn_mask = ht.broadcast_shape_op(attn_mask, (qlen, klen, bsz, self.n_head))
                attn_score = ht.masked_fill_op(attn_score, attn_mask, mask_value)

            elif len(attn_mask_shape) == 3:
                a, b, c = attn_mask_shape
                attn_mask = ht.array_reshape_op(attn_mask, (a, b, c, 1))
                attn_mask = ht.broadcast_shape_op(attn_mask, (qlen, klen, bsz, self.n_head))
                attn_score = ht.masked_fill_op(attn_score, attn_mask, mask_value)                


        attn_prob = ht.transpose_op(attn_score, (2, 3, 0, 1))
        attn_prob = ht.softmax_op(attn_prob)
        attn_prob = self.dropatt(attn_prob)

        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        w_head_v = ht.transpose_op(w_head_v, (1, 2, 0, 3))
        attn_vec = ht.batch_matmul_op(attn_prob, w_head_v)
        attn_vec = ht.transpose_op(attn_vec, (2, 0, 1, 3))

        attn_vec_size = (qlen, bsz, self.n_head, self.d_head)
        attn_vec = ht.array_reshape_op(attn_vec,(-1, self.n_head * self.d_head))

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        attn_out = ht.array_reshape_op(attn_out, (qlen, bsz, -1))

        if self.pre_lnorm:
            outputs = [w + attn_out]
        else:
            outputs = [self.layer_norm(w + attn_out)]

        if output_attentions:
            outputs.append(attn_prob)

        return outputs        


class RelPartialLearnableDecoderLayer(object):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, layer_norm_epsilon=1e-5, name='RelPartialLearnableDecoderLayer', **kwargs):
        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, layer_norm_epsilon=layer_norm_epsilon, name=name+'.dec_attn', **kwargs
        )
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get("pre_lnorm"), layer_norm_epsilon=layer_norm_epsilon, name=name+'.pos_ff'
        )

    def __call__(self, dec_inp, dec_inp_shape, r, r_shape, dec_attn_mask=None, dec_attn_mask_shape=None, mems=None, mems_shape=None, head_mask=None, output_attentions=False):
        attn_outputs = self.dec_attn(
            dec_inp,
            r,
            dec_inp_shape,
            r_shape,
            attn_mask=dec_attn_mask,
            attn_mask_shape=dec_attn_mask_shape,
            mems=mems,
            mems_shape=mems_shape,
            head_mask=head_mask,
            output_attentions=output_attentions
        )

        ff_output = self.pos_ff(attn_outputs[0], dec_inp_shape)
        outputs = [ff_output] + attn_outputs[1:]

        return outputs


class AdaptiveEmbedding(object):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, sample_softmax=False, name='AdaptiveEmbedding'):
        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj**0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = list()
        self.emb_projs = list()

        if div_val == 1:
            self.emb_layers.append(ht.layers.Embedding(n_token, d_embed, name=name+".emb_layers.0"))
            if d_proj != d_embed:
                self.emb_projs.append(ht.init.random_normal((d_proj, d_embed), name=name+'.emb_projs.0'))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)
                self.emb_layers.append(ht.layers.Embedding(r_idx - l_idx, d_emb_i, name=name+'.emb_layers.'+str(i)))
                self.emb_projs.append(ht.init.random_normal((d_proj, d_emb_i), name=name+'.emb_projs.'+str(i)))

    def __call__(self, inp, inp_shape):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = ht.matmul_op(embed, self.emb_projs[0], trans_B=True)
        else:
            size = np.prod(inp_shape)
            inp_flat = ht.array_reshape_op(inp, (-1, ))
            emb_flat = ht.init.zeros([size, self.d_proj], name='emb_flat', trainable=False)

            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i_1 = ht.bool_op(inp_flat, l_idx, 4)
                mask_i_2 = ht.bool_op(inp_flat, r_idx, 1)
                mask_i = mask_i_1 + mask_i_2
                mask_i = ht.bool_op(mask_i, 2, 0)

                inp_i = ht.where_const_op(mask_i, inp_flat, 0.0) + (- l_idx)

                emb_i = self.emb_layers[i](inp_i)
                emb_i = ht.matmul_op(emb_i, self.emb_projs[i], trans_B=True)
                emb_flat += emb_i


            embed_shape = inp_shape + (self.d_proj,)
            embed = ht.array_reshape_op(emb_flat, embed_shape)
 
        embed= embed * self.emb_scale
        return embed


class TransfoXLModel(object):
    def __init__(self, config):
        self.config = config

        self.n_token = config.vocab_size
        self.d_embed = config.d_embed
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = config.d_head

        self.word_emb = AdaptiveEmbedding(
            config.vocab_size, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val, name='word_emb'
        )

        self.drop = ht.layers.DropOut(config.dropout)

        self.n_layer = config.n_layer
        self.mem_len = config.mem_len
        self.attn_type = config.attn_type

        if not config.untie_r:
            self.r_r_bias = ht.init.random_normal((self.n_head, self.d_head), name='r_r_bias')
            self.r_w_bias = ht.init.random_normal((self.n_head, self.d_head), name='r_w_bias')
        self.layers = list()
        if config.attn_type == 0:  
            for i in range(config.n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        config.n_head,
                        config.d_model,
                        config.d_head,
                        config.d_inner,
                        config.dropout,
                        dropatt=config.dropatt,
                        pre_lnorm=config.pre_lnorm,
                        r_w_bias=None if config.untie_r else self.r_w_bias,
                        r_r_bias=None if config.untie_r else self.r_r_bias,
                        layer_norm_epsilon=config.layer_norm_epsilon,
                        name = 'layers.'+str(i)
                    )
                )
        else: 
            raise NotImplementedError 

        self.same_length = config.same_length
        self.clamp_len = config.clamp_len

        if self.attn_type == 0: 
            self.pos_emb = PositionalEmbedding(self.d_model)
        else: 
            raise NotImplementedError  

    def init_mems(self, bsz):
        if self.mem_len > 0:
            mems = []
            mems_shape = []
            for i in range(self.n_layer):
                empty = ht.init.zeros((self.mem_len, bsz, self.config.d_model), trainable=False)
                mems.append(empty)
                mems_shape.append([self.mem_len, bsz, self.config.d_model])
            return mems, mems_shape
        else:
            return None, None

    def __call__(
        self,
        input_ids=None,
        input_shape=None,
        mems=None,
        mems_shape=None,
        head_mask=None,
        head_mask_shape=None,
        inputs_embeds=None,
        inputs_embeds_shape=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        word_emb_shape = None
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = ht.transpose_op(input_ids, (1, 0))
            bsz, qlen = input_shape
            word_emb_shape = (qlen, bsz, self.d_model)
        elif inputs_embeds is not None:
            inputs_embeds = ht.transpose_op(inputs_embeds, (1, 0, 2))
            qlen, bsz = inputs_embeds_shape[1], inputs_embeds_shape[0]
            word_emb_shape = (qlen, bsz, self.inputs_embeds_shape[2])
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if mems is None:
            mems, mems_shape = self.init_mems(bsz)

        if head_mask is not None:
            if len(head_mask_shape) == 1:
                head_mask = ht.broadcast_shape_op(head_mask, (self.n_layer, 1, 1, 1, -1), add_axes=(0, 1, 2, 3))
            elif len(head_mask_shape) == 2:
                head_mask = ht.array_reshape_op(head_mask, (head_mask_shape[0], 1, 1, 1, head_mask_shape[1]))
        else:
            head_mask = [None] * self.n_layer

        if inputs_embeds is not None:
            word_emb = inputs_embeds
        else:
            word_emb = self.word_emb(input_ids, (qlen, bsz))

        mlen = mems_shape[0][0] if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = ht.init.ones((qlen, klen), trainable=False)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (ht.triu_op(all_ones, 1 + mlen) + ht.tril_op(all_ones, -mask_shift_len))
            dec_attn_mask_shape = (qlen, klen, 1)
            dec_attn_mask = ht.array_reshape_op(dec_attn_mask, dec_attn_mask_shape)
        else:
            all_ones = ht.init.ones((qlen, klen), trainable=False)
            dec_attn_mask = ht.triu_op(all_ones, 1 + mlen) 
            dec_attn_mask_shape = (qlen, klen, 1)
            dec_attn_mask = ht.array_reshape_op(dec_attn_mask, dec_attn_mask_shape)

        hids = []
        attentions = [] if output_attentions else None
        
        if self.attn_type == 0:
            pos_seq = ht.arange_op(klen - 1, -1, -1.0)
            if self.clamp_len > 0:
                pos_seq = ht.clamp_op(pos_seq, max=self.clamp_len)
            pos_emb, pos_emb_shape = self.pos_emb(pos_seq, klen)
            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)
            
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                layer_outputs = layer(
                    core_out,
                    word_emb_shape,
                    pos_emb,
                    pos_emb_shape,
                    dec_attn_mask=dec_attn_mask,
                    dec_attn_mask_shape=dec_attn_mask_shape,
                    mems=mems_i,
                    mems_shape=mems_shape[i],
                    head_mask=head_mask[i],
                    output_attentions=output_attentions,
                )
                core_out = layer_outputs[0]
                if output_attentions:
                    attentions.append(layer_outputs[1])
        else:  
            raise NotImplementedError  

        core_out = self.drop(core_out)
        core_out = ht.transpose_op(core_out, (1, 0, 2))

        return [core_out]
