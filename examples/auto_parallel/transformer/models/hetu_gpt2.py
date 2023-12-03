"""PyTorch OpenAI GPT-2 model."""

import numpy as np
import hetu as ht
import hetu.layers as htl


class GPT2Attention(object):
    def __init__(self, config, name):
        super().__init__()

        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.seq_len = config.n_positions
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.q_linear = htl.Linear(
            self.embed_dim, self.embed_dim, name=name+'_q_linear')
        self.k_linear = htl.Linear(
            self.embed_dim, self.embed_dim, name=name+'_k_linear')
        self.v_linear = htl.Linear(
            self.embed_dim, self.embed_dim, name=name+'_v_linear')
        self.attention = htl.MultiHeadAttention(
            self.embed_dim, self.num_heads, self.seq_len, config.attn_pdrop, causal_mask=True)
        self.c_proj = htl.Linear(
            self.embed_dim, self.embed_dim, name=name+'_c_proj')

        self.resid_dropout = htl.DropOut(config.resid_pdrop)

    def __call__(
        self,
        hidden_states,  # (batch_size*seq_len, embed_dim)
        attention_mask=None,  # (batch_size, 1, 1, seq_len)
    ):
        query = self.q_linear(hidden_states)
        # (batch_size*seq_len, embed_dim)
        key = self.k_linear(hidden_states)
        # (batch_size*seq_len, embed_dim)
        value = self.v_linear(hidden_states)
        # (batch_size*seq_len, embed_dim)

        attn_output = self.attention(query, key, value, attention_mask)
        # (batch_size*seq_len, embed_dim)

        attn_output = self.c_proj(attn_output)
        # (batch_size*seq_len, embed_dim)
        attn_output = self.resid_dropout(attn_output)
        # (batch_size*seq_len, embed_dim)

        return attn_output


class GPT2MLP(object):
    def __init__(self, intermediate_size, config, name):
        super().__init__()
        embed_dim = config.n_embd
        self.c_fc = htl.Linear(embed_dim, intermediate_size,
                               activation=config.activation_function, name=name+'_c_fc')
        self.c_proj = htl.Linear(
            intermediate_size, embed_dim, name=name+'_c_proj')
        self.dropout = htl.DropOut(config.resid_pdrop)

    def __call__(self, hidden_states):
        # (batch_size*seq_len, embed_dim)
        hidden_states = self.c_fc(hidden_states)
        # (batch_size*seq_len, 4*embed_dim)
        hidden_states = self.c_proj(hidden_states)
        # (batch_size*seq_len, embed_dim)
        hidden_states = self.dropout(hidden_states)
        # (batch_size*seq_len, embed_dim)
        return hidden_states


class GPT2Block(object):
    def __init__(self, config, name):
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = htl.LayerNorm(
            hidden_size, eps=config.layer_norm_epsilon, name=name+'_ln_1')
        self.attn = GPT2Attention(config, name=name+'_attn')
        self.ln_2 = htl.LayerNorm(
            hidden_size, eps=config.layer_norm_epsilon, name=name+'_ln_2')

        self.mlp = GPT2MLP(inner_dim, config, name=name+'_mlp')

    def __call__(
        self,
        hidden_states,  # (batch_size*seq_len, embed_dim)
        attention_mask=None,  # (batch_size, 1, 1, seq_len)
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        # (batch_size*seq_len, embed_dim)
        attn_output = self.attn(
            hidden_states,
            attention_mask=attention_mask,
        )
        # (batch_size*seq_len, embed_dim)

        # residual connection
        hidden_states = ht.sum_op([residual, attn_output])
        # (batch_size*seq_len, embed_dim)

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        # (batch_size*seq_len, embed_dim)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # (batch_size*seq_len, embed_dim)

        # residual connection
        hidden_states = ht.sum_op([residual, feed_forward_hidden_states])
        # (batch_size*seq_len, embed_dim)

        return hidden_states


class GPT2Model(object):
    def __init__(self, config, name):
        self.embed_dim = config.n_embd
        self.seq_len = config.n_positions

        self.wte = htl.Embedding(
            config.vocab_size, self.embed_dim, name=name+'_wte_weight')
        self.wpe = htl.Embedding(
            config.n_positions, self.embed_dim, name=name+'_wpe_weight')

        self.drop = htl.DropOut(config.embd_pdrop)
        self.h = [GPT2Block(config, name=name+'_h_{}'.format(i))
                  for i in range(config.n_layer)]
        self.ln_f = htl.LayerNorm(
            self.embed_dim, eps=config.layer_norm_epsilon, name=name+'_ln_f')

    def __call__(
        self,
        input_ids,  # (batch_size*seq_len, )
        attention_mask=None,  # (batch_size, seq_len)
        position_ids=None,  # (batch_size, seq_len)
    ):

        # GPT2Attention mask.
        if attention_mask is not None:
            attention_mask = (attention_mask+(-1.0)) * 10000.0
            attention_mask = ht.array_reshape_op(
                attention_mask, [-1, 1, 1, self.seq_len])
            # (batch_size, 1, 1, seq_len)

        inputs_embeds = self.wte(input_ids)
        # (batch_size*seq_len, embed_dim)
        position_embeds = self.wpe(position_ids)
        # (batch_size*seq_len, embed_dim)
        hidden_states = ht.sum_op(
            [inputs_embeds, position_embeds])
        # (batch_size*seq_len, embed_dim)

        hidden_states = self.drop(hidden_states)
        # (batch_size*seq_len, embed_dim)

        for block in self.h:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
            )
        # (batch_size*seq_len, embed_dim)

        hidden_states = self.ln_f(hidden_states)
        # (batch_size*seq_len, embed_dim)

        return hidden_states


class GPT2LMHeadModel(object):

    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2Model(config, name='transformer')
        if config.share_embedding is None:
            self.lm_head = htl.Linear(
                config.n_embd, config.vocab_size, weight_transpose=True, bias=False)
        else:
            self.lm_head = htl.Linear(config.n_embd, config.vocab_size,
                                      initializer=self.transformer.wte.embedding_table, weight_transpose=True, bias=False)
        self.logits_handle = htl.ReserveSplitLayer(htl.Sequence(
            htl.Reshape((-1, config.n_positions, config.vocab_size)),
            htl.Slice((0, 0, 0), (-1, config.n_positions - 1, -1)),
            htl.Reshape((-1, config.vocab_size)),
        ))
        self.labels_handle = htl.BatchSplitOnlyLayer(htl.Sequence(
            htl.Reshape((-1, config.n_positions)),
            htl.Slice((0, 1), (-1, -1)),
            htl.Reshape((-1,))
        ))

    def __call__(
        self,
        input_ids=None,  # (batch_size*seq_len,)
        attention_mask=None,  # (batch_size, seq_len)
        position_ids=None,  # (batch_size*seq_len,)
        labels=None,  # (batch_size*seq_len,)
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        hidden_states = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        # (batch_size*seq_len, embed_dim)

        lm_logits = self.lm_head(hidden_states)
        # (batch_size*seq_len, vocab_size)

        loss = None
        if labels is not None:
            shift_logits = self.logits_handle(lm_logits)
            # (batch_size, seq_len-1, vocab_size)

            shift_labels = self.labels_handle(labels)
            # (batch_size, seq_len-1)

            # Flatten the tokens
            loss = ht.softmaxcrossentropy_sparse_op(shift_logits, shift_labels)
            # (batch_size, seq_len-1)
            loss = ht.reduce_mean_op(loss, (0,))
            # (1,)

        return loss, lm_logits
