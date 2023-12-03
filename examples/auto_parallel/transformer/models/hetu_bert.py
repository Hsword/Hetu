"""PyTorch BERT model."""
import numpy as np

import hetu as ht
import hetu.layers as htl


def ACT2FN(act_func):
    if act_func == "relu":
        return htl.Relu()
    else:
        assert(False)


class BertEmbeddings(object):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, name):
        super().__init__()
        self.word_embeddings = htl.Embedding(
            config.vocab_size, config.hidden_size, name=name+'_word_embeddings_weight')
        self.position_embeddings = htl.Embedding(
            config.max_position_embeddings, config.hidden_size, name=name+'_position_embeddings_weight')
        self.token_type_embeddings = htl.Embedding(
            config.type_vocab_size, config.hidden_size, name=name+'_token_type_embeddings_weight')

        self.LayerNorm = htl.LayerNorm(
            config.hidden_size, eps=1e-12, name=name+'_LayerNorm')
        self.dropout = htl.DropOut(config.hidden_dropout_prob)

    def __call__(
        self,
        input_ids,  # (batch_size*seq_len)
        token_type_ids,  # (batch_size*seq_len)
        position_ids,  # (batch_size*seq_len)
    ):
        words_embeddings = self.word_embeddings(input_ids)
        # (batch_size*seq_len, hidden_size)
        position_embeddings = self.position_embeddings(position_ids)
        # (batch_size*seq_len, hidden_size)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # (batch_size*seq_len, hidden_size)
        embeddings = ht.sum_op(
            [words_embeddings, position_embeddings, token_type_embeddings])
        # (batch_size*seq_len, hidden_size)
        embeddings = self.LayerNorm(embeddings)
        # (batch_size*seq_len, hidden_size)
        embeddings = self.dropout(embeddings)
        # (batch_size*seq_len, hidden_size)
        return embeddings


class BertSelfAttention(object):
    def __init__(self, config, name):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.seq_len = config.max_position_embeddings
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = htl.Linear(
            config.hidden_size, self.all_head_size, initializer=ht.init.GenXavierNormal(), name=name+'_query')
        self.key = htl.Linear(
            config.hidden_size, self.all_head_size, initializer=ht.init.GenXavierNormal(), name=name+'_key')
        self.value = htl.Linear(
            config.hidden_size, self.all_head_size, initializer=ht.init.GenXavierNormal(), name=name+'_value')

        self.dropout = htl.DropOut(config.attention_probs_dropout_prob)

        self.attention = htl.MultiHeadAttention(
            config.hidden_size, self.num_attention_heads, self.seq_len, config.attention_probs_dropout_prob)

    def __call__(
        self,
        hidden_states,  # (batch_size*seq_len, hidden_size)
        attention_mask,  # (batch_size, 1, 1, seq_len)
    ):
        mixed_query_layer = self.query(hidden_states)
        # (batch_size*seq_len, hidden_size)
        mixed_key_layer = self.key(hidden_states)
        # (batch_size*seq_len, hidden_size)
        mixed_value_layer = self.value(hidden_states)
        # (batch_size*seq_len, hidden_size)

        context_layer = self.attention(
            mixed_query_layer, mixed_key_layer, mixed_value_layer, attention_mask)
        # (batch_size*seq_len, hidden_size)
        return context_layer


class BertSelfOutput(object):
    def __init__(self, config, name):
        super().__init__()
        self.dense = htl.Linear(
            config.hidden_size, config.hidden_size, initializer=ht.init.GenXavierNormal(), name=name+'_dense')
        self.LayerNorm = htl.LayerNorm(
            config.hidden_size, eps=1e-12, name=name+'_LayerNorm')
        self.dropout = htl.DropOut(config.hidden_dropout_prob)

    def __call__(
        self,
        hidden_states,  # (batch_size*seq_len, hidden_size)
        input_tensor,  # (batch_size*seq_len, hidden_size)
    ):
        hidden_states = self.dense(hidden_states)
        # (batch_size*seq_len, hidden_size)
        hidden_states = self.dropout(hidden_states)
        # (batch_size*seq_len, hidden_size)
        hidden_states = self.LayerNorm(
            ht.sum_op([hidden_states, input_tensor]))
        # (batch_size*seq_len, hidden_size)
        return hidden_states


class BertAttention(object):
    def __init__(self, config, name):
        super().__init__()
        self.self = BertSelfAttention(config, name=name+'_self')
        self.output = BertSelfOutput(config, name=name+'_output')

    def __call__(
        self,
        hidden_states,  # (batch_size*seq_len, hidden_size)
        attention_mask,  # (batch_size, 1, 1, seq_len)
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
        )
        # (batch_size*seq_len, hidden_size)
        attention_output = self.output(self_outputs, hidden_states)
        # (batch_size*seq_len, hidden_size)
        return attention_output


class BertIntermediate(object):
    def __init__(self, config, name):
        super().__init__()
        self.dense = htl.Linear(
            config.hidden_size, config.intermediate_size, activation=ACT2FN(config.hidden_act), initializer=ht.init.GenXavierNormal(), name=name+'_dense')

    def __call__(
        self,
        hidden_states,  # (batch_size*seq_len, hidden_size)
    ):
        hidden_states = self.dense(hidden_states)
        # (batch_size*seq_len, 4*hidden_size)
        return hidden_states


class BertOutput(object):
    def __init__(self, config, name):
        super().__init__()
        self.dense = htl.Linear(
            config.intermediate_size, config.hidden_size, initializer=ht.init.GenXavierNormal(), name=name+'_dense')
        self.LayerNorm = htl.LayerNorm(
            config.hidden_size, eps=1e-12, name=name+'_LayerNorm')
        self.dropout = htl.DropOut(config.hidden_dropout_prob)

    def __call__(
        self,
        hidden_states,  # (batch_size*seq_len, 4*hidden_size)
        input_tensor,  # (batch_size*seq_len, hidden_size)
    ):
        hidden_states = self.dense(hidden_states)
        # (batch_size*seq_len, hidden_size)
        hidden_states = self.dropout(hidden_states)
        # (batch_size*seq_len, hidden_size)
        hidden_states = self.LayerNorm(
            ht.sum_op([hidden_states, input_tensor]))
        # (batch_size*seq_len, hidden_size)
        return hidden_states


class BertLayer(object):
    def __init__(self, config, name):
        super().__init__()
        self.attention = BertAttention(config, name=name+'_attention')
        self.intermediate = BertIntermediate(config, name=name+'_intermediate')
        self.output = BertOutput(config, name=name+'_output')

    def __call__(
        self,
        hidden_states,  # (batch_size*seq_len, hidden_size)
        attention_mask,  # (batch_size, 1, 1, seq_len)
    ):
        attention_output = self.attention(hidden_states, attention_mask)
        # (batch_size*seq_len, hidden_size)
        intermediate_output = self.intermediate(attention_output)
        # (batch_size*seq_len, 4*hidden_size)
        layer_output = self.output(intermediate_output, attention_output)
        # (batch_size*seq_len, hidden_size)
        return layer_output


class BertEncoder(object):
    def __init__(self, config, name):
        super().__init__()
        self.config = config
        self.layer = [BertLayer(config, name=name+'_layer_{}'.format(i))
                      for i in range(config.num_hidden_layers)]

    def __call__(
        self,
        hidden_states,  # (batch_size*seq_len, hidden_size)
        attention_mask,  # (batch_size, 1, 1, seq_len)
    ):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
        # (batch_size*seq_len, hidden_size)
        return hidden_states


class BertPooler(object):
    def __init__(self, config, name):
        super().__init__()
        self.slice = htl.BatchSplitOnlyLayer(
            htl.Sequence(
                htl.Reshape(
                    (-1, config.max_position_embeddings, config.hidden_size)),
                htl.Slice((0, 0, 0), (-1, 1, -1)),
                htl.Reshape((-1, config.hidden_size)),
            )
        )
        self.dense = htl.Linear(config.hidden_size, config.hidden_size,
                                initializer=ht.init.GenXavierNormal(), activation=ht.tanh_op, name=name+'_dense')

    def __call__(
        self,
        hidden_states,  # (batch_size*seq_len, hidden_size)
    ):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = self.slice(hidden_states)
        # (batch_size, hidden_size)
        pooled_output = self.dense(first_token_tensor)
        # (batch_size, hidden_size)
        return pooled_output


class BertPredictionHeadTransform(object):
    def __init__(self, config, name):
        super().__init__()
        self.dense = htl.Linear(
            config.hidden_size, config.hidden_size, activation=ACT2FN(config.hidden_act), initializer=ht.init.GenXavierNormal(), name=name+'_dense')
        self.LayerNorm = htl.LayerNorm(
            config.hidden_size, eps=1e-12, name=name+'_LayerNorm')

    def __call__(
        self,
        hidden_states,  # (batch_size*seq_len, hidden_size)
    ):
        hidden_states = self.dense(hidden_states)
        # (batch_size*seq_len, hidden_size)
        hidden_states = self.LayerNorm(hidden_states)
        # (batch_size*seq_len, hidden_size)
        return hidden_states


class BertLMPredictionHead(object):
    def __init__(self, config, bert_model_embedding_weights, name):
        super().__init__()
        self.transform = BertPredictionHeadTransform(
            config, name=name+'_transform')

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        if bert_model_embedding_weights is None:
            self.decoder = htl.Linear(
                config.hidden_size, config.vocab_size, weight_transpose=True, name=name+'_decoder')
        else:
            self.decoder = htl.Linear(
                config.hidden_size, config.vocab_size, initializer=bert_model_embedding_weights, weight_transpose=True, name=name+'_decoder')

    def __call__(
        self,
        hidden_states,  # (batch_size*seq_len, hidden_size)
    ):
        hidden_states = self.transform(hidden_states)
        # (batch_size*seq_len, hidden_size)
        hidden_states = self.decoder(hidden_states)
        # (batch_size*seq_len, vocab_size)
        return hidden_states


class BertPreTrainingHeads(object):
    def __init__(self, config, bert_model_embedding_weights, name):
        super().__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights, name=name+'_predictions')
        self.seq_relationship = htl.Linear(
            config.hidden_size, 2, initializer=ht.init.GenXavierNormal(), name=name+'_seq_relationship')

    def __call__(
        self,
        sequence_output,  # (batch_size*seq_len, hidden_size)
        pooled_output,  # (batch_size, hidden_size)
    ):
        prediction_scores = self.predictions(sequence_output)
        # (batch_size*seq_len, vocab_size)
        seq_relationship_score = self.seq_relationship(pooled_output)
        # (batch_size, 2)
        return prediction_scores, seq_relationship_score


class BertModel(object):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the __call__ pass.
    """

    def __init__(self, config, name):
        super().__init__()
        self.config = config
        self.target_shape = (-1, 1, 1, config.max_position_embeddings)

        self.embeddings = BertEmbeddings(config, name=name+'_embeddings')
        self.encoder = BertEncoder(config, name=name+'_encoder')

        self.pooler = BertPooler(config, name=name+'_pooler')

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids):
        extended_attention_mask = (attention_mask+(-1.0)) * 10000.0
        extended_attention_mask = ht.array_reshape_op(
            extended_attention_mask, self.target_shape)

        embedding_output = self.embeddings(
            input_ids, token_type_ids, position_ids)
        # (batch_size*seq_len, hidden_size)
        sequence_output = self.encoder(
            embedding_output, extended_attention_mask)
        # (batch_size*seq_len, hidden_size)
        pooled_output = self.pooler(sequence_output)
        # (batch_size, hidden_size)
        return sequence_output, pooled_output


class BertForPreTraining(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel(config, name='bert')
        if config.share_embedding:
            table = self.bert.embeddings.word_embeddings.embedding_table
        else:
            table = None
        self.cls = BertPreTrainingHeads(
            config, table, name='cls')

    def __call__(self,
                 input_ids,  # (batch_size*seq_len,)
                 attention_mask,  # (batch_size, seq_len)
                 token_type_ids,  # (batch_size*seq_len,)
                 position_ids,  # (batch_size*seq_len,)
                 masked_lm_labels=None,  # (batch_size*seq_len,)
                 next_sentence_label=None,  # (batch_size,)
                 ):

        sequence_output, pooled_output = self.bert(
            input_ids, attention_mask, token_type_ids, position_ids)
        # (batch_size*seq_len, hidden_size), (batch_size, hidden_size)

        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)
        # (batch_size*seq_len, vocab_size), (batch_size, 2)

        return_op = [prediction_scores, seq_relationship_score]
        if masked_lm_labels is not None and next_sentence_label is not None:
            masked_lm_loss = ht.softmaxcrossentropy_sparse_op(
                prediction_scores, masked_lm_labels)
            # (batch_size*seq_len,)
            next_sentence_loss = ht.softmaxcrossentropy_sparse_op(
                seq_relationship_score, next_sentence_label)
            # (batch_size,)
            return_op += [masked_lm_loss, next_sentence_loss]
        return return_op
