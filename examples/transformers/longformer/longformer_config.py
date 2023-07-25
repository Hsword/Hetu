""" Longformer configuration"""
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Union


class LongformerConfig(object):
    def __init__(
        self,
        attention_window=512,
        sep_token_id=2,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=4098,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        chunk_size_feed_forward=0,
        output_attentions=False,
        output_hidden_states=False,
        is_decoder=False,
        num_labels=2,
        **kwargs
    ):
        self.attention_window = attention_window
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.is_decoder = is_decoder
        self.num_labels = num_labels
