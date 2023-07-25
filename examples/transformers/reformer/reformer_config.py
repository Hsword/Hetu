
""" Reformer configuration"""
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Union


class ReformerConfig(object):
    def __init__(
        self,
        attention_head_size=64,
        attn_layers=["local", "lsh", "local", "lsh", "local", "lsh"],
        axial_norm_std=1.0,
        axial_pos_embds=True,
        axial_pos_shape=[8, 16],
        axial_pos_embds_dim=[64, 192],
        chunk_size_lm_head=0,
        eos_token_id=2,
        feed_forward_size=512,
        hash_seed=None,
        hidden_act="relu",
        hidden_dropout_prob=0.0,
        hidden_size=256,
        initializer_range=0.02,
        is_decoder=False,
        layer_norm_eps=1e-12,
        local_num_chunks_before=1,
        local_num_chunks_after=0,
        local_attention_probs_dropout_prob=0.0,
        local_attn_chunk_length=256,
        lsh_attn_chunk_length=256,
        lsh_attention_probs_dropout_prob=0.0,
        lsh_num_chunks_before=1,
        lsh_num_chunks_after=0,
        max_position_embeddings=4096,
        num_attention_heads=12,
        num_buckets=None,
        num_hashes=1,
        pad_token_id=0,
        vocab_size=320,
        tie_word_embeddings=False,
        use_cache=True,
        classifier_dropout=None,
        chunk_size_feed_forward=0,
        output_attentions=False,
        output_hidden_states=False,
        **kwargs
    ):
        self.hash_seed = hash_seed
        self.vocab_size = vocab_size
        self.attention_head_size = attention_head_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hashes = num_hashes
        self.num_hidden_layers = len(attn_layers)
        self.num_buckets = tuple(num_buckets) if isinstance(num_buckets, list) else num_buckets
        self.lsh_attn_chunk_length = lsh_attn_chunk_length
        self.local_attn_chunk_length = local_attn_chunk_length
        self.lsh_num_chunks_after = lsh_num_chunks_after
        self.lsh_num_chunks_before = lsh_num_chunks_before
        self.local_num_chunks_after = local_num_chunks_after
        self.local_num_chunks_before = local_num_chunks_before
        self.hidden_act = hidden_act
        self.feed_forward_size = feed_forward_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.lsh_attention_probs_dropout_prob = lsh_attention_probs_dropout_prob
        self.local_attention_probs_dropout_prob = local_attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.axial_pos_embds = axial_pos_embds
        self.axial_pos_shape = tuple(axial_pos_shape)
        self.axial_pos_embds_dim = tuple(axial_pos_embds_dim)
        self.axial_norm_std = axial_norm_std
        self.chunk_size_lm_head = chunk_size_lm_head
        self.attn_layers = attn_layers
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.is_decoder = is_decoder
        self.tie_word_embeddings = tie_word_embeddings
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states