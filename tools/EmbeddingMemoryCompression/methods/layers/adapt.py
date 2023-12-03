import hetu as ht
from hetu.layers import Embedding
import numpy as np


class AdaptiveEmbedding(Embedding):
    def __init__(self, num_freq_emb, num_rare_emb, remap_indices, embedding_dim, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        # DeepRec Adaptive Embedding
        self.num_freq_emb = num_freq_emb
        self.num_rare_emb = num_rare_emb
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.freq_emb = initializer(
            shape=(num_freq_emb, embedding_dim), name=f'{name}_freq', ctx=ctx)
        self.rare_emb = initializer(
            shape=(num_rare_emb, embedding_dim), name=f'{name}_rare', ctx=ctx)
        remap_indices = ht.array(remap_indices.reshape(
            (-1, 1)), dtype=np.int32, ctx=self.ctx)
        self.remap_indices = ht.placeholder_op(
            f'{name}_remap', value=remap_indices, dtype=np.int32, trainable=False)

    def __call__(self, x):
        with ht.context(self.ctx):
            remap = ht.embedding_lookup_op(self.remap_indices, x)
            high_freq = ht.embedding_lookup_op(self.freq_emb, remap)
            low_freq_inds = ht.mod_hash_negative_op(
                remap, self.num_rare_emb)
            low_freq = ht.embedding_lookup_op(self.rare_emb, low_freq_inds)
            result = ht.add_op(high_freq, low_freq)
            return ht.array_reshape_op(result, (-1, self.embedding_dim))

    def __repr__(self):
        return f'{self.name}({self.num_freq_emb},{self.num_rare_emb},{self.embedding_dim})'
