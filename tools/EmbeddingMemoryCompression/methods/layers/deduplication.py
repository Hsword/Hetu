import hetu as ht
from hetu.layers import Embedding
import numpy as np


class DedupEmbedding(Embedding):
    def __init__(self, emb, remap_indices, nemb_per_block, trainable=True, name='embedding', ctx=None):
        self.num_embeddings = emb.shape[0]
        self.embedding_dim = emb.shape[1]
        self.nemb_per_block = nemb_per_block
        self.name = name
        self.ctx = ctx
        embedding_table = ht.array(emb, dtype=np.float32, ctx=ctx)
        self.embedding_table = ht.placeholder_op(self.name, value=embedding_table, dtype=np.float32, trainable=trainable)
        # self.decompressed_nrow = len(remap_indices)
        remap_indices = ht.array(remap_indices.reshape(
            (-1, 1)), dtype=np.int32, ctx=self.ctx)
        self.remap_indices = ht.placeholder_op(
            f'{name}_remap', value=remap_indices, dtype=np.int32, trainable=False)

    def __call__(self, x):
        with ht.context(self.ctx):
            remap = ht.embedding_lookup_op(self.remap_indices, ht.div_hash_op(x, self.nemb_per_block))
            real_indices = ht.add_op(ht.mul_byconst_op(ht.reshape_to_op(remap, x), self.nemb_per_block), ht.mod_hash_op(x, self.nemb_per_block))
            results = ht.embedding_lookup_op(self.embedding_table, real_indices)
            return results

    def __repr__(self):
        return f'{self.name}({self.num_embeddings},{self.embedding_dim})'
