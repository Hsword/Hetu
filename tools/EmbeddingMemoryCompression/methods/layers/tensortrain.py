import hetu as ht
from hetu.layers import Embedding
import numpy as np


class TensorTrainEmbedding(Embedding):
    def __init__(self, decomp_nemb, decomp_ndim, rank, name='embedding', ctx=None):
        self.num_tables = len(decomp_nemb)
        assert len(decomp_ndim) == self.num_tables
        self.decomp_nemb = decomp_nemb
        self.decomp_ndim = decomp_ndim
        self.ranks = [1, rank, rank, 1]
        self.name = name
        self.ctx = ctx
        cur_shapes = []
        for i in range(self.num_tables):
            nrow = decomp_nemb[i]
            ndim = decomp_ndim[i]
            prerank = self.ranks[i]
            postrank = self.ranks[i+1]
            ncol = prerank * ndim * postrank
            cur_shapes.append((nrow, ncol))
        ttcore_initializer = ht.init.GenReversedTruncatedNormal(
            stddev=1 / ((np.sqrt(1 / 3 * np.prod(decomp_nemb))) ** (1/3)))
        self.tt_cores = tuple(ttcore_initializer(
            shape=sh, name=f'{name}_{i}') for i, sh in enumerate(cur_shapes))

    def __call__(self, x):
        indices = x
        accum_embed = None
        accum_dim = 1
        for i in range(self.num_tables):
            if i == self.num_tables - 1:
                cur_ind = indices
            else:
                cur_ind = ht.mod_hash_op(indices, self.decomp_nemb[i])
                indices = ht.div_hash_op(indices, self.decomp_nemb[i])
            partial_embed = ht.embedding_lookup_op(self.tt_cores[i], cur_ind)
            if i == 0:
                accum_embed = partial_embed
            else:
                accum_embed = ht.array_reshape_op(
                    accum_embed, (-1, accum_dim, self.ranks[i]))
                partial_embed = ht.array_reshape_op(
                    partial_embed, (-1, self.ranks[i], self.decomp_ndim[i] * self.ranks[i+1]))
                accum_embed = ht.batch_matmul_op(
                    accum_embed, partial_embed)
            accum_dim *= self.decomp_ndim[i]
        accum_embed = ht.array_reshape_op(
            accum_embed, (-1, accum_dim))
        return accum_embed

    def __repr__(self):
        return f'{self.name}({self.ranks[1]})'
