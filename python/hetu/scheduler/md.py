from .base import EmbeddingTrainer
from ..layers import MDEmbedding
import numpy as np


class MDETrainer(EmbeddingTrainer):
    def assert_use_multi(self):
        assert self.use_multi == self.separate_fields == 1

    def _md_solver(self, alpha, mem_cap=None, round_dim=True, freq=None):
        # inherited from dlrm repo
        num_dim = self.embedding_dim
        indices, num_embed_fields = zip(
            *sorted(enumerate(self.num_embed_separate), key=lambda x: x[1]))
        num_embed_fields = np.array(num_embed_fields)
        if freq is not None:
            num_embed_fields /= freq[indices]
        if num_dim is not None:
            # use max dimension
            lamb = num_dim * (num_embed_fields[0] ** alpha)
        elif mem_cap is not None:
            # use memory capacity
            lamb = mem_cap / np.sum(num_embed_fields ** (1 - alpha))
        else:
            raise ValueError("Must specify either num_dim or mem_cap")
        d = lamb * (num_embed_fields ** (-alpha))
        d = np.round(np.maximum(d, 1))
        if round_dim:
            d = 2 ** np.round(np.log2(d))
        d = d.astype(int)
        undo_sort = [0] * len(indices)
        for i, v in enumerate(indices):
            undo_sort[v] = i
        return d[undo_sort]

    def get_dims_from_compress_rate(self):

        def evaluate(x):
            cur_dims = self._md_solver(x, round_dim=round_dim)
            cur_memory = sum(
                [nemb * ndim for nemb, ndim in zip(self.num_embed_separate, cur_dims)])
            return target_memory - cur_memory
        round_dim = self.embedding_args['round_dim']
        target_memory = self.num_embed * self.embedding_dim * self.compress_rate
        left, right = self.binary_search(0., 1., evaluate, 1e-3)
        # test how close is left
        dims = self._md_solver(left, round_dim=round_dim)
        real_compress_rate = sum(
            [nemb * ndim for nemb, ndim in zip(self.num_embed_separate, dims)]) / self.num_embed / self.embedding_dim
        if real_compress_rate < self.compress_rate + 1e-3:
            alpha = left
        else:
            alpha = right
            dims = self._md_solver(alpha, round_dim=round_dim)
            real_compress_rate = sum(
                [nemb * ndim for nemb, ndim in zip(self.num_embed_separate, dims)]) / self.num_embed / self.embedding_dim
        self.log_func(
            f'Get alpha {alpha} with compression rate {real_compress_rate}({self.compress_rate})')
        return dims

    def get_single_embed_layer(self, nemb, cdim, name):
        return MDEmbedding(
            nemb,
            cdim,
            self.embedding_dim,
            initializer=self.initializer,
            name=name,
            ctx=self.ectx,
        )

    def get_embed_layer(self):
        dims = self.get_dims_from_compress_rate()
        assert max(dims) == self.embedding_dim
        emb = []
        for i, (nemb, cdim) in enumerate(zip(self.num_embed_separate, dims)):
            emb.append(self.get_single_embed_layer(nemb, cdim, f'MDEmb_{i}'))
        return emb
