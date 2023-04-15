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
        dims = self._md_solver(
            self.embedding_args['alpha'], round_dim=self.embedding_args['round_dim'])
        assert max(dims) == self.embedding_dim
        emb = []
        for i, (nemb, cdim) in enumerate(zip(self.num_embed_separate, dims)):
            emb.append(self.get_single_embed_layer(nemb, cdim, f'MDEmb_{i}'))
        return emb
