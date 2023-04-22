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


    def _get_alpha(self):

        def multi_evaluate(x):
            num_fields=[1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683,8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572]
            cur_memory = 0
            for item in num_fields:
                if self.embedding_dim*(num_fields[0]**x)*(item**(1-x))>=1:
                    cur_memory+=self.embedding_dim*(num_fields[0]**x)*(item**(1-x))
                else:
                    cur_memory+=1
            return cur_memory - target_memory
        target_memory = self.num_embed * self.embedding_dim * self.compress_rate
        evaluate = multi_evaluate
        res = self.md_binary_search(0, 1, evaluate)
        self.log_func(
            f'Alpha {res} given compression rate {self.compress_rate}.')
        return res


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
        alpha=self._get_alpha()
        dims = self._md_solver(
            alpha, round_dim=self.embedding_args['round_dim'])
        assert max(dims) == self.embedding_dim
        emb = []
        for i, (nemb, cdim) in enumerate(zip(self.num_embed_separate, dims)):
            emb.append(self.get_single_embed_layer(nemb, cdim, f'MDEmb_{i}'))
        return emb
