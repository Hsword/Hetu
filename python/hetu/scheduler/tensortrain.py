from .base import EmbeddingTrainer
from ..layers import TensorTrainEmbedding
import math


class TTEmbTrainer(EmbeddingTrainer):
    def _get_decomp_dim(self):
        embedding_dim = self.embedding_dim
        assert embedding_dim >= 8 and embedding_dim & (embedding_dim - 1) == 0
        decomp_ndim = [2, 2, 2]
        idx = 2
        embedding_dim = embedding_dim // 8
        while embedding_dim != 1:
            decomp_ndim[idx] *= 2
            embedding_dim = embedding_dim // 2
            idx = (idx - 1) % 3
        return decomp_ndim

    def _get_decomp_emb(self, nemb):
        n1 = math.ceil(nemb ** (1/3))
        n2 = math.ceil((nemb / n1) ** (1/2))
        n3 = math.ceil(nemb / n1 / n2)
        return [n3, n2, n1]

    def _get_single_memory(self, nemb, ndim, rank):
        return (nemb[0] * ndim[0] + nemb[1] * ndim[1] * rank + nemb[2] * ndim[2]) * rank

    def _get_rank(self, decomp_nembs, decomp_ndim):
        target_memory = self.num_embed * self.embedding_dim * self.compress_rate

        def multi_evaluate(x):
            memory = 0
            for nemb, dn in zip(self.num_embed_separate, decomp_nembs):
                orimem = nemb * self.embedding_dim
                newmem = self._get_single_memory(dn, decomp_ndim, x)
                memory += min(orimem, newmem)
            return memory - target_memory

        def single_evaluate(x):
            memory = self._get_single_memory(decomp_nembs, decomp_ndim, x)
            return memory - target_memory

        if self.use_multi:
            evaluate = multi_evaluate
        else:
            evaluate = single_evaluate

        res = self.binary_search(0, 1000, evaluate)
        res = math.floor(res[1])
        if evaluate(res) > 0:
            res -= 1
        print(f'Rank {res} given compression rate {self.compress_rate}.')
        return res

    def get_single_embed_layer(self, decomp_nemb, decomp_ndim, rank, name):
        return TensorTrainEmbedding(
            decomp_nemb,
            decomp_ndim,
            rank,
            ttcore_initializer=self.embedding_args['ttcore_initializer'],
            name=name,
            ctx=self.ectx
        )

    def get_embed_layer(self):
        decomp_ndim = self._get_decomp_dim()
        if self.use_multi:
            decomp_nembs = [self._get_decomp_emb(
                nemb) for nemb in self.num_embed_separate]
        else:
            decomp_nembs = self._get_decomp_emb(self.num_embed)
        rank = self._get_rank(decomp_nembs, decomp_ndim)
        all_size = 0
        if self.use_multi:
            emb = []
            for i, nemb in enumerate(self.num_embed_separate):
                newmem = self._get_single_memory(
                    decomp_nembs[i], decomp_ndim, rank)
                orimem = nemb * self.embedding_dim
                if newmem < orimem:
                    all_size += newmem
                    emb.append(self.get_single_embed_layer(
                        decomp_nembs[i], decomp_ndim, rank, f'TTEmb({self.compress_rate})_{i}'))
                else:
                    all_size += orimem
                    emb.append(super().get_single_embed_layer(
                        nemb, f'Embedding_{i}'))
        else:
            all_size = self._get_single_memory(decomp_nembs, decomp_ndim, rank)
            emb = self.get_single_embed_layer(
                decomp_nembs, decomp_ndim, rank, f'TTEmb({self.compress_rate})')
        self.log_func(
            f'Real compress rate: {all_size / self.num_embed / self.embedding_dim}')
        return emb
