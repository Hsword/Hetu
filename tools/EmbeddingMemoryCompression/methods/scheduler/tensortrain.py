from .base import EmbeddingTrainer
from .compressor import Compressor
from ..layers import TensorTrainEmbedding
import math
import numpy as np


class TTEmbTrainer(EmbeddingTrainer):
    def _get_decomp_dim(self):
        embedding_dim = self.embedding_dim
        if embedding_dim & (embedding_dim - 1) == 0:
            assert embedding_dim >= 8
            decomp_ndim = [2, 2, 2]
            idx = 2
            embedding_dim = embedding_dim // 8
            while embedding_dim != 1:
                decomp_ndim[idx] *= 2
                embedding_dim = embedding_dim // 2
                idx = (idx - 1) % 3
        else:
            n1 = math.ceil(embedding_dim ** (1/3))
            while embedding_dim % n1 != 0:
                n1 -= 1
            rest = embedding_dim // n1
            n2 = math.ceil(rest ** (1/2))
            while rest % n2 != 0:
                n2 -= 1
            n3 = rest // n2
            decomp_ndim = sorted([n1, n2, n3])
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
                if nemb > threshold:
                    newmem = self._get_single_memory(dn, decomp_ndim, x)
                    memory += min(orimem, newmem)
                else:
                    memory += orimem
            return memory - target_memory

        def single_evaluate(x):
            memory = self._get_single_memory(decomp_nembs, decomp_ndim, x)
            return memory - target_memory

        threshold = self.embedding_args['threshold']
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
            threshold = self.embedding_args['threshold']
            for i, nemb in enumerate(self.num_embed_separate):
                newmem = self._get_single_memory(
                    decomp_nembs[i], decomp_ndim, rank)
                orimem = nemb * self.embedding_dim
                if nemb > threshold and newmem < orimem:
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


class TTDecomposer(Compressor):
    @staticmethod
    def compress(embedding, compress_rate):
        nemb, ndim = embedding.shape
        nrows = TTDecomposer._get_decomp_emb(nemb)
        ncols = TTDecomposer._get_decomp_emb(ndim)
        print(nrows, ncols)
        nmuls = [r * c for r, c in zip(nrows, ncols)]
        from sympy import symbols, solve
        x = symbols('x')
        results = solve(TTDecomposer.get_memory(
            x, nmuls) - nemb * ndim * compress_rate)
        rank = [r for r in results if r > 0][0]
        rank = math.floor(rank)
        rank = int(rank)
        print("Real compression ratio:", TTDecomposer.get_memory(
            rank, nmuls) / nemb / ndim, "with rank", rank)
        padded_nemb = nrows[0] * nrows[1] * nrows[2]
        padded_ndim = ncols[0] * ncols[1] * ncols[2]
        embedding = np.concatenate((embedding, np.zeros(
            (padded_nemb - nemb, ndim), dtype=np.float32)))
        embedding = np.concatenate((embedding, np.zeros(
            (padded_nemb, padded_ndim - ndim), dtype=np.float32)), axis=1)
        embedding = embedding.reshape(nrows + ncols)
        embedding = np.transpose(embedding, (0, 3, 1, 4, 2, 5))
        embedding = embedding.reshape(nmuls[0], -1)
        from sklearn.decomposition import randomized_svd
        from hetu.random import get_np_rand
        nprs = get_np_rand(1)
        tensors = []
        U, Sigma, VT = randomized_svd(
            embedding,
            n_components=rank,
            n_oversamples=rank+10,
            random_state=nprs.randint(100))
        U = np.dot(U, np.diag(Sigma))
        tensors.append(U.reshape(nrows[0], ncols[0], rank))
        embedding = VT.reshape(rank * nmuls[1], nmuls[2])
        U, Sigma, VT = randomized_svd(
            embedding,
            n_components=rank,
            n_oversamples=rank+10,
            random_state=nprs.randint(100))
        U = np.dot(U, np.diag(Sigma))
        tensors.append(U.reshape(rank, nrows[1], ncols[1], rank).transpose(1, 0, 2, 3))
        tensors.append(VT.reshape(rank, nrows[2], ncols[2]).transpose(1, 0, 2))
        return tensors

    @staticmethod
    def decompress(compressed_embedding, ori_shape):
        a, b, c = compressed_embedding
        b = b.transpose(1, 0, 2, 3)
        c = c.transpose(1, 0, 2)
        rank = a.shape[-1]
        nrows = [a.shape[0], b.shape[1], c.shape[1]]
        ncols = [a.shape[1], b.shape[2], c.shape[2]]
        padded_nemb = nrows[0] * nrows[1] * nrows[2]
        padded_ndim = ncols[0] * ncols[1] * ncols[2]
        result = a.reshape(-1, rank) @ b.reshape(rank, -1)
        result = result.reshape(-1, rank) @ c.reshape(rank, -1)
        result = result.reshape(sum(list(zip(nrows, ncols)), ()))
        result = np.transpose(result, (0, 2, 4, 1, 3, 5))
        result = result.reshape(padded_nemb, padded_ndim)
        result = result[:ori_shape[0], :ori_shape[1]]
        return result

    @staticmethod
    def decompress_batch(compressed_embedding, batch_ids, ori_dim):
        ntables = len(compressed_embedding)
        a, b, c = compressed_embedding
        rank = a.shape[-1]
        nrows = [a.shape[0], b.shape[0], c.shape[0]]
        ncols = [a.shape[1], b.shape[2], c.shape[2]]
        ranks = [1, rank, rank, 1]
        indices = batch_ids
        accum_dim = 1
        for i in range(ntables)[::-1]:
            if i == 0:
                cur_ind = indices
            else:
                cur_ind = indices % nrows[i]
                indices = indices // nrows[i]
            emb = compressed_embedding[i]
            partial_embed = emb[cur_ind]
            if i == ntables - 1:
                accum_embed = partial_embed
            else:
                accum_embed = accum_embed.reshape(-1, ranks[i+1], accum_dim)
                partial_embed = partial_embed.reshape(-1, ranks[i] * ncols[i], ranks[i+1])
                accum_embed = np.matmul(partial_embed, accum_embed)
            accum_dim *= ncols[i]
        accum_embed = accum_embed.reshape((-1, accum_dim))[:, :ori_dim]
        return accum_embed

    @staticmethod
    def get_memory(rank, nmuls):
        return (nmuls[0] + nmuls[-1]) * rank + nmuls[1] * rank * rank

    @staticmethod
    def _get_decomp_emb(nemb):
        n1 = math.ceil(nemb ** (1/3))
        n2 = math.ceil((nemb / n1) ** (1/2))
        n3 = math.ceil(nemb / n1 / n2)
        return [n3, n2, n1]
