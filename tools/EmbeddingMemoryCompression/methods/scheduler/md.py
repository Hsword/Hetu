from .base import EmbeddingTrainer
from .compressor import Compressor
from ..layers import MDEmbedding
import numpy as np
import math


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
                [nemb * ndim + ndim * self.embedding_dim * (ndim != self.embedding_dim) for nemb, ndim in zip(self.num_embed_separate, cur_dims)])
            return target_memory - cur_memory
        round_dim = self.embedding_args['round_dim']
        target_memory = self.num_embed * self.embedding_dim * self.compress_rate
        left, right = self.binary_search(0., 1., evaluate, 1e-3)
        # test how close is left
        dims = self._md_solver(left, round_dim=round_dim)
        real_compress_rate = sum(
            [nemb * ndim for nemb, ndim in zip(self.num_embed_separate, dims)]) / self.num_embed / self.embedding_dim
        if real_compress_rate < self.compress_rate * (1 + 1e-3):
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
        # assert max(dims) == self.embedding_dim
        emb = []
        for i, (nemb, cdim) in enumerate(zip(self.num_embed_separate, dims)):
            emb.append(self.get_single_embed_layer(nemb, cdim, f'MDEmb_{i}'))
        return emb


class SVDDecomposer(Compressor):
    @staticmethod
    def compress(embedding, compress_rate):
        nemb, ndim = embedding.shape
        rank = nemb * ndim * compress_rate / (nemb + ndim)
        rank = math.floor(rank)
        print("Real compression ratio:", rank *
              (nemb + ndim) / nemb / ndim, "with rank", rank)
        from sklearn.decomposition import randomized_svd
        from hetu.random import get_np_rand
        seed = get_np_rand(1).randint(100)
        U, Sigma, VT = randomized_svd(
            embedding,
            n_components=rank,
            n_oversamples=rank+10,
            random_state=seed)
        U = np.dot(U, np.diag(Sigma))
        return U, VT

    @staticmethod
    def decompress(compressed_embedding, projection_matrix):
        return compressed_embedding @ projection_matrix

    @staticmethod
    def decompress_batch(compressed_embedding, batch_ids, projection_matrix):
        return compressed_embedding[batch_ids] @ projection_matrix


class MagnitudeSVDDecomposer(Compressor):
    @staticmethod
    def compress(embedding, compress_rate, ngroup):
        split_embeddings, remap = Compressor.split_by_magnitude(
            embedding, ngroup)
        nemb, ndim = embedding.shape
        mem_cap = nemb * ndim * compress_rate
        nsubembs = np.array([emb.shape[0] for emb in split_embeddings])
        dims = MagnitudeSVDDecomposer._md_solver(
            mem_cap - remap.shape[0], nsubembs, ndim)
        memory = sum([n * d for n, d in zip(nsubembs, dims)])
        print("Real compression ratio:", (memory +
              remap.shape[0]) / nemb / ndim, "with nembs", nsubembs, "with ndims", dims)
        from sklearn.decomposition import randomized_svd
        from hetu.random import get_np_rand
        nprs = get_np_rand(1)
        all_embs = []
        for emb, rank in zip(split_embeddings, dims):
            U, Sigma, VT = randomized_svd(
                emb,
                n_components=rank,
                n_oversamples=rank+10,
                random_state=nprs.randint(100))
            U = np.dot(U, np.diag(Sigma))
            all_embs.append((U, VT))
        return all_embs, remap

    @staticmethod
    def decompress(compressed_embedding, remap):
        embedding = np.empty(
            (remap.shape[0], compressed_embedding[0][1].shape[1]), dtype=np.float32)
        start_index = 0
        reverse_remap = np.argsort(remap)
        for emb, proj in compressed_embedding:
            cur_emb = emb @ proj
            ending_index = start_index + cur_emb.shape[0]
            cur_idx = reverse_remap[start_index:ending_index]
            embedding[cur_idx] = cur_emb
            start_index = ending_index
        assert start_index == remap.shape[0]
        return embedding

    @staticmethod
    def decompress_batch(compressed_embedding, batch_ids, remap):
        embedding = np.empty(
            (batch_ids.shape[0], compressed_embedding[0][1].shape[1]), dtype=np.float32)
        remapped = remap[batch_ids]
        indind = np.zeros(remapped.shape, dtype=np.int32)
        embind = np.zeros(remapped.shape, dtype=np.int32)
        for g in range(len(compressed_embedding)):
            cur_nemb = compressed_embedding[g][0].shape[0]
            belong = (remapped >= 0) & (remapped < cur_nemb)
            indind[belong] = g
            embind[belong] = remapped[belong]
            remapped -= cur_nemb
        bidx = np.arange(batch_ids.shape[0])
        for g, (emb, proj) in enumerate(compressed_embedding):
            iscur = indind == g
            if sum(iscur) > 0:
                embedding[bidx[iscur]] = emb[embind[iscur]] @ proj
        return embedding

    @staticmethod
    def _md_solver(mem_cap, nemb_fields, dim, alpha=0.3):
        lamb = mem_cap / np.sum((nemb_fields ** (-alpha))
                                * (nemb_fields + dim))
        d = lamb * (nemb_fields ** (-alpha))
        d = np.round(np.maximum(d, 1))
        d = d.astype(int)
        return d
