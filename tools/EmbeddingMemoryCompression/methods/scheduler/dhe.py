from .base import EmbeddingTrainer
from ..layers import DeepHashEmbedding
from hetu.random import get_np_rand, set_random_seed
import math


class DHETrainer(EmbeddingTrainer):
    def _get_single_memory(self, fcdim, num_hash):
        return (num_hash + 1) * fcdim + 4 * fcdim * \
            (fcdim + 1) + (self.embedding_dim + 1) * fcdim + 10 * fcdim

    def _get_dim(self, num_hash):

        def multi_evaluate(x):
            newmem = self._get_single_memory(x, num_hash)
            memory = 0
            for nemb in self.num_embed_separate:
                orimem = nemb * self.embedding_dim
                if nemb > threshold:
                    memory += min(orimem, newmem)
                else:
                    memory += orimem
            return memory - target_memory

        def single_evaluate(x):
            memory = self._get_single_memory(x, num_hash)
            return memory - target_memory

        threshold = self.embedding_args['threshold']
        target_memory = self.num_embed * self.embedding_dim * self.compress_rate
        if self.use_multi:
            evaluate = multi_evaluate
        else:
            evaluate = single_evaluate
        res = self.binary_search(1, math.sqrt(
            self.num_embed * self.embedding_dim), evaluate)
        res = math.floor(res[1])
        if evaluate(res) > 0:
            res -= 1
        self.log_func(
            f'Dimension {res} given compression rate {self.compress_rate}.')
        return res

    def get_single_embed_layer(self, fcdim, nprs, name):
        return DeepHashEmbedding(
            self.embedding_dim,
            fcdim,
            self.embedding_args['num_buckets'],
            self.embedding_args['num_hash'],
            nprs,
            dist=self.embedding_args['dist'],
            initializer=self.initializer,
            name=name,
            ctx=self.ectx
        )

    def get_embed_layer(self):
        nhash = self.embedding_args['num_hash']
        fcdim = self._get_dim(nhash)
        set_random_seed(self.seed)
        nprs = get_np_rand(1)
        all_size = 0
        if self.use_multi:
            threshold = max(self._get_single_memory(fcdim, nhash),
                            self.embedding_args['threshold'] * self.embedding_dim)
            emb = []
            for i, nemb in enumerate(self.num_embed_separate):
                orimem = nemb * self.embedding_dim
                if orimem > threshold:
                    all_size += threshold
                    emb.append(self.get_single_embed_layer(
                        fcdim, nprs, f'DHEmb({self.compress_rate}_{i})'))
                else:
                    all_size += orimem
                    emb.append(super().get_single_embed_layer(
                        nemb, f'Embedding_{i}'))
        else:
            all_size = self._get_single_memory(fcdim, nhash)
            emb = self.get_single_embed_layer(
                fcdim, nprs, f'DHEmb({self.compress_rate})')
        self.log_func(
            f'Real compress rate: {all_size / self.num_embed / self.embedding_dim}')
        return emb
