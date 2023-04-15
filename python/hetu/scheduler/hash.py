from .base import EmbeddingTrainer
from ..layers import HashEmbedding
import math


class HashEmbTrainer(EmbeddingTrainer):
    def _compress_nemb(self, nemb):
        return math.ceil(nemb * self.compress_rate)

    def get_single_embed_layer(self, nemb, name):
        return HashEmbedding(
            nemb,
            self.embedding_dim,
            self.initializer,
            name,
            self.ectx,
            **self.embedding_args,
        )

    def get_embed_layer(self):
        if self.use_multi:
            emb = [self.get_single_embed_layer(
                self._compress_nemb(nemb), f'HashEmb({self.compress_rate})_{i}') for i, nemb in enumerate(self.num_embed_separate)]
        else:
            emb = self.get_single_embed_layer(
                self._compress_nemb(self.num_embed), f'HashEmb({self.compress_rate})')
        return emb
