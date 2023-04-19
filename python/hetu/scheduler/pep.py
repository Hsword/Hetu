from .base import EmbeddingTrainer
from ..layers import PEPEmbedding


class PEPEmbTrainer(EmbeddingTrainer):
    def assert_use_multi(self):
        assert self.use_multi == self.separate_fields == 0

    def get_embed_layer(self):
        return PEPEmbedding(
            self.num_embed,
            self.embedding_dim,
            self.embedding_args['threshold_type'],
            self.embedding_args['threshold_init'],
            initializer=self.initializer,
            name='PEPEmb',
            ctx=self.ectx,
        )
