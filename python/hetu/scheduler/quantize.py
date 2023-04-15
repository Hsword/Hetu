from .base import EmbeddingTrainer
from ..layers import QuantizedEmbedding


class QuantizeEmbTrainer(EmbeddingTrainer):
    def assert_use_multi(self):
        assert self.use_multi == self.separate_fields == 0

    def get_embed_layer(self):
        return QuantizedEmbedding(
            self.num_embed,
            self.embedding_dim,
            self.embedding_args['digit'],
            self.embedding_args['scale'],
            self.embedding_args['middle'],
            self.embedding_args['use_qparam'],
            initializer=self.initializer,
            name='QuanEmb',
            ctx=self.ectx,
        )
