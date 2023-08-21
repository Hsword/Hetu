from .base import EmbeddingTrainer
from ..layers import RobeEmbedding
from hetu.random import get_np_rand, set_random_seed
import math


class ROBETrainer(EmbeddingTrainer):
    def assert_use_multi(self):
        assert self.use_multi == 0

    def get_embed_layer(self):
        assert self.num_embed < 2038074743
        set_random_seed(self.seed)
        nprs = get_np_rand(1)
        emb = RobeEmbedding(
            math.floor(self.num_embed * self.embedding_dim *
                       self.compress_rate),
            self.embedding_dim,
            self.embedding_args['Z'],
            nprs,
            use_slot_coef=bool(self.separate_fields),
            initializer=self.initializer,
            name=f'RobeEmb{self.compress_rate}',
            ctx=self.ectx,
        )
        return emb
