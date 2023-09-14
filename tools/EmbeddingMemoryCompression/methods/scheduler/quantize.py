from .base import EmbeddingTrainer
from .compressor import Compressor
from ..layers import QuantizedEmbedding
from hetu.random import get_np_rand
import numpy as np


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


class Quantizer(Compressor):
    @staticmethod
    def compress(embedding, digit, middle=0, scale=0.01):
        nprs = get_np_rand(1)
        assert digit in (8, 16) and scale > 0
        embedding = np.floor((embedding - middle) / scale +
                             nprs.uniform(low=0, high=1, size=embedding.shape))
        if digit == 8:
            dtype = np.int8
        else:
            dtype = np.int16
        dtype_info = np.iinfo(dtype)
        embedding = np.maximum(embedding, dtype_info.min)
        embedding = np.minimum(embedding, dtype_info.max)
        embedding = embedding.astype(dtype)
        return embedding

    @staticmethod
    def decompress(compressed_embedding, middle=0, scale=0.01):
        embedding = compressed_embedding.astype(np.float32) * scale + middle
        return embedding

    @staticmethod
    def decompress_batch(compressed_embedding, batch_ids, middle=0, scale=0.01):
        embedding = compressed_embedding[batch_ids].astype(
            np.float32) * scale + middle
        return embedding
