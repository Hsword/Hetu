from .base import BaseLayer
import hetu as ht


class Embedding(BaseLayer):
    def __init__(self, num_embeddings, embedding_dim, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = None
        self.embedding_table = initializer(
            shape=(self.num_embeddings, self.embedding_dim), name=self.name+'.weight', ctx=self.ctx)

    def __call__(self, x):
        return ht.embedding_lookup_op(self.embedding_table, x, ctx=self.ctx)
