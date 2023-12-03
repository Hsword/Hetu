import hetu as ht
from hetu.layers import Embedding


class MDEmbedding(Embedding):
    def __init__(self, num_embeddings, compressed_dim, embedding_dim, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        self.num_embeddings = num_embeddings
        self.compressed_dim = compressed_dim
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(num_embeddings, compressed_dim), name=name, ctx=self.ctx)
        if compressed_dim < embedding_dim:
            self.projection = initializer(
                shape=(compressed_dim, embedding_dim), name=f'{name}_proj', ctx=self.ctx)
        else:
            self.projection = None

    def __call__(self, x):
        with ht.context(self.ctx):
            res = ht.embedding_lookup_op(self.embedding_table, x)
            if self.projection is not None:
                res = ht.matmul_op(res, self.projection)
        return res

    def __repr__(self):
        return f'{self.name}({self.num_embeddings},{self.compressed_dim},{self.embedding_dim})'
