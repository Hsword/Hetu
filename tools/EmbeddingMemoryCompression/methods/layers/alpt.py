import hetu as ht
from hetu.layers import Embedding


class ALPTEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, digit, init_scale, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        assert digit in (8, 16)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.digit = digit
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(self.num_embeddings, self.embedding_dim), name=self.name, ctx=ctx)
        self.scale = ht.init.constant(shape=(
            self.num_embeddings, 1), fill_value=init_scale, name=f'{self.name}_scale', trainable=False, ctx=ctx)
        self.middle = 0

    def __call__(self, x):
        with ht.context(self.ctx):
            lookup = ht.alpt_embedding_lookup_op(
                self.embedding_table, x, self.scale, self.middle, self.digit)
            return lookup

    def __repr__(self):
        return f'{self.name}({self.num_embeddings},{self.embedding_dim},{self.digit})'
