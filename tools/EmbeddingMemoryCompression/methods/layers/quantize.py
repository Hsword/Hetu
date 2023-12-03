import hetu as ht
from hetu.layers import Embedding


class QuantizedEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, digit, scale=0.01, middle=0, use_qparam=False, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        assert digit in (8, 16)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.digit = digit
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(self.num_embeddings, self.embedding_dim), name=self.name, ctx=ctx)
        if use_qparam:
            self.qparams = ht.init.GenEmpty()(shape=(self.num_embeddings, 2),
                                              name='qparams', trainable=False, ctx=ctx)
        else:
            self.scale = scale
            self.middle = middle
        self.use_qparam = use_qparam

    def __call__(self, x):
        with ht.context(self.ctx):
            if self.use_qparam:
                lookup = ht.quantized_embedding_lookup_op(
                    self.embedding_table, x, self.qparams, self.digit)
            else:
                lookup = ht.unified_quantized_embedding_lookup_op(
                    self.embedding_table, x, self.scale, self.middle, self.digit)
            return lookup

    def __repr__(self):
        return f'{self.name}({self.num_embeddings},{self.embedding_dim},{self.digit})'
