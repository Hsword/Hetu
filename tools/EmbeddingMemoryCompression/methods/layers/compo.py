import hetu as ht
from hetu.layers import Embedding


class CompositionalEmbedding(Embedding):
    def __init__(self, num_quotient, num_remainder, embedding_dim, aggregator='mul', initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        # KDD20, CompositionalHash
        # CIKM21, BinaryCodeHash
        # adapted from DLRM QREmbeddingBag
        aggregator = aggregator[:3]
        assert aggregator in ('sum', 'mul')
        self.aggregator = aggregator
        self.num_quotient = num_quotient
        self.num_remainder = num_remainder
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.qemb = initializer(
            shape=(self.num_quotient, self.embedding_dim), name=f'{name}_q', ctx=ctx)
        self.remb = initializer(
            shape=(self.num_remainder, self.embedding_dim), name=f'{name}_r', ctx=ctx)

    def __call__(self, x):
        with ht.context(self.ctx):
            qind = ht.div_hash_op(x, self.num_remainder)
            rind = ht.mod_hash_op(x, self.num_remainder)
            q = ht.embedding_lookup_op(self.qemb, qind)
            r = ht.embedding_lookup_op(self.remb, rind)
            if self.aggregator == 'sum':
                result = ht.add_op(q, r)
            elif self.aggregator == 'mul':
                result = ht.mul_op(q, r)
            return result

    def __repr__(self):
        return f'{self.name}({self.num_quotient},{self.num_remainder},{self.embedding_dim})'
