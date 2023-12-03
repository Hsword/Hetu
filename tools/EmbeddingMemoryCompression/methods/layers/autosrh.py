import hetu as ht
from .sparse import SparseEmbedding
import numpy as np


class AutoSrhEmbedding(SparseEmbedding):
    def __init__(self, num_embeddings, embedding_dim, nsplit, group_indices, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.nsplit = nsplit
        self.embedding_table = initializer(
            shape=(num_embeddings, embedding_dim), name=name, ctx=self.ctx)
        self.group_indices = ht.placeholder_op(
            name=f'{name}_groupind', value=group_indices.reshape(-1, 1), trainable=False, dtype=np.int32)
        self.alpha = ht.init.ones(
            shape=(self.nsplit, self.embedding_dim), name=f'{name}_alpha')

    def __call__(self, x):
        with ht.context(self.ctx):
            embeddings = ht.embedding_lookup_op(self.embedding_table, x)
            alpha_indices = ht.embedding_lookup_op(self.group_indices, x)
            alphas = ht.embedding_lookup_op(self.alpha, alpha_indices)
            return ht.mul_op(embeddings, ht.reshape_to_op(alphas, embeddings))


class AutoSrhRetrainEmbedding(AutoSrhEmbedding):
    def __init__(self, num_embeddings, embedding_dim, nsplit, group_indices, form='csr', initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        super().__init__(num_embeddings, embedding_dim,
                         nsplit, group_indices, initializer, name, ctx)
        self.form = form
        self.alpha.trainable = False
