import hetu as ht
from hetu.layers import Embedding
from .sparse import SparseEmbedding
import numpy as np


class PEPEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, threshold_type, threshold_init, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(self.num_embeddings, self.embedding_dim), name=self.name, ctx=ctx)
        assert threshold_type in (
            'dimension', 'feature', 'global', 'feature_dimension')
        self.threshold_type = threshold_type
        if threshold_type == 'feature_dimension':
            th_shape = (self.num_embeddings, self.embedding_dim)
        elif threshold_type == 'dimension':
            th_shape = (self.embedding_dim,)
        elif threshold_type == 'feature':
            th_shape = (self.num_embeddings, 1)
        else:
            th_shape = (1,)
        self.threshold = ht.init.constant(
            th_shape, threshold_init, name=f'{name}_threshold')

    def __call__(self, x):
        with ht.context(self.ctx):
            raw_embeddings = ht.embedding_lookup_op(self.embedding_table, x)
            if self.threshold_type.startswith('feature'):
                cur_threshold = ht.embedding_lookup_op(self.threshold, x)
            else:
                cur_threshold = self.threshold
            cur_threshold = ht.sigmoid_op(cur_threshold)
            if self.threshold_type != 'feature_dimension':
                cur_threshold = ht.broadcastto_op(
                    cur_threshold, raw_embeddings)
            embeddings = ht.mul_op(ht.sign_op(raw_embeddings), ht.relu_op(
                ht.minus_op(ht.abs_op(raw_embeddings), cur_threshold)))
            return embeddings


class PEPRetrainEmbedding(SparseEmbedding):
    def __init__(self, num_embeddings, embedding_dim, mask, form, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(self.num_embeddings, self.embedding_dim), name=self.name, ctx=ctx)
        self.mask = ht.placeholder_op(
            f'{name}_mask', value=mask, trainable=False, dtype=np.int32, ctx=self.ctx)
        self.form = form

    def __call__(self, x):
        with ht.context(self.ctx):
            lookups = ht.embedding_lookup_op(self.embedding_table, x)
            lookup_masks = ht.embedding_lookup_op(self.mask, x)
            return ht.mask_op(lookups, lookup_masks)
