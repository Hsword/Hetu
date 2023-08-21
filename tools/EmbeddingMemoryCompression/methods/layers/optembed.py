import hetu as ht
from hetu.layers import Embedding
import numpy as np


class OptEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, num_slot, batch_size, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_slot = num_slot
        self.batch_size = batch_size
        self.name = name
        self.ctx = ctx

        self.embedding_table = initializer(
            shape=(num_embeddings, embedding_dim), name=name, ctx=ctx)

        self.threshold = ht.init.zeros(
            shape=(self.num_slot, 1), name=f'{name}_threshold')
        self.potential_field_masks = ht.placeholder_op(
            name=f'{name}_pmask', value=self.pre_potential_field_mask(), trainable=False)

    def pre_potential_field_mask(self):
        masks = []
        for i in range(self.embedding_dim):
            zeros = np.zeros(self.embedding_dim - i - 1)
            ones = np.ones(i + 1)
            mask = np.concatenate(
                (ones, zeros), axis=0)[None, ...]
            masks.append(mask)
        total_masks = np.concatenate(masks, axis=0).astype(np.float32)
        return total_masks

    def get_batch_feature_mask(self, xv, tv):
        xv_norm = ht.reduce_norm1_op(xv, axes=2, keepdims=True)
        mask_f = ht.binary_step_op(ht.minus_op(
            xv_norm, ht.broadcastto_op(tv, xv_norm)))
        return mask_f

    def get_random_field_mask(self):
        indices = ht.randint_sample_op(
            shape=(self.batch_size, self.num_slot), low=0, high=self.embedding_dim)
        field_masks = ht.embedding_lookup_op(
            self.potential_field_masks, indices)
        return field_masks

    def __call__(self, x):
        with ht.context(self.ctx):
            xv = ht.embedding_lookup_op(self.embedding_table, x)
            mask_f = self.get_batch_feature_mask(xv, self.threshold)
            mask_f = ht.broadcastto_op(mask_f, xv)
            mask_e = self.get_random_field_mask()
            xe = ht.mul_op(mask_f, ht.mul_op(mask_e, xv))
        return xe

    def make_inference(self, x):
        with ht.context(self.ctx):
            xv = ht.embedding_lookup_op(self.embedding_table, x)
            mask_f = self.get_batch_feature_mask(xv, self.threshold)
            mask_f = ht.broadcastto_op(mask_f, xv)
            xe = ht.mul_op(mask_f, xv)
        return xe


class OptEmbeddingAfterRowPruning(OptEmbedding):
    def __init__(self, num_embeddings, original_num_embeddings, embedding_dim, num_slot, batch_size, name='embedding', ctx=None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_slot = num_slot
        self.batch_size = batch_size
        self.name = name
        self.ctx = ctx

        self.target_shape = (batch_size, num_slot, embedding_dim)
        self.embedding_table = ht.init.nulls(
            name=name, shape=(num_embeddings, embedding_dim), ctx=ctx)
        self.remap_indices = ht.init.nulls(
            name=f'{name}_remap', shape=(original_num_embeddings, 1), dtype=np.int32, trainable=False, ctx=ctx)
        self.potential_field_masks = ht.placeholder_op(
            name=f'{name}_pmask', value=self.pre_potential_field_mask(), trainable=False, ctx=ctx)
        self.candidate = ht.init.nulls(shape=(
            self.num_slot,), name=f'{name}_candidate', trainable=False, dtype=np.int32, ctx=ctx)

    def __call__(self, x):
        with ht.context(self.ctx):
            new_indices = ht.embedding_lookup_op(self.remap_indices, x)
            xe = ht.embedding_lookup_op(self.embedding_table, new_indices)
            mask_e = ht.embedding_lookup_op(
                self.potential_field_masks, self.candidate)
            xe = ht.mul_op(ht.broadcast_shape_op(
                mask_e, self.target_shape), ht.array_reshape_op(xe, self.target_shape))
        return xe
