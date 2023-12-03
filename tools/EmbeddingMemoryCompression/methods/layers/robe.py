import hetu as ht
from hetu.layers import Embedding
import numpy as np


class RobeEmbedding(Embedding):
    def __init__(self, robe_array_size, embedding_dim, Z, nprs, use_slot_coef=True, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        self.robe_array_size = robe_array_size
        self.embedding_dim = embedding_dim
        assert Z <= embedding_dim
        self.Z = Z
        self.use_slot_coef = use_slot_coef
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(self.robe_array_size, 1), name=self.name, ctx=ctx)
        random_numbers = np.concatenate(
            [np.array([2038074743]), nprs.randint(1, 2038074743, (9,))]).astype(np.int32)
        self.random_numbers = ht.placeholder_op(
            'random_numbers', value=random_numbers, dtype=np.int32, trainable=False)

    def __call__(self, x):
        with ht.context(self.ctx):
            expanded_indices = ht.robe_hash_op(
                x, self.random_numbers, self.robe_array_size, self.embedding_dim, self.Z, self.use_slot_coef)
            signs = ht.robe_sign_op(
                x, self.random_numbers, self.embedding_dim, self.use_slot_coef)
            lookups = ht.embedding_lookup_op(
                self.embedding_table, expanded_indices)
            lookups = ht.reshape_to_op(lookups, signs)
            lookups = ht.mul_op(lookups, signs)
            return lookups

    def __repr__(self):
        return f'{self.name}({self.robe_array_size})'
