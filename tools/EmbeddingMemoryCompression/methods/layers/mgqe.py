import hetu as ht
from .dpq import DPQEmbedding
import numpy as np


class MGQEmbedding(DPQEmbedding):
    def __init__(self, num_embeddings, embedding_dim, high_num_choices, low_num_choices, num_parts, frequency, batch_size, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        super().__init__(num_embeddings, embedding_dim, high_num_choices,
                         num_parts, batch_size, False, 'vq', initializer, name, ctx)
        self.low_num_choices = low_num_choices
        frequency = ht.array(frequency.reshape((-1, 1)),
                             dtype=np.int32, ctx=self.ctx)
        self.frequency = ht.placeholder_op(
            f'{name}_frequency', value=frequency, dtype=np.int32, trainable=False)

    def __call__(self, x):
        with ht.context(self.ctx):
            # table: (nembed, dim), x: (bs, slot)
            query_lookups = ht.embedding_lookup_op(
                self.embedding_table, x)
            # (bs, slot, dim)
            inputs = ht.array_reshape_op(
                query_lookups, (-1, self.num_parts, self.part_embedding_dim))
            query_lookups = ht.array_reshape_op(
                query_lookups, (-1, self.num_parts, 1, self.part_embedding_dim))
            # (bs * slot, npart, 1, pdim)
            query_lookups = ht.tile_op(query_lookups, [self.num_choices, 1])
            # (bs * slot, npart, nkey, pdim)
            key_mat = ht.array_reshape_op(
                self.key_matrix, (-1, self.num_choices, self.part_embedding_dim))
            key_mat = ht.broadcastto_op(key_mat, query_lookups)
            # (bs * slot, npart, nkey, pdim)
            # query metric: euclidean
            diff = ht.minus_op(query_lookups, key_mat)
            resp = ht.power_op(diff, 2)
            resp = ht.reduce_sum_op(resp, axes=[3])
            resp = ht.opposite_op(resp)
            # (bs * slot, npart, nkey)
            resp = self.bn_layer(resp)
            # !! only argmax op is changed, compared with DPQ
            mask = ht.embedding_lookup_op(self.frequency, x)
            mask = ht.array_reshape_op(mask, (-1,))
            codes = ht.argmax_partial_op(
                resp, mask, self.low_num_choices, dim=2)
            self.codebook_update = ht.sparse_set_op(self.codebooks, x, codes)
            # (bs * slot, npart)
            codes = ht.add_op(codes, self.dbase)
            outputs = ht.embedding_lookup_op(self.value_matrix, codes)
            # (bs * slot, npart, pdim)
            outputs_final = ht.add_op(ht.stop_gradient_op(
                ht.minus_op(outputs, inputs)), inputs)
            reg = ht.minus_op(outputs, ht.stop_gradient_op(inputs))
            reg = ht.power_op(reg, 2)
            self.reg = ht.reduce_mean_op(reg, axes=(0, 1, 2))

            outputs_final = ht.array_reshape_op(
                outputs_final, (-1, self.embedding_dim))
            return outputs_final
