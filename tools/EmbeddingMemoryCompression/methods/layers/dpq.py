import hetu as ht
from hetu.layers import Embedding
import numpy as np


class DPQEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, num_choices, num_parts, batch_size, share_weights=False, mode='vq', initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        from hetu.initializers import nulls
        from hetu.layers.normalization import BatchNorm
        assert mode in ('vq', 'sx')
        assert embedding_dim % num_parts == 0
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_choices = num_choices
        self.num_parts = num_parts
        self.batch_size = batch_size  # contains slot if use multi==0
        self.share_weights = share_weights
        self.mode = mode
        self.part_embedding_dim = embedding_dim // num_parts
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(shape=(
            num_embeddings, self.embedding_dim), name='{}_query'.format(name), ctx=ctx)
        self.key_matrix = self.make_matries(initializer, name+'_key')
        if mode == 'vq':
            self.value_matrix = self.key_matrix
        else:
            self.value_matrix = self.make_matries(initializer, name+'_value')
        self.bn_layer = BatchNorm(
            self.num_choices, scale=False, bias=False, name='{}_bn'.format(name))
        self.codebooks = nulls(shape=(num_embeddings, self.num_parts), name='{}_codebook'.format(
            name), ctx=ctx, dtype=np.int32, trainable=False)
        if not self.share_weights:
            dbase = np.array(
                [self.num_choices * d for d in range(self.num_parts)], dtype=int)
            dbase = np.tile(dbase, [self.batch_size, 1])
            dbase = ht.array(dbase, dtype=np.int32, ctx=self.ctx)
            self.dbase = ht.placeholder_op(
                'dbase', value=dbase, dtype=np.int32, trainable=False)

    def make_matries(self, initializer, name):
        if self.share_weights:
            shape = (self.num_choices, self.part_embedding_dim)
        else:
            shape = (self.num_parts * self.num_choices,
                     self.part_embedding_dim)
        return initializer(shape=shape, name='{}'.format(name), ctx=self.ctx)

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
            if self.mode == 'vq':
                # query metric: euclidean
                diff = ht.minus_op(query_lookups, key_mat)
                resp = ht.power_op(diff, 2)
                resp = ht.reduce_sum_op(resp, axes=[3])
                resp = ht.opposite_op(resp)
                # (bs * slot, npart, nkey)
            else:
                # query metric: dot
                dot = ht.mul_op(query_lookups, key_mat)
                resp = ht.reduce_sum_op(dot, axes=[3])
                # (bs * slot, npart, nkey)
            resp = self.bn_layer(resp)
            codes = ht.argmax_op(resp, 2)
            self.codebook_update = ht.sparse_set_op(self.codebooks, x, codes)
            # (bs * slot, npart)
            if self.mode == 'vq':
                if not self.share_weights:
                    codes = ht.add_op(codes, self.dbase)
                outputs = ht.embedding_lookup_op(self.value_matrix, codes)
                # (bs * slot, npart, pdim)
                outputs_final = ht.add_op(ht.stop_gradient_op(
                    ht.minus_op(outputs, inputs)), inputs)
                reg = ht.minus_op(outputs, ht.stop_gradient_op(inputs))
                reg = ht.power_op(reg, 2)
                self.reg = ht.reduce_mean_op(reg, axes=(0, 1, 2))
            else:
                resp_prob = ht.softmax_op(resp)
                # (bs * slot, npart, nkey)
                nb_idxs_onehot = ht.one_hot_op(codes, self.num_choices)
                # (bs * slot, npart, nkey)
                nb_idxs_onehot = ht.minus_op(resp_prob, ht.stop_gradient_op(
                    ht.minus_op(resp_prob, nb_idxs_onehot)))
                if self.share_weights:
                    outputs = ht.matmul_op(
                        # (bs * slot * npart, nkey)
                        ht.array_reshape_op(
                            nb_idxs_onehot, (-1, self.num_choices)),
                        self.value_matrix)  # (nkey, pdim)
                    # (bs * slot * npart, pdim)
                else:
                    outputs = ht.batch_matmul_op(
                        # (npart, bs * slot, nkey)
                        ht.transpose_op(nb_idxs_onehot, [1, 0, 2]),
                        ht.array_reshape_op(self.value_matrix, (-1, self.num_choices, self.part_embedding_dim)))  # (npart, nkey, pdim)
                    # (npart, bs * slot, pdim)
                    outputs = ht.transpose_op(outputs, [1, 0, 2])
                    # (bs * slot, npart, pdim)
                outputs_final = ht.array_reshape_op(
                    outputs, (-1, self.embedding_dim))
                # (bs * slot, dim)

            outputs_final = ht.array_reshape_op(
                outputs_final, (-1, self.embedding_dim))
            return outputs_final

    def make_inference(self, embed_input):
        with ht.context(self.ctx):
            codes = ht.embedding_lookup_op(self.codebooks, embed_input)
            # (bs, slot, npart)
            if not self.share_weights:
                codes = ht.add_op(codes, ht.reshape_to_op(self.dbase, codes))
            outputs = ht.embedding_lookup_op(self.value_matrix, codes)
            # (bs, slot, npart, pdim)
            outputs = ht.array_reshape_op(outputs, (-1, self.embedding_dim))
            # (bs * slot, dim)
            return outputs
