from .base import BaseLayer
import hetu as ht
import numpy as np


class Embedding(BaseLayer):
    def __init__(self, num_embeddings, embedding_dim, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(self.num_embeddings, self.embedding_dim), name=self.name, ctx=ctx)

    def compute_all(self, func, batch_size, val=True):
        # load data and lookup
        dense, sparse, y_ = self.load_data(func, batch_size, val)
        return self(sparse), dense, y_

    def __call__(self, x):
        with ht.context(self.ctx):
            return ht.embedding_lookup_op(self.embedding_table, x)

    def load_data(self, func, batch_size, val=True, sep=False, tr_name='train', va_name='validate', only_sparse=False):
        def make_op_data(tr_data, va_data=None, dtype=np.float32):
            op_data = [[tr_data, batch_size, tr_name], ]
            if val:
                op_data.append([va_data, batch_size, va_name])
            op_data = ht.dataloader_op(op_data, dtype=dtype)
            return op_data

        # define models for criteo
        dense, sparse, labels = func(return_val=val, separate_fields=sep)
        if val:
            tr_dense, va_dense = dense
            tr_sparse, va_sparse = sparse
            tr_labels, va_labels = labels
            va_labels = va_labels.reshape((-1, 1))
        else:
            tr_dense = dense
            tr_sparse = sparse
            tr_labels = labels
            va_dense = va_sparse = va_labels = None
        tr_labels = tr_labels.reshape((-1, 1))
        if only_sparse:
            dense_input = y_ = None
        else:
            dense_input = make_op_data(tr_dense, va_dense)
            y_ = make_op_data(tr_labels, va_labels)
        if sep:
            new_sparse_ops = []
            for i in range(tr_sparse.shape[1]):
                cur_data = make_op_data(
                    tr_sparse[:, i], None if va_sparse is None else va_sparse[:, i], dtype=np.int32)
                new_sparse_ops.append(cur_data)
            sparse_input = new_sparse_ops
        else:
            sparse_input = make_op_data(tr_sparse, va_sparse, dtype=np.int32)

        print("Data loaded.")
        self.dense_op = dense_input
        self.sparse_op = sparse_input
        self.y_op = y_
        return dense_input, sparse_input, y_


class MultipleEmbedding(Embedding):
    def __init__(self, num_embed_fields, embedding_dim, initializer=ht.init.GenXavierNormal(), names='embedding', ctx=None):
        self.num_embed_fields = num_embed_fields
        self.embedding_dim = embedding_dim
        if not isinstance(names, list):
            names = [f'{names}_{i}' for i in range(len(num_embed_fields))]
        self.name = names
        self.ctx = ctx
        self.embedding_table = [
            initializer(
                shape=(nemb, self.embedding_dim),
                name=nam,
                ctx=ctx,
            ) for nemb, nam in zip(self.num_embed_fields, self.name)
        ]

    def __call__(self, xs):
        with ht.context(self.ctx):
            results = []
            for emb, x in zip(self.embedding_table, xs):
                results.append(ht.embedding_lookup_op(emb, x))
            result = ht.concatenate_op(results, axis=1)
            return result

    def compute_all(self, func, batch_size, val=True):
        # load data and lookup
        dense, sparse, y_ = self.load_data(func, batch_size, val, sep=True)
        return self(sparse), dense, y_
