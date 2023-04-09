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

    def __call__(self, x):
        with ht.context(self.ctx):
            return ht.embedding_lookup_op(self.embedding_table, x)

    def get_eval_nodes(self, data_ops, model, opt):
        embed_input, dense_input, y_ = data_ops
        loss, prediction = model(self(embed_input), dense_input, y_)
        train_op = opt.minimize(loss)
        eval_nodes = {
            'train': [loss, prediction, y_, train_op],
            'validate': [loss, prediction, y_],
        }
        return eval_nodes

    def get_eval_nodes_inference(self, data_ops, model):
        embed_input, dense_input, y_ = data_ops
        loss, prediction = model(self(embed_input), dense_input, y_)
        eval_nodes = {
            'validate': [loss, prediction, y_],
        }
        return eval_nodes


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
