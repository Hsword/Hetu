from .base import EmbeddingTrainer
from ..layers import ALPTEmbedding


class ALPTEmbTrainer(EmbeddingTrainer):
    def assert_use_multi(self):
        assert self.use_multi == self.separate_fields == 0

    def get_embed_layer(self):
        return ALPTEmbedding(
            self.num_embed,
            self.embedding_dim,
            self.embedding_args['digit'],
            self.embedding_args['init_scale'],
            initializer=self.initializer,
            name='ALPTEmb',
            ctx=self.ectx,
        )

    def get_eval_nodes(self):
        raise NotImplementedError
        embed_input, dense_input, y_ = self.data_ops
        embeddings = self.embed_layer(embed_input)
        loss, prediction = self.model(
            embeddings, dense_input, y_)
        train_op = self.opt.minimize(loss)
        eval_nodes = {
            'train': [loss, prediction, y_, train_op],
            'validate': [loss, prediction, y_],
        }
        return eval_nodes

    def train_step(self):
        raise NotImplementedError
