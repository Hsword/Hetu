from .switchinference import SwitchInferenceTrainer
from ..layers import DeepLightEmbedding


class DeepLightTrainer(SwitchInferenceTrainer):
    def assert_use_multi(self):
        assert self.use_multi == self.separate_fields == 0

    @property
    def sparse_name(self):
        return 'DeepLightEmb'

    def get_embed_layer(self):
        return DeepLightEmbedding(
            self.num_embed,
            self.embedding_dim,
            self.prune_rate,
            self.form,
            warm=self.embedding_args['warm'],
            initializer=self.initializer,
            name=self.sparse_name,
            ctx=self.ectx,
        )

    def get_eval_nodes(self):
        embed_input, dense_input, y_ = self.data_ops
        loss, prediction = self.model(
            self.embed_layer(embed_input), dense_input, y_)
        train_op = self.opt.minimize(loss)
        eval_nodes = {
            'train': [loss, prediction, y_, train_op, self.embed_layer.make_prune_op(y_)],
            'validate': [loss, prediction, y_],
        }
        return eval_nodes
