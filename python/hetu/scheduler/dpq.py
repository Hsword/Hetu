from .switchinference import SwitchInferenceTrainer
from ..layers import DPQEmbedding
from ..gpu_ops import add_op


class DPQTrainer(SwitchInferenceTrainer):
    def assert_use_multi(self):
        assert self.use_multi == self.separate_fields == 0

    def get_embed_layer(self):
        return DPQEmbedding(
            self.num_embed,
            self.embedding_dim,
            self.embedding_args['num_choices'],
            self.embedding_args['num_parts'],
            self.num_slot,
            self.batch_size,
            share_weights=self.embedding_args['share_weights'],
            mode=self.embedding_args['mode'],
            initializer=self.initializer,
            name='DPQEmb',
            ctx=self.ectx,
        )

    def get_eval_nodes(self):
        embed_input, dense_input, y_ = self.data_ops
        loss, prediction = self.model(
            self.embed_layer(embed_input), dense_input, y_)
        if self.embedding_args['mode'] == 'vq':
            loss = add_op(loss, self.embed_layer.reg)
        train_op = self.opt.minimize(loss)
        eval_nodes = {
            'train': [loss, prediction, y_, train_op, self.embed_layer.codebook_update],
        }
        test_embed_input = self.embed_layer.make_inference(embed_input)
        test_loss, test_prediction = self.model(
            test_embed_input, dense_input, y_)
        eval_nodes['validate'] = [test_loss, test_prediction, y_]
        return eval_nodes

    def get_eval_nodes_inference(self):
        embed_input, dense_input, y_ = self.data_ops
        test_embed_input = self.embed_layer.make_inference(embed_input)
        test_loss, test_prediction = self.model(
            test_embed_input, dense_input, y_)
        eval_nodes = {
            'validate': [test_loss, test_prediction, y_],
        }
        return eval_nodes
