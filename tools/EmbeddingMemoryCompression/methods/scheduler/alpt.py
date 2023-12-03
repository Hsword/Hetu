from .base import EmbeddingTrainer
from ..layers import ALPTEmbedding
from hetu.gpu_links import reorder_into_lookup, assign_alpt_embedding
import math


class ALPTEmbTrainer(EmbeddingTrainer):
    def assert_use_multi(self):
        assert self.use_multi == self.separate_fields == 0

    @ property
    def all_train_names(self):
        return (self.train_name, 'train_scale')

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
        from hetu.gpu_ops.AssignWithIndexedSlices import AssignWithIndexedSlicesOp, assign_with_indexedslices_op
        from hetu.gpu_ops.QuantizeALPTEmb import alpt_rounding_op
        from hetu.gpu_ops.Division import div_op
        from hetu.gpu_ops.Broadcast import broadcastto_op
        from hetu.gpu_ops.EmbeddingLookUp import embedding_lookup_op
        from hetu.gpu_ops.MultiplyConst import mul_byconst_op
        from hetu.gpu_ops.executor import gradients
        from hetu.initializers import GenEmpty
        embed_input, dense_input, y_ = self.data_ops
        embeddings = self.embed_layer(embed_input)
        loss, prediction = self.model(
            embeddings, dense_input, y_)
        train_op = self.opt.minimize(loss)
        idoffsets_op = None
        updated_emb_op = None
        dense_param_opt = []
        for op in train_op:
            if isinstance(op, AssignWithIndexedSlicesOp):
                updated_emb_op = op.inputs[2]
                idoffsets_op = updated_emb_op.inputs[2].inputs[1]
            else:
                dense_param_opt.append(op)

        self.var_lookup = GenEmpty()(
            (self.batch_size, self.num_slot, self.embedding_dim), name=f'lookup', trainable=False, ctx=self.ctx)
        scale = self.embed_layer.scale
        lookuped_scale = embedding_lookup_op(scale, embed_input, ctx=self.ctx)
        broadcasted_lookuped_scale = broadcastto_op(
            lookuped_scale, self.var_lookup)
        lookup = div_op(self.var_lookup, broadcasted_lookuped_scale)
        round_result = alpt_rounding_op(
            lookup, lookuped_scale, self.embed_layer.middle, self.embed_layer.digit, ctx=self.ctx)

        new_loss, new_prediction = self.model(
            round_result, dense_input, y_)
        dscale = gradients(new_loss, [scale])

        scale_unique, scale_deduplookup, scale_dedupgrad = dscale[0]

        scale_factor = 1 / \
            math.sqrt(self.batch_size * self.embedding_dim *
                      (2 ** (self.embed_layer.digit-1) - 1))
        scale_dedupgrad = mul_byconst_op(
            scale_dedupgrad, scale_factor, ctx=self.ctx)
        scale_opt = type(self.opt)(
            learning_rate=self.embedding_args['scale_lr'])
        if hasattr(self.opt, 'betatss'):
            scale_opt.betatss = self.opt.betatss
            scale_opt.betats_update_ops = self.opt.betats_update_ops
        scale_update = scale_opt.sparse_opt_op_type(
            scale_opt, scale, scale_unique, scale_deduplookup, scale_dedupgrad)
        scale_assign = assign_with_indexedslices_op(
            scale, scale_unique, scale_update)
        eval_nodes = {
            self.train_name: [loss, prediction, y_, updated_emb_op, idoffsets_op, dense_param_opt],
            'train_scale': [scale_unique, scale_update, scale_assign],
            self.validate_name: [loss, prediction, y_],
            self.test_name: [loss, prediction, y_],
        }
        return eval_nodes

    def train_step(self):
        var2arr = self.var2arr
        stream = self.stream
        first_stage_results = self.executor.run(
            self.train_name, dataloader_step=False)
        loss_val, predict_y, y_val = first_stage_results[:3]
        updated_emb = first_stage_results[3]
        idoffsets = first_stage_results[4]
        reorder_into_lookup(idoffsets, updated_emb,
                            var2arr[self.var_lookup], stream)
        second_stage_results = self.executor.run('train_scale')
        unique_indices, updated_scale = second_stage_results[:2]
        assign_alpt_embedding(var2arr[self.embed_layer.embedding_table], unique_indices,
                              updated_emb, updated_scale, self.embed_layer.middle, self.embed_layer.digit, stream)
        return loss_val.asnumpy(), predict_y.asnumpy(), y_val.asnumpy()

    def init_executor(self, eval_nodes):
        super().init_executor(eval_nodes)
        if self.phase == 'train':
            self.executor.subexecutor['train_scale'].inference = False
