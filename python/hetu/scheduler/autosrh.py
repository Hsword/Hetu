from .switchinference import SwitchInferenceTrainer
from ..layers import AutoSrhEmbedding, AutoSrhRetrainEmbedding
from ..ndarray import empty
from ..gpu_links import multiply_grouping_alpha, num_less_than, set_mask_less_than
from ..optimizer import SGDOptimizer
import numpy as np


class AutoSrhTrainer(SwitchInferenceTrainer):
    def assert_use_multi(self):
        assert self.use_multi == self.separate_fields == 0

    @property
    def all_train_names(self):
        return (self.train_name, 'embed')

    @property
    def all_validate_names(self):
        return (self.validate_name, 'alpha')

    @property
    def sparse_name(self):
        return 'AutoSrhEmb'

    def get_data(self):
        nsplit = self.embedding_args['nsplit']
        embed_input, dense_input, y_ = super().get_data()
        grouping_indices = self.dataset.get_whole_frequency_grouping(
            embed_input.dataloaders[self.train_name].raw_data, nsplit)
        self.grouping_indices = grouping_indices.astype(np.int32)
        counter = [(grouping_indices == value).sum().item()
                   for value in range(nsplit)]
        self.log_func(f"AutoSrh feature nums (from low to high): {counter}")
        return embed_input, dense_input, y_

    def get_embed_layer(self):
        return AutoSrhEmbedding(
            self.num_embed,
            self.embedding_dim,
            self.embedding_args['nsplit'],
            self.grouping_indices,
            self.form,
            initializer=self.initializer,
            name=self.sparse_name,
            ctx=self.ectx,
        )

    def get_embed_layer_retrain(self, embedding_table, mask):
        return AutoSrhRetrainEmbedding(
            self.num_embed,
            self.embedding_dim,
            embedding_table,
            mask,
            self.form,
            name=self.sparse_name,
            ctx=self.ectx,
        )

    def fit(self):
        self.save_dir = self.args['save_dir']
        meta = self.try_load_ckpt()
        # warming up
        warm_up_eps = self.embedding_args['warm_start_epochs']
        warmed_up_embeddings = None
        if warm_up_eps > 0 and (meta is None or meta['args']['stage'] == 1):
            self.args['stage'] = 1
            assert self.save_topk > 0, 'Need to load the best ckpt for dimension selection; please set save_topk a positive integer.'
            self.log_func(f'Warming up {warm_up_eps} epoch')
            self.embed_layer = super().get_embed_layer()
            eval_nodes = super().get_eval_nodes()
            self.init_executor(eval_nodes)
            self.load_into_executor(meta)

            self._original_nepoch = self.nepoch
            self.nepoch = warm_up_eps
            self.run_once()
            self.nepoch = self._original_nepoch
            self.load_best_ckpt()
            self.executor.return_tensor_values()
            del self.executor
            warmed_up_embeddings = self.embed_layer.embedding_table
            del self.embed_layer

        if meta is None or meta['args']['stage'] <= 2:
            # start training
            self.log_func('Start training...')
            self.args['stage'] = 2
            assert self.save_topk > 0, 'Need to load the best ckpt for dimension selection; please set save_topk a positive integer.'
            self.train_step = self.first_stage_train_step
            self.embed_layer = self.get_embed_layer()
            if warmed_up_embeddings is not None:
                self.embed_layer.embedding_table = warmed_up_embeddings
            eval_nodes = self.get_eval_nodes()
            self.init_executor(eval_nodes)
            self.executor.subexecutor['alpha'].inference = False
            self.executor.subexecutor['embed'].inference = False

            if meta is None or meta['args']['stage'] == 2:
                self.load_into_executor(meta)
            self.prepare_path_for_retrain('main')
            self.init_ckpts()
            self.run_once()
            # prune and test
            stream = self.executor.config.comp_stream
            stream.sync()
            self.load_best_ckpt()
            self.executor.return_tensor_values()
            workspace = empty((self.num_embed, self.embedding_dim),
                              ctx=self.ectx, dtype=np.int32)
            prune_rate = empty((1,), ctx=self.ectx)
            embed_arr = self.embed_layer.embedding_table.tensor_value
            grouping_arr = self.embed_layer.group_indices.tensor_value
            alpha_arr = self.embed_layer.alpha.tensor_value
            multiply_grouping_alpha(embed_arr, grouping_arr, alpha_arr, stream)
            l, r = 0., 100.
            cnt = 0
            while l < r:
                cnt += 1
                mid = (l + r) / 2
                num_less_than(
                    embed_arr, workspace, prune_rate, mid, None, stream)
                stream.sync()
                sparse_items = prune_rate.asnumpy()[0]
                sparse_rate = sparse_items / self.num_embed / self.embedding_dim
                if abs(sparse_rate - self.prune_rate) < 1e-3 * self.compress_rate:
                    break
                elif sparse_rate > self.prune_rate:
                    r = mid
                else:
                    l = mid
                if cnt > 100:
                    break
            self.log_func(
                f'Final prune rate: {sparse_rate} (target {self.prune_rate}); final threshold: {mid}')
            set_mask_less_than(embed_arr, workspace, mid, stream)
        else:
            embed_arr = meta['state_dict'][self.sparse_name]
            workspace = meta['state_dict'][f'{self.sparse_name}_mask']

        # re-train
        self.args['stage'] = 3
        self.log_func('Start retraining...')
        self.prepare_path_for_retrain('retrain')
        self.embed_layer = self.get_embed_layer_retrain(embed_arr, workspace)
        eval_nodes = super().get_eval_nodes()
        self.init_executor(eval_nodes)
        self.init_ckpts()
        self.train_step = super().train_step
        self.run_once()
        stream = self.executor.config.comp_stream
        stream.sync()
        self.load_best_ckpt()
        self.executor.return_tensor_values()

        self.check_inference()

    def get_eval_nodes(self):
        from ..gpu_ops import add_op, reduce_sum_op, mul_byconst_op, abs_op, param_clip_op
        from ..gpu_ops import div_op, broadcastto_op, reduce_mean_op, addbyconst_op
        embed_input, dense_input, y_ = self.data_ops
        embeddings = self.embed_layer(embed_input)
        loss,loss2, prediction = self.model(
            embeddings, dense_input, y_)
        # add l1 loss for alpha
        loss = add_op(loss, mul_byconst_op(reduce_sum_op(
            abs_op(self.embed_layer.alpha), axes=[0, 1]), self.embedding_args['alpha_l1']))
        loss2 = add_op(loss2, mul_byconst_op(reduce_sum_op(
            abs_op(self.embed_layer.alpha), axes=[0, 1]), self.embedding_args['alpha_l1']))
        train_op = self.opt.minimize(loss)
        alpha_update = None
        embed_update = None
        dense_param_updates = []
        for op in train_op:
            if op.inputs[0] is self.embed_layer.alpha:
                alpha_update = op
            elif op.inputs[0] is self.embed_layer.embedding_table:
                embed_update = op
            else:
                dense_param_updates.append(op)
        alpha_grad = alpha_update.inputs[1]
        # add alpha norm onto the grad
        alpha_norm = div_op(alpha_grad, broadcastto_op(addbyconst_op(
            reduce_mean_op(abs_op(alpha_grad), axes=-1, keepdims=True), 1e-8), alpha_grad))
        # use a separate sgd optimizer for alpha
        alpha_opt = SGDOptimizer(self.embedding_args['alpha_lr'])
        alpha_update = alpha_opt.opt_op_type(
            alpha_update.inputs[0], alpha_norm, alpha_opt)
        alpha_clip_op = param_clip_op(
            self.embed_layer.alpha, alpha_update, 0., 1., self.ctx)
        # train here is for warm-up and re-train
        eval_nodes = {
            self.train_name: [loss,loss2, prediction, y_, embed_update, *dense_param_updates],
            'alpha': [alpha_clip_op],
            'embed': [loss,loss2, prediction, y_, embed_update],
            self.validate_name: [loss,loss2, prediction, y_],
            self.test_name: [loss,loss2, prediction, y_],
        }
        return eval_nodes

    def first_stage_train_step(self):
        loss_val,loss2_val, predict_y, y_val = self.executor.run(
            'embed', convert_to_numpy_ret_vals=True)[:4]
        self.executor.run('alpha')
        return loss_val,loss2_val, predict_y, y_val
