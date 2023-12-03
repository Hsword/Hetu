from .base import EmbeddingTrainer
from .multistage import MultiStageTrainer
from .switchinference import SwitchInferenceTrainer
from ..layers import AutoSrhEmbedding, AutoSrhRetrainEmbedding
from hetu.optimizer import SGDOptimizer
import numpy as np
import os.path as osp


class AutoSrhOverallTrainer(MultiStageTrainer):
    # in autosrh, the parameters are inherited from the previous stage;
    # but the optimizer states is new in every stage

    @property
    def legal_stages(self):
        return (1, 2, 3)

    def get_grouping_indices(self):
        nsplit = self.embedding_args['nsplit']
        grouping_indices = self.dataset.get_whole_frequency_grouping(
            self.data_ops[0].dataloaders[self.train_name].raw_data, nsplit).astype(np.int32)
        self.grouping_indices = grouping_indices
        counter = [(grouping_indices == value).sum().item()
                   for value in range(nsplit)]
        self.log_func(f"AutoSrh feature nums (from low to high): {counter}")

    def fit(self):
        stage = self.stage
        self.get_grouping_indices()
        # three stages: warmup, training, retraining
        if stage == 1:
            self.warmup_trainer = EmbeddingTrainer(
                self.dataset, self.model, self.opt, self.copy_args_with_stage(1), self.data_ops)
        else:
            self.warmup_trainer = None
        if stage <= 2:
            self.trainer = AutoSrhTrainer(
                self.dataset, self.model, self.opt, self.copy_args_with_stage(2))
            self.trainer.set_grouping_indices(self.grouping_indices)
            self.trainer.prepare_path_for_retrain('train')
        else:
            self.trainer = None
        self.retrainer = AutoSrhRetrainer(
            self.dataset, self.model, self.opt, self.copy_args_with_stage(3), self.data_ops)
        self.retrainer.set_grouping_indices(self.grouping_indices)
        self.retrainer.prepare_path_for_retrain('retrain')

        if self.warmup_trainer is not None:
            self.warmup_trainer.fit()
            ep, part = self.warmup_trainer.get_best_meta()
            self.trainer.load_ckpt = self.warmup_trainer.join(
                f'ep{ep}_{part}.pkl')
            self.warmup_trainer.executor.return_tensor_values()
            del self.warmup_trainer
        if self.trainer is not None:
            self.trainer.fit()
            ep, part = self.trainer.get_best_meta()
            self.retrainer.load_ckpt = self.trainer.join(f'ep{ep}_{part}.pkl')
            del self.trainer
        self.retrainer.fit()

    def test(self):
        stage = self.stage
        self.get_grouping_indices()
        assert stage >= 2
        if stage == 2:
            trainer = AutoSrhTrainer(
                self.dataset, self.model, self.opt, self.args, self.data_ops)
        else:
            trainer = AutoSrhRetrainer(
                self.dataset, self.model, self.opt, self.args, self.data_ops)
        trainer.set_grouping_indices(self.grouping_indices)
        trainer.test()


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

    def set_grouping_indices(self, grouping_indices):
        self.grouping_indices = grouping_indices

    def try_load_ckpt(self):
        meta = super().try_load_ckpt()
        assert meta is not None
        st = meta['state_dict']
        stage = meta['args']['embedding_args']['stage']
        assert stage in (1, 2)
        if stage == 1:
            assert self.sparse_name not in st
            st[self.sparse_name] = st.pop('Embedding')
            meta = self.load_only_parameters(meta)
        return meta

    def get_embed_layer(self):
        return AutoSrhEmbedding(
            self.num_embed,
            self.embedding_dim,
            self.embedding_args['nsplit'],
            self.grouping_indices,
            initializer=self.initializer,
            name=self.sparse_name,
            ctx=self.ectx,
        )

    def fit(self):
        self.log_func('Start training...')
        assert self.load_ckpt is not None, 'Need to load the best ckpt for warm up!'
        assert self.save_topk > 0, 'Need to load the best ckpt for dimension selection; please set save_topk a positive integer.'

        # start training
        self.init_ckpts()
        self.embed_layer = self.get_embed_layer()
        self.log_func(f'Embedding layer: {self.embed_layer}')
        eval_nodes = self.get_eval_nodes()
        self.init_executor(eval_nodes)
        self.executor.subexecutor['alpha'].inference = False
        self.executor.subexecutor['embed'].inference = False

        self.load_into_executor(self.try_load_ckpt())
        self.run_once()

        # prune
        stream = self.executor.config.comp_stream
        stream.sync()
        self.load_best_ckpt()
        self.executor.return_tensor_values()

    def get_eval_nodes(self):
        from hetu.gpu_ops import add_op, reduce_sum_op, mul_byconst_op, abs_op, param_clip_op
        from hetu.gpu_ops import div_op, broadcastto_op, reduce_mean_op, addbyconst_op
        embed_input, dense_input, y_ = self.data_ops
        embeddings = self.embed_layer(embed_input)
        loss, prediction = self.model(
            embeddings, dense_input, y_)
        # add l1 loss for alpha
        loss = add_op(loss, mul_byconst_op(reduce_sum_op(
            abs_op(self.embed_layer.alpha), axes=[0, 1]), self.embedding_args['alpha_l1']))
        # loss2 = add_op(loss2, mul_byconst_op(reduce_sum_op(
        # abs_op(self.embed_layer.alpha), axes=[0, 1]), self.embedding_args['alpha_l1']))
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
            self.train_name: [loss, prediction, y_, embed_update, *dense_param_updates],
            'alpha': [alpha_clip_op],
            'embed': [loss, prediction, y_, embed_update],
            self.validate_name: [loss, prediction, y_],
            self.test_name: [loss, prediction, y_],
        }
        return eval_nodes

    def train_step(self):
        loss_val, predict_y, y_val = self.executor.run(
            'embed', convert_to_numpy_ret_vals=True)[:3]
        self.executor.run('alpha')
        return loss_val, predict_y, y_val

    def test(self):
        EmbeddingTrainer.test(self)


class AutoSrhRetrainer(SwitchInferenceTrainer):
    @property
    def sparse_name(self):
        return 'AutoSrhEmb'

    def set_grouping_indices(self, grouping_indices):
        self.grouping_indices = grouping_indices

    def try_load_ckpt(self):
        meta = super().try_load_ckpt()
        assert meta is not None
        stage = meta['args']['embedding_args']['stage']
        assert stage in (2, 3)
        if stage == 2:
            meta = self.load_only_parameters(meta)
        return meta

    def get_embed_layer(self):
        return AutoSrhRetrainEmbedding(
            self.num_embed,
            self.embedding_dim,
            self.embedding_args['nsplit'],
            self.grouping_indices,
            self.form,
            initializer=self.initializer,
            name=self.sparse_name,
            ctx=self.ectx,
        )

    def fit(self):
        # re-train
        self.log_func('Start retraining...')
        # self.prepare_path_for_retrain('retrain')
        self.embed_layer = self.get_embed_layer()
        eval_nodes = self.get_eval_nodes()
        self.init_executor(eval_nodes)
        self.init_ckpts()
        self.load_into_executor(self.try_load_ckpt())
        self.run_once()
        stream = self.executor.config.comp_stream
        stream.sync()
        self.load_best_ckpt()
        self.executor.return_tensor_values()

        embed_arr = self.embed_layer.embedding_table.tensor_value.asnumpy()
        grouping_arr = self.grouping_indices
        alpha_arr = self.embed_layer.alpha.tensor_value.asnumpy()
        embed_arr = embed_arr * alpha_arr[grouping_arr]
        l, r = 0., 100.
        cnt = 0
        while l < r:
            cnt += 1
            mid = (l + r) / 2
            sparse_items = (np.abs(embed_arr) < mid).sum()
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
        mask = (np.abs(embed_arr) < mid)
        embed_arr[mask] = 0
        self.embed_layer.embedding_table.tensor_value[:] = embed_arr

        self.check_inference()

    def test(self):
        # use filename to distinguish whether test middle ckpt or final ckpt
        test_final = osp.split(self.load_ckpt)[-1].startswith('final')
        if test_final:
            super().test()
        else:
            EmbeddingTrainer.test(self)
