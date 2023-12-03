from .base import EmbeddingTrainer
from .multistage import MultiStageTrainer
from ..layers import AutoDimEmbedding, AutoDimRetrainEmbedding
from hetu.ndarray import empty
from hetu.gpu_links import all_fro_norm, matrix_elementwise_divide_const, \
    all_add_, div_n_mul_, matrix_elementwise_minus, matrix_elementwise_add_simple, \
    sgd_update, assign_embedding_with_indexedslices
from hetu.optimizer import SGDOptimizer
import numpy as np
from copy import deepcopy


class AutoDimOverallTrainer(MultiStageTrainer):
    @property
    def legal_stages(self):
        return (1, 2)

    def assert_use_multi(self):
        pass

    def get_data(self):
        pass

    def copy_args_with_stage(self, stage):
        new_args = deepcopy(self.args)
        new_args['embedding_args']['stage'] = stage
        if stage == 1:
            new_args['use_multi'] = new_args['separate_fields'] = 0
        else:
            new_args['use_multi'] = new_args['separate_fields'] = 1
        return new_args

    def get_dim_candidates(self):
        self.dim_candidates = [2]
        while self.dim_candidates[-1] < self.embedding_dim:
            self.dim_candidates.append(2 * self.dim_candidates[-1])
        self.dim_candidates[-1] = self.embedding_dim
        self.log_func(f'Dimension candidates: {self.dim_candidates}')
        self.num_cands = len(self.dim_candidates)

    def fit(self):
        stage = self.stage
        self.get_dim_candidates()
        # two stages: training, retraining
        if stage == 1:
            self.trainer = AutoDimTrainer(
                self.dataset, self.model, self.opt, self.copy_args_with_stage(1))
            self.trainer.set_dim_candidates(self.dim_candidates)
        else:
            self.trainer = None
        self.retrainer = AutoDimRetrainer(
            self.dataset, self.model, self.opt, self.copy_args_with_stage(2))
        self.retrainer.set_dim_candidates(self.dim_candidates)
        self.retrainer.prepare_path_for_retrain('retrain')

        if self.trainer is not None:
            self.trainer.fit()
            ep, part = self.trainer.get_best_meta()
            self.retrainer.load_ckpt = self.trainer.join(f'ep{ep}_{part}.pkl')
            del self.trainer
        self.retrainer.fit()

    def test(self):
        stage = self.stage
        self.get_dim_candidates()
        if stage == 1:
            trainer = AutoDimTrainer(
                self.dataset, self.model, self.opt, self.copy_args_with_stage(1))
        else:
            trainer = AutoDimRetrainer(
                self.dataset, self.model, self.opt, self.copy_args_with_stage(2))
        trainer.set_dim_candidates(self.dim_candidates)
        trainer.test()


class AutoDimTrainer(EmbeddingTrainer):
    def assert_use_multi(self):
        assert self.use_multi == self.separate_fields == 0

    def set_dim_candidates(self, dim_candidates):
        self.dim_candidates = dim_candidates
        self.num_cands = len(dim_candidates)

    def fit(self):
        self.r = self.embedding_args['r']
        self.alpha_lr = self.embedding_args['alpha_lr']

        # first stage
        self.log_func('Training the first stage')
        self.init_ckpts()
        self.embed_layer = self.get_embed_layer()
        self.alpha = self.embed_layer.alpha
        self.log_func(f'Embedding layer: {self.embed_layer}')
        if self.embedding_args['ignore_second']:
            eval_nodes = self.get_eval_nodes_without_second_order()
        else:
            eval_nodes = self.get_eval_nodes()
        self.lookups = self.embed_layer.lookups
        self.init_executor(eval_nodes)
        self.load_into_executor(self.try_load_ckpt())

        if self.embedding_args['ignore_second']:
            self.executor.subexecutor['alpha'].inference = True
            self.train_step = self.first_stage_train_step_without_second_order
        else:
            self.executor.subexecutor['alpha'].inference = False
            self.executor.subexecutor['lookups'].inference = False
            self.executor.subexecutor['allgrads'].inference = False
            self.get_arch_params(
                self.executor.config.placeholder_to_arr_map)
            self.train_step = self.first_stage_train_step

        self.run_once()

        self.load_best_ckpt()
        self.executor.return_tensor_values()

    @property
    def all_train_names(self):
        if self.embedding_args['ignore_second']:
            return (self.train_name,)
        else:
            return (self.train_name, 'alpha', 'lookups')

    @property
    def all_validate_names(self):
        if self.embedding_args['ignore_second']:
            return (self.validate_name, 'alpha')
        else:
            return (self.validate_name, 'allgrads')

    def get_arch_params(self, var2arr):
        copy_params = {}
        ori_params = {}
        copy_lookups = {}
        copy_unique_indices = {}
        copy_dedup_lookups = {}
        for node, arr in var2arr.items():
            if node is not self.alpha and node.trainable and not node.is_embed:
                copy_params[node] = empty(arr.shape, ctx=arr.ctx)
                ori_params[node] = arr
        for dim, lookup in self.lookups.items():
            copy_lookups[lookup] = empty(
                (self.batch_size, self.num_slot, dim), ctx=self.ctx)
            copy_unique_indices[lookup] = empty(
                (self.batch_size, self.num_slot), ctx=self.ctx, dtype=np.int32)
            copy_dedup_lookups[lookup] = empty(
                (self.batch_size, self.num_slot, dim), ctx=self.ctx)
        self.copy_params = copy_params
        self.ori_params = ori_params
        self.copy_lookups = copy_lookups
        self.copy_unique_indices = copy_unique_indices
        self.copy_dedup_lookups = copy_dedup_lookups
        self.workspace = empty(
            (len(self.copy_params) + len(self.copy_lookups),), ctx=self.ctx)
        self.norm = empty((1,), ctx=self.ctx)
        self.dalpha_values = [empty(
            (self.num_slot, self.num_cands), ctx=self.ctx) for _ in range(2)]

    def copy_from(self, stream):
        for node, arr in self.ori_params.items():
            self.copy_params[node]._async_copyfrom(arr, stream)

    def copy_from_lookups(self, lookups, dedup_lookups, unique_indices, stream):
        for node, l, dl, ui in zip(self.lookups.values(), lookups, dedup_lookups, unique_indices):
            self.copy_lookups[node]._async_copyfrom(l, stream)
            self.copy_dedup_lookups[node]._async_copyfrom(dl, stream)
            self.copy_unique_indices[node]._async_copyfrom(ui, stream)

    def copy_to(self, var2arr, stream):
        for node, arr in self.ori_params.items():
            arr._async_copyfrom(self.copy_params[node], stream)
        for dim, node in self.lookups.items():
            var2arr[self.var_lookups[dim]]._async_copyfrom(
                self.copy_lookups[node], stream)
            assign_embedding_with_indexedslices(
                var2arr[node.inputs[0]], self.copy_unique_indices[node], self.copy_dedup_lookups[node], stream)

    def first_stage_train_step(self):
        var2arr = self.var2arr
        stream = self.stream
        # copy original parameters (if embedding, copy lookuped ones) and update using train data to get temp model
        self.copy_from(stream)
        lookups = self.executor.run(
            'lookups', dataloader_step=False, inference=False)  # train data
        self.copy_from_lookups(
            lookups[:self.num_cands], lookups[self.num_cands:self.num_cands * 2], lookups[self.num_cands * 2:self.num_cands * 3], stream)

        # get all gradients using validation data; memorize dalpha
        allgrads = self.executor.run('allgrads', inference=True)  # valid data

        # return to the original model parameters
        self.copy_to(var2arr, stream)

        dalpha = allgrads[0]
        dup_demb = allgrads[1:1+self.num_cands]
        demb = allgrads[1+self.num_cands:1+self.num_cands*2]
        dparams = allgrads[1+self.num_cands*2:]
        self.dalpha_values[0]._async_copyfrom(dalpha, stream)

        # get real r with gradients
        grads_for_norm = demb + dparams
        all_fro_norm(grads_for_norm, self.workspace, self.norm, stream)
        matrix_elementwise_divide_const(self.r, self.norm, self.norm, stream)

        # add r to parameters; embedding only lookups; get gradp
        # WARNING: if the topology changes, need to carefully align tensors and others
        tensors = [var2arr[x] for x in self.var_lookups.values()] + \
            list(self.ori_params.values())
        grads_for_tensors = dup_demb + dparams

        all_add_(tensors, grads_for_tensors, self.norm, stream=stream)
        gradp = self.executor.run(
            'alpha', dataloader_step=False, inference=False)  # train data
        self.dalpha_values[1]._async_copyfrom(gradp[0], stream)

        # minus 2r from parameters; embedding only lookups; get gradn
        all_add_(tensors, grads_for_tensors, self.norm, -2, stream=stream)
        gradn = self.executor.run(
            'alpha', dataloader_step=False, inference=True)  # train data

        # add r for parameters to recover; no need for lookups
        all_add_(tensors[self.num_cands:], dparams, self.norm, stream=stream)

        # compute real dalpha and update
        matrix_elementwise_minus(
            self.dalpha_values[1], gradn[0], self.dalpha_values[1], stream)
        div_n_mul_(self.dalpha_values[1],
                   self.norm, -self.alpha_lr / 2, stream)
        matrix_elementwise_add_simple(
            self.dalpha_values[0], self.dalpha_values[1], self.dalpha_values[0], stream)
        sgd_update(var2arr[self.alpha], self.dalpha_values[0],
                   self.alpha_lr, 0, stream)

        results = self.executor.run(
            self.train_name, convert_to_numpy_ret_vals=True, inference=True)  # train data
        return results[:3]

    def first_stage_train_step_without_second_order(self):
        results = self.executor.run(
            self.train_name, convert_to_numpy_ret_vals=True)
        self.executor.run('alpha')
        return results[:3]

    def get_embed_layer(self):
        return AutoDimEmbedding(
            self.num_embed,
            self.dim_candidates,
            self.num_slot,
            self.batch_size,
            initializer=self.initializer,
            name='AutoDimEmb',
            ctx=self.ectx,
        )

    def get_eval_nodes(self):
        from hetu.gpu_ops.AssignWithIndexedSlices import AssignWithIndexedSlicesOp
        from hetu.gpu_ops import gradients
        from hetu.initializers import GenEmpty
        embed_input, dense_input, y_ = self.data_ops
        loss, prediction = self.model(
            self.embed_layer(embed_input), dense_input, y_)
        train_op = self.opt.minimize(loss)
        lookups = []
        dedup_lookups = []
        unique_indices = []
        dembed_ops = []
        dup_dembed_ops = []
        dparam_ops = []
        dalpha_op = None
        param_opts = []
        for op in train_op:
            if op.inputs[0] is self.alpha:
                dalpha_op = op.inputs[1]
            else:
                param_opts.append(op)
                if isinstance(op, AssignWithIndexedSlicesOp):
                    sparse_opt = op.inputs[2]
                    deduplookup = sparse_opt.inputs[2]
                    lookups.append(deduplookup.inputs[0])
                    unique_indices.append(sparse_opt.inputs[1])
                    dedup_lookups.append(deduplookup)
                    dedupgrad = sparse_opt.inputs[3]
                    dembed_ops.append(dedupgrad)
                    dup_dembed_ops.append(deduplookup.inputs[0])
                else:
                    dparam_ops.append(op.inputs[1])
        assert dalpha_op is not None

        self.var_lookups = {dim: GenEmpty()(
            (self.batch_size, self.num_slot, dim), name=f'lookups{dim}', trainable=False, ctx=self.ctx) for dim in self.dim_candidates}
        new_loss, new_pred = self.model(self.embed_layer.make_embed(
            self.var_lookups), dense_input, y_)
        alpha_grad = gradients(new_loss, [self.alpha])

        eval_nodes = {
            self.train_name: [loss, prediction, y_, param_opts],
            'lookups': lookups + dedup_lookups + unique_indices + param_opts,
            self.validate_name: [loss, prediction, y_],
            self.test_name: [loss, prediction, y_],
            'allgrads': [dalpha_op] + dup_dembed_ops + dembed_ops + dparam_ops,
            'alpha': [alpha_grad],
        }

        return eval_nodes

    def get_eval_nodes_without_second_order(self):
        embed_input, dense_input, y_ = self.data_ops
        loss, prediction = self.model(
            self.embed_layer(embed_input), dense_input, y_)
        train_op = self.opt.minimize(loss)
        param_opts = []
        alpha_update = None
        for op in train_op:
            if op.inputs[0] is self.alpha:
                alpha_update = op
            else:
                param_opts.append(op)
        assert alpha_update is not None
        alpha_optimizer = SGDOptimizer(learning_rate=self.alpha_lr)
        new_alpha_update = alpha_optimizer.opt_op_type(
            self.alpha, alpha_update.inputs[1], alpha_optimizer)

        eval_nodes = {
            self.train_name: [loss, prediction, y_, param_opts],
            self.validate_name: [loss, prediction, y_],
            self.test_name: [loss, prediction, y_],
            'alpha': [new_alpha_update],
        }

        return eval_nodes


class AutoDimRetrainer(EmbeddingTrainer):
    def assert_use_multi(self):
        assert self.use_multi == self.separate_fields == 1

    def assert_load_args(self, load_args):
        assert self.args['model'].__name__ == load_args['model']
        for k in ['method', 'dim', 'dataset', 'compress_rate']:
            assert load_args[k] == self.args[
                k], f'Current argument({k}) {self.args[k]} different from loaded {load_args[k]}'
        for k in ['bs', 'opt', 'lr', 'num_test_every_epoch', 'seed', 'embedding_args']:
            if load_args[k] != self.args[k]:
                self.log_func(
                    f'Warning: current argument({k}) {self.args[k]} different from loaded {load_args[k]}')

    def set_dim_candidates(self, dim_candidates):
        self.dim_candidates = dim_candidates
        self.num_cands = len(dim_candidates)

    def try_load_ckpt(self):
        meta = super().try_load_ckpt()
        assert meta is not None
        stage = meta['args']['embedding_args']['stage']
        assert stage in (1, 2)
        if stage == 1:
            dim_choices = self.make_retrain(meta['state_dict']['alphas'])
            self.args['dim_choices'] = dim_choices
            meta = self.load_only_parameters(meta)
            st = meta['state_dict']
            if self.embedding_args['reset_retrain']:
                meta['state_dict'] = {}
                self.log_func('Reset parameters! Retrain from scratch.')
            else:
                src_embeddings = {k: st.pop(f'AutoDimEmb{k}')
                                  for k in self.dim_candidates}
                src_weights = {k: st.pop(f'weight{k}')
                               for k in self.dim_candidates}
                src_biases = {k: st.pop(f'bias{k}')
                              for k in self.dim_candidates}
                feature_offset = 0
                for i, (nemb, ndim) in enumerate(zip(self.num_embed_separate, dim_choices)):
                    ending_offset = feature_offset + nemb
                    st[f'AutoDimNew_{i}'] = src_embeddings[ndim][feature_offset:ending_offset]
                    st[f'AutoDimNew_{i}_weight'] = src_weights[ndim][i]
                    st[f'AutoDimNew_{i}_bias'] = src_biases[ndim][i].squeeze(0)
                    feature_offset = ending_offset
                self.log_func('Retrain from pre-trained ckpt.')
        else:
            dim_choices = meta['args']['dim_choices']
            self.args['dim_choices'] = dim_choices
        return meta

    def fit(self):
        self.log_func('Start retraining...')
        # self.prepare_path_for_retrain()
        self.init_ckpts()
        meta = self.try_load_ckpt()
        self.embed_layer = self.get_embed_layer(self.args['dim_choices'])
        self.log_func(f'New embedding layer: {self.embed_layer}')
        eval_nodes = self.get_eval_nodes()
        self.init_executor(eval_nodes)
        self.load_into_executor(meta)
        self.run_once()

    def test(self):
        assert self.phase == 'test' and self.load_ckpt is not None, 'Checkpoint should be given in testing.'
        meta = self.try_load_ckpt()
        self.embed_layer = self.get_embed_layer(self.args['dim_choices'])
        self.log_func(f'Embedding layer: {self.embed_layer}')
        eval_nodes = self.get_eval_nodes_inference()
        self.init_executor(eval_nodes)
        self.load_into_executor(meta)

        log_file = open(self.result_file,
                        'w') if self.result_file is not None else None
        with self.timing():
            test_loss, test_metric, _ = self.test_once()
        test_time = self.temp_time[0]
        results = {
            'avg_test_loss': test_loss,
            f'test_{self.monitor}': test_metric,
            'test_time': test_time,
        }
        printstr = ', '.join(
            [f'{key}: {value:.4f}' for key, value in results.items()])
        self.log_func(printstr)
        if log_file is not None:
            print(printstr, file=log_file, flush=True)

    def get_embed_layer(self, dim_choices):
        emb = []
        for i, (nemb, cdim) in enumerate(zip(self.num_embed_separate, dim_choices)):
            emb.append(AutoDimRetrainEmbedding(
                nemb,
                cdim,
                self.embedding_dim,
                initializer=self.initializer,
                name=f'AutoDimNew_{i}',
                ctx=self.ectx,
            ))
        return emb

    def make_retrain(self, alpha):
        dim_choice = np.argmax(alpha, axis=1)
        dim_choice = [self.dim_candidates[int(ind)] for ind in dim_choice]
        self.log_func('Dimension choices:', dim_choice)
        return dim_choice
