import numpy as np

from .base import EmbeddingTrainer
from ..layers import AutoDimEmbedding, AutoDimRetrainEmbedding
from ..ndarray import empty
from ..gpu_links import all_fro_norm, matrix_elementwise_divide_const, \
    all_add_, div_n_mul_, matrix_elementwise_minus, matrix_elementwise_add_simple, \
    sgd_update, assign_embedding_with_indexedslices


class AutoDimTrainer(EmbeddingTrainer):
    def __init__(self, dataset, model, opt, args, **kargs):
        self.retraining = (args.phase == 'test')
        super().__init__(dataset, model, opt, args, **kargs)

    def assert_use_multi(self):
        assert self.use_multi == self.separate_fields == int(self.retraining)

    def fit(self):
        self.embed_layer = self.get_embed_layer()
        self.log_func(f'Embedding layer: {self.embed_layer}')
        self.alpha = self.embed_layer.alpha
        self.r = self.embedding_args['r']
        self.alpha_lr = self.embedding_args['alpha_lr']
        eval_nodes = self.get_eval_nodes()
        self.lookups = self.embed_layer.lookups
        self.init_executor(eval_nodes)

        self.executor.subexecutor['alpha'].inference = False
        self.executor.subexecutor['lookups'].inference = False
        self.executor.subexecutor['allgrads'].inference = False
        self.get_arch_params(self.executor.config.placeholder_to_arr_map)

        self.try_load_ckpt()
        self.train_step = self.first_stage_train_step

        self.run_once()

        self.log_func('Switch to re-training stage!!!')
        # re-init save topk
        self.prepare_path_for_retrain()
        self.init_ckpts()

        # re-init executor
        self.executor.return_tensor_values()
        self.set_use_multi(1)
        self.retraining = True
        self.data_ops = self.get_data()
        dim_choices = self.make_retrain(self.executor.config.comp_stream)

        self.reset_for_retrain()
        self.embed_layer = self.get_embed_layer_retrain(dim_choices)
        self.log_func(f'New embedding layer: {self.embed_layer}')
        eval_nodes = super().get_eval_nodes()

        self.seed += 1  # use a different seed

        self.init_executor(eval_nodes)
        self.train_step = super().train_step

        # re-run
        self.run_once()

    @property
    def all_train_names(self):
        return (self.train_name, 'alpha', 'lookups')

    @property
    def all_validate_names(self):
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

    def test(self):
        raise NotImplementedError

    def get_embed_layer(self):
        self.dim_candidates = [2]
        while self.dim_candidates[-1] < self.embedding_dim:
            self.dim_candidates.append(2 * self.dim_candidates[-1])
        self.dim_candidates[-1] = self.embedding_dim
        self.log_func(f'Dimension candidates: {self.dim_candidates}')
        self.num_cands = len(self.dim_candidates)
        return AutoDimEmbedding(
            self.num_embed,
            self.dim_candidates,
            self.num_slot,
            self.batch_size,
            initializer=self.initializer,
            name='AutoDimEmb',
            ctx=self.ectx,
        )

    def get_embed_layer_retrain(self, dim_choices):
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

    def get_eval_nodes(self):
        from ..gpu_ops.AssignWithIndexedSlices import AssignWithIndexedSlicesOp
        from ..gpu_ops import gradients
        from ..initializers import GenEmpty
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
            (self.batch_size, self.num_slot, dim), f'lookups{dim}', False, self.ctx) for dim in self.dim_candidates}
        new_loss, new_pred = self.model(self.embed_layer.make_embed(
            self.var_lookups), dense_input, y_)
        alpha_grad = gradients(new_loss, [self.alpha])

        eval_nodes = {
            'train': [loss, prediction, y_, param_opts],
            'lookups': lookups + dedup_lookups + unique_indices + param_opts,
            'validate': [loss, prediction, y_],
            'allgrads': [dalpha_op] + dup_dembed_ops + dembed_ops + dparam_ops,
            'alpha': [alpha_grad],
        }

        return eval_nodes

    def make_retrain(self, stream):
        from ..gpu_links import argmax
        from ..ndarray import empty
        dim_choice = empty((self.num_slot, ), ctx=self.ctx, dtype=np.int32)
        argmax(self.alpha.tensor_value, dim_choice, 1, stream=stream)
        stream.sync()
        dim_choice = [self.dim_candidates[int(ind)]
                      for ind in dim_choice.asnumpy()]
        self.log_func('Dimension choices:', dim_choice)
        return dim_choice
