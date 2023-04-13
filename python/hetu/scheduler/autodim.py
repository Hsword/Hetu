import os
import os.path as osp
import numpy as np

from .base import BaseTrainer
from ..gpu_ops import Executor
from ..ndarray import empty, IndexedSlices
from ..gpu_links import all_fro_norm, matrix_elementwise_divide_const, \
    all_add_, div_n_mul_, matrix_elementwise_minus, matrix_elementwise_add_simple, \
    sgd_update, assign_embedding_with_indexedslices


class AutoDimTrainer(BaseTrainer):
    def fit(self):
        eval_nodes = self.embed_layer.get_eval_nodes(
            self.data_ops, self.model, self.opt)
        self.alpha = self.embed_layer.alpha
        self.lookups = self.embed_layer.lookups
        self.var_lookups = self.embed_layer.var_lookups
        self.num_slot = self.embed_layer.num_slot
        self.num_cands = self.embed_layer.num_cands
        self.r = self.embed_layer.r
        self.alpha_lr = self.embed_layer.alpha_lr
        self.init_executor(eval_nodes)

        self.executor.subexecutor['alpha'].inference = False
        self.executor.subexecutor['lookups'].inference = False
        self.executor.subexecutor['allgrads'].inference = False
        self.get_arch_params(self.executor.config.placeholder_to_arr_map)

        self.total_epoch = int(self.nepoch * self.num_test_every_epoch)
        train_batch_num = self.executor.get_batch_num('train')
        npart = self.num_test_every_epoch
        self.base_batch_num = train_batch_num // npart
        self.residual = train_batch_num % npart
        self.try_load_ckpt()
        self.train_step = self.first_stage_train_step

        log_file = open(self.result_file,
                        'w') if self.result_file is not None else None
        for ep in range(self.start_ep, self.total_epoch):
            real_ep = ep // npart
            real_part = ep % npart
            self.log_func(f"Epoch {real_ep}({real_part})")
            _, early_stopping = self.run_epoch(
                self.base_batch_num + (real_part < self.residual), real_ep, real_part, log_file)
            self.cur_ep = real_ep
            self.cur_part = real_part
            if early_stopping:
                self.log_func('Early stop!')
                break

        self.log_func('Switch to re-training stage!!!')
        # re-init save topk
        if self.save_topk > 0:
            self.save_dir = self.save_dir + '_retrain'
            os.makedirs(self.save_dir)
        real_save_topk = max(1, self.save_topk)
        init_value = float('-inf')
        self.best_results = [init_value for _ in range(real_save_topk)]
        self.best_ckpts = [None for _ in range(real_save_topk)]

        # re-init executor
        self.executor.return_tensor_values()
        self.use_multi = 1
        self.data_ops = self.get_data()
        eval_nodes = self.embed_layer.get_eval_nodes_retrain(
            self.data_ops, self.model, self.opt, stream=self.executor.config.comp_stream)

        del self.executor

        resf_parts = osp.split(self.result_file)
        self.result_file = osp.join(resf_parts[0], 'retrain_' + resf_parts[1])

        run_name = osp.split(self.result_file)[1][:-4]
        executor = Executor(
            eval_nodes,
            ctx=self.ctx,
            seed=self.seed + 1,  # use a different seed
            log_path=self.log_dir,
            logger=self.logger,
            project=self.proj_name,
            run_name=run_name,
            run_id=self.run_id,
        )
        executor.set_config(self.args)
        self.executor = executor
        self.train_step = super().train_step

        # re-run
        self.total_epoch = int(self.nepoch * self.num_test_every_epoch)
        train_batch_num = self.executor.get_batch_num('train')
        npart = self.num_test_every_epoch
        self.base_batch_num = train_batch_num // npart
        self.residual = train_batch_num % npart

        log_file = open(self.result_file,
                        'w') if self.result_file is not None else None
        for ep in range(self.start_ep, self.total_epoch):
            real_ep = ep // npart
            real_part = ep % npart
            self.log_func(f"Epoch {real_ep}({real_part})")
            _, early_stopping = self.run_epoch(
                self.base_batch_num + (real_part < self.residual), real_ep, real_part, log_file)
            self.cur_ep = real_ep
            self.cur_part = real_part
            if early_stopping:
                self.log_func('Early stop!')
                break

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
        copy_dedup_lookups = {}
        for node, arr in var2arr.items():
            if node is not self.alpha and node.trainable and not node.is_embed:
                copy_params[node] = empty(arr.shape, ctx=arr.ctx)
                ori_params[node] = arr
        for dim, lookup in self.lookups.items():
            copy_lookups[lookup] = empty(
                (self.batch_size, self.num_slot, dim), ctx=self.ctx)
            indices = empty(
                (self.batch_size, self.num_slot), ctx=self.ctx, dtype=np.int32)
            values = empty(
                (self.batch_size, self.num_slot, dim), ctx=self.ctx)
            copy_dedup_lookups[lookup] = IndexedSlices(indices, values)
        self.copy_params = copy_params
        self.ori_params = ori_params
        self.copy_lookups = copy_lookups
        self.copy_dedup_lookups = copy_dedup_lookups
        self.workspace = empty(
            (len(self.copy_params) + len(self.copy_lookups),), ctx=self.ctx)
        self.norm = empty((1,), ctx=self.ctx)
        self.dalpha_values = [empty(
            (self.num_slot, self.num_cands), ctx=self.ctx) for _ in range(2)]

    def copy_from(self, stream):
        for node, arr in self.ori_params.items():
            self.copy_params[node]._async_copyfrom(arr, stream)

    def copy_from_lookups(self, lookups, dedup_lookups, stream):
        for node, l, dl in zip(self.lookups.values(), lookups, dedup_lookups):
            self.copy_lookups[node]._async_copyfrom(l, stream)
            self.copy_dedup_lookups[node].values._async_copyfrom(
                dl.values, stream)
            self.copy_dedup_lookups[node].indices._async_copyfrom(
                dl.indices, stream)

    def copy_to(self, var2arr, stream):
        for node, arr in self.ori_params.items():
            arr._async_copyfrom(self.copy_params[node], stream)
        for dim, node in self.lookups.items():
            var2arr[self.var_lookups[dim]]._async_copyfrom(
                self.copy_lookups[node], stream)
            assign_embedding_with_indexedslices(
                var2arr[node.inputs[0]], self.copy_dedup_lookups[node], stream)

    def first_stage_train_step(self):
        var2arr = self.executor.config.placeholder_to_arr_map
        stream = self.executor.config.comp_stream
        # copy original parameters (if embedding, copy lookuped ones) and update using train data to get temp model
        self.copy_from(stream)
        lookups = self.executor.run(
            'lookups', dataloader_step=False, inference=False)  # train data
        self.copy_from_lookups(
            lookups[:self.num_cands], lookups[self.num_cands:self.num_cands * 2], stream)

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
        assert self.load_ckpt is not None, 'Checkpoint should be given in testing.'
        eval_nodes = self.embed_layer.get_eval_nodes_inference(
            self.data_ops, self.model, False)
        self.init_executor(eval_nodes)

        self.try_load_ckpt()

        log_file = open(self.result_file,
                        'w') if self.result_file is not None else None
        with self.timing():
            test_loss, test_metric, _ = self.validate_once(
                self.executor.get_batch_num('validate'))
        test_time = self.temp_time[0]
        results = {
            'test_loss': test_loss,
            f'test_{self.monitor}': test_metric,
            'test_time': test_time,
        }
        printstr = ', '.join(
            [f'{key}: {value:.4f}' for key, value in results.items()])
        self.log_func(printstr)
        if log_file is not None:
            print(printstr, file=log_file, flush=True)
