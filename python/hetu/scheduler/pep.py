from .switchinference import SwitchInferenceTrainer
from ..layers import PEPEmbedding, PEPRetrainEmbedding
from ..ndarray import empty
from ..gpu_links import get_larget_than, sigmoid, num_less_than_tensor_threshold
from ..random import set_random_seed, reset_seed_seqnum
import os
import os.path as osp
import numpy as np


class PEPEmbTrainer(SwitchInferenceTrainer):
    def __init__(self, dataset, model, opt, args, **kargs):
        super().__init__(dataset, model, opt, args, **kargs)
        self.best_prune_rate = 0.

    @property
    def sparse_name(self):
        return 'PEPRetrainEmb'

    def assert_use_multi(self):
        assert self.use_multi == self.separate_fields == 0

    def get_embed_layer(self):
        return PEPEmbedding(
            self.num_embed,
            self.embedding_dim,
            self.embedding_args['threshold_type'],
            self.embedding_args['threshold_init'],
            initializer=self.initializer,
            name='PEPEmb',
            ctx=self.ectx,
        )

    def get_embed_layer_retrain(self):
        return PEPRetrainEmbedding(
            self.num_embed,
            self.embedding_dim,
            self.mask,
            form=self.form,
            initializer=self.initializer,
            name=self.sparse_name,
            ctx=self.ectx
        )

    def fit(self):
        self.save_dir = self.args['save_dir']
        # the first stage
        self.run_epoch = self.run_epoch_first_stage
        self.embed_layer = self.get_embed_layer()
        self.log_func(f'Embedding layer: {self.embed_layer}')

        self.mask = empty((self.num_embed, self.embedding_dim),
                          ctx=self.ectx, dtype=np.int32)
        self.sigmoid_threshold = empty(
            self.embed_layer.threshold.shape, ctx=self.ectx)
        self.cur_prune_rate = empty((1,), ctx=self.ectx)

        eval_nodes = self.get_eval_nodes()
        self.init_executor(eval_nodes)

        self.try_load_ckpt()
        self.run_once()

        # switch
        self.run_epoch = super().run_epoch

        self.executor.return_tensor_values()
        stream = self.executor.config.comp_stream
        stream.sync()

        # get mask
        threshold = empty(self.embed_layer.threshold.shape, ctx=self.ectx)
        sigmoid(self.embed_layer.threshold.tensor_value, threshold, stream)
        get_larget_than(
            self.embed_layer.embedding_table.tensor_value, threshold, self.mask, stream)
        stream.sync()

        # the second stage
        self.log_func('Switch to re-training stage!!!')
        # re-init save topk
        if self.save_topk > 0:
            self.save_dir = self.save_dir + '_retrain'
            os.makedirs(self.save_dir)
        self.init_ckpts()

        self.reset_for_retrain()
        self.embed_layer = self.get_embed_layer_retrain()
        npmask = self.mask.asnumpy()
        dense = npmask.sum()
        sparse_rate = dense / self.num_embed / self.embedding_dim
        self.log_func(f'Retrain with sparse rate: {sparse_rate}')
        eval_nodes = self.get_eval_nodes()

        resf_parts = osp.split(self.result_file)
        self.result_file = osp.join(resf_parts[0], 'retrain_' + resf_parts[1])

        reset_seed_seqnum()
        self.init_executor(eval_nodes)
        self.seed += 1
        set_random_seed(self.seed)  # use new seed for retraining

        # re-run
        self.run_once()

        # prune and test
        self.executor.config.comp_stream.sync()
        self.executor.return_tensor_values()
        self.check_inference()

    def run_epoch_first_stage(self, train_batch_num, epoch, part, log_file=None):
        results, early_stop = super().run_epoch(train_batch_num, epoch, part, log_file)
        # the stop condition is whether reach the parameter limit
        early_stop = False
        var2arr = self.executor.config.placeholder_to_arr_map
        stream = self.executor.config.comp_stream
        sigmoid(var2arr[self.embed_layer.threshold],
                self.sigmoid_threshold, stream)
        num_less_than_tensor_threshold(var2arr[self.embed_layer.embedding_table],
                                       self.mask, self.cur_prune_rate, self.sigmoid_threshold, stream)
        stream.sync()
        cur_prune_rate = self.cur_prune_rate.asnumpy().item() / self.num_embed / \
            self.embedding_dim
        if cur_prune_rate > self.best_prune_rate:
            self.best_prune_rate = cur_prune_rate
        self.log_func(
            f'Current prune rate: {cur_prune_rate} (best {self.best_prune_rate}); target prune rate: {self.prune_rate}')
        if cur_prune_rate >= self.prune_rate:
            early_stop = True
        return results, early_stop
