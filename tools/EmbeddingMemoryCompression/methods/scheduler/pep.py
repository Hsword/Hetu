from .switchinference import SwitchInferenceTrainer
from ..layers import PEPEmbedding, PEPRetrainEmbedding
from hetu.ndarray import empty, array
from hetu.gpu_links import get_larger_than, sigmoid, num_less_than_tensor_threshold, mask_func
from hetu.random import set_random_seed, reset_seed_seqnum
import numpy as np


class PEPEmbTrainer(SwitchInferenceTrainer):
    # TODO: need improvements; using MultiStageTrainer
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
        self.init_ckpts()
        meta = self.try_load_ckpt()
        if meta is None or meta['args']['stage'] == 1:
            # the first stage
            self.args['stage'] = 1
            assert self.save_topk > 0, 'Need to load the best ckpt for dimension selection; please set save_topk a positive integer.'
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
            self.load_into_executor(meta)

            self.run_once()

            self.load_best_ckpt()

            self.executor.return_tensor_values()
            stream = self.executor.config.comp_stream
            stream.sync()

            # get mask
            threshold = empty(self.embed_layer.threshold.shape, ctx=self.ectx)
            sigmoid(self.embed_layer.threshold.tensor_value, threshold, stream)
            get_larger_than(
                self.embed_layer.embedding_table.tensor_value, threshold, self.mask, stream)
            stream.sync()
            self.reset_for_retrain()
            npmask = self.mask.asnumpy()
        else:
            npmask = meta['state_dict']['PEPRetrainEmb_mask']
            self.mask = array(npmask, ctx=self.ectx, dtype=np.int32)
        dense = npmask.sum()
        sparse_rate = dense / self.num_embed / self.embedding_dim
        self.log_func(f'Retrain with sparse rate: {sparse_rate}')

        # the second stage
        self.log_func('Switch to re-training stage!!!')
        self.args['stage'] = 2
        # re-init save topk
        self.run_epoch = super().run_epoch
        self.prepare_path_for_retrain()
        self.init_ckpts()

        self.embed_layer = self.get_embed_layer_retrain()
        eval_nodes = self.get_eval_nodes()

        reset_seed_seqnum()
        self.init_executor(eval_nodes)
        if meta is not None and meta['args']['stage'] == 2:
            self.load_into_executor(meta)
        self.seed += 1
        set_random_seed(self.seed)  # use new seed for retraining

        # re-run
        self.run_once()
        self.load_best_ckpt()

        # prune and test
        stream = self.executor.config.comp_stream
        stream.sync()
        self.executor.return_tensor_values()
        embed_arr = self.embed_layer.embedding_table.tensor_value
        mask_func(embed_arr, self.embed_layer.mask.tensor_value,
                  embed_arr, stream)
        self.check_inference()

    def run_epoch_first_stage(self, train_batch_num, epoch, part, log_file=None):
        results, early_stop = super().run_epoch(train_batch_num, epoch, part, log_file)
        # the stop condition is whether reach the parameter limit
        early_stop = False
        var2arr = self.var2arr
        stream = self.stream
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
