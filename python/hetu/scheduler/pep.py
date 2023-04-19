from .base import EmbeddingTrainer
from .switchinference import SwitchInferenceTrainer
from ..layers import PEPEmbedding, PEPRetrainEmbedding
from ..ndarray import empty
from ..gpu_links import get_larget_than, sigmoid
from ..random import set_random_seed, reset_seed_seqnum
import os
import os.path as osp
import numpy as np


class PEPEmbTrainer(SwitchInferenceTrainer):
    @property
    def form(self):
        return 'csr'

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

    def get_embed_layer_retrain(self, mask):
        return PEPRetrainEmbedding(
            self.num_embed,
            self.embedding_dim,
            mask,
            initializer=self.initializer,
            name=self.sparse_name,
            ctx=self.ectx
        )

    def fit(self):
        # the first stage
        EmbeddingTrainer.fit(self)

        self.executor.return_tensor_values()
        stream = self.executor.config.comp_stream
        stream.sync()

        # get mask
        mask = empty((self.num_embed, self.embedding_dim),
                     ctx=self.ectx, dtype=np.int32)
        threshold = empty(self.embed_layer.threshold.shape, ctx=self.ectx)
        sigmoid(self.embed_layer.threshold.tensor_value, threshold, stream)
        get_larget_than(
            self.embed_layer.embedding_table.tensor_value, threshold, mask, stream)
        stream.sync()

        # the second stage
        self.log_func('Switch to re-training stage!!!')
        # re-init save topk
        if self.save_topk > 0:
            self.save_dir = self.save_dir + '_retrain'
            os.makedirs(self.save_dir)
        self.init_ckpts()

        self.embed_layer = self.get_embed_layer_retrain(mask)
        npmask = mask.asnumpy()
        dense = npmask.sum()
        sparse_rate = dense / self.num_embed / self.embedding_dim
        self.log_func(f'Retrain with sparse rate: {sparse_rate}')
        eval_nodes = self.get_eval_nodes()

        del self.executor

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
