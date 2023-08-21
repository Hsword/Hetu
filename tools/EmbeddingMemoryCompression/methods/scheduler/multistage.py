from .base import EmbeddingTrainer
from copy import deepcopy


class MultiStageTrainer(EmbeddingTrainer):
    def __init__(self, dataset, model, opt, args, data_ops=None, **kargs):
        super().__init__(dataset, model, opt, args, data_ops, **kargs)
        assert self.save_topk > 0, 'Need to load the best ckpt for multi-stage training; please set save_topk a positive integer.'
        assert self.stage in self.legal_stages, f'Stage {self.stage} is illegal! Candidates are {self.legal_stages}.'

    def assert_use_multi(self):
        # currently all multi stage uses only 1 embedding table; except autodim the second stage
        assert self.use_multi == self.separate_fields == 0

    def copy_args_with_stage(self, stage):
        new_args = deepcopy(self.args)
        new_args['embedding_args']['stage'] = stage
        return new_args

    @property
    def stage(self):
        return self.args['embedding_args']['stage']

    @property
    def legal_stages(self):
        return NotImplementedError

    def fit(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
