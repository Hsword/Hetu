import os

from ..layers import SparseEmbedding
from .base import EmbeddingTrainer
from ..gpu_ops import Executor


class SwitchInferenceTrainer(EmbeddingTrainer):
    def __init__(self, dataset, model, opt, args, **kargs):
        super().__init__(dataset, model, opt, args, **kargs)
        from .deeplight import DeepLightTrainer
        from .pep import PEPEmbTrainer
        self.use_sparse = isinstance(self, (DeepLightTrainer, PEPEmbTrainer))
        if self.use_sparse:
            real_dim = self.compress_rate * self.embedding_dim
            if real_dim >= 3:
                form = 'csr'
                real_target_sparse = (real_dim - 1) / 2 / self.embedding_dim
            else:
                form = 'coo'
                real_target_sparse = self.compress_rate / 3
            self.prune_rate = 1 - real_target_sparse
            self.form = form
            self.log_func(
                f'Use {form} for sparse storage; final prune rate {self.prune_rate}, given target sparse rate {self.compress_rate}.')

    def fit(self):
        self.save_dir = self.args['save_dir']
        super().fit()
        self.executor.config.comp_stream.sync()
        self.executor.return_tensor_values()
        self.check_inference()

    def check_inference(self):
        # check inference; use sparse embedding
        if self.use_sparse:
            infer_eval_nodes = self.get_eval_nodes_test_inference()
        else:
            infer_eval_nodes = self.get_eval_nodes_inference()
        infer_executor = Executor(infer_eval_nodes, ctx=self.ctx,
                                  seed=self.seed, log_path=self.log_dir)
        del self.executor
        self.executor = infer_executor
        with self.timing():
            test_loss, test_metric, _ = self.validate_once(
                infer_executor.get_batch_num('validate'))
        test_time = self.temp_time[0]
        os.makedirs(self.save_dir, exist_ok=True)
        infer_executor.save(self.save_dir, f'final_ep{self.cur_ep}_{self.cur_part}.pkl', {
            'epoch': self.cur_ep, 'part': self.cur_part, 'npart': self.num_test_every_epoch})
        results = {
            'test_loss': test_loss,
            f'test_{self.monitor}': test_metric,
            'test_time': test_time,
        }
        printstr = ', '.join(
            [f'{key}: {value:.4f}' for key, value in results.items()])
        self.log_func(printstr)
        log_file = open(self.result_file,
                        'a') if self.result_file is not None else None
        if log_file is not None:
            print(printstr, file=log_file, flush=True)

    def test(self):
        if self.use_sparse:
            self.embed_layer = self.get_embed_layer_inference()
        else:
            self.embed_layer = self.get_embed_layer()
        self.log_func(f'Embedding layer: {self.embed_layer}')
        assert self.load_ckpt is not None, 'Checkpoint should be given in testing.'
        eval_nodes = self.get_eval_nodes_inference()
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

    @property
    def sparse_name(self):
        raise NotImplementedError

    def get_eval_nodes_test_inference(self):
        assert self.use_sparse
        # check inference; use sparse embedding
        embed_input, dense_input, y_ = self.data_ops
        test_embed_input = self.embed_layer.make_inference(embed_input)
        test_loss, test_prediction = self.model(
            test_embed_input, dense_input, y_)
        eval_nodes = {'validate': [test_loss, test_prediction, y_]}
        return eval_nodes

    def get_embed_layer_inference(self):
        assert self.use_sparse
        return SparseEmbedding(
            self.num_embed,
            self.embedding_dim,
            self.form,
            name=self.sparse_name,
            ctx=self.ectx,
        )
