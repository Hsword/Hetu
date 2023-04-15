import os

from .base import BaseTrainer
from ..gpu_ops import Executor


class SwitchInferenceTrainer(BaseTrainer):
    def fit(self):
        self.save_dir = self.args['save_dir']
        super().fit()
        self.executor.config.comp_stream.sync()
        self.executor.return_tensor_values()
        # check inference; use sparse embedding
        infer_eval_nodes = self.embed_layer.get_eval_nodes_inference(
            self.data_ops, self.model)
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
        assert self.load_ckpt is not None, 'Checkpoint should be given in testing.'
        from ..layers import DeepLightEmbedding
        if isinstance(self.embed_layer, DeepLightEmbedding):
            eval_nodes = self.embed_layer.get_eval_nodes_inference(
                self.data_ops, self.model, False)
        else:
            eval_nodes = self.embed_layer.get_eval_nodes_inference(
                self.data_ops, self.model)
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
