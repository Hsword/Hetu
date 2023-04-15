import os
import os.path as osp
from time import time
from tqdm import tqdm
from argparse import Namespace
import numpy as np
from sklearn import metrics
import contextlib
import pickle

from ..layers import Embedding
from ..initializers import GenUniform
from ..dataloader import Dataloader, dataloader_op
from ..ndarray import cpu, gpu
from ..gpu_ops import Executor, concatenate_op


class EmbeddingTrainer(object):
    def __init__(self, dataset, model, opt, args, **kargs):
        self.model = model
        self.opt = opt

        if isinstance(args, Namespace):
            args = vars(args)
        elif args is None:
            args = {}
        if not isinstance(args, dict):
            args = dict(args)
        self.args = args
        self.args.update(kargs)

        self.num_embed = dataset.num_embed
        self.num_embed_separate = dataset.num_embed_separate
        self.num_slot = dataset.num_sparse
        self.embedding_dim = self.args['dim']
        self.compress_rate = self.args['compress_rate']
        self.initializer = self.args.get('initializer', None)
        if self.initializer is None:
            border = np.sqrt(1 / max(self.num_embed_separate))
            self.initializer = GenUniform(minval=-border, maxval=border)
        self.embedding_args = self.args.get('embedding_args', {})

        self.ctx = self.get_ctx(self.args['ctx'])
        self.ectx = self.get_ctx(self.args['ectx'])
        self.seed = self.args['seed']
        self.log_dir = self.args['log_dir']
        self.proj_name = self.args.get('project_name', 'embedmem')
        self.logger = self.args.get('logger', 'hetu')
        self.run_id = self.args.get('run_id', None)
        self.batch_size = self.args.get('batch_size', None)
        if self.batch_size is None:
            self.batch_size = self.args['bs']
        self.use_multi = self.args.get('use_multi', 0)
        self.separate_fields = self.args.get('separate_fields', self.use_multi)

        self.nepoch = self.args.get('nepoch', 0.1)
        self.tqdm_enabled = self.args.get('tqdm_enabled', True)
        self.save_topk = self.args.get('save_topk', 0)
        assert self.save_topk >= 0, f'save_topk should not be smaller than 0; got {self.save_topk}'
        if self.save_topk > 0:
            self.save_dir = self.args['save_dir']
            assert osp.isdir(
                self.save_dir), f'save_dir {self.save_dir} not exists'
        self.load_ckpt = self.args.get('load_ckpt', False)
        self.train_name = self.args.get('train_name', 'train')
        self.validate_name = self.args.get('validate_name', 'validate')
        self.monitor = self.args.get('monitor', 'auc')
        assert self.monitor in (
            'auc', 'acc', 'loss'), f'monitor should be in (auc, acc, loss); got {self.monitor}'
        self.num_test_every_epoch = self.args.get('num_test_every_epoch', 10)
        self.log_func = self.args.get('log_func', print)
        self.early_stop_steps = self.args.get('early_stop_steps', -1)
        if self.monitor in ('acc', 'auc'):
            self.monitor_type = 1  # maximize
        else:
            self.monitor_type = -1  # minimize
        self.check_acc = self.monitor == 'acc'
        self.check_auc = self.monitor == 'auc'
        self.result_file = self.args.get('result_file', None)

        real_save_topk = max(1, self.save_topk)
        init_value = float('-inf')
        self.best_results = [init_value for _ in range(real_save_topk)]
        self.best_ckpts = [None for _ in range(real_save_topk)]

        if self.early_stop_steps > 0:
            self.early_stop_counter = 0

        self.temp_time = [None]

        self.start_ep = 0
        self.dataset = dataset
        self.data_ops = self.get_data()
        self.embed_layer = self.get_embed_layer()
        self.log_func(f'Embedding layer: {self.embed_layer}')

    def set_use_multi(self, new_use_multi):
        self.use_multi = new_use_multi
        self.separate_fields = new_use_multi

    def assert_use_multi(self):
        assert self.use_multi == self.separate_fields

    @property
    def all_train_names(self):
        return self.train_name

    @property
    def all_validate_names(self):
        return self.validate_name

    def make_dataloader_op(self, tr_data, va_data, dtype=np.float32):
        train_dataloader = Dataloader(
            tr_data, self.batch_size, self.all_train_names, dtype=dtype)
        valid_dataloader = Dataloader(
            va_data, self.batch_size, self.all_validate_names, dtype=dtype)
        data_op = dataloader_op(
            [train_dataloader, valid_dataloader], dtype=dtype)
        return data_op

    def get_data(self):
        self.assert_use_multi()
        all_data = self.dataset.process_all_data_by_day(
            use_test=True, separate_fields=self.separate_fields)

        # define models for criteo
        tr_sparse, va_sparse = all_data[-2]
        tr_labels, va_labels = all_data[-1]
        tr_labels = tr_labels.reshape((-1, 1))
        va_labels = va_labels.reshape((-1, 1))
        if len(all_data) == 3:
            tr_dense, va_dense = all_data[0]
            dense_input = self.make_dataloader_op(tr_dense, va_dense)
        else:
            dense_input = None
        y_ = self.make_dataloader_op(tr_labels, va_labels)
        if self.use_multi:
            new_sparse_ops = []
            for i in range(tr_sparse.shape[1]):
                cur_data = self.make_dataloader_op(
                    tr_sparse[:, i], None if va_sparse is None else va_sparse[:, i], dtype=np.int32)
                new_sparse_ops.append(cur_data)
            embed_input = new_sparse_ops
        else:
            embed_input = self.make_dataloader_op(
                tr_sparse, va_sparse, dtype=np.int32)
        self.log_func("Data loaded.")
        return embed_input, dense_input, y_

    def get_embed_layer(self):
        if self.use_multi:
            emb = [self.get_single_embed_layer(
                nemb, f'Embedding_{i}') for i, nemb in enumerate(self.num_embed_separate)]
        else:
            emb = self.get_single_embed_layer(self.num_embed, 'Embedding')
        return emb

    def get_single_embed_layer(self, nemb, name):
        return Embedding(
            nemb,
            self.embedding_dim,
            self.initializer,
            name,
            self.ectx,
            **self.embedding_args,
        )

    def run_epoch(self, train_batch_num, epoch, part, log_file=None):
        with self.timing():
            train_loss, train_metric = self.train_once(
                train_batch_num, epoch, part)
        train_time = self.temp_time[0]
        with self.timing():
            test_loss, test_metric, early_stop = self.validate_once(
                self.executor.get_batch_num('validate'), epoch, part)
        test_time = self.temp_time[0]
        results = {
            'train_loss': train_loss,
            f'train_{self.monitor}': train_metric,
            'train_time': train_time,
            'test_loss': test_loss,
            f'test_{self.monitor}': test_metric,
            'test_time': test_time,
        }
        printstr = ', '.join(
            [f'{key}: {value:.4f}' for key, value in results.items()])
        results.update({'epoch': epoch, 'part': part, })
        results['avg_train_loss'] = results.pop('train_loss')
        results['avg_test_loss'] = results.pop('test_loss')
        self.executor.multi_log(results)
        self.executor.step_logger()
        self.log_func(printstr)
        if log_file is not None:
            print(printstr, file=log_file, flush=True)
        return results, early_stop

    def train_step(self):
        loss_val, predict_y, y_val = self.executor.run(
            self.train_name, convert_to_numpy_ret_vals=True)[:3]
        return loss_val, predict_y, y_val

    def train_once(self, step_num, epoch, part):
        localiter = range(step_num)
        if self.tqdm_enabled:
            localiter = tqdm(localiter)
        train_loss = []
        if self.check_auc:
            ground_truth_y = []
            predicted_y = []
        elif self.check_acc:
            train_acc = []
        for it in localiter:
            loss_val, predict_y, y_val = self.train_step()
            self.executor.multi_log(
                {'epoch': epoch, 'part': part, 'train_loss': loss_val})
            self.executor.step_logger()
            train_loss.append(loss_val[0])
            if self.check_auc:
                ground_truth_y.append(y_val)
                predicted_y.append(predict_y)
            elif self.check_acc:
                acc_val = self.get_acc(y_val, predict_y)
                train_acc.append(acc_val)
        train_loss = np.mean(train_loss)
        if self.check_auc:
            train_auc = self.get_auc(ground_truth_y, predicted_y)
            result = train_auc
        elif self.check_acc:
            train_acc = np.mean(train_acc)
            result = train_acc
        return train_loss, result

    def validate_once(self, step_num, epoch=None, part=None):
        localiter = range(step_num)
        if self.tqdm_enabled:
            localiter = tqdm(localiter)
        test_loss = []
        if self.check_auc:
            ground_truth_y = []
            predicted_y = []
        elif self.check_acc:
            test_acc = []
        for it in localiter:
            loss_value, test_y_predicted, y_test_value = self.executor.run(
                self.validate_name, convert_to_numpy_ret_vals=True)
            correct_prediction = self.get_acc(y_test_value, test_y_predicted)
            test_loss.append(loss_value[0])
            if self.check_auc:
                ground_truth_y.append(y_test_value)
                predicted_y.append(test_y_predicted)
            elif self.check_acc:
                test_acc.append(correct_prediction)
        test_loss = np.mean(test_loss)
        new_result = test_loss
        if self.check_auc:
            test_auc = self.get_auc(ground_truth_y, predicted_y)
            new_result = test_auc
        elif self.check_acc:
            test_acc = np.mean(test_acc)
            new_result = test_acc
        if epoch is not None:
            early_stopping = self.try_save_ckpt(new_result, (epoch, part))
        else:
            early_stopping = False
        return test_loss, new_result, early_stopping

    def try_save_ckpt(self, new_result, cur_meta):
        new_result = self.monitor_type * new_result
        if self.save_topk > 0 and new_result >= self.best_results[-1]:
            idx = None
            for i, res in enumerate(self.best_results):
                if new_result >= res:
                    idx = i
                    break
            if idx is not None:
                self.best_results.insert(idx, new_result)
                self.best_ckpts.insert(idx, cur_meta)
                ep, part = cur_meta
                self.executor.save(self.save_dir, f'ep{ep}_{part}.pkl', {
                    'epoch': ep, 'part': part, 'npart': self.num_test_every_epoch})
                rm_res = self.best_results.pop()
                rm_meta = self.best_ckpts.pop()
                self.log_func(
                    f'Save ep{ep}_{part}.pkl with {self.monitor}:{new_result}.')
                self.log_func(
                    f'Current ckpts {self.best_ckpts} with aucs {self.best_results}.')
                if rm_meta is not None:
                    ep, part = rm_meta
                    os.remove(
                        osp.join(self.save_dir, f'ep{ep}_{part}.pkl'))
                    self.log_func(
                        f'Remove ep{ep}_{part}.pkl with {self.monitor}:{rm_res}.')
        early_stopping = False
        if self.early_stop_steps > 0:
            if new_result >= self.best_results[0]:
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            if self.early_stop_counter >= self.early_stop_steps:
                early_stopping = True
        return early_stopping

    @contextlib.contextmanager
    def timing(self):
        start = time()
        yield
        ending = time()
        self.temp_time[0] = ending - start

    def get_auc(self, ground_truth_y, predicted_y):
        # auc for an epoch
        cur_gt = np.concatenate(ground_truth_y)
        cur_pr = np.concatenate(predicted_y)
        cur_gt = self.inf_nan_to_zero(cur_gt)
        cur_pr = self.inf_nan_to_zero(cur_pr)
        return metrics.roc_auc_score(cur_gt, cur_pr)

    def get_acc(self, y_val, predict_y):
        if y_val.shape[1] == 1:
            # binary output
            acc_val = np.equal(
                y_val,
                predict_y > 0.5).astype(np.float32)
        else:
            acc_val = np.equal(
                np.argmax(y_val, 1),
                np.argmax(predict_y, 1)).astype(np.float32)
        return acc_val

    def inf_nan_to_zero(self, arr):
        arr[np.isnan(arr)] = 0
        arr[np.isinf(arr)] = 0
        return arr

    def try_load_ckpt(self):
        if self.load_ckpt is not None:
            with open(self.load_ckpt, 'rb') as fr:
                meta = pickle.load(fr)
            self.executor.load_dict(meta['state_dict'])
            self.executor.load_seeds(meta['seed'])
            start_epoch = meta['epoch']
            start_part = meta['part'] + 1
            assert meta['npart'] == self.num_test_every_epoch
            self.start_ep = start_epoch * self.num_test_every_epoch + start_part
            if self.train_name in self.executor.subexecutor:
                self.executor.set_dataloader_batch_index(
                    self.train_name, start_part * self.base_batch_num)
            self.log_func(f'Load ckpt from {osp.split(self.load_ckpt)[-1]}.')

    def get_ctx(self, idx):
        if idx < 0:
            ctx = cpu(0)
        else:
            assert idx < 8
            ctx = gpu(idx)
        return ctx

    def init_executor(self, eval_nodes):
        run_name = osp.split(self.result_file)[1][:-4]
        executor = Executor(
            eval_nodes,
            ctx=self.ctx,
            seed=self.seed,
            log_path=self.log_dir,
            logger=self.logger,
            project=self.proj_name,
            run_name=run_name,
            run_id=self.run_id,
        )
        executor.set_config(self.args)
        self.executor = executor

    def fit(self):
        eval_nodes = self.get_eval_nodes()
        self.init_executor(eval_nodes)

        self.total_epoch = int(self.nepoch * self.num_test_every_epoch)
        train_batch_num = self.executor.get_batch_num('train')
        npart = self.num_test_every_epoch
        self.base_batch_num = train_batch_num // npart
        self.residual = train_batch_num % npart
        self.try_load_ckpt()

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

    def test(self):
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

    def get_embeddings(self, embed_input):
        if self.use_multi:
            results = [emb_layer(x) for emb_layer, x in zip(
                self.embed_layer, embed_input)]
            result = concatenate_op(results, axis=-1)
        else:
            result = self.embed_layer(embed_input)
        return result

    def get_eval_nodes(self):
        embed_input, dense_input, y_ = self.data_ops
        embeddings = self.get_embeddings(embed_input)
        loss, prediction = self.model(
            embeddings, dense_input, y_)
        train_op = self.opt.minimize(loss)
        eval_nodes = {
            'train': [loss, prediction, y_, train_op],
            'validate': [loss, prediction, y_],
        }
        return eval_nodes

    def get_eval_nodes_inference(self):
        embed_input, dense_input, y_ = self.data_ops
        embeddings = self.get_embeddings(embed_input)
        loss, prediction = self.model(
            embeddings, dense_input, y_)
        eval_nodes = {
            'validate': [loss, prediction, y_],
        }
        return eval_nodes

    @staticmethod
    def binary_search(left, right, evaluator):
        assert evaluator(left) < 0 < evaluator(right)
        while right - left > 0.5:
            middle = (left + right) / 2
            mid_score = evaluator(middle)
            if mid_score < 0:
                left = middle
            elif mid_score > 0:
                right = middle
        return left, right
