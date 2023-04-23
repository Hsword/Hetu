from .base import EmbeddingTrainer
from ..layers import OptEmbedding, OptEmbeddingAfterRowPruning
from ..ndarray import empty
from ..random import get_np_rand
import numpy as np


class OptEmbedTrainer(EmbeddingTrainer):
    def get_embed_layer(self):
        return OptEmbedding(
            self.num_embed,
            self.embedding_dim,
            self.num_slot,
            self.batch_size,
            initializer=self.initializer,
            name='OptEmbed',
            ctx=self.ectx,
        )

    def get_embed_layer_evolution(self, new_embeddings, remap_indices):
        return OptEmbeddingAfterRowPruning(
            self.pruned_num_embed,
            self.embedding_dim,
            self.num_slot,
            self.batch_size,
            new_embeddings,
            remap_indices.reshape(-1, 1),
            name='OptEmbedAfterRowPruning',
            ctx=self.ectx,
        )

    def get_eval_nodes(self):
        from ..gpu_ops import mul_byconst_op, reduce_sum_op, exp_op, opposite_op, add_op
        embed_input, dense_input, y_ = self.data_ops
        embeddings = self.embed_layer(embed_input)
        loss, prediction = self.model(
            embeddings, dense_input, y_)
        regloss = mul_byconst_op(reduce_sum_op(exp_op(opposite_op(
            self.embed_layer.threshold)), axes=0, keepdims=False,), self.embedding_args['alpha'])
        loss = add_op(loss, regloss)
        train_op = self.opt.minimize(loss)
        eval_nodes = {
            self.train_name: [loss, prediction, y_, train_op],
        }
        val_embeddings = self.embed_layer.make_inference(embed_input)
        val_loss, val_pred = self.model(
            val_embeddings, dense_input, y_)
        eval_nodes[self.validate_name] = [val_loss, val_pred, y_],
        eval_nodes[self.test_name] = [val_loss, val_pred, y_],
        return eval_nodes

    def calc_row_sparsity(self, embedding_table, threshold, stream):
        from ..gpu_links import num_less_than_tensor_threshold, array_set, concatenate, reduce_norm1
        stream.sync()
        npthreshold = threshold.asnumpy().reshape(-1)
        for field, value in zip(self.fields_arr, npthreshold):
            array_set(field, value, stream)
        concatenate(self.fields_arr, self.row_thresh_arr, 0, stream)
        reduce_norm1(embedding_table, self.norm1_arr, [1], stream)
        num_less_than_tensor_threshold(
            self.norm1_arr, self.norm1_arr, self.row_sparse_arr, self.row_thresh_arr, stream)
        stream.sync()
        row_sparse_num = self.row_sparse_arr.asnumpy()[0]
        return 1 - row_sparse_num / self.num_embed

    def prune_row(self):
        var2arr = self.var2arr
        stream = self.stream
        stream.sync()
        npembedding = var2arr[self.embed_layer.embedding_table].asnumpy()
        nprow_threhold = var2arr[self.embed_layer.threshold].asnumpy(
        ).reshape(-1)
        npnorm1 = np.abs(npembedding).sum(axis=1)
        all_reserving = []
        offset = 0
        pruned_num_embed_separate = []
        for i, nemb in enumerate(self.num_embed_separate):
            cur_reserving = (
                np.tile(nprow_threhold[i], nemb) < npnorm1[offset:offset+nemb])
            offset += nemb
            all_reserving.append(cur_reserving)
            pruned_num_embed_separate.append(cur_reserving.sum())
        reserving = np.concatenate(all_reserving, axis=0)
        num_reserving = sum(pruned_num_embed_separate)
        self.pruned_num_embed_separate = pruned_num_embed_separate
        self.pruned_num_embed = num_reserving
        self.args['pruned_num_embed'] = num_reserving
        self.args['pruned_num_embed_separate'] = pruned_num_embed_separate
        self.log_func(
            f'Pruning with final row sparse rate {num_reserving / self.num_embed}')
        remap_indices = np.full(
            (self.num_embed, ), fill_value=-1, dtype=np.int32)
        new_indices = np.arange(num_reserving, dtype=np.int32)
        remap_indices[reserving] = new_indices
        new_embedding_table = npembedding[reserving]
        self.log_func(
            f'New num embed separate: {self.pruned_num_embed_separate}')
        return new_embedding_table, remap_indices

    def supernet_run_epoch(self, train_batch_num, epoch, part, log_file=None):
        results = super().run_epoch(train_batch_num, epoch, part, log_file)
        var2arr = self.var2arr
        stream = self.stream
        row_sparse_rate = self.calc_row_sparsity(
            var2arr[self.embed_layer.embedding_table], var2arr[self.embed_layer.threshold], stream)
        self.log_func(f'Row sparse rate: {row_sparse_rate}')
        return results

    def fit(self):
        # prepare for calculate row sparsity
        self.norm1_arr = empty((self.num_embed, 1), ctx=self.ctx)
        self.row_sparse_arr = empty((1,), ctx=self.ctx)
        self.fields_arr = [empty((nemb, 1), ctx=self.ctx)
                           for nemb in self.num_embed_separate]
        self.row_thresh_arr = empty((self.num_embed, 1), ctx=self.ctx)

        meta = self.try_load_ckpt()

        if meta is None or meta['args']['stage'] == 1:
            # train supernet
            self.args['stage'] = 1
            assert self.save_topk > 0, 'Need to load the best ckpt for dimension selection; please set save_topk a positive integer.'
            self.run_epoch = self.supernet_run_epoch
            self.embed_layer = self.get_embed_layer()
            self.log_func(f'Embedding layer: {self.embed_layer}')
            eval_nodes = self.get_eval_nodes()
            self.init_executor(eval_nodes)
            self.load_into_executor(meta)

            self.run_once()

            self.load_best_ckpt()

            # then prune the row to zero
            self.executor.return_tensor_values()
            new_embeddings, remap_indices = self.prune_row()
            del self.executor
        else:
            state_dict = meta['state_dict']
            new_embeddings = state_dict['OptEmbedAfterRowPruning']
            remap_indices = state_dict['OptEmbedAfterRowPruning_remap']
            self.pruned_num_embed = meta['args']['pruned_num_embed']
            self.pruned_num_embed_separate = meta['args']['pruned_num_embed_separate']

        # prepare for search and retrain
        self.args['stage'] = 2
        self.run_epoch = super().run_epoch
        self.embed_layer = self.get_embed_layer_evolution(
            new_embeddings, remap_indices)
        self.log_func(f'Evolution embedding layer: {self.embed_layer}')
        eval_nodes = super().get_eval_nodes()
        self.init_executor(eval_nodes)
        if meta is None or meta['args']['stage'] <= 2:
            if meta is None or meta['args']['stage'] == 1:
                self.executor.save(self.save_dir, f'evo_start.pkl', {
                    'epoch': 0, 'part': 0, 'npart': self.num_test_every_epoch, 'args': self.get_args_for_saving()})
            else:
                self.load_into_executor(meta)

            # Evolutionary Search Hyper-params
            self.keep_num = self.embedding_args['keep_num']
            self.mutation_num = self.embedding_args['mutation_num']
            self.crossover_num = self.embedding_args['crossover_num']
            self.population_num = self.keep_num + self.mutation_num + self.crossover_num
            self.m_prob = self.embedding_args['m_prob']
            nepoch_search = self.embedding_args['nepoch_search']
            self.evolution_search(nepoch_search)

        # then retrain, start from supernet parameters
        self.args['stage'] = 3
        self.log_func(f'Switch to retraining.')
        if meta is not None and meta['args']['stage'] == 3:
            self.load_into_executor(meta)
        self.prepare_path_for_retrain()
        self.init_ckpts()
        self.run_once()

    def calc_col_sparsity(self, cand=None):
        base = self.num_embed * self.embedding_dim
        params = 0
        for i, nemb in enumerate(self.pruned_num_embed_separate):
            params += nemb * (cand[i] + 1)
        percentage = params / base
        return percentage, int(params)

    def calc_col_sparsity_all_candidates(self):
        crs = []
        for cand in self.cands:
            cr, _ = self.calc_col_sparsity(cand)
            crs.append(cr)
        return crs

    def set_candidate(self, cand):
        var2arr = self.var2arr
        stream = self.stream
        stream.sync()
        var2arr[self.embed_layer.candidate][:] = cand

    def eval_one_candidate(self, cand):
        self.set_candidate(cand)
        loss, auc, _ = self.validate_once()
        return auc, loss

    def eval_all_candidates(self):
        aucs, losses = [], []
        for cand in self.cands:
            auc, loss = self.eval_one_candidate(cand)
            aucs.append(auc)
            losses.append(loss)
        return aucs, losses

    def init_random_candidates(self, nprs):
        self.log_func("Generating random embedding masks ...")
        self.cands = []
        for i in range(self.population_num):
            cand = nprs.randint(
                low=0, high=self.embedding_dim, size=(self.num_slot,))
            self.cands.append(cand)

    def sort_cands(self, metrics):
        reverse = [1-i for i in metrics]
        indexlist = np.argsort(reverse)
        self.cands = [self.cands[i] for i in indexlist]

    def get_mutation(self, nprs):
        mutation = []
        assert self.m_prob > 0

        for i in range(self.mutation_num):
            origin = self.cands[i]
            for j in range(self.num_slot):
                if nprs.random() < self.m_prob:
                    origin[j] = nprs.randint(low=0, high=self.embedding_dim)
            mutation.append(origin)
        return mutation

    def get_crossover(self, nprs):
        crossover = []

        def indexes_gen(m, n):
            seen = set()
            x, y = nprs.randint(m, n), nprs.randint(m, n)
            while True:
                seen.add((x, y))
                yield (x, y)
                x, y = nprs.randint(m, n), nprs.randint(m, n)
                while (x, y) in seen:
                    x, y = nprs.randint(m, n), nprs.randint(m, n)
        gen = indexes_gen(0, self.crossover_num)

        for i in range(self.crossover_num):
            point = nprs.randint(1, self.embedding_dim)
            x, y = next(gen)
            origin_x, origin_y = self.cands[x], self.cands[y]
            xy = np.concatenate((origin_x[:point], origin_y[point:]))
            crossover.append(xy)
        return crossover

    def evolution_search(self, max_epoch):
        # all use this numpy random state
        nprs = get_np_rand(1)
        # initialize random
        self.init_random_candidates(nprs)
        acc_auc = 0.0
        acc_cand = None

        for epoch_idx in range(int(max_epoch)):
            aucs, losses = self.eval_all_candidates()
            self.log_func(
                f'Epoch {epoch_idx}: best AUC {max(aucs)}; worst AUC {min(aucs)}')
            self.sort_cands(aucs)
            # crs = self.calc_col_sparsity_all_candidates()

            if acc_auc < aucs[0]:
                # acc_auc, acc_cr, acc_cand = aucs[0], crs[0], self.cands[0]
                acc_auc, acc_cand = aucs[0], self.cands[0]

            mutation = self.get_mutation(nprs)
            crossover = self.get_crossover(nprs)
            self.cands = self.cands[:self.keep_num] + mutation + crossover

        acc_auc, acc_loss = self.eval_one_candidate(cand=acc_cand)
        acc_cr, _ = self.calc_col_sparsity(acc_cand)
        self.log_func(
            f'Best candidate: {acc_cand}')
        self.log_func(f'with AUC {acc_auc}; Loss {acc_loss}; CR {acc_cr}')
        self.set_candidate(acc_cand)

    def test(self):
        assert self.phase == 'test' and self.load_ckpt is not None, 'Checkpoint should be given in testing.'
        meta = self.try_load_ckpt()
        assert meta['args']['stage'] >= 2, 'Not support test the first stage.'
        state_dict = meta['state_dict']
        new_embeddings = state_dict['OptEmbedAfterRowPruning']
        remap_indices = state_dict['OptEmbedAfterRowPruning_remap']
        self.pruned_num_embed = meta['args']['pruned_num_embed']
        self.pruned_num_embed_separate = meta['args']['pruned_num_embed_separate']

        self.embed_layer = self.get_embed_layer_evolution(
            new_embeddings, remap_indices)
        self.data_ops = self.get_data()
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
