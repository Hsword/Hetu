""" adapted from 'Serving Deep Learning Models with Deduplication from Relational Databases' """
import hetu as ht
from hetu.random import get_np_rand
from .base import EmbeddingTrainer
from .compressor import Compressor
from .multistage import MultiStageTrainer
from ..layers import DedupEmbedding
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os.path as osp


class DedupOverallTrainer(MultiStageTrainer):
    # first pre-train, then LSH, finally fine-tune

    @property
    def legal_stages(self):
        return (1, 2)

    def assert_use_multi(self):
        assert self.use_multi == self.separate_fields

    def fit(self):
        stage = self.stage
        # two stages: warmup, training(pruning)
        if stage == 1:
            self.warmup_trainer = EmbeddingTrainer(
                self.dataset, self.model, self.opt, self.copy_args_with_stage(1), self.data_ops)
        else:
            self.warmup_trainer = None
        self.trainer = DedupTrainer(
            self.dataset, self.model, self.opt, self.copy_args_with_stage(2), self.data_ops)
        self.trainer.prepare_path_for_retrain('train')

        if self.warmup_trainer is not None:
            self.warmup_trainer.fit()
            ep, part = self.warmup_trainer.get_best_meta()
            self.trainer.load_ckpt = self.warmup_trainer.join(
                f'ep{ep}_{part}.pkl')
            self.warmup_trainer.executor.return_tensor_values()
            del self.warmup_trainer

        # finetune
        self.trainer.fit()

    def test(self):
        assert self.stage == 2
        trainer = DedupTrainer(
            self.dataset, self.model, self.opt, self.args, self.data_ops)
        trainer.test()


class DedupTrainer(EmbeddingTrainer):
    def __init__(self, dataset, model, opt, args, data_ops=None, **kargs):
        super().__init__(dataset, model, opt, args, data_ops, **kargs)
        self.fix_emb = self.embedding_args['fix_emb']
        self.block_cap = self.embedding_args['block_cap']
        self.lsh_threshold = self.embedding_args['lsh_threshold']
        self.fp = self.embedding_args['fp']
        self.sim = self.embedding_args['sim']
        self.nemb_per_block = self.block_cap // self.embedding_dim

    def try_load_ckpt(self):
        meta = self.load(self.load_ckpt)
        self.assert_load_args(meta['args'])
        assert meta['npart'] == self.num_test_every_epoch
        same_stage = meta['args']['embedding_args'].get(
            'stage', None) == self.args['embedding_args'].get('stage', None)
        if same_stage:
            if self.phase == 'train' and 'metric' in meta and meta['args'].get('monitor', 'auc') == self.monitor:
                load_metric = meta['metric']
                self.best_ckpts[0] = (meta['epoch'], meta['part'])
                self.best_results[0] = load_metric
                self.log_func(
                    f'Load ckpt from {osp.split(self.load_ckpt)[-1]} with the same metric {self.monitor}: {load_metric}')
            else:
                self.log_func(
                    f'Load ckpt from {osp.split(self.load_ckpt)[-1]}.')
            self.loaded_ep = (meta['epoch'], meta['part'])
            start_epoch = meta['epoch']
            start_part = meta['part'] + 1
            self.start_ep = start_epoch * self.num_test_every_epoch + start_part
        else:
            meta['epoch'] = 0
            meta['part'] = -1
            self.start_ep = 0
            self.log_func(
                f'Load ckpt from {osp.split(self.load_ckpt)[-1]}.')
        return meta

    def fit(self):
        assert self.phase == 'train' and self.load_ckpt is not None
        self.init_ckpts()
        meta = self.try_load_ckpt()
        assert meta is not None
        meta_stage = meta['args']['embedding_args']['stage']
        if meta_stage == 1:
            ori_mem = 0
            new_mem = 0
            with self.timing():
                if self.separate_fields:
                    new_embs = []
                    new_maps = []
                    for i in range(self.num_slot):
                        emb = meta['state_dict'][f'Embedding_{i}']
                        ori_mem += (emb.shape[0] * emb.shape[1])
                        new_emb, old_to_new_map = self.deduplicate(
                            emb, fp=self.fp, sim=self.sim)
                        if new_emb.shape[0] >= emb.shape[0]:
                            # not compress if larger memory
                            new_embs.append(emb)
                            new_maps.append(None)
                            new_mem += (emb.shape[0] * emb.shape[1])
                        else:
                            new_embs.append(new_emb)
                            new_maps.append(old_to_new_map)
                            new_mem += (new_emb.shape[0] *
                                        new_emb.shape[1] + old_to_new_map.shape[0])
                else:
                    emb = meta['state_dict']['Embedding']
                    ori_mem = emb.shape[0] * emb.shape[1]
                    new_embs, new_maps = self.deduplicate(
                        emb, fp=self.fp, sim=self.sim)
                    assert new_embs.shape[0] < emb.shape[0]
                    new_mem = new_embs.shape[0] * \
                        new_embs.shape[1] + new_maps.shape[0]
            dedup_time = self.temp_time[0]
            self.log_func(f'Deduplication time: {dedup_time}s.')
            self.log_func(
                f'Real compress rate: {new_mem / ori_mem} ({new_mem} / {ori_mem})')
        else:
            state_dict = meta['state_dict']
            if self.separate_fields:
                new_embs = []
                new_maps = []
                for i in range(self.num_slot):
                    key = f'DedupEmb_{i}'
                    if key in state_dict:
                        emb = state_dict[key]
                        mapping = state_dict[f'DedupEmb_{i}_remap']
                    else:
                        emb = state_dict[f'Embedding_{i}']
                        mapping = None
                    new_embs.append(emb)
                    new_maps.append(mapping)
            else:
                new_embs = state_dict['DedupEmb']
                new_maps = state_dict['DedupEmb_remap']
        self.embed_layer = self.get_embed_layer(new_embs, new_maps)
        self.log_func(f'Embedding layer: {self.embed_layer}')
        eval_nodes = self.get_eval_nodes()
        self.init_executor(eval_nodes)

        self.load_into_executor(meta)
        self.executor.save(self.save_dir, f'start.pkl', {
            'epoch': 0, 'part': -1, 'npart': self.num_test_every_epoch, 'args': self.get_args_for_saving(), 'metric': None})
        loss, auc, _ = self.test_once()
        self.log_func(f'Initial metrics: test loss {loss}, test AUC {auc}')
        self.run_once()

    def assert_load_args(self, load_args):
        assert self.args['model'].__name__ == load_args['model']
        for k in ['method', 'dim', 'dataset', 'separate_fields', 'use_multi']:
            assert load_args[k] == self.args[
                k], f'Current argument({k}) {self.args[k]} different from loaded {load_args[k]}'
        for k in ['compress_rate', 'bs', 'opt', 'lr', 'num_test_every_epoch', 'seed', 'embedding_args']:
            if load_args[k] != self.args[k]:
                if k == 'compress_rate' and load_args['embedding_args']['stage'] == self.args['embedding_args']['stage']:
                    assert load_args[k] == self.args[
                        k], f'Current argument({k}) {self.args[k]} different from loaded {load_args[k]}'
                else:
                    self.log_func(
                        f'Warning: current argument({k}) {self.args[k]} different from loaded {load_args[k]}')

    @property
    def dedup_name(self):
        return 'DedupEmb'

    def get_embed_layer(self, embs, mappings):
        if self.separate_fields:
            emb_layers = []
            for i, (emb, mapping) in enumerate(zip(embs, mappings)):
                if mapping is None:
                    emb_layers.append(ht.layers.Embedding(
                        emb.shape[0],
                        self.embedding_dim,
                        ht.init.GenEmpty(),
                        name=f'Embedding_{i}',
                        ctx=self.ectx,
                    ))
                else:
                    emb_layers.append(DedupEmbedding(
                        emb,
                        mapping,
                        self.nemb_per_block,
                        trainable=not self.fix_emb,
                        name=f'{self.dedup_name}_{i}',
                        ctx=self.ectx,
                    ))
        else:
            assert mappings is not None
            emb_layers = DedupEmbedding(
                embs,
                mappings,
                self.nemb_per_block,
                trainable=not self.fix_emb,
                name=self.dedup_name,
                ctx=self.ectx,
            )
        return emb_layers

    def deduplicate(self, embedding, fp=0.01, sim=0.7):
        # step 1 & 2: calculate 3rd quartile, sort in ascending order
        ori_num_emb = embedding.shape[0]
        block_cap = self.block_cap
        num_emb_per_block = self.nemb_per_block
        num_pad_emb = - ori_num_emb % num_emb_per_block
        embedding = np.concatenate((embedding, np.zeros(
            (num_pad_emb, self.embedding_dim), dtype=embedding.dtype)), axis=0)
        embedding = embedding.reshape(-1, block_cap)
        pad_idx = embedding.shape[0] - 1
        val_3qs = np.percentile(embedding, 75, axis=1)
        orders = np.argsort(val_3qs)
        # step 3: select k blocks
        lsh_indexer = L2LSH(prob_dim=block_cap, r=0.09, num_k=1, num_l=90)
        dup_map = {}
        threshold = (embedding.shape[0] * block_cap - ori_num_emb *
                     self.embedding_dim * self.compress_rate) / (block_cap - 1)
        assert threshold > 0, f'Cannot reach compress rate {self.compress_rate}.'
        transformed_embedding = lsh_indexer.compute_lsh(embedding)
        # we do not evaluate or finetune during deduplication for the following reasons:
        # 1 we stop the deduplication according to memory budget rather than acc
        # 2 the embeddings are separate parameters, can be finetuned together after dedup phase
        for oi, idx in enumerate(tqdm(orders, desc='Deduplicate')):
            # Flag to indicate whether the block can be deduplicated
            has_dedup = False
            # Maximum similarity
            max_sim = 0
            # Index of the most similar block
            max_b2_index = None

            b1_index = idx

            # The block needs to be deduplicated
            b1 = embedding[idx]
            tb1 = transformed_embedding[idx]

            query_result = lsh_indexer.query(tb1, self.lsh_threshold)
            if idx != pad_idx:
                lsh_indexer.insert(tb1, idx)

            # for b2_index in tqdm(query_result, leave=False):
            for b2_index in query_result:
                if b1_index == b2_index:
                    continue
                b2 = embedding[b2_index]

                if pad_idx in (b1_index, b2_index):
                    b1_tmp = b1[:-num_pad_emb * self.embedding_dim]
                    b2_tmp = b2[:-num_pad_emb * self.embedding_dim]
                else:
                    b1_tmp = b1
                    b2_tmp = b2

                # compute the similarity between a candidate block and the query block
                diff = np.abs(b1_tmp-b2_tmp)
                block_sim = np.sum(diff <= fp) / b1_tmp.shape[0]

                if block_sim > max_sim and block_sim >= sim:
                    max_sim = block_sim
                    max_b2_index = b2_index
                    has_dedup = True

            # If there is a deduplicable block, then deduplicate it
            if has_dedup:
                dup_map[b1_index] = max_b2_index

            if len(dup_map) >= threshold:
                progress = oi / len(orders)
                self.log_func(
                    f'Break at progress {progress}; please consider use larger threshold if too early.')
                break

        new_shape = (embedding.shape[0] - len(dup_map), block_cap)
        new_embedding = np.empty(new_shape, dtype=embedding.dtype)
        old_to_new_map = np.empty((embedding.shape[0],), dtype=np.int32)
        new_idx = 0
        for i in orders:
            if i in dup_map:
                old_to_new_map[i] = old_to_new_map[dup_map[i]]
            else:
                new_embedding[new_idx] = embedding[i]
                old_to_new_map[i] = new_idx
                new_idx += 1
        assert new_idx == new_embedding.shape[0]
        return new_embedding.reshape(-1, self.embedding_dim), old_to_new_map


class Deduplicator(Compressor):
    @staticmethod
    def compress(embedding, compress_rate, block_cap=(10000, 100), lsh_threshold=0, fp=0.01, sim=0.7):
        # step 1 & 2: calculate 3rd quartile, sort in ascending order
        nemb, ndim = embedding.shape
        ori_size = nemb * ndim
        dtype = embedding.dtype
        block_size_x, block_size_y = block_cap
        block_cap_size = block_size_x * block_size_y
        pad_size_x = -nemb % block_size_x
        pad_size_y = -ndim % block_size_y
        has_pad_x = pad_size_x > 0
        has_pad_y = pad_size_y > 0
        padded_embedding = embedding
        if has_pad_x:
            padded_embedding = np.concatenate(
                (padded_embedding, np.zeros((pad_size_x, ndim), dtype=dtype)))
        if has_pad_y:
            padded_embedding = np.concatenate((padded_embedding, np.zeros(
                (padded_embedding.shape[0], pad_size_y), dtype=dtype)), axis=1)
        block_num_x = padded_embedding.shape[0] // block_size_x
        block_num_y = padded_embedding.shape[1] // block_size_y
        last_row_index = block_num_y * (block_num_x - 1)
        new_emb = []
        for i in range(block_num_x):
            for j in range(block_num_y):
                start_x = i * block_size_x
                ending_x = start_x + block_size_x
                start_y = j * block_size_y
                ending_y = start_y + block_size_y
                new_emb.append(
                    padded_embedding[start_x:ending_x, start_y:ending_y].reshape(-1))
        padded_embedding = np.stack(new_emb)
        val_3qs = np.percentile(padded_embedding, 75, axis=1)
        orders = np.argsort(val_3qs)
        # step 3: select k blocks
        lsh_indexer = L2LSH(prob_dim=block_cap_size, r=0.09, num_k=1, num_l=90)
        dup_map = {}
        block_num = block_num_x * block_num_y
        target_dup_length = (block_num * block_cap_size -
                             compress_rate * ori_size) / (block_cap_size - 1)
        assert target_dup_length < block_num, f'Cannot compress to {compress_rate}'
        transformed_embedding = lsh_indexer.compute_lsh(padded_embedding)

        def check_pad(idx):
            res = 0
            if has_pad_x and idx >= last_row_index:
                res += 2
            if has_pad_y and (idx + 1) % block_num_y == 0:
                res += 1
            return res

        def remove_pad(x, is_pad):
            if is_pad:
                x_tmp = x.reshape(block_size_x, block_size_y)
                if is_pad == 3:  # the last
                    x_tmp = x_tmp[:-pad_size_x, :-pad_size_y]
                elif is_pad == 2:  # last row
                    x_tmp = x_tmp[:-pad_size_x, :]
                elif is_pad == 1:  # last col
                    x_tmp = x_tmp[:, :-pad_size_y]
                x_tmp = x_tmp.reshape(-1)
            else:
                x_tmp = x
            return x_tmp
        # we do not evaluate or finetune during deduplication for the following reasons:
        # 1 we stop the deduplication according to memory budget rather than acc
        # 2 the embeddings are separate parameters, can be finetuned together after dedup phase
        for oi, idx in enumerate(tqdm(orders, desc='Deduplicate')):
            # Flag to indicate whether the block can be deduplicated
            has_dedup = False
            # Maximum similarity
            max_sim = 0
            # Index of the most similar block
            max_b2_index = None

            b1_index = idx

            # The block needs to be deduplicated
            b1 = padded_embedding[idx]
            tb1 = transformed_embedding[idx]

            query_result = lsh_indexer.query(tb1, lsh_threshold)
            is_b1_pad = check_pad(idx)
            if not is_b1_pad:
                lsh_indexer.insert(tb1, idx)
            if len(query_result) > 0:
                b1_tmp = remove_pad(b1, is_b1_pad)

            # for b2_index in tqdm(query_result, leave=False):
            for b2_index in query_result:
                if b1_index == b2_index:
                    continue
                # is_b2_pad = check_pad(b2_index) # assert False
                b2 = padded_embedding[b2_index]
                b2_tmp = remove_pad(b2, is_b1_pad)

                # compute the similarity between a candidate block and the query block
                diff = np.abs(b1_tmp - b2_tmp)
                block_sim = np.sum(diff <= fp) / b1_tmp.shape[0]

                if block_sim > max_sim and block_sim >= sim:
                    max_sim = block_sim
                    max_b2_index = b2_index
                    has_dedup = True

            # If there is a deduplicable block, then deduplicate it
            if has_dedup:
                dup_map[b1_index] = max_b2_index

                if len(dup_map) >= target_dup_length:
                    progress = oi / len(orders)
                    print(
                        f'Break at progress {progress}; please consider use larger threshold if too early.')
                    break

        new_embedding = []
        new_map = np.full((block_num,), -1, dtype=np.int32)
        new_idx = 0
        for idx in orders:
            tar = dup_map.get(idx, idx)
            if new_map[tar] < 0:
                new_embedding.append(padded_embedding[tar])
                new_map[tar] = new_idx
                new_idx += 1
            if idx != tar:
                new_map[idx] = new_map[tar]
        new_embedding = np.stack(new_embedding)
        assert new_idx == new_embedding.shape[0]
        print('Final compression ratio:',
              (new_embedding.shape[0] * new_embedding.shape[1] + len(new_map)) / ori_size)
        return new_embedding, new_map

    @staticmethod
    def decompress(compressed_embedding, dup_map, ori_size, block_cap):
        block_size_x, block_size_y = block_cap
        nemb, ndim = ori_size
        block_num_x = (nemb - 1) // block_size_x + 1
        block_num_y = (ndim - 1) // block_size_y + 1
        embedding = np.empty(
            (block_num_x * block_size_x, block_num_y * block_size_y), dtype=np.float32)
        for i in range(block_num_x):
            idx = i * block_num_y
            start_x = i * block_size_x
            ending_x = start_x + block_size_x
            cur_embedding = embedding[start_x:ending_x]
            for j in range(block_num_y):
                start_y = j * block_size_y
                ending_y = start_y + block_size_y
                cur_embedding[:, start_y:ending_y] = compressed_embedding[dup_map[idx+j]
                                                                          ].reshape(block_cap)
        embedding = embedding[:nemb, :ndim]
        return embedding

    @staticmethod
    def decompress_batch(compressed_embedding, batch_ids, dup_map, ori_size, block_cap):
        block_size_x, block_size_y = block_cap
        ndim = ori_size[1]
        block_num_y = (ndim - 1) // block_size_y + 1
        block_idx_x = np.unique(batch_ids // block_size_x)
        # potential_embeddings = {x: np.empty(
        #     (block_size_x, block_num_y * block_size_y), dtype=np.float32) for x in block_idx_x}
        potential_embeddings = {}
        for k in block_idx_x:
            idx = k * block_num_y
            # cur_embedding = potential_embeddings[k]
            cur_embeddings = []
            for j in range(block_num_y):
                # start_y = j * block_size_y
                # ending_y = start_y + block_size_y
                cur_embeddings.append(compressed_embedding[dup_map[idx+j]].reshape(block_cap))
                # cur_embedding[:, start_y:ending_y] = compressed_embedding[dup_map[idx+j]
                #                                                           ].reshape(block_cap)
            if len(cur_embeddings) > 1:
                cur_embeddings = np.concatenate(cur_embeddings, axis=1)
            else:
                cur_embeddings = cur_embeddings[0]
            potential_embeddings[k] = cur_embeddings
        result_embeddings = np.empty(
            (batch_ids.shape[0], ndim), dtype=np.float32)
        for i, ridx in enumerate(batch_ids):
            result_embeddings[i, :] = potential_embeddings[ridx //
                                                           block_size_x][ridx % block_size_x, :ndim]
        return result_embeddings


class L2LSH(object):
    """L2LSH class, each hash value is computed by np.ceil((a @ X + b) / r)

    Args:
        prob_dim (int): size of probability vector
        r (float, optional): r is a real positive number
        num_l (int, optional): number of band,
            default=40
        num_k (int, optional): number of concatenated hash function, 
            default=4

    """

    def __init__(self, prob_dim, r=8, num_l=40, num_k=4):
        self.nprs = get_np_rand(1)

        self.num_l = num_l
        self.num_k = num_k
        self.r = r

        # Initiate the coefficients, num_l*num_k hash functions in total
        self.a = self.nprs.normal(size=(prob_dim, num_l*num_k))
        # b is uniformly at random on [0, r]
        self.b = self.nprs.uniform(low=0, high=self.r, size=(num_l*num_k, ))

        # generate the hash ranges for each band
        self.hash_ranges = [(i*self.num_k, (i+1)*self.num_k)
                            for i in range(self.num_l)]
        self.hash_table = defaultdict(set)

    def compute_lsh(self, prob):
        """Compute hash value for the given probability vector

        Args:
            prob (array): a square root probability vector
        """
        prob = np.reshape(prob, (-1, self.a.shape[0]))
        return np.ceil((prob @ self.a + self.b) / self.r)

    def insert(self, prob, key):
        """Insert a probability vector with key name and store them in hash_table

        Args:
            prob (array): a square root probability vector
            key (string): key name of the inserted probability vector 
        """
        # Reshape from (dim, ) to (dim, 1) to avoid Python broadcasting error
        # prob = np.reshape(prob, (-1,1))

        # lsh_values = self.compute_lsh(prob).flatten()
        lsh_values = prob

        # Insert each band hash value into hash table
        index = 0

        # Concatenation AND
        for start, end in self.hash_ranges:
            dict_key = hash((index, *lsh_values[start:end].tolist()))
            self._insert(dict_key, key)
            index += 1

    def query(self, prob, threshold=0):
        """Retrieve the keys of probs that are similiar to the 
        given probability vector

        Args:
            prob (array): a square root probability vector
            threshold (int, optional): number of collided hash value

        """
        # Reshape from (dim, ) to (dim, 1) to avoid Python broadcasting error
        # prob = np.reshape(prob, (-1,1))

        # Compute LSH for the given probability vector
        candidates = list()
        # lsh_values = self.compute_lsh(prob).flatten()
        lsh_values = prob

        # Construct a hit dictonary
        candidate_hit = defaultdict(int)

        # Retrieve the probs
        index = 0
        for start, end in self.hash_ranges:
            dict_key = hash((index, *lsh_values[start:end].tolist()))
            for val in self.hash_table.get(dict_key, []):
                candidate_hit[val] += 1

            index += 1

        for k, v in candidate_hit.items():
            if v >= threshold:
                # add to result multiple times
                # for _ in range(v):
                candidates.append(k)

        # TODO Can be optimized to in the future
        # if allow_duplicate:
        return candidates
        # else:
        #     return set(candidates)

    def _insert(self, key, value):
        """Insert a band hash value to the hash table

        Args:
            key (int): band hash value
            value (str): key name of the given probability vector

        """
        self.hash_table[key].add(value)
