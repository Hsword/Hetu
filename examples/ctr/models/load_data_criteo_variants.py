import os
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
if __name__ == '__main__':
    from load_data import CTRDataset
else:
    from .load_data import CTRDataset

default_data_path = osp.join(
    osp.split(osp.abspath(__file__))[0], '../datasets')
default_criteo_path = osp.join(default_data_path, 'criteo')
default_criteo2core_path = osp.join(default_data_path, 'criteo2core')


class Criteo2CoreDataset(CTRDataset):
    def __init__(self, path=None, criteo_path=None):
        if path is None:
            path = default_criteo2core_path
        if criteo_path is None:
            criteo_path = default_criteo_path
        assert criteo_path != path
        self.criteo_path = criteo_path
        self.criteo2core_path = path
        super().__init__(path)

    @property
    def num_dense(self):
        return 13

    @property
    def num_sparse(self):
        return 26

    @property
    def num_embed(self):
        return 6779211

    @property
    def num_embed_separate(self):
        return [1460, 570, 1545892, 668221, 305, 24, 12399, 633, 3, 74829, 5557, 1518203, 3194,
                27, 13883, 1231446, 10, 5310, 2150, 4, 1437677, 18, 15, 157462, 103, 99816]

    def get_sparsity(self):
        # the average samples per feature
        label = np.memmap(self.join('label.bin'), mode='r', dtype=np.int32)
        nsamples = label.shape[0]
        del label
        return nsamples * self.num_sparse / self.num_embed

    def get_skewness(self, nreport=10):
        # the percent of samples of top % features
        sparse = np.fromfile(self.join('sparse.bin'), dtype=np.int32)
        cnts = np.bincount(sparse)
        sorted_indices = np.argsort(-cnts)
        nsamples = []
        nfeature_per_interval = round(self.num_embed / nreport)
        for i in range(nreport):
            start = i * nfeature_per_interval
            if i == nreport - 1:
                ending = self.num_embed
            else:
                ending = start + nfeature_per_interval
            nsamples.append(cnts[sorted_indices[start:ending]].sum())
        nsamples = np.array(nsamples, dtype=np.float32) / sparse.shape[0]
        return nsamples.tolist()

    def criteo_join(self, path):
        return osp.join(self.criteo_path, path)

    def criteo2core_join(self, path):
        return osp.join(self.criteo2core_path, path)

    def symlink(self):
        if not osp.exists(self.join('dense.bin')):
            os.symlink(self.criteo2core_join(
                'dense.bin'), self.join('dense.bin'))
        if not osp.exists(self.join('label.bin')):
            os.symlink(self.criteo2core_join(
                'label.bin'), self.join('label.bin'))

    def save_counts_n_sep(self, counts, sparse):
        print('Processed counts:', counts)
        assert counts == self.num_embed_separate
        counts = np.array(counts, dtype=np.int32)
        counts.tofile(self.join('count.bin'))
        accum = self.count_to_accum(counts)
        accum = np.array(accum, dtype=np.int32)
        accum.tofile(self.join('accum.bin'))
        for i in range(self.num_sparse):
            sparse[:, i] -= accum[i]
        sparse.tofile(self.join('sparse_sep.bin'))

    def transform_from_criteo(self):
        from sklearn.preprocessing import LabelEncoder
        df = pd.read_csv(self.criteo_join('train.csv'))
        sparse_feats = [col for col in df.columns if col.startswith('C')]
        hasnull = df[sparse_feats].isnull().any().to_numpy()
        criteo_sparse = np.fromfile(self.criteo_join(
            'kaggle_processed_sparse.bin'), dtype=np.int32).reshape(-1, self.num_sparse)
        offset = 0
        counts = []
        for i in range(self.num_sparse):
            cur_column = criteo_sparse[:, i]
            cur_column = cur_column - cur_column.min()
            cnt = np.bincount(cur_column)
            has1 = (cnt == 1).any()
            if has1:
                if not hasnull[i]:
                    cur_column = cur_column + 1
                    cnt = np.concatenate(
                        (np.zeros((1,), dtype=cnt.dtype), cnt))
                cur_column[cnt[cur_column] == 1] = 0
                label_encoder = LabelEncoder()
                cur_column = label_encoder.fit_transform(cur_column)
            criteo_sparse[:, i] = cur_column + offset
            counts.append(cur_column.max() + 1)
            offset += counts[-1]
        criteo_sparse.tofile(self.join('sparse.bin'))
        self.save_counts_n_sep(counts, criteo_sparse)

        os.symlink(self.criteo_join('kaggle_processed_dense.bin'),
                   self.join('dense.bin'))
        os.symlink(self.criteo_join('kaggle_processed_label.bin'),
                   self.join('label.bin'))

    def get_split_indices(self, num_samples):
        total_per_file = []
        days = 7
        num_data_per_split, extras = divmod(num_samples, days)
        total_per_file = np.full(
            (days,), fill_value=num_data_per_split, dtype=np.int32)
        total_per_file[:extras] += 1
        offset_per_file = np.cumsum(total_per_file)

        train_ending = offset_per_file[-2]
        train_indices = slice(train_ending)
        nlast = total_per_file[-1]
        nval = nlast // 2
        ntest = nlast - nval
        test_ending = train_ending + ntest
        test_indices = slice(train_ending, test_ending)
        val_indices = slice(test_ending, offset_per_file[-1])
        print('Slices:', train_indices, val_indices, test_indices)
        return train_indices, val_indices, test_indices

    def process_all_data_by_day(self, separate_fields=False):
        all_data_path = [self.join(f'{k}.bin') for k in self.keys]
        if separate_fields:
            sparse_index = self.keys.index('sparse')
            all_data_path[sparse_index] = self.join('sparse_sep.bin')

        data_ready = self.all_exists(all_data_path)

        if not data_ready:
            self.transform_from_criteo()

        memmap_data = [np.memmap(p, mode='r', dtype=dtype).reshape(
            shape) for p, dtype, shape in zip(all_data_path, self.dtypes, self.shapes)]

        slices = self.get_split_indices(memmap_data[0].shape[0])
        dense_data, sparse_data, label_data = [
            [data[sli] for sli in slices] for data in memmap_data]

        return dense_data, sparse_data, label_data


class Criteo2CoreSparsifiedDataset(Criteo2CoreDataset):
    def __init__(self, path=None, criteo2core_path=None):
        if path is None:
            path = osp.join(default_data_path, 'criteo2core_sparsified')
        if criteo2core_path is None:
            criteo2core_path = default_criteo2core_path
        assert criteo2core_path != path
        self.criteo2core_path = criteo2core_path
        CTRDataset.__init__(self, path)

    @property
    def num_embed(self):
        return 13558422

    @property
    def num_embed_separate(self):
        return [2920, 1140, 3091784, 1336442, 610, 48, 24798, 1266, 6, 149658, 11114, 3036406, 6388,
                54, 27766, 2462892, 20, 10620, 4300, 8, 2875354, 36, 30, 314924, 206, 199632]

    def transform_from_criteo(self):
        # actually transform from criteo2core
        self.symlink()

        ori_sparse_path = self.criteo2core_join('sparse.bin')
        ori_sparse = np.fromfile(
            ori_sparse_path, dtype=np.int32).reshape(-1, self.num_sparse)
        # transpose to alleviate the burden of dict
        x = ori_sparse.T.reshape(-1)
        ad = np.empty(x.shape, dtype=x.dtype)
        y = np.bincount(x)
        d = {}
        np.random.seed(123)
        for i in tqdm(range(x.shape[0]), desc='Sparsifying'):
            value = x[i]
            if value not in d:
                nx = y[value]
                seq = np.zeros(nx, dtype=np.int32)
                nones = round(nx/2+np.random.random()-0.5)
                seq[:int(nones)] = 1
                np.random.shuffle(seq)
                d[value] = [seq, 0]
            dseq, dind = d[value]
            ad[i] = dseq[dind]
            dind += 1
            if dind == len(dseq):
                _ = d.pop(value)
            else:
                d[value][1] = dind
        ad.tofile(self.join('addend.bin'))
        x = 2 * x + ad
        ny = np.bincount(x)
        assert len(ny) == 2 * len(y) and (ny > 0).all()  # test
        new_sparse = x.reshape(self.num_sparse, -1).T.reshape(-1)
        new_sparse.tofile(self.join('sparse.bin'))
        new_sparse = new_sparse.reshape(-1, self.num_sparse)

        counts = []
        for i in range(self.num_sparse):
            cur_column = new_sparse[:, i]
            counts.append(cur_column.max() - cur_column.min() + 1)
        self.save_counts_n_sep(counts, new_sparse)


class Criteo2CoreDensifiedDataset(Criteo2CoreDataset):
    def __init__(self, path=None, criteo2core_path=None):
        if path is None:
            path = osp.join(default_data_path, 'criteo2core_densified')
        if criteo2core_path is None:
            criteo2core_path = default_criteo2core_path
        assert criteo2core_path != path
        self.criteo2core_path = criteo2core_path
        CTRDataset.__init__(self, path)

    @property
    def num_embed(self):
        return 3389612

    @property
    def num_embed_separate(self):
        return [730, 285, 772946, 334111, 153, 12, 6200, 317, 2, 37415, 2779, 759102, 1597,
                14, 6942, 615723, 5, 2655, 1075, 2, 718839, 9, 8, 78731, 52, 49908]

    def transform_from_criteo(self):
        # actually transform from criteo2core
        self.symlink()

        ori_sparse_path = self.criteo2core_join('sparse.bin')
        sparse = np.fromfile(
            ori_sparse_path, dtype=np.int32).reshape(-1, self.num_sparse)

        np.random.seed(124)
        counts = []
        offset = 0
        for i in range(self.num_sparse):
            # here features with the same frequency are hashed together
            # so as to maintain the skewness
            ori_column = sparse[:, i]
            ori_column = ori_column - ori_column.min()
            ori_count = super().num_embed_separate[i]
            cnts = np.bincount(ori_column)
            assert ori_count == cnts.shape[0]
            new_count = (ori_count + 1) // 2
            counts.append(new_count)
            sorted_indices = np.argsort(-cnts)
            new_indices = np.empty(ori_count, dtype=np.int32)
            new_indices[sorted_indices] = np.arange(
                ori_count, dtype=np.int32) // 2
            new_column = new_indices[ori_column]
            sparse[:, i] = new_column + offset
            offset += new_count
        sparse.tofile(self.join('sparse.bin'))

        self.save_counts_n_sep(counts, sparse)


class Criteo2CoreMoreSkewedDataset(Criteo2CoreDataset):
    def __init__(self, path=None, criteo2core_path=None):
        if path is None:
            path = osp.join(default_data_path, 'criteo2core_moreskewed')
        if criteo2core_path is None:
            criteo2core_path = default_criteo2core_path
        assert criteo2core_path != path
        self.criteo2core_path = criteo2core_path
        CTRDataset.__init__(self, path)

    @property
    def num_embed(self):
        return 6779218

    @property
    def num_embed_separate(self):
        return [1461, 570, 1545892, 668221, 306, 24, 12399, 633, 3, 74829, 5557, 1518204, 3195,
                27, 13884, 1231446, 10, 5310, 2151, 4, 1437678, 18, 15, 157462, 103, 99816]

    def transform_from_criteo(self):
        # actually transform from criteo2core
        self.symlink()

        ori_sparse_path = self.criteo2core_join('sparse.bin')
        sparse = np.fromfile(
            ori_sparse_path, dtype=np.int32).reshape(-1, self.num_sparse)

        np.random.seed(125)
        counts = []
        offset = 0
        if osp.exists(self.join('addend.bin')):
            ads = np.fromfile(self.join('addend.bin'),
                              dtype=np.int32).reshape(26, -1)
        else:
            ads = []
        for i in range(self.num_sparse):
            ori_column = sparse[:, i]
            ori_column = ori_column - ori_column.min()
            ori_count = super().num_embed_separate[i]
            assert ori_count == ori_column.max() + 1
            # first split frequency
            cnts = np.bincount(ori_column)
            n_tosparsify = round(ori_count / 3)
            n_todensify = ori_count - n_tosparsify
            sorted_indices = np.argsort(cnts)
            sp_indices = sorted_indices[:n_tosparsify]
            de_indices = sorted_indices[n_tosparsify:]
            sp_mask = np.zeros((ori_count,), dtype=np.int32)
            sp_mask[sp_indices] = 1
            if isinstance(ads, list):
                ad = np.full(ori_column.shape, fill_value=-1, dtype=np.int32)
                d = {}
                for j in tqdm(range(ori_column.shape[0]), desc=f'Sparsifying {i}'):
                    value = ori_column[j]
                    if sp_mask[value]:
                        if value not in d:
                            nx = cnts[value]
                            seq = np.zeros(nx, dtype=np.int32)
                            nones = round(nx/2+np.random.random()-0.5)
                            seq[:int(nones)] = 1
                            np.random.shuffle(seq)
                            d[value] = [seq, 0]
                        dseq, dind = d[value]
                        ad[j] = dseq[dind]
                        dind += 1
                        if dind == len(dseq):
                            _ = d.pop(value)
                        else:
                            d[value][1] = dind
                ads.append(ad)
            else:
                ad = ads[i]
            ind_mapping = np.empty((ori_count,), dtype=np.int32)
            ind_mapping[sp_indices] = np.arange(
                n_tosparsify, dtype=np.int32) * 2
            new_de_indices = np.arange(n_todensify, dtype=np.int32)
            if len(new_de_indices) % 2 == 1:
                new_de_indices += 1
            new_sp_count = n_tosparsify * 2
            new_de_count = (n_todensify + 1) // 2
            new_de_indices = new_de_indices // 2 + new_sp_count
            ind_mapping[de_indices] = new_de_indices
            new_column = ind_mapping[ori_column]
            new_column = new_column + np.maximum(ad, 0)
            sparse[:, i] = new_column + offset
            new_count = new_sp_count + new_de_count
            counts.append(new_count)
            offset += new_count

        if isinstance(ads, list):
            ads = np.concatenate(ads)
            ads.tofile(self.join('addend.bin'))

        sparse.tofile(self.join('sparse.bin'))

        self.save_counts_n_sep(counts, sparse)


class Criteo2CoreLessSkewedDataset(Criteo2CoreDataset):
    def __init__(self, path=None, criteo2core_path=None):
        if path is None:
            path = osp.join(default_data_path, 'criteo2core_lessskewed')
        if criteo2core_path is None:
            criteo2core_path = default_criteo2core_path
        assert criteo2core_path != path
        self.criteo2core_path = criteo2core_path
        CTRDataset.__init__(self, path)

    @property
    def num_embed(self):
        return 6779218

    @property
    def num_embed_separate(self):
        return [1461, 570, 1545892, 668221, 306, 24, 12399, 633, 3, 74829, 5557, 1518204, 3195,
                27, 13884, 1231446, 10, 5310, 2151, 4, 1437678, 18, 15, 157462, 103, 99816]

    def transform_from_criteo(self):
        # actually transform from criteo2core
        self.symlink()

        ori_sparse_path = self.criteo2core_join('sparse.bin')
        sparse = np.fromfile(
            ori_sparse_path, dtype=np.int32).reshape(-1, self.num_sparse)

        np.random.seed(126)
        counts = []
        offset = 0
        if osp.exists(self.join('addend.bin')):
            ads = np.fromfile(self.join('addend.bin'),
                              dtype=np.int32).reshape(26, -1)
        else:
            ads = []
        for i in range(self.num_sparse):
            ori_column = sparse[:, i]
            ori_column = ori_column - ori_column.min()
            ori_count = super().num_embed_separate[i]
            assert ori_count == ori_column.max() + 1
            # first split frequency
            cnts = np.bincount(ori_column)
            n_tosparsify = round(ori_count / 3)
            n_todensify = ori_count - n_tosparsify
            sorted_indices = np.argsort(-cnts)
            sp_indices = sorted_indices[:n_tosparsify]
            de_indices = sorted_indices[n_tosparsify:]
            sp_mask = np.zeros((ori_count,), dtype=np.int32)
            sp_mask[sp_indices] = 1
            if isinstance(ads, list):
                ad = np.full(ori_column.shape, fill_value=-1, dtype=np.int32)
                d = {}
                for j in tqdm(range(ori_column.shape[0]), desc=f'Sparsifying {i}'):
                    value = ori_column[j]
                    if sp_mask[value]:
                        if value not in d:
                            nx = cnts[value]
                            seq = np.zeros(nx, dtype=np.int32)
                            nones = round(nx/2+np.random.random()-0.5)
                            seq[:int(nones)] = 1
                            np.random.shuffle(seq)
                            d[value] = [seq, 0]
                        dseq, dind = d[value]
                        ad[j] = dseq[dind]
                        dind += 1
                        if dind == len(dseq):
                            _ = d.pop(value)
                        else:
                            d[value][1] = dind
                ads.append(ad)
            else:
                ad = ads[i]
            ind_mapping = np.empty((ori_count,), dtype=np.int32)
            ind_mapping[sp_indices] = np.arange(
                n_tosparsify, dtype=np.int32) * 2
            new_de_indices = np.arange(n_todensify, dtype=np.int32)
            np.random.shuffle(new_de_indices)
            new_sp_count = n_tosparsify * 2
            new_de_count = (n_todensify + 1) // 2
            new_de_indices = new_de_indices % new_de_count + new_sp_count
            ind_mapping[de_indices] = new_de_indices
            new_column = ind_mapping[ori_column]
            new_column = new_column + np.maximum(ad, 0)
            sparse[:, i] = new_column + offset
            new_count = new_sp_count + new_de_count
            counts.append(new_count)
            offset += new_count

        if isinstance(ads, list):
            ads = np.concatenate(ads)
            ads.tofile(self.join('addend.bin'))

        sparse.tofile(self.join('sparse.bin'))

        self.save_counts_n_sep(counts, sparse)


def _gen_data_n_test(dataset):
    # generate and check data
    dense, sparse, label = dataset.process_all_data()
    print([d.shape for d in dense])
    print([d.shape for d in sparse])
    print([d.shape for d in label])

    # get sparsity and skewness
    print('sparsity', dataset.get_sparsity())
    print('skewness', dataset.get_skewness())
    print()


if __name__ == '__main__':
    # 2-core
    print('2-core dataset')
    target_path = default_criteo2core_path
    os.makedirs(target_path, exist_ok=True)
    dataset = Criteo2CoreDataset(target_path)
    _gen_data_n_test(dataset)

    # sparsified
    print('sparsified dataset')
    target_path = osp.join(default_data_path, 'criteo2core_sparsified')
    os.makedirs(target_path, exist_ok=True)
    dataset = Criteo2CoreSparsifiedDataset(target_path)
    _gen_data_n_test(dataset)

    # densified
    print('densified dataset')
    target_path = osp.join(default_data_path, 'criteo2core_densified')
    os.makedirs(target_path, exist_ok=True)
    dataset = Criteo2CoreDensifiedDataset(target_path)
    _gen_data_n_test(dataset)

    # more skewed
    print('more skewed dataset')
    target_path = osp.join(default_data_path, 'criteo2core_moreskewed')
    os.makedirs(target_path, exist_ok=True)
    dataset = Criteo2CoreMoreSkewedDataset(target_path)
    _gen_data_n_test(dataset)

    # less skewed
    print('less skewed dataset')
    target_path = osp.join(default_data_path, 'criteo2core_lessskewed')
    os.makedirs(target_path, exist_ok=True)
    dataset = Criteo2CoreLessSkewedDataset(target_path)
    _gen_data_n_test(dataset)
