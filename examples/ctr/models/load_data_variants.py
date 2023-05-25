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


class Criteo2CoreDataset(CTRDataset):
    def __init__(self, path, criteo_path=None):
        super().__init__(path)
        if criteo_path is None:
            criteo_path = default_criteo_path
        assert criteo_path != self.path
        self.criteo_path = criteo_path

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

    def criteo_join(self, path):
        return osp.join(self.criteo_path, path)

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
        print('Processed counts:', counts)
        assert counts == self.num_embed_separate
        counts = np.array(counts, dtype=np.int32)
        counts.tofile(self.join('count.bin'))
        accum = self.count_to_accum(counts)
        accum = np.array(accum, dtype=np.int32)
        accum.tofile(self.join('accum.bin'))
        for i in range(self.num_sparse):
            criteo_sparse[:, i] -= accum[i]
        criteo_sparse.tofile(self.join('sparse_sep.bin'))

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


def sparsify(x):
    x = x.reshape(-1, 26).T.reshape(-1)
    ad = np.empty(x.shape, dtype=x.dtype)
    y = np.bincount(x)
    d = {}
    for i in tqdm(range(x.shape[0])):
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
    x = 2 * x + ad
    ny = np.bincount(x)
    assert len(ny) == 2 * len(y) and (ny > 0).all()  # test
    return x


def densify(x):
    ...


if __name__ == '__main__':
    target_path = osp.join(default_data_path, 'criteo2core')
    os.makedirs(target_path, exist_ok=True)
    dataset = Criteo2CoreDataset(target_path)
    dense, sparse, label = dataset.process_all_data()
    print([d.shape for d in dense])
    print([d.shape for d in sparse])
    print([d.shape for d in label])
