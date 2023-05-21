import os
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

default_data_path = osp.join(
    osp.split(osp.abspath(__file__))[0], '../datasets')
default_ml20m_path = osp.join(default_data_path, 'ml-20m')
default_amazonbooks_path = osp.join(default_data_path, 'amazon-books')


def get_dataset(dataset):
    if dataset in ('ml20m', 'ml-20m', 'ml_20m'):
        return ML20MDataset
    elif dataset in ('amazonbooks', 'amazon-books', 'amazon_books'):
        return AmazonBooksDataset


class RatingDataset(object):
    def __init__(self, path, raw_path):
        # default ml 20m
        self.path = path
        self.raw_path = self.join(raw_path)
        self.phases = ['train', 'val', 'test']
        self.keys = ['sparse', 'rating']
        self.dtypes = [np.int32, np.float32]
        self.shapes = [(-1, 2), (-1,)]

    @property
    def num_dense(self):
        return 0

    @property
    def num_sparse(self):
        return 2

    @property
    def num_embed(self):
        return self.num_user_embed + self.num_item_embed

    @property
    def num_embed_separate(self):
        return [self.num_user_embed, self.num_item_embed]

    @property
    def columns(self):
        raise NotImplementedError

    @property
    def num_user_embed(self):
        raise NotImplementedError

    @property
    def num_item_embed(self):
        raise NotImplementedError

    def all_exists(self, paths):
        return all([osp.exists(pa) for pa in paths])

    def join(self, fpath):
        return osp.join(self.path, fpath)

    def read_from_raw(self, nrows=-1):
        df = self.read_csv(nrows)
        df = df.sort_values('timestamp')
        df, counts = self.process_sparse_feats(df, self.columns[:2])
        assert counts == self.num_embed_separate
        df[self.columns[1]] = df[self.columns[1]] + self.num_user_embed
        df[self.columns[2]] = df[self.columns[2]] / df[self.columns[2]].max()
        df_users = df.groupby(self.columns[0], group_keys=False)
        tra_data = df_users.apply(lambda x: x[:-2])
        val_data = df_users.apply(lambda x: x[-2:-1])
        inf_data = df_users.apply(lambda x: x[-1:])
        return tra_data, val_data, inf_data, counts

    def read_csv(self, nrows=-1):
        path = self.raw_path
        if not osp.exists(path):
            assert False, f'Raw path {path} not exists.'
        if nrows > 0:
            df = pd.read_csv(path, nrows=nrows)
        else:
            df = pd.read_csv(path)
        return df

    def process_sparse_feats(self, data, feats):
        from sklearn.preprocessing import LabelEncoder
        # inplace process for embeddings
        for f in feats:
            label_encoder = LabelEncoder()
            data[f] = label_encoder.fit_transform(data[f])
        counts = []
        for f in feats:
            counts.append(data[f].nunique())
        return data, counts

    def get_separate_fields(self, path, sparse, num_embed_fields):
        if osp.exists(path):
            return np.memmap(path, mode='r', dtype=np.int32).reshape(-1, self.num_sparse)
        else:
            accum = 0
            sparse = np.array(sparse)
            for i in range(self.num_sparse):
                sparse[:, i] -= accum
                accum += num_embed_fields[i]
            sparse.tofile(path)
            return sparse

    def process_all_data(self, separate_fields=False):
        all_data_path = [
            [self.join(f'{ph}_{k}.bin') for k in self.keys] for ph in self.phases]

        data_ready = self.all_exists(sum(all_data_path, []))

        if not data_ready:
            print("Reading raw data={}".format(self.raw_path))
            all_data = self.read_from_raw()
            for data, paths in zip(all_data[:3], all_data_path):
                for col, dtype, path in zip([self.columns[:2], self.columns[2]], self.dtypes, paths):
                    cur_data = np.array(data[col], dtype=dtype)
                    cur_data.tofile(path)

        def get_data(phase):
            index = {'train': 0, 'val': 1, 'test': 2}[phase]
            sparse_index = self.keys.index('sparse')
            memmap_data = [np.memmap(p, mode='r', dtype=dtype).reshape(shape)
                           for p, dtype, shape in zip(all_data_path[index], self.dtypes, self.shapes)]
            if separate_fields:
                memmap_data[sparse_index] = self.get_separate_fields(
                    self.join(f'{phase}_sparse_sep.bin'), memmap_data[sparse_index], self.num_embed_separate)
            return memmap_data

        training_data = get_data('train')
        validation_data = get_data('val')
        testing_data = get_data('test')
        return tuple(zip(training_data, validation_data, testing_data))

    def _binary_search_frequency(self, left, right, counter, target):
        if left >= right - 1:
            return left
        middle = (left + right) // 2
        high_freq_num = np.sum(counter >= middle)
        if high_freq_num > target:
            return self._binary_search_frequency(middle, right, counter, target)
        elif high_freq_num < target:
            return self._binary_search_frequency(left, middle, counter, target)
        else:
            return middle

    def get_frequency_counter(self, train_data, num_embed, fpath):
        if osp.exists(fpath):
            counter = np.fromfile(fpath, dtype=np.int32)
        else:
            counter = np.zeros((num_embed,), dtype=np.int32)
            for idx in tqdm(train_data.reshape(-1)):
                counter[idx] += 1
            counter.tofile(fpath)
        return counter

    def get_single_frequency_split(self, train_data, num_embed, top_percent, fpath):
        fpath_parts = fpath.split('.')
        real_fpath = '.'.join(
            fpath_parts[:-1]) + '_' + str(top_percent) + '.' + fpath_parts[-1]
        if osp.exists(real_fpath):
            result = np.fromfile(real_fpath, dtype=np.int32)
        else:
            dirpath, filepath = osp.split(fpath)
            cache_fpath = osp.join(dirpath, 'counter_' + filepath)
            counter = self.get_frequency_counter(
                train_data, num_embed, cache_fpath)
            nhigh = len(counter) * top_percent
            kth = self._binary_search_frequency(
                np.min(counter), np.max(counter) + 1, counter, nhigh)
            print(f'The threshold for high frequency is {kth}.')
            result = (counter >= kth).astype(np.int32)
            print(
                f'Real ratio of high frequency is {np.sum(result) / len(result)}.')
            result.tofile(real_fpath)
        return result

    def get_whole_frequency_split(self, train_data, top_percent):
        # now the filename is not correlated to top percent;
        # if modify top percent, MUST modify the fpath!
        freq_path = self.join('freq_split.bin')
        result = self.get_single_frequency_split(
            train_data, self.num_embed, top_percent, freq_path)
        return result

    def get_separate_frequency_split(self, train_data, top_percent):
        # TODO: merge with CTRDataset; this function is the same in CTRDataset
        # now the filename is not correlated to top percent;
        # if modify top percent, MUST modify the fpath!
        separate_dir = self.join('freq_split_separate')
        os.makedirs(separate_dir, exist_ok=True)
        freq_paths = [
            osp.join(separate_dir, f'fields{i}.bin') for i in range(self.num_sparse)]
        results = [self.get_single_frequency_split(data, nemb, top_percent, fp) for data, nemb, fp in zip(
            train_data, self.num_embed_separate, freq_paths)]
        return results

    def remap_split_frequency(self, frequency):
        hidx = 0
        lidx = 1
        remap_indices = np.zeros(frequency.shape, dtype=np.int32)
        for i, ind in enumerate(frequency):
            if ind:
                remap_indices[i] = hidx
                hidx += 1
            else:
                remap_indices[i] = -lidx
                lidx += 1
        return remap_indices

    def get_separate_remap(self, train_data, top_percent):
        # TODO: merge with CTRDataset; this function is the same in CTRDataset
        # now the filename is not correlated to top percent;
        # if modify top percent, MUST modify the fpath!
        separate_dir = self.join('freq_split_separate')
        os.makedirs(separate_dir, exist_ok=True)
        remap_path = [
            osp.join(separate_dir, f'remap_fields{i}_{top_percent}.bin') for i in range(self.num_sparse)]
        if self.all_exists(remap_path):
            remap_indices = [np.fromfile(rp, dtype=np.int32)
                             for rp in remap_path]
        else:
            results = self.get_separate_frequency_split(
                train_data, top_percent)
            remap_indices = [self.remap_split_frequency(
                res) for res in results]
            for ind, rp in zip(remap_indices, remap_path):
                ind.tofile(rp)
        return remap_indices

    def get_whole_frequency_grouping(self, train_data, nsplit):
        cache_path = self.join(f'freq_grouping_{nsplit}.bin')
        if osp.exists(cache_path):
            grouping = np.fromfile(cache_path, dtype=np.int32)
        else:
            counter_path = self.join('counter_freq_split.bin')
            counter = self.get_frequency_counter(
                train_data, self.num_embed, counter_path)
            indices = np.argsort(counter)
            nignore = 0
            group_index = 0
            grouping = np.zeros(counter.shape, dtype=np.int32)
            cur_nsplit = nsplit
            while cur_nsplit != 0:
                threshold = (self.num_embed - nignore) / cur_nsplit
                assert int(threshold) > 0
                minvalue = counter[indices[nignore]]
                places = (counter == minvalue)
                cnt = places.sum()
                if cnt >= threshold:
                    nignore += cnt
                    grouping[places] = group_index
                    assert cur_nsplit > 1
                else:
                    cur_index = nignore + int(threshold) - 1
                    target_value = counter[indices[cur_index]]
                    assert target_value > minvalue
                    offset = 1
                    while True:
                        if cur_index + offset >= self.num_embed or counter[indices[cur_index + offset]] != target_value:
                            ending = cur_index + offset
                            break
                        elif cur_index - offset <= nignore or counter[indices[cur_index - offset]] != target_value:
                            ending = cur_index - offset + 1
                            break
                        offset += 1
                    assert ending > nignore
                    grouping[indices[nignore:ending]] = group_index
                    nignore = ending
                cur_nsplit -= 1
                group_index += 1
            if nignore < self.num_embed:
                grouping[indices[nignore:self.num_embed]] = group_index - 1
            grouping.tofile(cache_path)
        return grouping


class ML20MDataset(RatingDataset):
    def __init__(self, path=None, raw_path=None):
        if path is None:
            path = default_ml20m_path
        if raw_path is None:
            raw_path = 'ratings.csv'
        super().__init__(path, raw_path)

    @property
    def columns(self):
        return ['userId', 'movieId', 'rating']

    @property
    def num_user_embed(self):
        return 138493

    @property
    def num_item_embed(self):
        return 26744


class AmazonBooksDataset(RatingDataset):
    def __init__(self, path=None, raw_path=None):
        if path is None:
            path = default_amazonbooks_path
        if raw_path is None:
            raw_path = 'ratings.csv'
        super().__init__(path, raw_path)

    @property
    def columns(self):
        return ['userId', 'itemId', 'rating']

    @property
    def num_user_embed(self):
        return 1856344

    @property
    def num_item_embed(self):
        return 704093

    def read_from_raw(self, nrows=-1):
        # preprocess
        csv_file = self.raw_path
        if not osp.exists(csv_file):
            real_raw_path = self.join('Books_5.json')
            assert osp.exists(real_raw_path)
            with open(real_raw_path, 'r') as fr, open(csv_file, 'w') as fw:
                print(','.join(self.columns + ['timestamp']), file=fw)
                for line in tqdm(fr):
                    data = json.loads(line)
                    uid = data['reviewerID']
                    bid = data['asin']
                    rating = data['overall']
                    timestamp = data['unixReviewTime']
                    print(f'{uid},{bid},{rating},{timestamp}', file=fw)
        return super().read_from_raw(nrows)


if __name__ == '__main__':
    dataset = ML20MDataset()
    dataset.handle()
