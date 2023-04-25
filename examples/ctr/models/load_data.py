import os
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm

default_data_path = osp.join(
    osp.split(osp.abspath(__file__))[0], '../datasets')
default_criteo_path = osp.join(default_data_path, 'criteo')
default_avazu_path = osp.join(default_data_path, 'avazu')


def get_dataset(dataset):
    if dataset == 'criteo':
        return CriteoDataset
    elif dataset == 'avazu':
        return AvazuDataset
    else:
        raise NotImplementedError


class CTRDataset(object):
    def __init__(self, path):
        self.path = path
        self.raw_path = self.join('train.csv')
        self.phases = ['train', 'val', 'test']
        self.keys = ['dense', 'sparse', 'label']
        self.dtypes = [np.float32, np.int32, np.int32]
        self.shapes = [(-1, self.num_dense), (-1, self.num_sparse), (-1,)]

    @property
    def num_dense(self):
        raise NotImplementedError

    @property
    def num_sparse(self):
        raise NotImplementedError

    @property
    def num_embed(self):
        raise NotImplementedError

    @property
    def num_embed_separate(self):
        raise NotImplementedError

    def all_exists(self, paths):
        return all([osp.exists(pa) for pa in paths])

    def join(self, fpath):
        return osp.join(self.path, fpath)

    def download(self, path):
        raise NotImplementedError

    def read_from_raw(self, nrows=-1):
        raise NotImplementedError

    def read_csv(self, nrows=-1):
        path = self.raw_path
        if not osp.exists(path):
            self.download(path)
        if nrows > 0:
            df = pd.read_csv(path, nrows=nrows)
        else:
            df = pd.read_csv(path)
        return df

    def process_dense_feats(self, data, feats, inplace=True):
        if inplace:
            d = data
        else:
            d = data.copy()
        d = d[feats].fillna(0.0)
        for f in feats:
            d[f] = d[f].apply(lambda x: np.log(
                x+1) if x > 0 else 0)  # for criteo
        return d

    def process_sparse_feats(self, data, feats, inplace=True):
        from sklearn.preprocessing import LabelEncoder
        # process to embeddings.
        if inplace:
            d = data
        else:
            d = data.copy()
        d = d[feats].fillna("0")
        for f in feats:
            label_encoder = LabelEncoder()
            d[f] = label_encoder.fit_transform(d[f])
        feature_cnt = 0
        counts = [0]
        for f in feats:
            d[f] += feature_cnt
            feature_cnt += d[f].nunique()
            counts.append(feature_cnt)
        return d, counts

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

    def accum_to_count(self, accum):
        count = []
        for i in range(len(accum) - 1):
            count.append(accum[i+1] - accum[i])
        return count

    def count_to_accum(self, count):
        accum = [0]
        for c in count:
            accum.append(accum[-1] + c)
        return accum

    def get_split_indices(self, num_samples):
        raise NotImplementedError

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

    def get_whole_remap(self, train_data, top_percent):
        # now the filename is not correlated to top percent;
        # if modify top percent, MUST modify the fpath!
        remap_path = self.join(f'freq_remap{top_percent}.bin')
        if osp.exists(remap_path):
            remap_indices = np.fromfile(remap_path, dtype=np.int32)
        else:
            result = self.get_whole_frequency_split(train_data, top_percent)
            remap_indices = self.remap_split_frequency(result)
            remap_indices.tofile(remap_path)
        return remap_indices

    def get_separate_remap(self, train_data, top_percent):
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
        cache_path = self.join('freq_grouping.bin')
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

    def process_all_data_by_day(self, separate_fields=False):
        all_data_path = [
            [self.join(f'kaggle_processed_{ph}_{k}.bin') for k in self.keys] for ph in self.phases]

        data_ready = self.all_exists(sum(all_data_path, []))

        if not data_ready:
            ckeys = self.keys + ['count']
            cdtypes = self.dtypes + [np.int32]
            cshapes = self.shapes + [(-1,)]
            pro_paths = [
                self.join(f'kaggle_processed_{k}.bin') for k in ckeys]
            pro_data_ready = self.all_exists(pro_paths)

            if not pro_data_ready:
                print("Reading raw data={}".format(self.raw_path))
                all_data = self.read_from_raw()
                for data, path, dtype in zip(all_data, pro_paths, cdtypes):
                    data = np.array(data, dtype=dtype)
                    data.tofile(path)
            data_with_accum = [np.fromfile(path, dtype=dtype).reshape(
                shape) for path, dtype, shape in zip(pro_paths, cdtypes, cshapes)]
            accum = data_with_accum[-1]
            input_data = data_with_accum[:-1]
            counts = self.accum_to_count(accum)

            indices = self.get_split_indices(input_data[0].shape[0])

            # create training, validation, and test sets
            for ind, cur_paths in zip(indices, all_data_path):
                cur_data = [d[ind] for d in input_data]
                for d, p in zip(cur_data, cur_paths):
                    d.tofile(p)
        else:
            count = np.fromfile(
                self.join(f'kaggle_processed_count.bin'), dtype=np.int32)
            counts = self.accum_to_count(count)
        assert counts == self.num_embed_separate

        def get_data(phase):
            index = {'train': 0, 'val': 1, 'test': 2}[phase]
            sparse_index = self.keys.index('sparse')
            memmap_data = [np.memmap(p, mode='r', dtype=dtype).reshape(
                shape) for p, dtype, shape in zip(all_data_path[index], self.dtypes, self.shapes)]
            if separate_fields:
                memmap_data[sparse_index] = self.get_separate_fields(
                    self.join(f'kaggle_processed_{phase}_sparse_sep.bin'), memmap_data[sparse_index], counts)
            return memmap_data

        training_data = get_data('train')
        validation_data = get_data('val')
        testing_data = get_data('test')
        return tuple(zip(training_data, validation_data, testing_data))


class CriteoDataset(CTRDataset):
    def __init__(self, path=default_criteo_path):
        super().__init__(path)

    @property
    def num_dense(self):
        return 13

    @property
    def num_sparse(self):
        return 26

    @property
    def num_embed(self):
        return 33762577

    @property
    def num_embed_separate(self):
        return [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683,
                8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572]

    def download(self, path=None):
        import tarfile
        from six.moves import urllib
        if path is None:
            path = self.path
        if not osp.exists(path):
            os.makedirs(path)
        assert osp.isdir(path), 'Please provide a directory path.'
        # this source may be invalid, please use other valid sources.
        origin = (
            'https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz'
        )
        print('Downloading data from %s' % origin)
        dataset = osp.join(path, 'criteo.tar.gz')
        urllib.request.urlretrieve(origin, dataset)
        print("Extracting criteo zip...")
        with tarfile.open(dataset) as f:
            f.extractall(path=path)
        print("Create local files...")

    def read_from_raw(self, nrows=-1):
        df = self.read_csv(nrows)
        dense_feats = [col for col in df.columns if col.startswith('I')]
        sparse_feats = [col for col in df.columns if col.startswith('C')]
        assert len(dense_feats) == self.num_dense
        assert len(sparse_feats) == self.num_sparse
        labels = df['label']
        dense = self.process_dense_feats(df, dense_feats)
        sparse, counts = self.process_sparse_feats(df, sparse_feats)
        return dense, sparse, labels, counts

    def get_split_indices(self, num_samples):
        total_per_file = []
        days = 7
        num_data_per_split, extras = divmod(num_samples, days)
        total_per_file = [num_data_per_split] * days
        for j in range(extras):
            total_per_file[j] += 1
        offset_per_file = np.array([0] + [x for x in total_per_file])
        for i in range(days):
            offset_per_file[i + 1] += offset_per_file[i]
        print("File offsets: {}".format(offset_per_file))

        # create reordering
        indices = np.arange(num_samples)
        indices = np.array_split(indices, offset_per_file[1:-1])
        train_indices = np.concatenate(indices[:-1])
        test_indices = indices[-1]
        test_indices, val_indices = np.array_split(test_indices, 2)
        # randomize
        train_indices = np.random.permutation(train_indices)
        print("Randomized indices across days ...")
        indices = [train_indices, val_indices, test_indices]

        return indices


class AvazuDataset(CTRDataset):
    def __init__(self, path=default_avazu_path):
        # please download manually from https://www.kaggle.com/c/avazu-ctr-prediction/data
        super().__init__(path)
        self.keys = ['sparse', 'label']
        self.dtypes = [np.int32, np.int32]
        self.shapes = [(-1, self.num_sparse), (-1,)]

    @property
    def num_dense(self):
        return 0

    @property
    def num_sparse(self):
        return 22

    @property
    def num_embed(self):
        return 9449445

    @property
    def num_embed_separate(self):
        return [240, 7, 7, 4737, 7745, 26, 8552, 559, 36,
                2686408, 6729486, 8251, 5, 4, 2626, 8, 9, 435, 4, 68, 172, 60]

    def read_from_raw(self, nrows=-1):
        df = self.read_csv(nrows)
        sparse_feats = [
            col for col in df.columns if col not in ('click', 'id')]
        assert len(sparse_feats) == self.num_sparse
        labels = df['click']
        sparse, counts = self.process_sparse_feats(df, sparse_feats)
        return sparse, labels, counts

    def get_split_indices(self, num_samples):
        # get exactly number of samples of last day by 'hour' column
        n_last_day = 4218938
        indices = np.arange(num_samples)
        train_indices = indices[:-n_last_day]
        test_indices = indices[-n_last_day:]
        test_indices, val_indices = np.array_split(test_indices, 2)
        # randomize
        train_indices = np.random.permutation(train_indices)
        print("Randomized indices across days ...")
        indices = [train_indices, val_indices, test_indices]
        return indices


if __name__ == '__main__':
    criteo = CriteoDataset(default_criteo_path)
    criteo.download()
