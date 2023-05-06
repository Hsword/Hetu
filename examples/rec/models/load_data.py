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
        df_users = df.groupby(self.columns[0], group_keys=False)
        tra_data = df_users.apply(lambda x: x[:-2])
        val_data = df_users.apply(lambda x: x[-2:-1])
        inf_data = df_users.apply(lambda x: x[-1:])
        assert counts == self.num_embed_separate
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

    def process_all_data(self, separate_fields=True):
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
            memmap_data = [np.memmap(p, mode='r', dtype=dtype).reshape(shape)
                           for p, dtype, shape in zip(all_data_path[index], self.dtypes, self.shapes)]
            return memmap_data

        training_data = get_data('train')
        validation_data = get_data('val')
        testing_data = get_data('test')
        return tuple(zip(training_data, validation_data, testing_data))


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
