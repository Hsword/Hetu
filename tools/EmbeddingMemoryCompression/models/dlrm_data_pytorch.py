from __future__ import absolute_import, division, print_function, unicode_literals

import os.path as osp
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset


def make_criteo_data_and_loaders(args, offset_to_length_converter=False):
    train_data = CriteoDataset(
        args.data_set,
        args.data_randomize,
        "train",
        args.raw_data_file,
    )

    test_data = CriteoDataset(
        args.data_set,
        args.data_randomize,
        "test",
        args.raw_data_file,
    )

    collate_wrapper_criteo = collate_wrapper_criteo_offset
    if offset_to_length_converter:
        collate_wrapper_criteo = collate_wrapper_criteo_length

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_criteo,
        pin_memory=False,
        drop_last=False,  # True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_mini_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        collate_fn=collate_wrapper_criteo,
        pin_memory=False,
        drop_last=False,  # True
    )

    return train_data, train_loader, test_data, test_loader


# Kaggle Display Advertising Challenge Dataset
# dataset (str): name of dataset (Kaggle or Terabyte)
# randomize (str): determines randomization scheme
#            "none": no randomization
#            "day": randomizes each day"s data (only works if split = True)
#            "total": randomizes total dataset
# split (bool) : to split into train, test, validation data-sets

class CriteoDataset(Dataset):

    def __init__(
            self,
            dataset,
            randomize,
            split="train",
            in_path="",
            out_path="",
    ):
        self.den_fea = 13
        self.spa_fea = 26
        cur_dir = osp.dirname(__file__)
        criteo_dir = osp.join(cur_dir, '../datasets/criteo')
        if dataset == "kaggle":
            days = 7
            if in_path == "":
                in_path = osp.join(criteo_dir, 'train.txt')
            if out_path == "":
                out_path = osp.join(criteo_dir, 'kaggle_processed.npz')
        elif dataset == "terabyte":
            days = 24
            if in_path == "":
                in_path = osp.join(criteo_dir, 'train_terabyte.txt')
            if out_path == "":
                out_path = osp.join(criteo_dir, 'terabyte_processed.npz')
        else:
            raise(ValueError("Data set option is not supported"))

        assert split in ("none", "train", "val", "test")
        split_out_path = out_path[:-4] + '_' + split + '.npz'

        self.in_path = in_path
        self.out_path = out_path

        data_ready = osp.exists(split_out_path)

        if data_ready:
            data = np.load(split_out_path)
            dense, sparse, label = data['dense'], data['sparse'], data['label']
            self.counts = data['count']
        else:
            dense, sparse, label = self.getAllData()

            # compute offsets per file
            num_samples = dense.shape[0]
            total_per_file = []
            num_data_per_split, extras = divmod(num_samples, days)
            total_per_file = [num_data_per_split] * days
            for j in range(extras):
                total_per_file[j] += 1
            self.offset_per_file = np.array([0] + [x for x in total_per_file])
            for i in range(days):
                self.offset_per_file[i + 1] += self.offset_per_file[i]
            print("File offsets: {}".format(self.offset_per_file))

            # create reordering
            indices = np.arange(dense.shape[0])

            if split == "none":
                # randomize all data
                if randomize == "total":
                    indices = np.random.permutation(indices)
                    print("Randomized indices...")

                dense = dense[indices]
                sparse = sparse[indices]
                label = label[indices]

            else:
                indices = np.array_split(indices, self.offset_per_file[1:-1])

                # randomize train data (per day)
                if randomize == "day":  # or randomize == "total":
                    for i in range(len(indices) - 1):
                        indices[i] = np.random.permutation(indices[i])
                    print("Randomized indices per day ...")

                train_indices = np.concatenate(indices[:-1])
                test_indices = indices[-1]
                test_indices, val_indices = np.array_split(test_indices, 2)

                print("Defined %s indices..." % (split))

                # randomize train data (across days)
                if randomize == "total":
                    train_indices = np.random.permutation(train_indices)
                    print("Randomized indices across days ...")

                # create training, validation, and test sets
                if split == 'train':
                    dense = dense[train_indices]
                    sparse = sparse[train_indices]
                    label = label[train_indices]
                elif split == 'val':
                    dense = dense[val_indices]
                    sparse = sparse[val_indices]
                    label = label[val_indices]
                elif split == 'test':
                    dense = dense[test_indices]
                    sparse = sparse[test_indices]
                    label = label[test_indices]
            np.savez_compressed(
                split_out_path,
                sparse=sparse,
                dense=dense,
                label=label,
                count=self.counts,
            )

        self.dense = dense
        self.sparse = sparse
        self.label = label

        print("Split data according to indices...")

    def getAllData(self):
        in_path = self.in_path
        out_path = self.out_path
        data_ready = osp.exists(out_path)

        if not data_ready:
            print("Reading raw data={}".format(in_path))
            self.getCriteoAdData()
        print("Reading pre-processed data={}".format(out_path))
        data = np.load(out_path)
        dense, sparse, label, count = data['dense'], data['sparse'], data['label'], data['count']
        counts = []
        for i in range(len(count) - 1):
            counts.append(count[i+1] - count[i])
        self.counts = counts
        print("Count of each feature: {}".format(counts))
        return dense, sparse, label

    def getCriteoAdData(self):
        df = pd.read_csv(self.in_path, sep='\t', header=None)
        df.columns = ['label'] + ["I" + str(i) for i in range(1, 1 + self.den_fea)] + [
            "C"+str(i) for i in range(1 + self.den_fea, 1 + self.den_fea + self.spa_fea)]
        dense_feats = [col for col in df.columns if col.startswith('I')]
        sparse_feats = [col for col in df.columns if col.startswith('C')]
        labels = df['label']
        dense = df[dense_feats].fillna(0.0)
        for f in dense_feats:
            dense[f] = dense[f].apply(lambda x: np.log(x+1) if x > 0 else 0)
        sparse = df[sparse_feats].fillna("0")
        for f in sparse_feats:
            label_encoder = LabelEncoder()
            sparse[f] = label_encoder.fit_transform(sparse[f])
        feature_cnt = 0
        counts = [0]
        for f in sparse_feats:
            # sparse[f] += feature_cnt
            feature_cnt += sparse[f].nunique()
            counts.append(feature_cnt)

        np.savez_compressed(
            self.out_path,
            sparse=sparse,
            dense=dense,
            label=labels,
            count=counts,
        )

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [
                self[idx] for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]

        i = index

        return self.dense[i], self.sparse[i], self.label[i]

    def _default_preprocess(self, dense, sparse, label):
        dense = torch.tensor(dense, dtype=torch.float)
        sparse = torch.tensor(sparse, dtype=torch.long)
        label = torch.tensor(label.astype(np.float32))

        return dense, sparse, label

    def __len__(self):
        return len(self.label)


def collate_wrapper_criteo_offset(list_of_tuples):
    # where each tuple is (dense, sparse, label)
    transposed_data = list(zip(*list_of_tuples))
    dense = torch.tensor(transposed_data[0], dtype=torch.float)
    sparse = torch.tensor(transposed_data[1], dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = sparse.shape[0]
    featureCnt = sparse.shape[1]

    lS_i = [sparse[:, i] for i in range(featureCnt)]
    lS_o = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]

    return dense, torch.stack(lS_o), torch.stack(lS_i), T


# Conversion from offset to length
def offset_to_length_converter(lS_o, lS_i):
    def diff(tensor):
        return tensor[1:] - tensor[:-1]

    return torch.stack(
        [
            diff(torch.cat((S_o, torch.tensor(lS_i[ind].shape))).int())
            for ind, S_o in enumerate(lS_o)
        ]
    )


def collate_wrapper_criteo_length(list_of_tuples):
    # where each tuple is (dense, sparse, label)
    transposed_data = list(zip(*list_of_tuples))
    dense = torch.tensor(transposed_data[0], dtype=torch.float)
    sparse = torch.tensor(transposed_data[1], dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = sparse.shape[0]
    featureCnt = sparse.shape[1]

    lS_i = torch.stack([sparse[:, i] for i in range(featureCnt)])
    lS_o = torch.stack(
        [torch.tensor(range(batchSize)) for _ in range(featureCnt)]
    )

    lS_l = offset_to_length_converter(lS_o, lS_i)

    return dense, lS_l, lS_i, T
