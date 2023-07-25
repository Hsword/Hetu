import os
import wget
import zipfile
from collections import defaultdict as dd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


DATASETS = ["ml-1m", "ml-20m", "ml-25m"]
urls = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}


def download(dataset, data_dir, num_negatives=4):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    assert dataset in ["ml-1m", "ml-20m",
                       "ml-25m"], 'Invalid dataset: %s.' % dataset
    data_subdir = os.path.join(data_dir, dataset)
    print('Data in', data_subdir)
    zip_file = os.path.join(data_dir, dataset + '.zip')
    ratings = os.path.join(data_subdir, 'ratings.csv')
    if not os.path.exists(ratings):
        if not os.path.exists(zip_file):
            print('Downloading movielens %s...' % dataset)
            wget.download(urls[dataset], zip_file)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            print('Extracting movielens %s...' % dataset)
            zip_ref.extractall(data_dir)
    ratings = os.path.join(data_subdir, 'ratings.csv')

    num_users, num_items = {
        'ml-1m': (6040, 3706),
        'ml-20m': (138493, 26744),
        'ml-25m': (162541, 59047),
    }[dataset]

    # Generate raw training and testing files
    item_reverse_mapping = {}
    cur_item_idx = 0
    latest = [(0, -1)] * num_users
    mat = sp.dok_matrix((num_users, num_items), dtype=np.float32)
    with open(ratings, 'r') as fr:
        fr.readline()
        for line in tqdm(fr):
            entries = line.strip().split(',')
            user = int(entries[0])
            item = int(entries[1])
            if item not in item_reverse_mapping:
                item_reverse_mapping[item] = cur_item_idx
                cur_item_idx += 1
            rating = float(entries[2])
            if rating <= 0:
                continue
            reitem = item_reverse_mapping[item]
            mat[user-1, reitem] = 1
            timestamp = int(entries[-1])
            if latest[user-1][0] < timestamp:
                latest[user-1] = (timestamp, reitem)
    print('#users:', num_users, '#items:', num_items)

    new_lates = np.concatenate((np.array(latest, dtype=np.int32)[
                               :, 1:], np.empty((num_users, 99), dtype=np.int32)), 1)

    # sample for test data first, each user 99 items, using all data
    for i, lat in enumerate(latest):
        new_lates[i][0] = lat[1]
        for k in range(1, 100):
            j = np.random.randint(num_items)
            while (i, j) in mat.keys():
                j = np.random.randint(num_items)
            new_lates[i][k] = j
    np.save(os.path.join(data_subdir, 'test.npy'), new_lates)

    # sample for train data, each data with num_negative negative samples
    all_num = (1 + num_negatives) * (len(mat.keys()) - num_users)
    user_input = np.empty((all_num,), dtype=np.int32)
    item_input = np.empty((all_num,), dtype=np.int32)
    labels = np.empty((all_num,), dtype=np.int32)
    idx = 0
    for (i, j) in mat.keys():
        if new_lates[i][0] == j:
            continue
        # positive instance
        user_input[idx] = i
        item_input[idx] = j
        labels[idx] = 1
        idx += 1
        # negative instances
        for t in range(num_negatives):
            k = np.random.randint(num_items)
            while (i, k) in mat.keys():
                k = np.random.randint(num_items)
            user_input[idx] = i
            item_input[idx] = k
            labels[idx] = 0
            idx += 1
    assert all_num == idx
    np.savez(os.path.join(data_subdir, 'train.npz'),
             user_input=user_input, item_input=item_input, labels=labels)


def getdata(dataset, data_dir='datasets'):
    assert dataset in ["ml-1m", "ml-20m",
                       "ml-25m"], 'Invalid dataset: %s.' % dataset
    data_subdir = os.path.join(data_dir, dataset)
    file_paths = [os.path.join(data_subdir, data)
                  for data in ['train.npz', 'test.npy']]
    if any([not os.path.exists(path) for path in file_paths]):
        download(dataset, data_dir)
    return np.load(file_paths[0]), np.load(file_paths[1])


if __name__ == "__main__":
    download('ml-25m', 'datasets')
