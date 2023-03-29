import os
import os.path as osp
import numpy as np
import pandas as pd

default_data_path = osp.join(
    osp.split(osp.abspath(__file__))[0], '../datasets')
default_criteo_path = osp.join(default_data_path, 'criteo')
default_avazu_path = osp.join(default_data_path, 'avazu')

###########################################################################
# criteo
###########################################################################


def download_criteo(path):
    import tarfile
    from six.moves import urllib
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

    dense_feats, sparse_feats, labels = read_criteo_from_raw(
        osp.join(path, 'train.txt'))

    # save numpy arrays
    target_path = [osp.join(path, filename) for filename in [
        'train_dense_feats.npy', 'train_sparse_feats.npy', 'train_labels.npy',
        'test_dense_feats.npy', 'test_sparse_feats.npy', 'test_labels.npy']]
    num_data = dense_feats.shape[0]
    perm = np.random.permutation(num_data)
    # split data in 2 parts
    test_num = num_data // 10
    processed_data = [
        dense_feats[perm[:-test_num]],  # train dense
        sparse_feats[perm[:-test_num]],  # train sparse
        labels[perm[:-test_num]],       # train labels
        dense_feats[perm[-test_num:]],  # validate dense
        sparse_feats[perm[-test_num:]],  # validate sparse
        labels[perm[-test_num:]],       # validate labels
    ]
    print('Array shapes:')
    for i in range(len(processed_data)):
        print(osp.split(target_path[i])
              [-1].split('.')[0], processed_data[i].shape)
        np.save(target_path[i], processed_data[i])
    print('Numpy arrays saved.')


def process_dense_feats(data, feats, inplace=True):
    if inplace:
        d = data
    else:
        d = data.copy()
    d = d[feats].fillna(0.0)
    for f in feats:
        d[f] = d[f].apply(lambda x: np.log(x+1) if x > 0 else 0)
    return d


def process_sparse_feats(data, feats, inplace=True, return_counts=False):
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
    if return_counts:
        return d, counts
    else:
        return d


def read_criteo_from_raw(path=osp.join(default_criteo_path, 'train.txt'), return_counts=False, nrows=-1):
    den_fea = 13
    spa_fea = 26
    if nrows > 0:
        df = pd.read_csv(path, sep='\t', header=None, nrows=nrows)
    else:
        df = pd.read_csv(path, sep='\t', header=None)
    df.columns = ['label'] + ["I" + str(i) for i in range(1, 1 + den_fea)] + [
        "C"+str(i) for i in range(1 + den_fea, 1 + den_fea + spa_fea)]
    dense_feats = [col for col in df.columns if col.startswith('I')]
    sparse_feats = [col for col in df.columns if col.startswith('C')]
    labels = df['label']
    dense = process_dense_feats(df, dense_feats)
    if return_counts:
        sparse, counts = process_sparse_feats(
            df, sparse_feats, return_counts=True)
        return dense, sparse, labels, counts
    else:
        sparse = process_sparse_feats(df, sparse_feats)
        return dense, sparse, labels


def process_head_criteo_data(path=default_criteo_path, nrows=20000, return_val=True):
    raw_path = osp.join(path, "train.txt")
    if not osp.exists(raw_path):
        download_criteo(path)
    dense_feats, sparse_feats, labels = read_criteo_from_raw(
        raw_path, nrows=nrows)
    dense_feats = np.array(dense_feats)
    sparse_feats = np.array(sparse_feats).astype(np.int32)
    labels = np.array(labels).reshape(-1, 1)
    if return_val:
        test_num = nrows // 10
        train_dense = dense_feats[:-test_num]
        train_sparse = sparse_feats[:-test_num]
        train_label = labels[:-test_num]
        validate_dense = dense_feats[-test_num:]
        validate_sparse = sparse_feats[-test_num:]
        validate_label = labels[-test_num:]
        return (train_dense, validate_dense), (train_sparse, validate_sparse), (train_label, validate_label)
    else:
        return dense_feats, sparse_feats, labels


def process_sampled_criteo_data(path=default_criteo_path):
    # all data should be available! no checking.
    processed_data = [np.load(osp.join(path, filename))
                      for filename in ['sampled_dense_feats.npy', 'sampled_sparse_feats.npy', 'sampled_labels.npy']]
    return tuple(processed_data)


def process_all_criteo_data(path=default_criteo_path, return_val=True, separate_fields=False):
    file_paths = [osp.join(path, filename) for filename in [
        'train_dense_feats.npy', 'test_dense_feats.npy', 'train_sparse_feats.npy',
        'test_sparse_feats.npy',  'train_labels.npy', 'test_labels.npy']]
    if not all([osp.exists(p) for p in file_paths]):
        download_criteo(path)
    files = [np.load(filename) for filename in file_paths]
    if separate_fields:
        fields_files = ['train_sparse_feats_fields.npy',
                        'test_sparse_feats_fields.npy']
        if not all([osp.exists(osp.join(path, p)) for p in fields_files]):
            ori_train_sparse = files[2]
            ori_test_sparse = files[3]
            ori_sparse = np.concatenate(
                (ori_train_sparse, ori_test_sparse), axis=0)
            cur_offset = 0
            for i in range(26):
                ori_sparse[:, i] -= cur_offset
                uni = np.unique(ori_sparse[:, i])
                cur_num = len(uni)
                cur_offset += cur_num
            test_num = ori_sparse.shape[0] // 10
            new_train_sparse = ori_sparse[:-test_num]
            new_test_sparse = ori_sparse[-test_num:]
            new_train_sparse = np.transpose(new_train_sparse, (1, 0))
            new_test_sparse = np.transpose(new_test_sparse, (1, 0))
            np.save(osp.join(path, 'train_sparse_feats_fields.npy'),
                    new_train_sparse)
            np.save(osp.join(path, 'test_sparse_feats_fields.npy'),
                    new_test_sparse)
        else:
            new_train_sparse = np.load(osp.join(
                path, 'train_sparse_feats_fields.npy'))
            new_test_sparse = np.load(osp.join(
                path, 'test_sparse_feats_fields.npy'))
        files[2] = new_train_sparse
        files[3] = new_test_sparse
    if return_val:
        return (files[0], files[1]), (files[2], files[3]), (files[4], files[5])
    else:
        return files[0], files[2], files[4]


def get_separate_fields(path, sparse, num_embed_fields=None):
    if osp.exists(path):
        return np.memmap(path, mode='r', dtype=np.int32).reshape(-1, 26)
    else:
        if num_embed_fields is None:
            num_embed_fields = [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683,
                                8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572]
        accum = 0
        sparse = np.array(sparse)
        for i in range(26):
            sparse[:, i] -= accum
            accum += num_embed_fields[i]
        sparse.tofile(path)
        return sparse


def process_all_criteo_data_by_day(path=default_criteo_path, return_val=True, separate_fields=False):
    days = 7
    in_path = osp.join(path, 'train.txt')
    phases = ['train', 'val', 'test']
    keys = ['dense', 'sparse', 'label']
    dtypes = [np.float32, np.int32, np.int32]
    shapes = [(-1, 13), (-1, 26), (-1,)]
    all_data_path = [
        [osp.join(path, f'kaggle_processed_{ph}_{k}.bin') for k in keys] for ph in phases]

    data_ready = all([osp.exists(p) for value in all_data_path for p in value])

    if not data_ready:
        ckeys = keys + ['count']
        cdtypes = dtypes + [np.int32]
        cshapes = shapes + [(-1,)]
        pro_paths = [
            osp.join(path, f'kaggle_processed_{k}.bin') for k in ckeys]
        pro_data_ready = all([osp.exists(path) for path in pro_paths])

        if not pro_data_ready:
            print("Reading raw data={}".format(in_path))
            all_data = read_criteo_from_raw(in_path, return_counts=True)
            for data, path, dtype in zip(all_data, pro_paths, cdtypes):
                data = np.array(data, dtype=dtype)
                data.tofile(path)
        dense, sparse, label, count = [np.fromfile(path, dtype=dtype).reshape(
            shape) for path, dtype, shape in zip(pro_paths, cdtypes, cshapes)]
        counts = []
        for i in range(len(count) - 1):
            counts.append(count[i+1] - count[i])
        print("Count of each feature: {}".format(counts))
        num_samples = dense.shape[0]
        total_per_file = []
        num_data_per_split, extras = divmod(num_samples, days)
        total_per_file = [num_data_per_split] * days
        for j in range(extras):
            total_per_file[j] += 1
        offset_per_file = np.array([0] + [x for x in total_per_file])
        for i in range(days):
            offset_per_file[i + 1] += offset_per_file[i]
        print("File offsets: {}".format(offset_per_file))

        # create reordering
        indices = np.arange(dense.shape[0])
        indices = np.array_split(indices, offset_per_file[1:-1])
        train_indices = np.concatenate(indices[:-1])
        test_indices = indices[-1]
        test_indices, val_indices = np.array_split(test_indices, 2)
        # randomize
        train_indices = np.random.permutation(train_indices)
        print("Randomized indices across days ...")
        indices = [train_indices, val_indices, test_indices]

        # create training, validation, and test sets
        for ind, cur_paths in zip(indices, all_data_path):
            cur_data = [dense[ind], sparse[ind], label[ind]]
            for d, p in zip(cur_data, cur_paths):
                d.tofile(p)

    def get_data(phase):
        index = {'train': 0, 'val': 1, 'test': 2}[phase]
        # train and val
        dense, sparse, label = [np.memmap(p, mode='r', dtype=dtype).reshape(
            shape) for p, dtype, shape in zip(all_data_path[index], dtypes, shapes)]
        if separate_fields:
            sparse = get_separate_fields(
                osp.join(path, f'kaggle_processed_{phase}_sparse_sep.bin'), sparse)
        return dense, sparse, label

    train_dense, train_sparse, train_label = get_data('train')
    # val_dense, val_sparse, val_label = get_data('val')
    test_dense, test_sparse, test_label = get_data('test')
    if return_val:
        return (train_dense, test_dense), (train_sparse, test_sparse), (train_label, test_label)
    else:
        return train_dense, train_sparse, train_label


###########################################################################
# adult
###########################################################################


def maybe_download(train_data, test_data):
    """if adult data "train.csv" and "test.csv" are not in your directory,
    download them.
    """

    COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "gender",
               "capital_gain", "capital_loss", "hours_per_week", "native_country",
               "income_bracket"]

    if not osp.exists(train_data):
        print("downloading training data...")
        df_train = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                               names=COLUMNS, skipinitialspace=True)
    else:
        df_train = pd.read_csv("train.csv")

    if not osp.exists(test_data):
        print("downloading testing data...")
        df_test = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                              names=COLUMNS, skipinitialspace=True, skiprows=1)
    else:
        df_test = pd.read_csv("test.csv")

    return df_train, df_test


def cross_columns(x_cols):
    """simple helper to build the crossed columns in a pandas dataframe
    """
    crossed_columns = dict()
    colnames = ['_'.join(x_c) for x_c in x_cols]
    for cname, x_c in zip(colnames, x_cols):
        crossed_columns[cname] = x_c
    return crossed_columns


def val2idx(df, cols):
    """helper to index categorical columns before embeddings.
    """
    val_types = dict()
    for c in cols:
        val_types[c] = df[c].unique()

    val_to_idx = dict()
    for k, v in val_types.items():
        val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}

    for k, v in val_to_idx.items():
        df[k] = df[k].apply(lambda x: v[x])

    unique_vals = dict()
    for c in cols:
        unique_vals[c] = df[c].nunique()

    return df, unique_vals


def onehot(x):
    from sklearn.preprocessing import OneHotEncoder
    return np.array(OneHotEncoder().fit_transform(x).todense())


def wide(df_train, df_test, wide_cols, x_cols, target):
    print('Processing wide data')
    df_train['IS_TRAIN'] = 1
    df_test['IS_TRAIN'] = 0
    df_wide = pd.concat([df_train, df_test])

    crossed_columns_d = cross_columns(x_cols)
    categorical_columns = list(
        df_wide.select_dtypes(include=['object']).columns)

    wide_cols += list(crossed_columns_d.keys())

    for k, v in crossed_columns_d.items():
        df_wide[k] = df_wide[v].apply(lambda x: '-'.join(x), axis=1)

    df_wide = df_wide[wide_cols + [target] + ['IS_TRAIN']]

    dummy_cols = [
        c for c in wide_cols if c in categorical_columns + list(crossed_columns_d.keys())]
    df_wide = pd.get_dummies(df_wide, columns=[x for x in dummy_cols])

    train = df_wide[df_wide.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
    test = df_wide[df_wide.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)
    assert all(train.columns == test.columns)

    cols = [c for c in train.columns if c != target]
    X_train = train[cols].values
    y_train = train[target].values.reshape(-1, 1)
    X_test = test[cols].values
    y_test = test[target].values.reshape(-1, 1)
    return X_train, y_train, X_test, y_test


def load_adult_data(return_val=True):
    df_train, df_test = maybe_download("train.csv", "test.csv")

    df_train['income_label'] = (
        df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    df_test['income_label'] = (
        df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    age_groups = [0, 25, 65, 90]
    age_labels = range(len(age_groups) - 1)
    df_train['age_group'] = pd.cut(
        df_train['age'], age_groups, labels=age_labels)
    df_test['age_group'] = pd.cut(
        df_test['age'], age_groups, labels=age_labels)

    # columns for wide model
    wide_cols = ['workclass', 'education', 'marital_status', 'occupation',
                 'relationship', 'race', 'gender', 'native_country', 'age_group']
    x_cols = (['education', 'occupation'], ['native_country', 'occupation'])

    # columns for deep model
    embedding_cols = ['workclass', 'education', 'marital_status', 'occupation',
                      'relationship', 'race', 'gender', 'native_country']
    cont_cols = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']

    target = 'income_label'

    x_train_wide, y_train_wide, x_test_wide, y_test_wide = wide(
        df_train, df_test, wide_cols, x_cols, target)
    x_train_wide = np.array(x_train_wide).astype(np.float32)
    x_test_wide = np.array(x_test_wide).astype(np.float32)

    print('Processing deep data')
    df_train['IS_TRAIN'] = 1
    df_test['IS_TRAIN'] = 0
    df_deep = pd.concat([df_train, df_test])

    deep_cols = embedding_cols + cont_cols
    df_deep = df_deep[deep_cols + [target, 'IS_TRAIN']]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_deep[cont_cols] = pd.DataFrame(scaler.fit_transform(df_train[cont_cols]),
                                      columns=cont_cols)
    df_deep, unique_vals = val2idx(df_deep, embedding_cols)

    train = df_deep[df_deep.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
    test = df_deep[df_deep.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)

    x_train_deep = np.array([train[c] for c in deep_cols]).astype(np.float32)
    y_train = np.array(train[target].values).reshape(-1, 1).astype(np.int32)
    x_test_deep = np.array([test[c] for c in deep_cols]).astype(np.float32)
    y_test = np.array(test[target].values).reshape(-1, 1).astype(np.int32)

    x_train_deep = np.transpose(x_train_deep)
    x_test_deep = np.transpose(x_test_deep)
    y_train = onehot(y_train)
    y_test = onehot(y_test)

    if return_val:
        return x_train_deep, x_train_wide, y_train, x_test_deep, x_test_wide, y_test
    else:
        return x_train_deep, x_train_wide, y_train


###########################################################################
# avazu
###########################################################################

def process_avazu(path=default_avazu_path):
    # please download in advance from https://www.kaggle.com/c/avazu-ctr-prediction/data
    train_file = osp.join(path, 'train.csv')
    # test_file = osp.join(path, 'test.csv') # useless, no labels

    df_train = pd.read_csv(train_file)
    sparse_feats = process_sparse_feats(df_train, df_train.columns[2:])
    # the embedding num for each feature:
    # [240, 7, 7, 4737, 7745, 26, 8552, 559, 36, 2686408, 6729486, 8251, 5, 4, 2626, 8, 9, 435, 4, 68, 172, 60]
    # sum: 9449445

    np.save(osp.join(path, 'sparse.npy'), sparse_feats)


if __name__ == '__main__':
    download_criteo(osp.join(osp.split(
        osp.abspath(__file__)), '../datasets/criteo'))
