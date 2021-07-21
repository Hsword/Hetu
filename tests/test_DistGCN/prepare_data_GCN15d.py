from graphmix import Graph

import numpy as np
import scipy.sparse as sp
import os
import sys
import math
import argparse
import pickle as pkl
import networkx as nx

'''
Usage example: (in Dir Hetu/)
    python ./tests/test_DistGCN/prepare_data_GCN15d.py --size 8 --replication 2 --dataset Reddit
'''


def coo_slice(a, row_range, col_range):
    a = a.tocoo()
    condition = np.where((a.row >= row_range[0]) & (a.row < row_range[1]) & (
        a.col >= col_range[0]) & (a.col < col_range[1]))
    return sp.coo_matrix((a.data[condition], (a.row[condition]-row_range[0], a.col[condition]-col_range[0])), shape=(row_range[1]-row_range[0], col_range[1]-col_range[0]))


def get_adj_matrix_all(A, replication, size, dir_name):
    node_count = A.shape[0]

    n_per_proc = math.ceil(float(node_count) / (size // replication))
    stages = size // (replication ** 2)
    col_block = stages*n_per_proc
    row_block = math.ceil(float(node_count)/(size//replication))

    for rank in range(size):
        rank_row = rank // replication  # i
        rank_col = rank % replication  # j

        col_start = int(col_block*rank_col)
        col_end = int(col_block*(rank_col+1))
        if col_end > node_count:
            col_end = node_count

        row_start = int(row_block*rank_row)
        row_end = int(row_block*(rank_row+1))
        if row_end > node_count:
            row_end = node_count

        a = coo_slice(A.tocoo(), row_range=(row_start, row_end),
                      col_range=(col_start, col_end))
        sp.save_npz(dir_name+"adj_part"+str(rank)+".npz", a)
        print("adj_part: rank = %d" % rank, a.shape, len(a.data))


def get_inputs(H, replication, rank, size):
    node_count = H.shape[0]
    rank_row = rank // replication  # i
    row_block = math.ceil(float(node_count)/(size//replication))
    row_start = int(row_block*rank_row)
    row_end = int(row_block*(rank_row+1))
    if row_end > node_count:
        row_end = node_count
    h = H[row_start:row_end, :]
    print("inputs_part: rank = %d" % rank, h.shape)
    return h


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def check_sparsity(adj):
    if args.size == -1:
        return
    adj = adj.tocoo()
    node_count = adj.shape[0]
    block_num = args.size//args.replication
    p = math.ceil(float(node_count)/(args.size//args.replication))
    starts = list(range(0, node_count, p))
    ends = list(range(p, node_count, p))+[node_count]
    sparsity = np.zeros(shape=(block_num, block_num), dtype=int)
    for i in range(block_num):
        for j in range(block_num):
            sparsity[i, j] = np.where((adj.row >= starts[i]) & (adj.row < ends[i]) & (
                adj.col >= starts[j]) & (adj.col < ends[j]))[0].shape[0]
    print(sparsity)


def load_data(args):
    dataset = args.dataset
    data_dir = './tests/test_DistGCN/datasets/%s/' % dataset

    # ---load data---
    if dataset == "Reddit":
        adj = sp.load_npz(data_dir+'raw/reddit_graph.npz')
        inputs = np.load(data_dir+'raw/reddit_data.npz')
        x, y = inputs['feature'], inputs['label']
    elif dataset == 'Proteins':
        adj = sp.load_npz(data_dir+'protein_adj.npz')
        y = np.load(data_dir+'protein_labels.npy')
        y = y.astype(int)
        np.random.seed(123)
        bounds = np.sqrt(6.0 / (132534 + 602))
        x = np.random.uniform(low=-bounds, high=bounds,
                              size=[132534, 602]).astype(np.float32)
    elif dataset == 'Arch':
        adj = sp.load_npz(data_dir+'arch_adj.npz')
        y = np.random.randint(10, size=adj.shape[0])
        np.random.seed(123)
        bounds = np.sqrt(6.0 / (adj.shape[0] + 602))
        x = np.random.uniform(low=-bounds, high=bounds,
                              size=[adj.shape[0], 602]).astype(np.float32)
    elif dataset == 'Products':
        adj = sp.load_npz(data_dir+'products_adj.npz')
        x = np.load(data_dir+'products_feat.npy')
        y = np.load(data_dir+'products_label.npy').astype(np.int)
    elif dataset == 'Youtube':
        adj = np.load(data_dir+'youtube_coo.npy', allow_pickle=True).item()
        np.random.seed(123)
        bounds = np.sqrt(6.0 / (adj.shape[0] + 602))
        x = np.random.uniform(low=-bounds, high=bounds,
                              size=[adj.shape[0], 602]).astype(np.float32)
        y = np.load(data_dir+'youtube_label.npy')

    graph = Graph(edge_index=np.vstack(
        [adj.row, adj.col]), num_nodes=x.shape[0])

    # ---preprocess graph---
    graph.add_self_loop()
    normed_val = graph.gcn_norm(True)
    node_count = graph.num_nodes

    # ---construct adj,x,y---
    edge_index = graph.edge_index
    adj = sp.coo_matrix(
        (normed_val, (edge_index[0], edge_index[1])), shape=(node_count, node_count))

    # ---check block sparsity---
    print('Sparsity before reordering:')
    check_sparsity(adj)

    if args.shuffle == 1:
        print("Shuffle the graph...")
        order = np.random.permutation(node_count)
        adj = adj.tocsr()[:, order][order]
        x = x[order, :]
        y = y[order]
        print('Sparsity after Shuffle:')
        check_sparsity(adj)

    print('node_count = %d, num_features = %d, num_classes = %d, edge_count = %d' % (
        adj.shape[0], x.shape[1], np.max(y)+1, len(adj.data)))
    return adj, x, y


def prepare_data(args, prepare_all_data=False):
    dataset, replication, size = args.dataset, args.replication, args.size
    print("Preparing data...")

    adj_all, input_all, label_all = load_data(args)

    if prepare_all_data:
        size_set = [1, 2, 4, 8, 4, 8]
        replication_set = [1, 1, 1, 1, 2, 2]
    else:
        size_set = [size]
        replication_set = [replication]

    for i in range(len(size_set)):
        replication, size = replication_set[i], size_set[i]
        print("size=%d, replication=%s, dataset=%s" %
              (size, replication, dataset))

        if size == 1:  # whole graph for single GPU
            replication = 1
            dir_name = "./tests/test_DistGCN/data_GCN15d/%s_whole_graph/" % dataset
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            adj_all = adj_all.tocoo()
            sp.save_npz(dir_name+"adj_whole.npz", adj_all)
            print("adj_whole: ", adj_all.shape, len(adj_all.data))
            np.save(dir_name+"input_whole.npy", input_all)
            print("inputs_all: ", input_all.shape)
            np.save(dir_name+"label_whole.npy", label_all)
            print("labels_all: ", label_all.shape)
            print("Data preparation done!")
        else:  # partitioned graph for multiple GPU
            dir_name = "./tests/test_DistGCN/data_GCN15d/%s_size_%d_rep_%d/" % (
                dataset, size, replication)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            for rank in range(size):
                input_part = get_inputs(input_all, replication, rank, size)
                label_part = get_inputs(
                    label_all.reshape(-1, 1), replication, rank, size).reshape(-1)
                np.save(dir_name+"input"+str(rank)+".npy", input_part)
                np.save(dir_name+"label"+str(rank)+".npy", label_part)
            print("Done inputs and labels!")

            get_adj_matrix_all(adj_all, replication, size, dir_name)
            print("Data preparation done!")


def get_dataset(args):
    if args.dataset in ['Reddit', 'reddit']:
        args.dataset = 'Reddit'
    elif args.dataset in ['Proteins', 'proteins']:
        args.dataset = 'Proteins'
    elif args.dataset in ['Arch', 'arch']:
        args.dataset = 'Arch'
    elif args.dataset in ['Products', 'products']:
        args.dataset = 'Products'
    elif args.dataset in ['All', 'all']:
        args.dataset = 'All'
    else:
        print(
            "Dataset should be in ['Reddit','Proteins','Arch','Products','All']")
        assert False


parser = argparse.ArgumentParser()
parser.add_argument('--replication', type=int, default=1,
                    help='Replication of distGCN1.5D.')
parser.add_argument('--size', type=int, default=8,
                    help='Number of devices')
parser.add_argument('--dataset', type=str, default="Reddit",
                    help='Choose dataset [Reddit, Proteins, Arch, Products].')
parser.add_argument('--shuffle', type=int, default=1,
                    help='Whether to shuffle the graph before algorithm.')
args = parser.parse_args()

get_dataset(args)

if args.size == -1:
    prepare_data(args, True)
elif args.dataset == 'All':
    dataset = ['Reddit', 'Proteins', 'Arch', 'Products']
    for i in range(len(dataset)):
        args.dataset = dataset[i]
        prepare_data(args)
else:
    prepare_data(args)
