from graphmix import Graph

import numpy as np
import scipy.sparse as sp
import os
import sys
import math
import argparse
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse.csgraph import reverse_cuthill_mckee
import pickle as pkl
import time

'''
Usage example: (in Dir Hetu/)
    python ./tests/test_DistGCN/prepare_data_GCN15d_reorder.py --size 8 --replication 2 --dataset Reddit  --reorder_alg metis
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


def get_inputs(H, replication, rank, size, block_no=None):
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


def matrix_visualize(m, title, img_name, args):
    print("Visualization matrix after partitioning...")
    dir_name = "./tests/test_DistGCN/matrix_visualization/%s/" % args.dataset
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    plt.clf()
    if not isinstance(m, sp.coo_matrix):
        m = sp.coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='white')
    ax.plot(m.col, m.row, ',', color='black')
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 400
    plt.title(title)
    plt.savefig(dir_name+img_name)
    print("Visualization done!")


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

# ------------------------
# ---Graph reorder utils---


def metis_reorder(adj, nparts=1000):
    adj = adj.tocoo()
    node_count = adj.shape[0]
    # construct the graph, x/y/num_classes doesn't matter
    graph = Graph(edge_index=np.vstack(
        [adj.row, adj.col]), num_nodes=node_count)
    # ---partition the graph using metis and calculate reorder index---
    print("Metis reorder nparts = %d" % nparts)
    nodes = graph.partition(nparts)

    reindex = np.zeros(shape=(node_count,), dtype=int)
    class_dic = {i: [] for i in range(nparts)}
    for i in range(node_count):
        class_dic[nodes[i]].append(i)
    cnt = 0
    np.random.seed(123)
    part_order = np.array(range(nparts))

    for i in range(nparts):
        for j in class_dic[part_order[i]]:
            reindex[j] = cnt
            cnt += 1
    return reindex


def rcm_reorder(adj):
    # ---graph reordering using RCM---
    node_count = adj.shape[0]
    reindex_reverse = np.array(
        list(reverse_cuthill_mckee(adj.tocsr(), symmetric_mode=True)))
    reindex = np.zeros((node_count,), int)
    for i in range(node_count):
        reindex[reindex_reverse[i]] = i
    return reindex


def slashburn_reorder(adj):
    node_count = adj.shape[0]
    reindex = np.zeros((node_count,), int)

    G = nx.Graph()
    edges = []
    for i in range(adj.row.shape[0]):
        if(adj.row[i] < adj.col[i]):
            edges.append((adj.row[i], adj.col[i]))
    G.add_nodes_from([i for i in range(node_count)])
    G.add_edges_from(edges)
    front = 0
    end = node_count-1

    def slash_burn(G, front, end):
        deg = list(G.degree)
        d = sorted(deg, key=lambda deg: deg[1], reverse=True)
        for i in range(int(0.005*node_count)):
            if(i < len(d)):
                reindex[front] = d[i][0]
                front += 1
                G.remove_node(d[i][0])

        print(len(list(G.nodes)))
        if(len(list(G.nodes)) == 0):
            return
        components = list(
            sorted(nx.connected_components(G), key=len, reverse=False))
        nCom = len(components)
        if(len(components[nCom-1]) > 1):
            for i in range(nCom-1):
                cur_com = components[i]
                for node in cur_com:
                    reindex[end] = node
                    end -= 1
                    G.remove_node(node)

            if(len(list(G.nodes)) == 0):
                return
            slash_burn(G, front, end)
        else:
            nodes = list(G.nodes)
            for n in nodes:
                reindex[front] = n
                G.remove_node(n)
                front += 1
            return
        return

    slash_burn(G, front, end)
    reverse_reindex = np.zeros((node_count,), int)
    for i in range(node_count):
        reverse_reindex[reindex[i]] = i
    return reverse_reindex


def deg_reorder(adj):
    node_count = adj.shape[0]
    degree = np.zeros((node_count))
    for i in range(adj.nnz):
        degree[adj.row[i]] += 1
    reindex = np.argsort(-degree)
    reverse_reindex = np.zeros((node_count,), int)
    for i in range(node_count):
        reverse_reindex[reindex[i]] = i
    return reverse_reindex

# return reverse reorder index


def graph_reorder(adj, reorder_alg='metis'):
    print("Calculating the reordering index...")
    print('Reorder_alg = %s' % (reorder_alg))
    node_count = adj.shape[0]

    if args.size == 1:
        adj = adj.tocoo()
        if reorder_alg == 'metis':
            nparts = node_count//args.part_size
            reindex = metis_reorder(adj, nparts=nparts)
        elif reorder_alg == 'rcm':
            reindex = rcm_reorder(adj)
        elif reorder_alg == 'slashburn':
            reindex = slashburn_reorder(adj)
        elif reorder_alg == 'deg':
            reindex = deg_reorder(adj)
        else:
            print(
                "Supported reordering algorithms are [metis, rcm, slashburn, deg].")
            exit(-1)
    elif args.size//args.replication in [2, 4, 8]:
        s = args.size//args.replication
        reorder_count = math.ceil(float(node_count)/s)
        starts = list(range(0, node_count, reorder_count))
        ends = list(range(reorder_count, node_count,
                          reorder_count))+[node_count]
        reindexs = []
        for i in range(s):
            index0, index1 = starts[i], ends[i]
            a = coo_slice(adj, row_range=(index0, index1),
                          col_range=(index0, index1))
            if reorder_alg == 'metis':
                nparts = reorder_count//args.part_size
                reindex_part = metis_reorder(a, nparts=nparts)
            elif reorder_alg == 'rcm':
                reindex_part = rcm_reorder(a)
            elif reorder_alg == 'slashburn':
                reindex_part = slashburn_reorder(a)
            elif reorder_alg == 'deg':
                reindex_part = deg_reorder(a)
            else:
                print(
                    "Supported reordering algorithms are [metis, rcm, slashburn, deg].")
                exit(-1)
            reindex_part = np.array(reindex_part)+index0
            reindexs.append(reindex_part)
        reindex = np.concatenate(reindexs)
    reverse_reindex = np.zeros((node_count,), int)
    for i in range(node_count):
        reverse_reindex[reindex[i]] = i
    print("Got reordered index!")
    return reverse_reindex

# ------------------------


def check_sparsity(adj):
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
    part_size, vis, dataset = args.part_size, args.visualize, args.dataset
    data_dir = './tests/test_DistGCN/datasets/%s/' % dataset

    # Original graph data should be in ./tests/test_DistGCN/datasets/
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
    nparts = node_count//part_size

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

    # ---graph reordering---
    reindex_reverse = graph_reorder(adj, args.reorder_alg)

    # ---reorder the graph
    print("Reordering the graph...")
    adj = adj.tocsr()[:, reindex_reverse][reindex_reverse]
    x = x[reindex_reverse, :]
    y = y[reindex_reverse]
    print("Reordering done!")

    # ---check block sparsity---
    print('Sparsity after reordering:')
    check_sparsity(adj)

    # ---visualize adj---
    if vis:
        if args.reorder_alg == 'metis':
            img_name = "partitioned_%d_metis.png" % (nparts)
            title = "Matrix Reordered by METIS %d parts" % nparts
        elif args.reorder_alg == 'rcm':
            img_name = "partitioned_rcm.png"
            title = "Matrix Reordered by RCM"
        elif args.reorder_alg == 'slashburn':
            img_name = "partitioned_%d_slashburn.png" % (nparts)
            title = "Matrix Reordered by slashburn %d parts" % nparts
        elif args.reorder_alg == 'deg':
            img_name = "partitioned_deg.png"
            title = "Matrix Reordered by deg"
        matrix_visualize(adj, title, img_name, args)

    print('node_count = %d, num_features = %d, num_classes = %d, edge_count = %d' % (
        adj.shape[0], x.shape[1], np.max(y)+1, len(adj.data)))
    return adj, x, y


def prepare_data(args):
    replication, size, dataset, reorder_alg = args.replication, args.size, args.dataset, args.reorder_alg
    print("Preparing data...")

    adj_all, input_all, label_all = load_data(args)
    print("size=%d, replication=%s, reorder_alg=%s, dataset=%s" %
          (size, replication, reorder_alg, dataset))

    if size == 1:  # whole graph for single GPU
        replication = 1
        dir_name = "./tests/test_DistGCN/data_GCN15d_reorder/%s/%s_whole_graph/" % (
            reorder_alg, dataset)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        sp.save_npz(dir_name+"adj_whole.npz", adj_all)
        print("adj_whole: ", adj_all.shape, len(adj_all.data))
        np.save(dir_name+"input_whole.npy", input_all)
        print("inputs_all: ", input_all.shape)
        np.save(dir_name+"label_whole.npy", label_all)
        print("labels_all: ", label_all.shape)
        print("Data preparation done!")
    else:  # partitioned graph for multiple GPU
        dir_name = "./tests/test_DistGCN/data_GCN15d_reorder/%s/%s_size_%d_rep_%d/" % (
            reorder_alg, dataset, size, replication)
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
                    help='Replication of distGCN1.5D [1,2 ].')
parser.add_argument('--size', type=int, default=2,
                    help='Number of devices [2, 4, 8, 16]')
parser.add_argument('--visualize', type=int, default=0,
                    help='Visualize matrix after partitioning or not [0, 1].')
parser.add_argument('--part_size', type=int, default=200,
                    help='Metis cluster size.')
parser.add_argument('--reorder_alg', type=str, default="metis",
                    help='Graph reordering algorithm [rcm, metis, slashburn, deg, go].')
parser.add_argument('--dataset', type=str, default="Reddit",
                    help='Choose dataset [Reddit, Proteins, Arch, Products].')
parser.add_argument('--shuffle', type=int, default=1,
                    help='Whether to shuffle the graph before algorithm.')
args = parser.parse_args()

get_dataset(args)
if args.size == -1:
    size_set = [1, 2, 4, 8, 4, 8]
    replication_set = [1, 1, 1, 1, 2, 2]
    for i in range(len(size_set)):
        args.replication, args.size = replication_set[i], size_set[i]
        prepare_data(args)
elif args.dataset == 'All':
    dataset = ['Reddit', 'Proteins', 'Arch', 'Products']
    for i in range(len(dataset)):
        args.dataset = dataset[i]
        prepare_data(args)
else:
    prepare_data(args)
