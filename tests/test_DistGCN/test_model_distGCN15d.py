import numpy as np
import scipy.sparse as sp
import hetu as ht
from hetu.communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t
import math
import time
import argparse

'''
Usage example: (in Dir Hetu/)
    Original graph data:
        Single GPU:
            mpirun -quiet --allow-run-as-root -np 1 python tests/test_DistGCN/test_model_distGCN15d.py --replication 1 --dataset Reddit
        Multiple GPU:
            mpirun -quiet --allow-run-as-root -np 8 python tests/test_DistGCN/test_model_distGCN15d.py --replication 2 --dataset Reddit
    Reordered graph data:
        Single GPU:
            mpirun -quiet --allow-run-as-root -np 1 python tests/test_DistGCN/test_model_distGCN15d.py --replication 1 --dataset Reddit --reorder 1 --reorder_alg metis
        Multiple GPU:
            mpirun -quiet --allow-run-as-root -np 8 python tests/test_DistGCN/test_model_distGCN15d.py --replication 2 --dataset Reddit --reorder 1 --reorder_alg metis
'''


def row_num(node_count, rank, size):
    n_per_proc = math.ceil(float(node_count) / size)
    if(node_count % size == 0):
        return int(node_count/size)
    if(rank < size-1):
        return int(n_per_proc)
    else:
        return int(node_count % n_per_proc)


def col_num(node_count, replication, rank):
    rank_col = rank % replication  # j
    col_block = math.ceil(float(node_count) / replication)
    col_start = int(col_block*rank_col)
    col_end = int(col_block*(rank_col+1))
    if col_end > node_count:
        col_end = node_count
    return col_end-col_start


def convert_to_one_hot(vals, max_val=0):
    """Helper method to convert label array to one-hot array."""
    if max_val == 0:
        max_val = vals.max() + 1
    one_hot_vals = np.zeros((vals.size, max_val))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals


def get_proc_groups(size, replication):
    if replication == 1:
        return None, None, None, None

    row_procs = []
    for i in range(0, size, replication):
        row_procs.append(list(range(i, i + replication)))

    col_procs = []
    for i in range(replication):
        col_procs.append(list(range(i, size, replication)))

    row_groups = []
    for i in range(len(row_procs)):
        row_groups.append(ht.new_group_comm(row_procs[i]))

    col_groups = []
    for i in range(len(col_procs)):
        col_groups.append(ht.new_group_comm(col_procs[i]))

    return row_procs, col_procs, row_groups, col_groups


def load_data(args, size, replication, rank):
    print("Loading data for rank %d..." % rank)
    dataset = args.dataset
    reorder_alg = args.reorder_alg
    dir_name = "./tests/test_DistGCN/data_GCN15d/%s_size_%d_rep_%d/" % (
        dataset, size, replication)
    if args.reorder:
        dir_name = "./tests/test_DistGCN/data_GCN15d_reorder/%s/%s_size_%d_rep_%d/" % (
            reorder_alg, dataset, size, replication)
    adj_part = sp.load_npz(dir_name+"adj_part"+str(rank)+".npz")
    data_part, row_part, col_part = adj_part.data, adj_part.row, adj_part.col
    input_part = np.load(dir_name+"input"+str(rank)+".npy")
    label_part = np.load(dir_name+"label"+str(rank)+".npy")
    print("Data loading done for rank %d." % rank)
    return adj_part, data_part, row_part, col_part, input_part, label_part


def load_data_whole(args):
    dataset = args.dataset
    reorder_alg = args.reorder_alg
    print("Loading dataset %s ..." % dataset)
    dir_name = "./tests/test_DistGCN/data_GCN15d/%s_whole_graph/" % (dataset)
    if args.reorder:
        dir_name = "./tests/test_DistGCN/data_GCN15d_reorder/%s/%s_whole_graph/" % (
            reorder_alg, dataset)
    adj_whole = sp.load_npz(dir_name+"adj_whole.npz")
    adj_whole = adj_whole.tocoo()
    data_whole, row_whole, col_whole = adj_whole.data, adj_whole.row, adj_whole.col
    input_whole = np.load(dir_name+"input_whole.npy")
    label_whole = np.load(dir_name+"label_whole.npy")
    print("Data loading done for dataset %s." % dataset)
    return adj_whole, data_whole, row_whole, col_whole, input_whole, label_whole


def test(args):
    comm = ht.wrapped_mpi_nccl_init()
    device_id = comm.dev_id
    rank = comm.rank
    size = comm.nrank

    dataset_info = {'Reddit': [232965, 602, 41], 'Proteins': [
        132534, 602, 8], 'Arch': [1644228, 602, 10], 'Products': [2449029, 100, 47]}

    node_count, num_features, num_classes = dataset_info[args.dataset]

    hidden_layer_size = 128
    if num_features < 128:
        hidden_layer_size = 64

    replication = args.replication

    node_Count_Self = row_num(
        node_count, rank//replication, size // replication)
    node_Count_All = node_count

    _, _, row_groups, col_groups = get_proc_groups(size, replication)

    executor_ctx = ht.gpu(device_id)

    if size > 1:
        adj_part, data_part, row_part, col_part, input_part, label_part = load_data(
            args, size, replication, rank)
    else:
        adj_part, data_part, row_part, col_part, input_part, label_part = load_data_whole(
            args)

    adj_matrix = ht.sparse_array(
        data_part, (row_part, col_part), shape=adj_part.shape, ctx=executor_ctx)

    # train:val:test=6:2:2
    # Our optimization on distributed GNN algorithm does NOT affect the correctness!
    # Here due to the limitation of current slice_op, data is split continuously.
    # Continuous split is unfriendly for reordered graph data where nodes are already clustered.
    # Specifically, training on some node clusters and testing on other clusters may cause poor test accuracy.
    # The better way is to split data randomly!
    train_split, test_split = 0.6, 0.8
    train_node = int(train_split*node_Count_Self)
    test_node = int(test_split*node_Count_Self)

    A = ht.Variable(name="A", trainable=False)
    H = ht.Variable(name="H")
    np.random.seed(123)
    bounds = np.sqrt(6.0 / (num_features + hidden_layer_size))
    W1_val = np.random.uniform(
        low=-bounds, high=bounds, size=[num_features, hidden_layer_size]).astype(np.float32)
    W1 = ht.Variable(name="W1", value=W1_val)
    bounds = np.sqrt(6.0 / (num_classes + hidden_layer_size))
    np.random.seed(123)
    W2_val = np.random.uniform(
        low=-bounds, high=bounds, size=[hidden_layer_size, num_classes]).astype(np.float32)

    W2 = ht.Variable(name="W2", value=W2_val)
    y_ = ht.Variable(name="y_")

    z = ht.distgcn_15d_op(A, H, W1, node_Count_Self, node_Count_All,
                          size, replication, device_id, comm, [row_groups, col_groups], True)
    H1 = ht.relu_op(z)
    y = ht.distgcn_15d_op(A, H1, W2, node_Count_Self, node_Count_All,
                          size, replication, device_id, comm, [row_groups, col_groups], True)

    y_train = ht.slice_op(y, (0, 0), (train_node, num_classes))
    label_train = ht.slice_op(y_, (0, 0), (train_node, num_classes))

    y_test = ht.slice_op(
        y, (test_node, 0), (node_Count_Self-test_node, num_classes))
    label_test = ht.slice_op(
        y_, (test_node, 0), (node_Count_Self-test_node, num_classes))

    loss = ht.softmaxcrossentropy_op(y_train, label_train)
    loss_test = ht.softmaxcrossentropy_op(y_test, label_test)
    opt = ht.optim.AdamOptimizer()
    train_op = opt.minimize(loss)

    executor = ht.Executor([loss, y, loss_test, train_op], ctx=executor_ctx)

    feed_dict = {
        A: adj_matrix,
        H: ht.array(input_part, ctx=executor_ctx),
        y_: ht.array(convert_to_one_hot(label_part, max_val=num_classes), ctx=executor_ctx),
    }

    epoch_num = 100
    epoch_all, epoch_0 = 0, 0

    for i in range(epoch_num):
        epoch_start_time = time.time()
        results = executor.run(feed_dict=feed_dict)
        loss = results[0].asnumpy().sum()
        y_out = results[1]
        loss_test = results[2].asnumpy().sum()
        epoch_end_time = time.time()
        epoch_time = epoch_end_time-epoch_start_time
        epoch_all += epoch_time
        if i == 0:
            epoch_0 = epoch_time

        print("[Epoch: %d, Rank: %d] Epoch time: %.3f, Total time: %.3f" %
              (i, rank, epoch_time, epoch_all))

        y_out_train, y_predict = y_out.asnumpy().argmax(
            axis=1)[:train_node], y_out.asnumpy().argmax(axis=1)[test_node:]
        label_train, label_test = label_part[:
                                             train_node], label_part[test_node:]
        train_acc = ht.array(
            np.array([(y_out_train == label_train).sum()]), ctx=executor_ctx)
        test_acc = ht.array(
            np.array([(y_predict == label_test).sum()]), ctx=executor_ctx)
        train_loss = ht.array(np.array([loss]), ctx=executor_ctx)
        test_loss = ht.array(np.array([loss_test]), ctx=executor_ctx)

        if replication > 1:
            col_groups[rank % replication].dlarrayNcclAllReduce(
                test_acc, test_acc, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum)
            col_groups[rank % replication].dlarrayNcclAllReduce(
                test_loss, test_loss, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum)
            col_groups[rank % replication].dlarrayNcclAllReduce(
                train_acc, train_acc, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum)
            col_groups[rank % replication].dlarrayNcclAllReduce(
                train_loss, train_loss, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum)
        else:
            comm.dlarrayNcclAllReduce(
                test_acc, test_acc, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum)
            comm.dlarrayNcclAllReduce(
                test_loss, test_loss, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum)
            comm.dlarrayNcclAllReduce(
                train_acc, train_acc, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum)
            comm.dlarrayNcclAllReduce(
                train_loss, train_loss, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum)

        test_acc = float(test_acc.asnumpy()[0]) / \
            (node_count-test_split*node_count)
        test_loss = test_loss.asnumpy()[0]/(node_count-test_split*node_count)
        train_acc = float(train_acc.asnumpy()[0])/(train_split*node_count)
        train_loss = train_loss.asnumpy()[0]/(train_split*node_count)

        if rank == 0:
            print("[Epoch: %d] Train Loss: %.3f, Train Accuracy: %.3f, Test Loss: %.3f, Test Accuracy: %.3f"
                  % (i, train_loss, train_acc, test_loss, test_acc))

    avg_epoch_time = (epoch_all-epoch_0)/(epoch_num-1)
    results = ht.array(np.array([epoch_all, avg_epoch_time]), ctx=executor_ctx)
    comm.dlarrayNcclAllReduce(
        results, results, ncclDataType_t.ncclFloat32, reduceop=ncclRedOp_t.ncclSum)
    results = results.asnumpy()/size

    if rank == 0:
        print("\nAverage Total Time: %.3f, Average Epoch Time: %.3f" %
              (results[0], results[1]))


def get_dataset(args):
    if args.dataset in ['Reddit', 'reddit']:
        args.dataset = 'Reddit'
    elif args.dataset in ['Proteins', 'proteins']:
        args.dataset = 'Proteins'
    elif args.dataset in ['Arch', 'arch']:
        args.dataset = 'Arch'
    elif args.dataset in ['Products', 'products']:
        args.dataset = 'Products'
    else:
        print("Dataset should be in ['Reddit','Proteins','Arch','Products']")
        assert False


parser = argparse.ArgumentParser()
parser.add_argument('--replication', type=int, default=1,
                    help='Replication of distGCN1.5D.')
parser.add_argument('--reorder', type=int, default=0,
                    help='Reorder graph or not.')
parser.add_argument('--reorder_alg', type=str, default="metis",
                    help='Graph reordering algorithm [rcm, metis, slashburn, deg].')
parser.add_argument('--dataset', type=str, default="Reddit",
                    help='Choose dataset [Reddit, Proteins, Arch, Products].')
args = parser.parse_args()
get_dataset(args)
test(args)
