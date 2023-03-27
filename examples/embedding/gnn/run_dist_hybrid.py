from gnn_tools.launcher import launch_graphmix_and_hetu_ps
from gnn_model.utils import get_norm_adj, prepare_data
from gnn_model.model import sparse_model
import graphmix

import hetu as ht
from hetu.communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t

import numpy as np
import time
import os
import sys
import multiprocessing
import argparse

# usage :
#   mpirun -np 4 --allow-run-as-root python3 run_dist_hybrid.py [configfile] [-p data_path]
#   python3 run_dist_hybrid.py [configfile] [-p data_path] --server


class TrainStat():
    def __init__(self, comm):
        self.file = open("log.txt", "w")
        self.train_stat = np.zeros(4)
        self.test_stat = np.zeros(4)
        self.count = 0
        self.time = []
        self.comm = comm

    def update_test(self, cnt, total, loss):
        self.test_stat += [1, cnt, total, loss]

    def update_train(self, cnt, total, loss):
        self.train_stat += [1, cnt, total, loss]

    def sync_and_clear(self):
        self.count += 1
        train_stat = ht.array(self.train_stat, ht.cpu())
        test_stat = ht.array(self.test_stat, ht.cpu())
        self.comm.dlarrayNcclAllReduce(
            train_stat, train_stat, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum, self.comm.stream)
        self.comm.dlarrayNcclAllReduce(
            test_stat, test_stat, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum, self.comm.stream)
        self.comm.stream.sync()
        train_stat, test_stat = train_stat.asnumpy(), test_stat.asnumpy()
        printstr = "epoch {}: test loss: {:.3f} test acc: {:.3f} train loss: {:.3f} train acc: {:.3f}".format(
            self.count,
            test_stat[3] / test_stat[0],
            test_stat[1] / test_stat[2],
            train_stat[3] / train_stat[0],
            train_stat[1] / train_stat[2],
        )
        logstr = "{} {} {} {}".format(
            test_stat[3] / test_stat[0],
            test_stat[1] / test_stat[2],
            train_stat[3] / train_stat[0],
            train_stat[1] / train_stat[2],
        )
        self.time.append(time.time())
        if self.comm.device_id.value == 0:
            print(printstr, flush=True)
            print(logstr, file=self.file, flush=True)
            if len(self.time) > 3:
                epoch_time = np.array(self.time[1:])-np.array(self.time[:-1])
                print(
                    "epoch time: {:.3f}+-{:.3f}".format(np.mean(epoch_time), np.var(epoch_time)))

        self.train_stat[:] = 0
        self.test_stat[:] = 0


def train_main(args):
    comm = ht.wrapped_mpi_nccl_init()
    device_id = comm.dev_id
    cli = graphmix.Client()
    meta = cli.meta
    hidden_layer_size = args.hidden_size
    num_epoch = args.num_epoch
    rank = cli.rank()
    nrank = cli.num_worker()
    ctx = ht.gpu(device_id)
    embedding_width = args.hidden_size
    # the last two is train label and other train mask
    num_int_feature = meta["int_feature"] - 2
    # sample some graphs
    ngraph = 10 * meta["train_node"] // (args.batch_size * nrank)
    graphs = prepare_data(ngraph)
    # build model
    [loss, y, train_op], [mask_, norm_adj_] = sparse_model(
        num_int_feature, args.hidden_size, meta["idx_max"], args.hidden_size, meta["class"], args.learning_rate)
    idx = 0
    graph = graphs[idx]
    idx = (idx + 1) % ngraph
    ht.GNNDataLoaderOp.step(graph)
    ht.GNNDataLoaderOp.step(graph)
    executor = ht.Executor([loss, y, train_op], ctx=ctx, comm_mode='Hybrid',
                           use_sparse_pull=False, cstable_policy=args.cache)
    nbatches = meta["train_node"] // (args.batch_size * nrank)
    train_state = TrainStat(comm)
    for epoch in range(num_epoch):
        for _ in range(nbatches):
            graph_nxt = graphs[idx]
            idx = (idx + 1) % ngraph
            ht.GNNDataLoaderOp.step(graph_nxt)
            train_mask = np.bitwise_and(
                graph.extra[:, 0], graph.i_feat[:, -1] == 1)
            eval_mask = np.bitwise_and(
                graph.extra[:, 0], graph.i_feat[:, -1] != 1)
            feed_dict = {
                norm_adj_: get_norm_adj(graph, ht.gpu(device_id)),
                mask_: train_mask
            }
            loss_val, y_predicted, _ = executor.run(feed_dict=feed_dict)
            y_predicted = y_predicted.asnumpy().argmax(axis=1)

            acc = np.sum((y_predicted == graph.i_feat[:, -2]) * eval_mask)
            train_acc = np.sum(
                (y_predicted == graph.i_feat[:, -2]) * train_mask)
            train_state.update_test(acc, eval_mask.sum(), np.sum(
                loss_val.asnumpy()*eval_mask)/eval_mask.sum())
            train_state.update_train(train_acc, train_mask.sum(), np.sum(
                loss_val.asnumpy()*train_mask)/train_mask.sum())
            ht.get_worker_communicate().BarrierWorker()
            graph = graph_nxt
        train_state.sync_and_clear()


def server_init(server):
    batch_size = args.batch_size
    server.init_cache(0.1, graphmix.cache.LFUOpt)
    worker_per_server = server.num_worker() // server.num_server()
    server.add_sampler(graphmix.sampler.GraphSage, batch_size=batch_size,
                       depth=2, width=2, thread=4 * worker_per_server, subgraph=True)
    server.is_ready()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--path", "-p", required=True)
    parser.add_argument("--num_epoch", default=300, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=1, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--cache", default="LFUOpt", type=str)
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()
    if args.server:
        launch_graphmix_and_hetu_ps(
            train_main, args, server_init, hybrid_config="server")
    else:
        launch_graphmix_and_hetu_ps(
            train_main, args, server_init, hybrid_config="worker")
