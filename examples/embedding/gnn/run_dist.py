from gnn_tools.launcher import launch_graphmix_and_hetu_ps
from gnn_model.utils import get_norm_adj, prepare_data
from gnn_model.model import sparse_model
from gnn_tools.log import SharedTrainingStat
import graphmix

import hetu as ht

import numpy as np
import argparse

# usage : on each machine
#   python3 run_dist.py [configfile] [-p data_path]


def train_main(args):
    cli = graphmix.Client()
    meta = cli.meta
    hidden_layer_size = args.hidden_size
    num_epoch = args.num_epoch
    rank = cli.rank()
    nrank = cli.num_worker()
    ctx = ht.gpu(rank % args.num_local_worker)
    embedding_width = args.hidden_size
    # the last two is train label and other train mask
    num_int_feature = meta["int_feature"] - 2
    # sample some graphs
    ngraph = meta["train_node"] // (args.batch_size * nrank)
    graphs = prepare_data(ngraph)
    # build model
    [loss, y, train_op], [mask_, norm_adj_] = sparse_model(
        num_int_feature, args.hidden_size, meta["idx_max"], args.hidden_size, meta["class"], args.learning_rate)

    idx = 0
    graph = graphs[idx]
    idx = (idx + 1) % ngraph
    ht.GNNDataLoaderOp.step(graph)
    ht.GNNDataLoaderOp.step(graph)
    executor = ht.Executor([loss, y, train_op], ctx=ctx, comm_mode='PS',
                           use_sparse_pull=False, cstable_policy=args.cache)
    nbatches = meta["train_node"] // (args.batch_size * nrank)
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
                norm_adj_: get_norm_adj(graph, ht.gpu(rank % args.num_local_worker)),
                mask_: train_mask
            }
            loss_val, y_predicted, _ = executor.run(feed_dict=feed_dict)
            y_predicted = y_predicted.asnumpy().argmax(axis=1)

            acc = np.sum((y_predicted == graph.i_feat[:, -2]) * eval_mask)
            train_acc = np.sum(
                (y_predicted == graph.i_feat[:, -2]) * train_mask)
            stat.update(acc, eval_mask.sum(), np.sum(
                loss_val.asnumpy()*eval_mask)/eval_mask.sum())
            stat.update_train(train_acc, train_mask.sum(), np.sum(
                loss_val.asnumpy()*train_mask)/train_mask.sum())
            ht.get_worker_communicate().BarrierWorker()
            graph = graph_nxt
        if rank == 0:
            stat.print(epoch)


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
    args = parser.parse_args()
    stat = SharedTrainingStat()
    launch_graphmix_and_hetu_ps(train_main, args, server_init=server_init)
