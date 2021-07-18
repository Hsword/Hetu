import hetu as ht
from hetu.launcher import launch

import os
import numpy as np
import yaml
import time
import math
import argparse
from tqdm import tqdm
from hetu_ncf import neural_mf
import heapq  # for retrieval topK


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0


class Logging(object):
    def __init__(self, path='logs/hetulog.txt'):
        with open(path, 'w') as fw:
            fw.write('')
        self.path = path

    def write(self, s):
        print(s)
        with open(self.path, 'a') as fw:
            fw.write(s + '\n')
            fw.flush()


def worker(args):
    def validate():
        hits, ndcgs = [], []
        for idx in range(testData.shape[0]):
            start_index = idx * 100
            predictions = executor.run(
                'validate', convert_to_numpy_ret_vals=True)
            map_item_score = {
                testItemInput[start_index + i]: predictions[0][i] for i in range(100)}
            gtItem = testItemInput[start_index]
            # Evaluate top rank list
            ranklist = heapq.nlargest(
                topK, map_item_score, key=map_item_score.get)
            hr = getHitRatio(ranklist, gtItem)
            ndcg = getNDCG(ranklist, gtItem)
            hits.append(hr)
            ndcgs.append(ndcg)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        return hr, ndcg

    def get_current_shard(data):
        if args.comm is not None:
            part_size = data.shape[0] // nrank
            start = part_size * rank
            end = start + part_size if rank != nrank - 1 else data.shape[0]
            return data[start:end]
        else:
            return data

    device_id = 0
    if args.comm == 'PS':
        rank = ht.get_worker_communicate().rank()
        nrank = int(os.environ['DMLC_NUM_WORKER'])
        device_id = rank % 8
    elif args.comm == 'Hybrid':
        comm = ht.wrapped_mpi_nccl_init()
        device_id = comm.dev_id
        rank = comm.rank
        nrank = int(os.environ['DMLC_NUM_WORKER'])

    from movielens import getdata
    if args.all:
        trainData, testData = getdata('ml-25m', 'datasets')
        trainUsers = get_current_shard(trainData['user_input'])
        trainItems = get_current_shard(trainData['item_input'])
        trainLabels = get_current_shard(trainData['labels'])
        testData = get_current_shard(testData)
        testUserInput = np.repeat(
            np.arange(testData.shape[0], dtype=np.int32), 100)
        testItemInput = testData.reshape((-1,))
    else:
        trainData, testData = getdata('ml-25m', 'datasets')
        trainUsers = get_current_shard(trainData['user_input'][:1024000])
        trainItems = get_current_shard(trainData['item_input'][:1024000])
        trainLabels = get_current_shard(trainData['labels'][:1024000])
        testData = get_current_shard(testData[:1470])
        testUserInput = np.repeat(
            np.arange(testData.shape[0], dtype=np.int32), 100)
        testItemInput = testData.reshape((-1,))

    num_users, num_items = {
        'ml-1m': (6040, 3706),
        'ml-20m': (138493, 26744),
        'ml-25m': (162541, 59047),
    }['ml-25m']
    # assert not args.all or num_users == testData.shape[0]
    batch_size = 1024
    num_negatives = 4
    topK = 10
    user_input = ht.dataloader_op([
        ht.Dataloader(trainUsers, batch_size, 'train'),
        ht.Dataloader(testUserInput, 100, 'validate'),
    ])
    item_input = ht.dataloader_op([
        ht.Dataloader(trainItems, batch_size, 'train'),
        ht.Dataloader(testItemInput, 100, 'validate'),
    ])
    y_ = ht.dataloader_op([
        ht.Dataloader(trainLabels.reshape((-1, 1)), batch_size, 'train'),
    ])

    loss, y, train_op = neural_mf(
        user_input, item_input, y_, num_users, num_items)

    executor = ht.Executor({'train': [loss, train_op], 'validate': [y]}, ctx=ht.gpu(device_id),
                           comm_mode=args.comm, cstable_policy=args.cache, bsp=args.bsp, cache_bound=args.bound, seed=123)

    path = 'logs/hetulog_%s' % ({None: 'local',
                                 'PS': 'ps', 'Hybrid': 'hybrid'}[args.comm])
    path += '_%d.txt' % rank if args.comm else '.txt'
    log = Logging(path=path)
    epoch = 7
    start = time.time()
    for ep in range(epoch):
        ep_st = time.time()
        log.write('epoch %d' % ep)
        train_loss = []
        for idx in tqdm(range(executor.get_batch_num('train'))):
            loss_val = executor.run('train', convert_to_numpy_ret_vals=True)
            train_loss.append(loss_val[0])

        tra_loss = np.mean(train_loss)
        ep_en = time.time()

        # validate phase
        if args.val:
            hr, ndcg = validate()
            printstr = "train_loss: %.4f, HR: %.4f, NDCF: %.4f, train_time: %.4f" % (
                tra_loss, hr, ndcg, ep_en - ep_st)
        else:
            printstr = "train_loss: %.4f, train_time: %.4f" % (
                tra_loss, ep_en - ep_st)
        log.write(printstr)
    log.write('all time: %f' % (time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", action="store_true",
                        help="whether to perform validation")
    parser.add_argument("--all", action="store_true",
                        help="whether to use all data, default to use 1024000 training data")
    parser.add_argument("--comm", default=None,
                        help="whether to use distributed setting, can be None, AllReduce, PS, Hybrid")
    parser.add_argument("--bsp", action="store_true",
                        help="whether to use bsp instead of asp")
    parser.add_argument("--cache", default=None, help="cache policy")
    parser.add_argument("--bound", default=100, help="cache bound")
    parser.add_argument(
        "--config", type=str, default="./settings/local_s1_w4.yml", help="configuration for ps")
    args = parser.parse_args()

    if args.comm is None:
        worker(args)
    elif args.comm == 'Hybrid':
        settings = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
        value = settings['shared']
        os.environ['DMLC_ROLE'] = 'worker'
        for k, v in value.items():
            os.environ[k] = str(v)
        worker(args)
    elif args.comm == 'PS':
        launch(worker, args)
    else:
        raise NotImplementedError
