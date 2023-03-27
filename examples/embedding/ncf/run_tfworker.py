import os
import json
import numpy as np
import tensorflow as tf
import time
import argparse
from tqdm import tqdm
from tf_ncf import neural_mf
import heapq  # for retrieval topK
import math


def pop_env():
    for k in ['https_proxy', 'http_proxy']:
        if k in os.environ:
            os.environ.pop(k)


pop_env()


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
    def __init__(self, path='logs/tflog.txt'):
        with open(path, 'w') as fw:
            fw.write('')
        self.path = path

    def write(self, s):
        print(s)
        with open(self.path, 'a') as fw:
            fw.write(s + '\n')
            fw.flush()


def train_ncf(cluster, rank, nrank, args):
    def validate():
        # validate phase
        hits, ndcgs = [], []
        for idx in range(testData.shape[0]):
            start_index = idx * 100
            my_feed_dict = {
                user_input: testUserInput[start_index:start_index+100],
                item_input: testItemInput[start_index:start_index+100],
            }
            predictions = sess.run([y], feed_dict=my_feed_dict)
            map_item_score = {
                testItemInput[start_index+i]: predictions[0][i] for i in range(100)}

            # Evaluate top rank list
            ranklist = heapq.nlargest(
                topK, map_item_score, key=map_item_score.get)
            hr = getHitRatio(ranklist, testItemInput[start_index])
            ndcg = getNDCG(ranklist, testItemInput[start_index])
            hits.append(hr)
            ndcgs.append(ndcg)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        return hr, ndcg

    def get_current_shard(data):
        part_size = data.shape[0] // nrank
        start = part_size * rank
        end = start + part_size if rank != nrank - 1 else data.shape[0]
        return data[start:end]

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
    batch_size = 1024
    num_negatives = 4
    topK = 10

    worker_device = "/job:worker/task:%d/gpu:0" % (rank)
    with tf.device(worker_device):
        user_input = tf.compat.v1.placeholder(tf.int32, [None, ])
        item_input = tf.compat.v1.placeholder(tf.int32, [None, ])
        y_ = tf.compat.v1.placeholder(tf.float32, [None, ])

    with tf.device(tf.compat.v1.train.replica_device_setter(cluster=cluster)):
        server_num = len(cluster.as_dict()['ps'])
        embed_partitioner = tf.fixed_size_partitioner(
            server_num, 0) if server_num > 1 else None
        loss, y, opt = neural_mf(
            user_input, item_input, y_, num_users, num_items, embed_partitioner)
        train_op = opt.minimize(loss)

    server = tf.train.Server(
        cluster, job_name="worker", task_index=rank)
    init = tf.compat.v1.global_variables_initializer()
    sv = tf.train.Supervisor(
        is_chief=(rank == 0),
        init_op=init,
        recovery_wait_secs=1)
    sess_config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps",
                        "/job:worker/task:%d" % rank])
    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

    log = Logging(path='logs/tflog%d.txt' % rank)
    epoch = 7
    iterations = trainUsers.shape[0] // batch_size
    start = time.time()
    for ep in range(epoch):
        ep_st = time.time()
        log.write('epoch %d' % ep)
        train_loss = []
        for idx in tqdm(range(iterations)):
            start_index = idx * batch_size
            my_feed_dict = {
                user_input: trainUsers[start_index:start_index+batch_size],
                item_input: trainItems[start_index:start_index+batch_size],
                y_: trainLabels[start_index:start_index+batch_size],
            }

            loss_val = sess.run([loss, train_op], feed_dict=my_feed_dict)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", action="store_true",
                        help="whether to perform validation")
    parser.add_argument("--rank", type=int, required=True,
                        help="rank of process")
    parser.add_argument(
        "--config", type=str, default='../ctr/settings/tf_local_s1_w2.json', help="config file path")
    parser.add_argument("--all", action="store_true",
                        help="whether to use all data")
    args = parser.parse_args()
    task_id = int(args.rank)
    raw_config = args.config

    config = json.load(open(raw_config))
    cluster = tf.train.ClusterSpec(config)

    train_ncf(cluster, task_id, len(config['worker']), args)


if __name__ == '__main__':
    main()
