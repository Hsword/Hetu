import os
import numpy as np
import tensorflow as tf
import time
import argparse
from tqdm import tqdm
from tf_ncf import neural_mf
import heapq  # for retrieval topK
import math

from autodist import AutoDist
from autodist.resource_spec import ResourceSpec
from autodist.strategy import PS, PSLoadBalancing, PartitionedPS, AllReduce, Parallax
from autodist.strategy.base import Strategy
from autodist.kernel.common.utils import get_op_name
from tensorflow.python.framework import ops


def pop_env():
    for k in ['https_proxy', 'http_proxy']:
        if k in os.environ:
            os.environ.pop(k)


pop_env()

# Please DO NOT modify /etc/bash.bashrc to activate conda environment.
# Use python_venv in spec yml file instead.
# Use absolute path of python file.
# Here we use the tf native partitioner instead of autodist's PartitionPS.


class Parallaxx(PSLoadBalancing, AllReduce):
    """
    Modify original parallax to remove replica on CPUs.
    """

    def __init__(self, chunk_size=128, local_proxy_variable=False, sync=True, staleness=0):
        PSLoadBalancing.__init__(self, local_proxy_variable, sync, staleness)
        AllReduce.__init__(self, chunk_size)

    # pylint: disable=attribute-defined-outside-init
    def build(self, graph_item, resource_spec):
        """Generate the strategy."""
        expr = Strategy()

        # For each variable, generate variable synchronizer config
        expr.graph_config.replicas.extend(
            [k for k, v in resource_spec.gpu_devices])
        reduction_device_names = [k for k, _ in resource_spec.cpu_devices]
        self.loads = {ps: 0.0 for ps in reduction_device_names}

        # Generate node config
        node_config = []
        for idx, var in enumerate(graph_item.trainable_var_op_to_var.values()):
            var_op_name = get_op_name(var.name)
            grad, _, _ = graph_item.var_op_name_to_grad_info[var_op_name]
            if isinstance(grad, ops.Tensor):  # this is a dense variable
                group_id = idx // self.chunk_size
                config = self._gen_all_reduce_node_config(
                    var.name, group=group_id)
            else:  # sparse updates
                # For Parallax Strategy, all PS vars are sparse so we don't use a proxy.
                # Sparse variables are likely larger, so keeping copies would be costlier,
                # and usually each device only requires a small part of the overall variable.
                config = self._gen_ps_node_config(
                    var,
                    # For Parallax Strategy, all PS vars are sparse which does not need proxy.
                    False,
                    self._sync,
                    self._staleness
                )
            node_config.append(config)
        expr.node_config.extend(node_config)

        return expr


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


def main():
    resource_spec_file = os.path.join(os.path.dirname(
        __file__), '../ctr/settings', 'plx_local_spec.yml')
    autodist = AutoDist(resource_spec_file, Parallaxx())
    respec = ResourceSpec(resource_spec_file)

    def validate():
        # validate phase
        hits, ndcgs = [], []
        for idx in range(num_users):
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

    from movielens import getdata
    trainData, testData = getdata('ml-25m', 'datasets')
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
    with tf.Graph().as_default() as g, autodist.scope():
        user_input = tf.compat.v1.placeholder(tf.int32, [None, ])
        item_input = tf.compat.v1.placeholder(tf.int32, [None, ])
        y_ = tf.compat.v1.placeholder(tf.float32, [None, ])

        loss, y, opt = neural_mf(
            user_input, item_input, y_, num_users, num_items)
        train_op = opt.minimize(loss)

        sess = autodist.create_distributed_session()

        log = Logging(path=os.path.join(
            os.path.dirname(__file__), 'logs', 'tfplx.txt'))
        epoch = 7
        iterations = trainData['user_input'].shape[0] // batch_size
        start = time.time()
        for ep in range(epoch):
            ep_st = time.time()
            log.write('epoch %d' % ep)
            train_loss = []
            for idx in range(iterations):
                start_index = idx * batch_size
                my_feed_dict = {
                    user_input: trainData['user_input'][start_index:start_index+batch_size],
                    item_input: trainData['item_input'][start_index:start_index+batch_size],
                    y_: trainData['labels'][start_index:start_index+batch_size],
                }

                loss_val = sess.run([loss, train_op], feed_dict=my_feed_dict)
                train_loss.append(loss_val[0])

            tra_loss = np.mean(train_loss)
            ep_en = time.time()

            # validate phase
            hr, ndcg = validate()
            printstr = "train_loss: %.4f, HR: %.4f, NDCF: %.4f, train_time: %.4f" % (
                tra_loss, hr, ndcg, ep_en - ep_st)
            log.write(printstr)
        log.write('all time:', (time.time() - start))


if __name__ == '__main__':
    main()
