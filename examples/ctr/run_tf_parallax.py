import os
import numpy as np
import tensorflow as tf
import time
import argparse
from tqdm import tqdm
from sklearn import metrics

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


def train_criteo(model, args):
    resource_spec_file = os.path.join(os.path.dirname(
        __file__), 'settings', 'plx_local_spec.yml')
    autodist = AutoDist(resource_spec_file, Parallaxx())
    respec = ResourceSpec(resource_spec_file)
    if args.all:
        from models.load_data import process_all_criteo_data
        dense, sparse, all_labels = process_all_criteo_data()
        dense_feature, val_dense = dense
        sparse_feature, val_sparse = sparse
        labels, val_labels = all_labels
    else:
        from models.load_data import process_sampled_criteo_data
        dense_feature, sparse_feature, labels = process_sampled_criteo_data()

    # autodist will split the feeding data
    batch_size = 128
    with tf.Graph().as_default() as g, autodist.scope():
        dense_input = tf.compat.v1.placeholder(tf.float32, [batch_size, 13])
        sparse_input = tf.compat.v1.placeholder(tf.int32, [batch_size, 26])
        y_ = y_ = tf.compat.v1.placeholder(tf.float32, [batch_size, 1])
        embed_partitioner = tf.fixed_size_partitioner(
            len(respec.nodes), 0) if len(respec.nodes) > 1 else None
        loss, y, opt = model(dense_input, sparse_input,
                             y_, embed_partitioner, False)
        train_op = opt.minimize(loss)

        sess = autodist.create_distributed_session()

        my_feed_dict = {
            dense_input: np.empty(shape=(batch_size, 13)),
            sparse_input: np.empty(shape=(batch_size, 26)),
            y_: np.empty(shape=(batch_size, 1)),
        }

        if args.all:
            raw_log_file = os.path.join(os.path.split(os.path.abspath(__file__))[
                                        0], 'logs', 'tf_plx_%s.log' % (args.model))
            print('Processing all data, log to', raw_log_file)
            log_file = open(raw_log_file, 'w')
            iterations = dense_feature.shape[0] // batch_size
            total_epoch = 11
            start_index = 0
            for ep in range(total_epoch):
                print("epoch %d" % ep)
                st_time = time.time()
                train_loss, train_acc, train_auc = [], [], []
                for it in tqdm(range(iterations // 10 + (ep % 10 == 9) * (iterations % 10))):
                    my_feed_dict[dense_input][:] = dense_feature[start_index: start_index + batch_size]
                    my_feed_dict[sparse_input][:] = sparse_feature[start_index: start_index + batch_size]
                    my_feed_dict[y_][:] = labels[start_index: start_index+batch_size]
                    start_index += batch_size
                    if start_index + batch_size > dense_feature.shape[0]:
                        start_index = 0
                    loss_val = sess.run(
                        [loss, y, y_, train_op], feed_dict=my_feed_dict)
                    pred_val = loss_val[1]
                    true_val = loss_val[2]
                    acc_val = np.equal(
                        true_val,
                        pred_val > 0.5)
                    train_loss.append(loss_val[0])
                    train_acc.append(acc_val)
                    train_auc.append(metrics.roc_auc_score(true_val, pred_val))
                tra_accuracy = np.mean(train_acc)
                tra_loss = np.mean(train_loss)
                tra_auc = np.mean(train_auc)
                en_time = time.time()
                train_time = en_time - st_time
                printstr = "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, train_time: %.4f"\
                    % (tra_loss, tra_accuracy, tra_auc, train_time)
                print(printstr)
                log_file.write(printstr + '\n')
                log_file.flush()

        else:
            iteration = dense_feature.shape[0] // batch_size

            epoch = 50
            for ep in range(epoch):
                print('epoch', ep)
                if ep == 5:
                    start = time.time()
                ep_st = time.time()
                train_loss = []
                train_acc = []
                for idx in range(iteration):
                    start_index = idx * batch_size
                    my_feed_dict[dense_input][:] = dense_feature[start_index: start_index + batch_size]
                    my_feed_dict[sparse_input][:] = sparse_feature[start_index: start_index + batch_size]
                    my_feed_dict[y_][:] = labels[start_index: start_index+batch_size]

                    loss_val = sess.run(
                        [loss, y, y_, train_op], feed_dict=my_feed_dict)
                    pred_val = loss_val[1]
                    true_val = loss_val[2]
                    if pred_val.shape[1] == 1:  # for criteo case
                        acc_val = np.equal(
                            true_val,
                            pred_val > 0.5)
                    else:
                        acc_val = np.equal(
                            np.argmax(pred_val, 1),
                            np.argmax(true_val, 1)).astype(np.float32)
                    train_loss.append(loss_val[0])
                    train_acc.append(acc_val)
                tra_accuracy = np.mean(train_acc)
                tra_loss = np.mean(train_loss)
                ep_en = time.time()
                print("train_loss: %.4f, train_acc: %.4f, train_time: %.4f"
                      % (tra_loss, tra_accuracy, ep_en - ep_st))
            print('all time:', (time.time() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="model to be tested")
    parser.add_argument("--all", action="store_true",
                        help="whether to use all data")
    args = parser.parse_args()
    raw_model = args.model
    import tf_models
    model = eval('tf_models.' + raw_model)
    dataset = raw_model.split('_')[-1]
    print('Model:', raw_model)

    if dataset == 'criteo':
        train_criteo(model, args)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
