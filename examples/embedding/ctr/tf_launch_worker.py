import tensorflow as tf
import numpy as np
import argparse
import os
import time
import json
from sklearn import metrics
from tqdm import tqdm


def pop_env():
    for k in ['https_proxy', 'http_proxy']:
        if k in os.environ:
            os.environ.pop(k)


pop_env()


def train_criteo(model, cluster, task_id, nrank, args):
    def get_current_shard(data):
        part_size = data.shape[0] // nrank
        start = part_size * task_id
        end = start + part_size if task_id != nrank - 1 else data.shape[0]
        return data[start:end]

    if args.all:
        from models.load_data import process_all_criteo_data
        dense, sparse, all_labels = process_all_criteo_data()
        dense_feature = get_current_shard(dense[0])
        sparse_feature = get_current_shard(sparse[0])
        labels = get_current_shard(all_labels[0])
        val_dense = get_current_shard(dense[1])
        val_sparse = get_current_shard(sparse[1])
        val_labels = get_current_shard(all_labels[1])
    else:
        from models.load_data import process_sampled_criteo_data
        dense_feature, sparse_feature, labels = process_sampled_criteo_data()
        dense_feature = get_current_shard(dense_feature)
        sparse_feature = get_current_shard(sparse_feature)
        labels = get_current_shard(labels)

    batch_size = 128
    worker_device = "/job:worker/task:%d/gpu:0" % (task_id)
    with tf.device(worker_device):
        dense_input = tf.compat.v1.placeholder(tf.float32, [batch_size, 13])
        sparse_input = tf.compat.v1.placeholder(tf.int32, [batch_size, 26])
        y_ = y_ = tf.compat.v1.placeholder(tf.float32, [batch_size, 1])

    with tf.device(tf.compat.v1.train.replica_device_setter(cluster=cluster)):
        server_num = len(cluster.as_dict()['ps'])
        embed_partitioner = tf.fixed_size_partitioner(
            server_num, 0) if server_num > 1 else None
        loss, y, opt = model(dense_input, sparse_input, y_,
                             embed_partitioner, param_on_gpu=False)
        train_op = opt.minimize(loss)

    server = tf.train.Server(
        cluster, job_name="worker", task_index=task_id)
    init = tf.compat.v1.global_variables_initializer()
    sv = tf.train.Supervisor(
        is_chief=(task_id == 0),
        init_op=init,
        recovery_wait_secs=1)
    sess_config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps",
                        "/job:worker/task:%d" % task_id])
    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
    # sess.run(init)
    if task_id == 0:
        writer = tf.compat.v1.summary.FileWriter('logs/board', sess.graph)

    my_feed_dict = {
        dense_input: np.empty(shape=(batch_size, 13)),
        sparse_input: np.empty(shape=(batch_size, 26)),
        y_: np.empty(shape=(batch_size, 1)),
    }

    if args.all:
        raw_log_file = './logs/tf_dist_%s_%d.log' % (args.model, task_id)
        print('Processing all data, log to', raw_log_file)
        log_file = open(raw_log_file, 'w')
        iterations = dense_feature.shape[0] // batch_size
        total_epoch = 21
        start_index = 0
        for ep in range(total_epoch):
            print("epoch %d" % ep)
            st_time = time.time()
            train_loss, train_acc, train_auc = [], [], []
            for it in range(iterations // 10 + (ep % 10 == 9) * (iterations % 10)):
                my_feed_dict[dense_input][:] = dense_feature[start_index: start_index + batch_size]
                my_feed_dict[sparse_input][:] = sparse_feature[start_index: start_index + batch_size]
                my_feed_dict[y_][:] = labels[start_index: start_index+batch_size]
                start_index += batch_size
                if start_index + batch_size > dense_feature.shape[0]:
                    start_index = 0
                loss_val = sess.run([loss, y, y_, train_op],
                                    feed_dict=my_feed_dict)
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

            if args.val:
                val_loss, val_acc, val_auc = [], [], []
                for it in range(val_dense.shape[0] // batch_size):
                    local_st = it * batch_size
                    my_feed_dict[dense_input][:] = val_dense[local_st: local_st + batch_size]
                    my_feed_dict[sparse_input][:] = val_sparse[local_st: local_st + batch_size]
                    my_feed_dict[y_][:] = val_labels[local_st: local_st+batch_size]
                    loss_val = sess.run([loss, y, y_], feed_dict=my_feed_dict)
                    pred_val = loss_val[1]
                    true_val = loss_val[2]
                    acc_val = np.equal(
                        true_val,
                        pred_val > 0.5)
                    val_loss.append(loss_val[0])
                    val_acc.append(acc_val)
                    val_auc.append(metrics.roc_auc_score(true_val, pred_val))
                v_accuracy = np.mean(val_acc)
                v_loss = np.mean(val_loss)
                v_auc = np.mean(val_auc)
                printstr = "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, test_loss: %.4f, test_acc: %.4f, test_auc: %.4f, train_time: %.4f"\
                    % (tra_loss, tra_accuracy, tra_auc, v_loss, v_accuracy, v_auc, train_time)
            else:
                printstr = "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, train_time: %.4f"\
                    % (tra_loss, tra_accuracy, tra_auc, train_time)

            print(printstr)
            log_file.write(printstr + '\n')
            log_file.flush()
    else:
        # here no val
        iteration = dense_feature.shape[0] // batch_size

        epoch = 10
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

                loss_val = sess.run([loss, y, y_, train_op],
                                    feed_dict=my_feed_dict)
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
        print("tensorflow: ", (time.time() - start))


def train_adult(model, cluster, task_id, nrank):
    from models.load_data import load_adult_data
    x_train_deep, x_train_wide, y_train = load_adult_data(return_val=False)
    part_size = len(x_train_deep) // nrank
    start = part_size * task_id
    end = start + part_size if task_id != nrank - 1 else len(x_train_deep)
    x_train_deep = x_train_deep[start:end]
    x_train_wide = x_train_wide[start:end]
    y_train = y_train[start:end]

    batch_size = 128
    total_epoch = 50
    dim_wide = 809

    worker_device = "/job:worker/task:%d/gpu:0" % (task_id)
    with tf.device(worker_device):
        X_deep = []
        for i in range(8):
            X_deep.append(tf.compat.v1.placeholder(tf.int32, [batch_size, 1]))
        for i in range(4):
            X_deep.append(tf.compat.v1.placeholder(
                tf.float32, [batch_size, 1]))
        X_wide = tf.compat.v1.placeholder(tf.float32, [batch_size, dim_wide])
        y_ = tf.compat.v1.placeholder(tf.float32, [batch_size, 2])
    loss, y, train_op, global_step = model(
        X_deep, X_wide, y_, cluster, task_id)

    with tf.device(
            tf.compat.v1.train.replica_device_setter(
                worker_device=worker_device,
                cluster=cluster)):
        server = tf.train.Server(
            cluster, job_name="worker", task_index=task_id)
        init = tf.global_variables_initializer()
        sv = tf.train.Supervisor(
            is_chief=(task_id == 0),
            init_op=init,
            recovery_wait_secs=1,
            global_step=global_step)
        sess_config = tf.ConfigProto(
            # allow_soft_placement=True,
            log_device_placement=False,
            device_filters=["/job:ps",
                            "/job:worker/task:%d" % task_id])
        sess = sv.prepare_or_wait_for_session(
            server.target, config=sess_config)

        sess.run(init)

        iterations = x_train_deep.shape[0] // batch_size
        for ep in range(total_epoch):
            print('epoch', ep)
            if ep == 5:
                start = time.time()
            ep_st = time.time()
            train_loss = []
            train_acc = []
            pre_index = 0

            for it in range(iterations):
                batch_x_deep = x_train_deep[pre_index:pre_index + batch_size]
                batch_x_wide = x_train_wide[pre_index:pre_index + batch_size]
                batch_y = y_train[pre_index:pre_index + batch_size]
                pre_index += batch_size

                my_feed_dict = dict()
                for i in range(12):
                    my_feed_dict[X_deep[i]] = np.array(
                        batch_x_deep[:, 1]).reshape(-1, 1)

                my_feed_dict[X_wide] = np.array(batch_x_wide)
                my_feed_dict[y_] = batch_y
                loss_val = sess.run([loss, y, y_, train_op],
                                    feed_dict=my_feed_dict)
                acc_val = np.equal(
                    np.argmax(loss_val[1], 1),
                    np.argmax(loss_val[2], 1)).astype(np.float32)
                train_loss.append(loss_val[0])
                train_acc.append(acc_val)
            tra_accuracy = np.mean(train_acc)
            tra_loss = np.mean(train_loss)
            ep_en = time.time()
            print("train_loss: %.4f, train_acc: %.4f, train_time: %.4f"
                  % (tra_loss, tra_accuracy, ep_en - ep_st))
        print("tensorflow: ", (time.time() - start))


def test_bandwidth(cluster, task_id):
    print('test bandwidth')
    iters = 1000
    params_size = 128 * 9
    ps_device = "/job:ps/task:0/cpu:0"
    worker_device = "/job:worker/task:%d/cpu:0" % (task_id)

    with tf.device(ps_device):
        dtype = tf.int32
        params = tf.get_variable("params", shape=[params_size], dtype=dtype,
                                 initializer=tf.zeros_initializer())
    with tf.device(tf.compat.v1.train.replica_device_setter(
            worker_device=worker_device,
            cluster=cluster)):
        update = tf.get_variable("update", shape=[params_size], dtype=dtype,
                                 initializer=tf.ones_initializer())
        add_op = params.assign(update)

        server = tf.train.Server(
            cluster, job_name="worker", task_index=task_id)
        init = tf.global_variables_initializer()
        sv = tf.train.Supervisor(
            is_chief=(task_id == 0),
            init_op=init,
            recovery_wait_secs=1)
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=["/job:ps",
                            "/job:worker/task:%d" % task_id])
        sess = sv.prepare_or_wait_for_session(
            server.target, config=sess_config)

        sess.run(init)
        # warm up
        for i in range(5):
            sess.run(add_op.op)

        start_time = time.time()
        for i in range(iters):
            sess.run(add_op.op)
        elapsed_time = time.time() - start_time
        ans = float(iters)*(params_size / 1024 / 1024)/elapsed_time
        print("transfer rate: %f MB/s" % (ans))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="model to be tested")
    parser.add_argument("--rank", type=int, required=True,
                        help="rank of process")
    parser.add_argument(
        "--config", type=str, default='./settings/tf_dist_s1_w2.json', help="config file path")
    parser.add_argument("--val", action="store_true",
                        help="whether to use validation")
    parser.add_argument("--all", action="store_true",
                        help="whether to use all data")
    args = parser.parse_args()
    raw_model = args.model
    task_id = int(args.rank)
    raw_config = args.config

    config = json.load(open(raw_config))
    cluster = tf.train.ClusterSpec(config)

    if raw_model != 'band':
        import tf_models
        model = eval('tf_models.' + raw_model)
        dataset = raw_model.split('_')[-1]
        print('Model:', raw_model)
        if dataset == 'criteo':
            train_criteo(model, cluster, task_id, len(config['worker']), args)
        elif dataset == 'adult':
            # not support val or all
            train_adult(model, cluster, task_id, len(config['worker']))
        else:
            raise NotImplementedError
    else:
        test_bandwidth(cluster, task_id)


if __name__ == '__main__':
    main()
