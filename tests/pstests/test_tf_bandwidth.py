import tensorflow as tf
import numpy as np
import argparse
import os
import time
import json
import multiprocessing
import signal


def pop_env():
    for k in ['https_proxy', 'http_proxy']:
        if k in os.environ:
            os.environ.pop(k)


pop_env()


def launch_server(cluster, task_id):
    server = tf.train.Server(cluster, job_name='ps', task_index=task_id)
    server.join()


def test_bandwidth(cluster, task_id):
    print('test bandwidth')
    iters = 1000
    params_size = 128 * 100
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


def signal_handler(signal, frame):
    print("SIGINT signal caught, stop Training")
    for ps in server_list:
        ps.kill()
    for worker in worker_list:
        worker.kill()
    exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default='./tf_local_s1_w2.json', help="config file path")
    args = parser.parse_args()

    config = json.load(open(args.config))
    print(config)
    # exit()
    cluster = tf.train.ClusterSpec(config)

    for i, ps in enumerate(config['ps']):
        proc = multiprocessing.Process(target=launch_server, args=[cluster, i])
        server_list.append(proc)
        proc.start()
    for i, worker in enumerate(config['worker']):
        proc = multiprocessing.Process(
            target=test_bandwidth, args=[cluster, i])
        worker_list.append(proc)
        proc.start()
    signal.signal(signal.SIGINT, signal_handler)
    for proc in worker_list:
        proc.join()
    for ps in server_list:
        ps.kill()


if __name__ == '__main__':
    server_list = []
    worker_list = []
    main()
