import os
import tensorflow as tf
import multiprocessing
import signal
import json
import argparse


def pop_env():
    for k in ['https_proxy', 'http_proxy']:
        if k in os.environ:
            os.environ.pop(k)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


pop_env()


def start_server(cluster, task_id):
    server = tf.train.Server(cluster, job_name='ps', task_index=task_id)
    server.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default='./settings/tf_dist_s4_w2.json', help="config file path")
    parser.add_argument("--id", type=int, required=True)
    args = parser.parse_args()
    raw_config = args.config
    config = json.load(open(raw_config))
    cluster = tf.train.ClusterSpec(config)
    global proc
    proc = multiprocessing.Process(
        target=start_server, args=[cluster, args.id, ])
    proc.start()
    signal.signal(signal.SIGINT, signal_handler)
    proc.join()


def signal_handler(signal, frame):
    print("SIGINT signal caught, stop Training")
    global proc
    proc.kill()
    exit(0)


if __name__ == '__main__':
    main()
