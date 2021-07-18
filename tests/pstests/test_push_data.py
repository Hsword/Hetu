import hetu as ht
import numpy as np

import argparse
import six.moves.cPickle as pickle
import gzip
import sys
import json
import os
import ctypes
import yaml
import multiprocessing
import signal


def start_process(settings):
    for key, value in settings.items():
        os.environ[key] = str(value)
    if os.environ['DMLC_ROLE'] == "server":
        ht.server_init()
        ht.server_finish()
    elif os.environ['DMLC_ROLE'] == "worker":
        ht.worker_init()
        test()
        ht.worker_finish()
    elif os.environ['DMLC_ROLE'] == "scheduler":
        ht.scheduler_init()
        ht.scheduler_finish()
    else:
        raise ValueError("Unknown role", os.environ['DMLC_ROLE'])


def signal_handler(signal, frame):
    print("SIGINT signal caught, stop Training")
    for proc in process_list:
        proc.kill()
    exit(0)


def pointer(arr):
    assert(arr.data.c_contiguous)
    assert(arr.dtype == np.long)
    return ctypes.cast(arr.ctypes.data, ctypes.POINTER(ctypes.c_long))


def test():
    ctx = ht.cpu(0)
    rank = int(os.environ["WORKER_ID"])
    nrank = int(os.environ["DMLC_NUM_WORKER"])
    arr = ht.array(np.random.rand(2, rank+100), ctx=ctx)
    print(arr.asnumpy())

    push_indices = np.array([2*rank+1, 2*rank+2])

    if rank == 0:
        pull_indices = np.array([3])
    elif rank == 1:
        pull_indices = np.array([1])

    push_length = np.array([rank+100, rank+100])

    if rank == 0:
        pull_length = np.array([101])
        out_arr = ht.array(np.zeros(101), ctx=ctx)
    elif rank == 1:
        pull_length = np.array([100])
        out_arr = ht.array(np.zeros(100), ctx=ctx)

    print(out_arr.asnumpy())

    worker_communicate = ht.get_worker_communicate()
    query = worker_communicate.PushData(
        pointer(push_indices), 2, arr.handle, pointer(push_length))

    worker_communicate.WaitData(query)

    worker_communicate.BarrierWorker()
    worker_communicate.PullData(
        pointer(pull_indices), 1, out_arr.handle, pointer(pull_length))
    worker_communicate.WaitData(query)

    print(out_arr.asnumpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='./local_s2_w2.yml')
    args = parser.parse_args()
    file_path = args.config
    settings = yaml.load(open(file_path).read(), Loader=yaml.FullLoader)
    process_list = []
    for key, value in settings.items():
        if key != 'shared':
            proc = multiprocessing.Process(
                target=start_process, args=[value, ])
            process_list.append(proc)
            proc.start()
    signal.signal(signal.SIGINT, signal_handler)
    for proc in process_list:
        proc.join()
