from hetu import get_worker_communicate
from hetu.launcher import launch
from hetu.cstable import CacheSparseTable

import ctypes
import argparse
import numpy as np
from tqdm import tqdm


def test(args):
    comm = get_worker_communicate()
    node_id = 0
    limit = 10000
    length = 10000
    width = 128
    comm.InitTensor(ctypes.c_int(node_id), ctypes.c_int(2), ctypes.c_int(length), ctypes.c_int(width), ctypes.c_int(2), ctypes.c_double(0), ctypes.c_double(0.1), ctypes.c_ulonglong(123),
                    ctypes.c_int(0), (ctypes.c_float * 1)(0.1), ctypes.c_int(1))
    cache = CacheSparseTable(limit, length, width, node_id, "LFUOpt")
    for i in tqdm(range(10000)):
        key = np.random.randint(10000, size=1000).astype(np.uint64)
        value = np.empty((key.size, width), np.float32)
        ts = cache.embedding_lookup(key, value)
        ts.wait()
        grad = np.random.rand(key.size, width).astype(np.float32)
        ts = cache.embedding_update(key, grad)
        ts.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    launch(test, args)
