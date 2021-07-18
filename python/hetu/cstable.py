from hetu.gpu_ops.executor import path_to_lib
from hetu import ndarray
from hetu import get_worker_communicate

import numpy as np
import sys
import os
from functools import partial

"""
    CacheSparseTable:
        length, width: the length and width of the whole embedding table
        limit: the max number of embedding lines stored in cache
        node_id: the unique node_id in the model
        policy: cache policy, LRU or LFU
"""


class CacheSparseTable:
    def __init__(self, limit, length, width, node_id, policy="LRU", bound=100):
        # make sure we open libps.so first
        comm = get_worker_communicate()
        sys.path.append(os.path.dirname(__file__)+"/../../build/lib")
        import hetu_cache
        policy = policy.lower()
        if policy == "lru":
            self.cache = hetu_cache.LRUCache(limit, length, width, node_id)
        elif policy == "lfu":
            self.cache = hetu_cache.LFUCache(limit, length, width, node_id)
        elif policy == "lfuopt":
            self.cache = hetu_cache.LFUOptCache(limit, length, width, node_id)
        else:
            raise NotImplementedError(policy)
        self.cache.pull_bound = bound
        self.cache.push_bound = bound
        comm.BarrierWorker()

    """
        embedding_lookup:
            keys: a list of keys to lookup
            dest: target memory space to write to
            sync: async call of sync call
            if async, a wait_t is returned, use wait.wait() to wait until it finish.
            if async, must make sure keys and dest are alive throughout the call
    """

    def embedding_lookup(self, keys, dest, sync=False):
        wait = None
        if type(keys) is np.ndarray and type(dest) is np.ndarray:
            assert dest.shape == (keys.size, self.width)
            assert keys.dtype == np.uint64
            assert dest.dtype == np.float32
            wait = self.cache.embedding_lookup(keys, dest)
        elif type(keys) is ndarray.NDArray and type(dest) is ndarray.NDArray:
            assert dest.shape == (*keys.shape, self.width)
            assert not ndarray.is_gpu_ctx(keys.ctx)
            assert not ndarray.is_gpu_ctx(dest.ctx)
            wait = self.cache.embedding_lookup_raw(
                keys.handle.contents.data, dest.handle.contents.data, np.prod(keys.shape))
        else:
            raise TypeError
        if sync:
            wait.wait()
        else:
            return wait
    """
        embedding_lookup:
            keys: a list of keys to update
            grads: gradients to send
            sync: async call of sync call
            if async, a wait_t is returned, use wait.wait() to wait until it finish.
            if async, must make sure keys and dest are alive throughout the call
    """

    def embedding_update(self, keys, grads, sync=False):
        wait = None
        if type(keys) is np.ndarray and type(grads) is np.ndarray:
            assert grads.shape == (keys.size, self.width)
            assert keys.dtype == np.uint64
            assert grads.dtype == np.float32
            wait = self.cache.embedding_update(keys, grads)
        elif type(keys) is ndarray.NDArray and type(grads) is ndarray.NDArray:
            assert grads.shape == (*keys.shape, self.width)
            assert not ndarray.is_gpu_ctx(keys.ctx)
            assert not ndarray.is_gpu_ctx(grads.ctx)
            wait = self.cache.embedding_update_raw(
                keys.handle.contents.data, grads.handle.contents.data, np.prod(keys.shape))
        else:
            raise TypeError
        if sync:
            wait.wait()
        else:
            return wait

    def embedding_push_pull(self, pullkeys, dest, pushkeys, grads, sync=False):
        wait = None
        if type(pullkeys) is ndarray.NDArray and type(dest) is ndarray.NDArray and \
                type(pushkeys) is ndarray.NDArray and type(grads) is ndarray.NDArray:
            assert grads.shape == (*pushkeys.shape, self.width)
            assert dest.shape == (*pullkeys.shape, self.width)
            assert not ndarray.is_gpu_ctx(pullkeys.ctx)
            assert not ndarray.is_gpu_ctx(pushkeys.ctx)
            assert not ndarray.is_gpu_ctx(grads.ctx)
            assert not ndarray.is_gpu_ctx(dest.ctx)
            wait = self.cache.embedding_push_pull_raw(
                pullkeys.handle.contents.data, dest.handle.contents.data, np.prod(
                    pullkeys.shape),
                pushkeys.handle.contents.data, grads.handle.contents.data, np.prod(
                    pushkeys.shape)
            )
        else:
            raise TypeError
        if sync:
            wait.wait()
        else:
            return wait

    @property
    def width(self):
        return self.cache.width

    @property
    def limit(self):
        return self.cache.limit

    def perf_enabled(self, enable=True):
        self.cache.perf_enabled = enable

    @property
    def perf(self):
        # perf data example [item1, item2...]
        # item = "type": pull_or_push, "is_full": is_cache_full, "num_all", num_of_key
        # "num_unique": num_of_unique_key, "num_miss": num_of_missed_unique_key,
        # "num_evict": num_push_of_eviction, "num_transfered"(if push): miss+outofpushbound+evict
        # "num_transfered"(if pull): miss+outofpullbound, "time": last_time_in_ms
        return self.cache.perf

    # if bypass, directly pull and push the server
    def bypass(self):
        self.cache.bypass()

    def undobypass(self):
        self.cache.undo_bypass()

    def __repr__(self):
        return self.cache.__repr__()

    # the following calls are single key call
    # for debugging
    def lookup(self, key):
        return self.cache.lookup(key)

    def count(self, key):
        return self.cache.count(key)

    def insert(self, key, embedding):
        return self.cache.insert(key, embedding)

    def keys(self):
        return self.cache.keys()

    # PerfHelperFunction

    # miss rate for pull
    def overall_miss_rate(self, include_cold_start=False):
        if not include_cold_start:
            perf = list(filter(lambda x: x["is_full"], self.perf))
        else:
            perf = self.perf
        if not perf:
            return -1
        pull_perf = list(filter(lambda x: x["type"] == "Pull", perf))
        num_all = [x["num_unique"] for x in pull_perf]
        num_miss = [x["num_miss"] for x in pull_perf]
        return np.sum(num_miss) / np.sum(num_all)

    # data rate compared with vanilla sparse pull (ignore cost for idx&version)
    def overall_data_rate(self, include_cold_start=False):
        if not include_cold_start:
            perf = list(filter(lambda x: x["is_full"], self.perf))
        else:
            perf = self.perf
        if not perf:
            return -1
        num_all = [x["num_all"] for x in perf]
        num_miss = [x["num_transfered"] for x in perf]
        return np.sum(num_miss) / np.sum(num_all)

    def debug_keys(self):
        comm = get_worker_communicate()
        nrank = comm.nrank()
        form = "w" if comm.rank() == 0 else "a"
        for i in range(nrank):
            if i == comm.rank():
                with open("_keys.log".format(comm.rank()), form) as f:
                    print(*self.keys(), file=f, flush=True)
            comm.BarrierWorker()

        if comm.rank() != 0:
            return
        keys = []
        with open("_keys.log".format(comm.rank()), "r") as f:
            for i in range(nrank):
                keys.append(set(map(int, f.readline().split())))
        rt = np.zeros([nrank, nrank])
        for i in range(nrank):
            for j in range(nrank):
                if not keys[i]:
                    continue
                rt[i][j] = len(keys[i].intersection(keys[j])) / len(keys[i])
        return rt
