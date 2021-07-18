import hetu as ht

import time
import os
import sys
import yaml
import multiprocessing
from multiprocessing.sharedctypes import RawArray as rarr
import argparse
import signal
import numpy as np
from scipy.stats import truncnorm
import ctypes
import matplotlib.pyplot as plt

nitem = 2000
item_len = 1000
indx1 = 30
indx2 = 40


def test_init_ps(rarr, init_type, init_a, init_b=1.0, sparse=False):
    assert init_type in ('constant', 'uniform', 'normal', 'truncated_normal')
    init_type_map = {'constant': 0, 'uniform': 1,
                     'normal': 2, 'truncated_normal': 3}
    ctx = ht.cpu(0)
    rank = int(os.environ["WORKER_ID"])
    nrank = int(os.environ["DMLC_NUM_WORKER"])
    local_arr = np.frombuffer(rarr, dtype=np.float32).reshape(nitem, item_len)
    if rank == 0:
        arr = ht.array(local_arr, ctx=ctx)
    else:
        arr = ht.empty((nitem, item_len), ctx=ctx)
    comm = ht.get_worker_communicate()
    if sparse:
        arr_len = ctypes.c_int(nitem)
        arr_wid = ctypes.c_int(item_len)
    else:
        arr_len = ctypes.c_int(nitem * item_len)
        arr_wid = ctypes.c_int(1)
    itype = ctypes.c_int(init_type_map[init_type])
    comm.InitTensor(ctypes.c_int(0), ctypes.c_int(sparse), arr_len, arr_wid, itype, ctypes.c_double(
        init_a), ctypes.c_double(init_b), ctypes.c_ulonglong(123), ctypes.c_int(0), (ctypes.c_float * 1)(0.1), ctypes.c_int(1))

    comm.Pull(ctypes.c_int(0), arr.handle)
    comm.Wait(ctypes.c_int(0))
    if rank == 0:
        local_arr[:] = arr.asnumpy()
    comm.BarrierWorker()
    if rank != 0:
        np.testing.assert_allclose(local_arr, arr.asnumpy(), rtol=5e-7)
    else:
        if init_type == 'constant':
            np.testing.assert_allclose(
                np.full((nitem, item_len), init_a), arr.asnumpy(), rtol=5e-7)
        else:
            if init_type == 'uniform':
                numpy_samples = np.random.uniform(
                    low=init_a, high=init_b, size=(nitem, item_len)).astype(np.float32)
            elif init_type == 'normal':
                numpy_samples = np.random.normal(
                    loc=init_a, scale=init_b, size=(nitem, item_len)).astype(np.float32)
            else:
                numpy_samples = truncnorm.rvs(-2.0, 2.0, loc=init_a,
                                              scale=init_b, size=(nitem, item_len)).astype(np.float32)
            fig, ax = plt.subplots(1, 1)
            ax.hist(numpy_samples.flatten(), histtype='stepfilled',
                    alpha=0.2, bins=50, label='numpy')
            ax.hist(local_arr.flatten(), histtype='step',
                    alpha=0.2, bins=50, label='ps')
            ax.legend(loc='best', frameon=False)
            # ax2.legend(loc='best', frameon=False)
            file_name = '%s_%.1f_%.1f_%d.png' % (
                init_type, init_a, init_b, int(sparse))
            plt.savefig(file_name)
            print('Check file %s.' % file_name)
    print('Init parameters %d/%d passed.' % (rank, nrank))
    if rank == 0:
        comm.ClearOnServer(0)
    comm.Clear(0)
    comm.BarrierWorker()


def test_api(rarr, rpush, rpull, sparse=False, lr=0.5):
    ctx = ht.cpu(0)
    rank = int(os.environ["WORKER_ID"])
    nrank = int(os.environ["DMLC_NUM_WORKER"])
    local_arr = np.frombuffer(rarr, dtype=np.float32).reshape(
        nitem, item_len).copy()
    local_push = np.frombuffer(rpush, dtype=np.float32).copy()
    local_pull = np.frombuffer(rpull, dtype=np.float32).copy()
    if rank == 0:
        arr = ht.array(local_arr, ctx=ctx)
    else:
        arr = ht.empty((nitem, item_len), ctx=ctx)
    comm = ht.get_worker_communicate()
    if sparse:
        arr_len = ctypes.c_int(nitem)
        arr_wid = ctypes.c_int(item_len)
    else:
        arr_len = ctypes.c_int(nitem * item_len)
        arr_wid = ctypes.c_int(1)
    comm.InitTensor(ctypes.c_int(0), ctypes.c_int(sparse), arr_len, arr_wid, ctypes.c_int(0), ctypes.c_double(0.0), ctypes.c_double(1.0), ctypes.c_ulonglong(123),
                    ctypes.c_int(0), (ctypes.c_float * 1)(lr), ctypes.c_int(1))
    if sparse:
        local_arr[:] = 0
        for j in local_push:
            local_arr[int(j)] += 1
        if rank == 0:
            push_ind = ht.array(local_push.reshape(indx1, indx2), ctx=ctx)
            push_val = ht.array(
                np.ones((indx1, indx2, item_len)).astype(np.float32), ctx=ctx)
            comm.SparsePush(0, push_ind.handle, push_val.handle, None)
            comm.Wait(0)
        comm.BarrierWorker()
        comm.Pull(0, arr.handle)
        comm.Wait(0)
        np.testing.assert_allclose(local_arr, arr.asnumpy(), rtol=5e-7)
        print('SparsePush DensePull %d/%d passed.' % (rank, nrank))
        comm.BarrierWorker()

        for j in local_push:
            local_arr[int(j)] += 1
        if rank == 0:
            push_ind = ht.array(local_push.reshape(indx1, indx2), ctx=ctx)
            push_val = ht.array(
                np.ones((indx1, indx2, item_len)).astype(np.float32), ctx=ctx)
            comm.SDPushPull(0, push_ind.handle,
                            push_val.handle, arr.handle, None)
            comm.Wait(0)
        comm.BarrierWorker()
        if rank != 0:
            comm.Pull(0, arr.handle)
            comm.Wait(0)
        np.testing.assert_allclose(local_arr, arr.asnumpy(), rtol=5e-7)
        print('SDPushPull %d/%d passed.' % (rank, nrank))
        comm.BarrierWorker()

        for j in local_push:
            local_arr[int(j)] += 1
        pull_ind = ht.array(local_pull.reshape(indx1, indx2), ctx=ctx)
        pull_val = ht.empty((indx1, indx2, item_len), ctx=ctx)
        if rank == 0:
            push_ind = ht.array(local_push.reshape(indx1, indx2), ctx=ctx)
            push_val = ht.array(
                np.ones((indx1, indx2, item_len)).astype(np.float32), ctx=ctx)
            comm.SSPushPull(0, push_ind.handle, push_val.handle,
                            pull_ind.handle, pull_val.handle, None)
            comm.Wait(0)
        comm.BarrierWorker()
        if rank != 0:
            comm.SparsePull(0, pull_ind.handle, pull_val.handle)
            comm.Wait(0)
        np.testing.assert_allclose(local_arr[local_pull.astype(int)].reshape(
            indx1, indx2, item_len), pull_val.asnumpy(), rtol=5e-7)
        print('SSPushPull and SparsePull %d/%d passed.' % (rank, nrank))
        comm.BarrierWorker()

    else:
        if rank == 0:
            comm.Push(0, arr.handle, None)
            comm.Wait(0)
        comm.BarrierWorker()
        comm.Pull(0, arr.handle)
        comm.Wait(0)
        np.testing.assert_allclose(local_arr, arr.asnumpy(), rtol=5e-7)
        print('DensePush DensePull %d/%d passed.' % (rank, nrank))
        comm.BarrierWorker()
        if rank == 0:
            temp_push_val = ht.array(
                np.ones((nitem, item_len)).astype(np.float32), ctx=ctx)
            comm.DDPushPull(0, temp_push_val.handle, arr.handle, None)
            comm.Wait(0)
        comm.BarrierWorker()
        if rank != 0:
            comm.Pull(0, arr.handle)
            comm.Wait(0)
        np.testing.assert_allclose(local_arr + 1, arr.asnumpy())
        print('DenseDensePushPull %d/%d passed.' % (rank, nrank))
        comm.BarrierWorker()
    if rank == 0:
        comm.ClearOnServer(0)
    comm.Clear(0)
    comm.BarrierWorker()


def start_process(settings, args, arr=None, push_arr=None, pull_arr=None):
    for key, value in settings.items():
        os.environ[key] = str(value)
    if os.environ['DMLC_ROLE'] == "server":
        ht.server_init()
        ht.server_finish()
    elif os.environ['DMLC_ROLE'] == "worker":
        ht.worker_init()
        test_api(arr, push_arr, pull_arr)
        test_init_ps(arr, 'constant', 1234.567)
        test_init_ps(arr, 'uniform', -0.5, 0.4)
        test_init_ps(arr, 'normal', 5.6, 2.0)
        test_init_ps(arr, 'truncated_normal', -2.3, 1.4)
        test_api(arr, push_arr, pull_arr, True)
        test_init_ps(arr, 'constant', 1234.567, True)
        test_init_ps(arr, 'uniform', -0.5, 0.4, True)
        test_init_ps(arr, 'normal', 5.6, 2.0, True)
        test_init_ps(arr, 'truncated_normal', -2.3, 1.4, True)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='./local_s2_w2.yml')
    args = parser.parse_args()
    settings = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
    process_list = []
    arr = rarr('f', np.random.rand(nitem * item_len,).astype(np.float32))
    push_arr = rarr('f', np.random.randint(
        0, nitem, (indx1 * indx2)).astype(np.float32))
    pull_arr = rarr('f', np.random.randint(
        0, nitem, (indx1 * indx2)).astype(np.float32))
    for key, value in settings.items():
        if key != 'shared':
            if key[0] != 'w':
                proc = multiprocessing.Process(
                    target=start_process, args=[value, args])
            else:
                proc = multiprocessing.Process(target=start_process, args=[
                                               value, args, arr, push_arr, pull_arr])
            process_list.append(proc)
            proc.start()
    signal.signal(signal.SIGINT, signal_handler)
    for proc in process_list:
        proc.join()
