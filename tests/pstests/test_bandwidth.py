import hetu as ht

import time
import os
import sys
import yaml
import multiprocessing
import argparse
import signal
import numpy as np
import ctypes
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
import threading


def pointer(arr):
    assert(arr.data.c_contiguous)
    assert(arr.dtype == np.long)
    return ctypes.cast(arr.ctypes.data, ctypes.POINTER(ctypes.c_long))


def test(func_name, nitem=2000, item_len=10000, ind_len=500, max_thread=10, ret_ans=False):
    func_name = func_name.lower()
    ctx = ht.cpu(0)
    rank = int(os.environ["WORKER_ID"])
    nrank = int(os.environ["DMLC_NUM_WORKER"])

    comm = ht.get_worker_communicate()
    byte_count = 0
    if func_name == 'pushnpull':
        inarr = ht.array(np.random.rand(nitem, item_len), ctx=ctx)
        outarr = ht.array(np.random.rand(nitem, item_len), ctx=ctx)

        def func(name):
            comm.Push(name, inarr.handle, None)
            comm.Pull(name, outarr.handle)
            comm.Wait(name)
            nonlocal byte_count
            byte_count += nitem * item_len * 4 * 2
    elif func_name == 'pushpull':
        inarr = ht.array(np.random.rand(nitem, item_len), ctx=ctx)
        outarr = ht.array(np.random.rand(nitem, item_len), ctx=ctx)

        def func(name):
            comm.DDPushPull(name, inarr.handle, outarr.handle, None)
            comm.Wait(name)
            nonlocal byte_count
            byte_count += nitem * item_len * 4 * 2
    elif func_name == 'sparsepushnpull':
        inarr = ht.array(np.random.rand(ind_len, item_len), ctx=ctx)
        outarr = ht.array(np.random.rand(nitem, item_len), ctx=ctx)

        def func(name):
            np_ind = np.random.randint(low=0, high=nitem, size=(ind_len,))
            inind = ht.array(np_ind.astype(np.float32), ctx=ctx)
            uni_ind_len = np.unique(np_ind).size
            comm.SparsePush(name, inind.handle, inarr.handle, None)
            comm.Pull(name, outarr.handle)
            comm.Wait(name)
            nonlocal byte_count
            byte_count += (nitem + uni_ind_len) * item_len * 4
    elif func_name == 'sparsepushnsparsepull':
        inarr = ht.array(np.random.rand(ind_len, item_len), ctx=ctx)
        outarr = ht.array(np.random.rand(ind_len, item_len), ctx=ctx)

        def func(name):
            np_inind = np.random.randint(low=0, high=nitem, size=(ind_len,))
            np_outind = np.random.randint(low=0, high=nitem, size=(ind_len,))
            inind = ht.array(np_inind.astype(np.float32), ctx=ctx)
            outind = ht.array(np_outind.astype(np.float32), ctx=ctx)
            uni_inind_len = np.unique(np_inind).size
            uni_outind_len = np.unique(np_outind).size
            comm.SparsePush(name, inind.handle, inarr.handle, None)
            comm.SparsePull(name, outind.handle, outarr.handle)
            comm.Wait(name)
            nonlocal byte_count
            byte_count += (uni_inind_len + uni_outind_len) * item_len * 4
    elif func_name == 'push':
        inarr = ht.array(np.random.rand(nitem, item_len), ctx=ctx)

        def func(name):
            comm.Push(name, inarr.handle, None)
            comm.Wait(name)
            nonlocal byte_count
            byte_count += nitem * item_len * 4
    elif func_name == 'pull':
        outarr = ht.array(np.random.rand(nitem, item_len), ctx=ctx)

        def func(name):
            comm.Pull(name, outarr.handle)
            comm.Wait(name)
            nonlocal byte_count
            byte_count += nitem * item_len * 4
    elif func_name == 'sparsepush':
        inarr = ht.array(np.random.rand(ind_len, item_len), ctx=ctx)

        def func(name):
            np_inind = np.random.randint(low=0, high=nitem, size=(ind_len,))
            inind = ht.array(np_inind.astype(np.float32), ctx=ctx)
            uni_inind_len = np.unique(np_inind).size
            comm.SparsePush(name, inind.handle, inarr.handle, None)
            comm.Wait(name)
            nonlocal byte_count
            byte_count += uni_inind_len * item_len * 4
    elif func_name == 'sparsepull':
        outarr = ht.array(np.random.rand(ind_len, item_len), ctx=ctx)

        def func(name):
            np_outind = np.random.randint(low=0, high=nitem, size=(ind_len,))
            outind = ht.array(np_outind.astype(np.float32), ctx=ctx)
            uni_outind_len = np.unique(np_outind).size
            comm.SparsePull(name, outind.handle, outarr.handle)
            comm.Wait(name)
            nonlocal byte_count
            byte_count += uni_outind_len * item_len * 4
    elif func_name == 'sdpushpull':
        inarr = ht.array(np.random.rand(ind_len, item_len), ctx=ctx)
        outarr = ht.array(np.random.rand(nitem, item_len), ctx=ctx)

        def func(name):
            np_inind = np.random.randint(low=0, high=nitem, size=(ind_len,))
            inind = ht.array(np_inind.astype(np.float32), ctx=ctx)
            uni_inind_len = np.unique(np_inind).size
            comm.SDPushPull(name, inind.handle, inarr.handle,
                            outarr.handle, None)
            comm.Wait(name)
            nonlocal byte_count
            byte_count += (uni_inind_len + nitem) * item_len * 4
    elif func_name == 'sspushpull':
        inarr = ht.array(np.random.rand(ind_len, item_len), ctx=ctx)
        outarr = ht.array(np.random.rand(ind_len, item_len), ctx=ctx)

        def func(name):
            np_inind = np.random.randint(low=0, high=nitem, size=(ind_len,))
            np_outind = np.random.randint(low=0, high=nitem, size=(ind_len,))
            inind = ht.array(np_inind.astype(np.float32), ctx=ctx)
            uni_inind_len = np.unique(np_inind).size
            outind = ht.array(np_outind.astype(np.float32), ctx=ctx)
            uni_outind_len = np.unique(np_outind).size
            comm.SSPushPull(name, inind.handle, inarr.handle,
                            outind.handle, outarr.handle, None)
            comm.Wait(name)
            nonlocal byte_count
            byte_count += (uni_inind_len + uni_outind_len) * item_len * 4
    else:
        assert False
    if 'sparse' in func_name or func_name in ('sdpushpull', 'sspushpull'):
        arr_len = ctypes.c_int(nitem)
        arr_wid = ctypes.c_int(item_len)
        sparse_init = ctypes.c_int(1)
    else:
        arr_len = ctypes.c_int(nitem * item_len)
        arr_wid = ctypes.c_int(1)
        sparse_init = ctypes.c_int(0)
    for i in range(max_thread):
        comm.InitTensor(i, sparse_init, arr_len, arr_wid, ctypes.c_int(0), ctypes.c_double(0), ctypes.c_double(1), ctypes.c_ulonglong(123),
                        ctypes.c_int(0), (ctypes.c_float * 1)(0.1), ctypes.c_int(1))
    t = ThreadPoolExecutor(max_workers=max_thread)
    if ret_ans:
        task_list = [None for i in range(max_thread)]
        for i in range(max_thread):
            task_list[i] = t.submit(func, i)
        curByte = byte_count
        start = time.time()
        cnt = 0
        while cnt < 30:
            for i in range(max_thread):
                if task_list[i].done():
                    cnt += 1
                    task_list[i] = t.submit(func, i)
        speed = (byte_count - curByte) / (time.time() - start) / 2 ** 20
        t.shutdown()
        for i in range(max_thread):
            comm.ClearOnServer(i)
            comm.Clear(i)
        return speed
    else:
        def watch():
            start = time.time()
            while True:
                time.sleep(1)
                speed = byte_count / (time.time() - start)
                print("speed : {} MB/s".format(speed / 2**20))
        task_list = [None for i in range(max_thread)]
        threading.Thread(target=watch).start()
        while True:
            for i in range(max_thread):
                if task_list[i] is None or task_list[i].done():
                    task_list[i] = t.submit(func, i)


def test_dense_n_draw(range_size, func, trial=5, use_text=False):
    assert func in ('pushpull', 'push', 'pull', 'pushnpull')
    assert trial >= 3
    ans = {}
    for i in tqdm(range_size):
        temps = []
        for _ in range(trial):
            temps.append(test(func, i, 1, ret_ans=True))
        temps.remove(max(temps))
        temps.remove(min(temps))
        ans[i] = sum(temps) / (trial - 2)
    print(ans)
    import matplotlib.pyplot as plt
    xs = list(ans.keys())
    ys = list(ans.values())
    plt.bar(xs, ys, width=range_size.step // 2)
    plt.xlabel('Data Size')
    plt.ylabel('Bandwidth MB/s')
    plt.title('Bandwidth of ' + func)
    if use_text:
        for xx, yy in zip(xs, ys):
            plt.text(xx, yy + 20, '%.0f' % yy, ha='center', va='bottom')
    plt.savefig('test_dense_bandwidth.png')


def test_sparse_n_draw(range_ind_len, range_item_len, func, trial=5, use_text=False):
    assert func in ('sparsepush', 'sparsepull')
    assert trial >= 3
    ans = {}
    for i in tqdm(range_ind_len):
        for j in range_item_len:
            nitem = 5 * i
            temps = []
            for _ in range(trial):
                temps.append(test(func, nitem, j, i, ret_ans=True))
            temps.remove(max(temps))
            temps.remove(min(temps))
            ans[(i, j)] = sum(temps) / (trial - 2)
    print(ans)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    xs, ys = [], []
    for k in ans.keys():
        xs.append(k[0])
        ys.append(k[1])
    ax = plt.subplot(111, projection='3d')
    zs = list(ans.values())
    ax.bar3d([xx - range_ind_len.step // 4 for xx in xs], [yy - range_item_len.step //
                                                           4 for yy in ys], np.zeros_like(xs), range_ind_len.step // 2, range_item_len.step // 2, zs)
    if use_text:
        for xx, yy, zz in zip(xs, ys, zs):
            ax.text(xx, yy, zz, '%.0f' % zz, ha='center', va='bottom')
    ax.set_xlabel('Index Size')
    ax.set_ylabel('Item Length')
    ax.set_zlabel('Bandwidth MB/s')
    ax.set_title('Bandwidth of ' + func)
    plt.savefig('test_sparse_bandwidth.png')


def start_process(settings, args):
    for key, value in settings.items():
        os.environ[key] = str(value)
    if os.environ['DMLC_ROLE'] == "server":
        ht.server_init()
        ht.server_finish()
    elif os.environ['DMLC_ROLE'] == "worker":
        ht.worker_init()
        test(args.func)
        # test_dense_n_draw(range(100000, 1000000, 100000), 'pushpull')
        # test_sparse_n_draw(range(100, 600, 100), range(1000, 6000, 1000), 'sparsepush')
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
    parser.add_argument("--config", default='./local_s2_w1.yml')
    parser.add_argument("--func", default='pushpull')
    args = parser.parse_args()
    assert args.func in ('pushpull', 'pushnpull', 'sparsepushnpull', 'sparsepushnsparsepull',
                         'push', 'pull', 'sparsepush', 'sparsepull', 'sdpushpull', 'sspushpull')
    file_path = args.config
    settings = yaml.load(open(file_path).read(), Loader=yaml.FullLoader)
    process_list = []
    for key, value in settings.items():
        if key != 'shared':
            proc = multiprocessing.Process(
                target=start_process, args=[value, args])
            process_list.append(proc)
            proc.start()
    signal.signal(signal.SIGINT, signal_handler)
    for proc in process_list:
        proc.join()
