import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from time import time

import hetu as ht
from hetu import stream
from hetu import cpu_links as cpu_op
from hetu import gpu_links as gpu_op


def test_normal(size, mean=0, std=1):
    ctx = ht.gpu(0)
    cuda_x = ht.empty(size, ctx=ctx)
    stre = stream.create_stream_handle(ctx)
    np_st = time()
    for i in range(10):
        x = np.random.normal(loc=mean, scale=std, size=size).astype(np.float32)
        cuda_x[:] = x
    np_en = time()
    print('numpy time: ', np_en - np_st)
    cu_st = time()
    for i in range(10):
        gpu_op.normal_init(cuda_x, mean, std, 123, stre)
    stre.sync()
    cu_en = time()
    print('cuda time: ', cu_en - cu_st)
    fig, ax = plt.subplots(1, 1)
    cuda_x = cuda_x.asnumpy()
    assert (cuda_x.shape == x.shape)
    ax.hist(x.flatten(), histtype='stepfilled',
            alpha=0.2, bins=50, label='numpy')
    ax.hist(cuda_x.flatten(), histtype='step',
            alpha=0.2, bins=50, label='cuda')
    ax.legend(loc='best', frameon=False)
    file_name = 'normal_%f_%f.png' % (mean, std)
    plt.savefig(file_name)
    plt.close()


test_normal((1024, 128), 0, 1)
test_normal((1024, 128), 4.5, 2.6)
test_normal((1024, 128), -2.6, 4.5)
test_normal((1024, 128, 128), -10, 9)


def test_uniform(size, lb=-1, ub=1):
    ctx = ht.gpu(0)
    cuda_x = ht.empty(size, ctx=ctx)
    stre = stream.create_stream_handle(ctx)
    np_st = time()
    for i in range(10):
        x = np.random.uniform(low=lb, high=ub, size=size).astype(np.float32)
        cuda_x[:] = x
    np_en = time()
    print('numpy time: ', np_en - np_st)
    cu_st = time()
    for i in range(10):
        gpu_op.uniform_init(cuda_x, lb, ub, 123, stre)
    stre.sync()
    cu_en = time()
    print('cuda time: ', cu_en - cu_st)
    fig, ax = plt.subplots(1, 1)
    cuda_x = cuda_x.asnumpy()
    assert (cuda_x.shape == x.shape)
    ax.hist(x.flatten(), histtype='stepfilled',
            alpha=0.2, bins=50, label='numpy')
    ax.hist(cuda_x.flatten(), histtype='step',
            alpha=0.2, bins=50, label='cuda')
    ax.legend(loc='best', frameon=False)
    file_name = 'uniform_%f_%f.png' % (lb, ub)
    plt.savefig(file_name)
    plt.close()


test_uniform((1024, 128), 0, 1)
test_uniform((1024, 128), -100, 100)
test_uniform((1024, 128), -4.5, -4.4)
test_uniform((1024, 128, 128), -10, 9)


def test_truncated_normal(size, mean=0, std=1):
    ctx = ht.gpu(0)
    cuda_x = ht.empty(size, ctx=ctx)
    stre = stream.create_stream_handle(ctx)
    np_st = time()
    for i in range(10):
        x = truncnorm.rvs(-2.0, 2.0, loc=mean, scale=std,
                          size=size).astype(np.float32)
        cuda_x[:] = x
    np_en = time()
    print('numpy time: ', np_en - np_st)
    cu_st = time()
    for i in range(10):
        gpu_op.truncated_normal_init(cuda_x, mean, std, 123, stre)
    stre.sync()
    cu_en = time()
    print('cuda time: ', cu_en - cu_st)
    fig, ax = plt.subplots(1, 1)
    cuda_x = cuda_x.asnumpy()
    assert (cuda_x.shape == x.shape)
    ax.hist(x.flatten(), histtype='stepfilled',
            alpha=0.2, bins=50, label='numpy')
    ax.hist(cuda_x.flatten(), histtype='step',
            alpha=0.2, bins=50, label='cuda')
    ax.legend(loc='best', frameon=False)
    file_name = 'truncated_normal_%f_%f.png' % (mean, std)
    plt.savefig(file_name)
    plt.close()


test_truncated_normal((1024, 128), 0, 1)
test_truncated_normal((1024, 128), 4.5, 2.6)
test_truncated_normal((1024, 128), -2.6, 4.5)
test_truncated_normal((1024, 128, 128), -10, 9)


def test_cpu_normal(size, mean=0, std=1):
    cpu_x = ht.empty(size, ctx=ht.cpu(0))
    np_st = time()
    for i in range(10):
        x = np.random.normal(loc=mean, scale=std, size=size).astype(np.float32)
        cpu_x[:] = x
    np_en = time()
    print('numpy time: ', np_en - np_st)
    cpu_st = time()
    for i in range(10):
        cpu_op.normal_init(cpu_x, mean, std, 123)
    cpu_en = time()
    print('cpu time: ', cpu_en - cpu_st)
    fig, ax = plt.subplots(1, 1)
    cpu_x = cpu_x.asnumpy()
    assert (cpu_x.shape == x.shape)
    ax.hist(x.flatten(), histtype='stepfilled',
            alpha=0.2, bins=50, label='numpy')
    ax.hist(cpu_x.flatten(), histtype='step', alpha=0.2, bins=50, label='cpu')
    ax.legend(loc='best', frameon=False)
    file_name = 'normal_%f_%f_cpu.png' % (mean, std)
    plt.savefig(file_name)
    plt.close()


test_cpu_normal((1024, 128), 0, 1)
test_cpu_normal((1024, 128), 4.5, 2.6)
test_cpu_normal((1024, 128), -2.6, 4.5)
test_cpu_normal((1024, 128, 128), -10, 9)


def test_cpu_uniform(size, lb=-1, ub=1):
    cpu_x = ht.empty(size, ctx=ht.cpu(0))
    np_st = time()
    for i in range(10):
        x = np.random.uniform(low=lb, high=ub, size=size).astype(np.float32)
        cpu_x[:] = x
    np_en = time()
    print('numpy time: ', np_en - np_st)
    cpu_st = time()
    for i in range(10):
        cpu_op.uniform_init(cpu_x, lb, ub, 123)
    cpu_en = time()
    print('cpu time: ', cpu_en - cpu_st)
    fig, ax = plt.subplots(1, 1)
    cpu_x = cpu_x.asnumpy()
    assert (cpu_x.shape == x.shape)
    ax.hist(x.flatten(), histtype='stepfilled',
            alpha=0.2, bins=50, label='numpy')
    ax.hist(cpu_x.flatten(), histtype='step', alpha=0.2, bins=50, label='cpu')
    ax.legend(loc='best', frameon=False)
    file_name = 'uniform_%f_%f_cpu.png' % (lb, ub)
    plt.savefig(file_name)
    plt.close()


test_cpu_uniform((1024, 128), 0, 1)
test_cpu_uniform((1024, 128), -100, 100)
test_cpu_uniform((1024, 128), -4.5, -4.4)
test_cpu_uniform((1024, 128, 128), -10, 9)


def test_cpu_truncated_normal(size, mean=0, std=1):
    cpu_x = ht.empty(size, ctx=ht.cpu(0))
    np_st = time()
    for i in range(10):
        x = truncnorm.rvs(-2.0, 2.0, loc=mean, scale=std,
                          size=size).astype(np.float32)
        cpu_x[:] = x
    np_en = time()
    print('numpy time: ', np_en - np_st)
    cpu_st = time()
    for i in range(10):
        cpu_op.truncated_normal_init(cpu_x, mean, std, 123)
    cpu_en = time()
    print('cpu time: ', cpu_en - cpu_st)
    fig, ax = plt.subplots(1, 1)
    cpu_x = cpu_x.asnumpy()
    assert (cpu_x.shape == x.shape)
    ax.hist(x.flatten(), histtype='stepfilled',
            alpha=0.2, bins=50, label='numpy')
    ax.hist(cpu_x.flatten(), histtype='step', alpha=0.2, bins=50, label='cpu')
    ax.legend(loc='best', frameon=False)
    file_name = 'truncated_normal_%f_%f.png' % (mean, std)
    plt.savefig(file_name)
    plt.close()


test_cpu_truncated_normal((1024, 128), 0, 1)
test_cpu_truncated_normal((1024, 128), 4.5, 2.6)
test_cpu_truncated_normal((1024, 128), -2.6, 4.5)
test_cpu_truncated_normal((1024, 128, 128), -10, 9)
