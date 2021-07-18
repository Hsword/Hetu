import numpy as np
import scipy.sparse as sp
import math
import hetu as ht
from hetu.context import DeviceGroup
from hetu.communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t


def convert_to_context(device_list):
    return DeviceGroup([ht.gpu(x) for x in device_list])


def test_default():
    comm1 = ht.new_group_comm()
    a = ht.array(np.array([1, 2, 3, 4, 5]), ctx=ctx)
    comm1.dlarrayNcclAllReduce(
        a, a, ncclDataType_t.ncclFloat32, reduceop=ncclRedOp_t.ncclSum)
    print("Default Allreduce device=%d" % comm1.dev_id, a.asnumpy())


def test_broadcast(group, root):
    device_group = convert_to_context(group)
    comm1 = ht.new_group_comm(device_group)
    a = ht.array(np.array([-1, -1, -1, -1, -1]), ctx=ctx)
    if rank == root:
        a = ht.array(np.array([2, 3, 4, 5, 6]), ctx=ctx)
    if rank in group:
        comm1.dlarrayBroadcast(
            a, a, ncclDataType_t.ncclFloat32, root=device_group.index(ht.gpu(root)))
    print("Broadcast device=%d" % comm1.dev_id, a.asnumpy())


def test_allreduce(group):
    device_group = convert_to_context(group)
    comm1 = ht.new_group_comm(device_group)
    a = ht.array(np.array([1, 2, 3, 4, 5]), ctx=ctx)
    if rank in group:
        comm1.dlarrayNcclAllReduce(
            a, a, ncclDataType_t.ncclFloat32, reduceop=ncclRedOp_t.ncclSum)
    print("Allreduce device=%d" % comm1.dev_id, a.asnumpy())


def test_allgather(group):
    device_group = convert_to_context(group)
    comm1 = ht.new_group_comm(device_group)
    a = ht.array(np.array([rank, rank]), ctx=ctx)
    b = ht.array(np.zeros(2*len(group)), ctx=ctx)
    if rank in group:
        comm1.dlarrayAllGather(a, b, ncclDataType_t.ncclFloat32)
    print("Allgather device=%d" % comm1.dev_id, b.asnumpy())


def test_group_broadcast():
    row_procs = []
    for i in range(0, 8, 2):
        row_procs.append(list(range(i, i+2)))

    col_procs = []
    for i in range(2):
        col_procs.append(list(range(i, 8, 2)))

    row_groups = []
    for i in range(len(row_procs)):
        row_groups.append(ht.new_group_comm(convert_to_context(row_procs[i])))

    col_groups = []
    for i in range(len(col_procs)):
        col_groups.append(ht.new_group_comm(convert_to_context(col_procs[i])))

    rank_row = rank // 2
    rank_col = rank % 2
    group_row = row_procs[rank_row]
    group_col = col_procs[rank_col]
    comm_row = row_groups[rank_row]
    comm_col = col_groups[rank_col]

    a = ht.array(np.array([rank, rank, rank, rank, rank]), ctx=ctx)
    comm_row.dlarrayBroadcast(
        a, a, ncclDataType_t.ncclFloat32, root=1)
    print("Broadcast device=%d, a:" % device_id, a.asnumpy())

    b = ht.array(np.array([rank, rank, rank, rank, rank]), ctx=ctx)
    comm_col.dlarrayBroadcast(
        b, b, ncclDataType_t.ncclFloat32, root=1)
    print("Broadcast device=%d, b:" % device_id, b.asnumpy())


comm = ht.wrapped_mpi_nccl_init()
device_id = comm.dev_id
rank = comm.rank
size = comm.nrank
ctx = ht.gpu(rank)
a = ht.array(np.array([1, 2, 3, 4, 5]), ctx=ctx)

test_default()

test_broadcast(group=[0, 2, 4, 5, 6], root=4)
test_broadcast(group=[1, 4, 2, 7], root=4)
test_allreduce(group=[1, 4, 2, 5])
test_allreduce(group=[0, 7, 6, 2, 4])
test_allgather(group=[2, 5, 3, 7])
test_allgather(group=[2, 6, 1, 7, 4])

test_group_broadcast()
