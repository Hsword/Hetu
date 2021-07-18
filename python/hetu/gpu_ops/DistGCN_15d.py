from __future__ import absolute_import
from .Node import Op
import math
from .. import ndarray
import numpy as np
from ctypes import *


def row_num(node_count, rank, size):
    n_per_proc = math.ceil(float(node_count) / size)
    if (node_count % size == 0):
        return int(node_count / size)
    if (rank < size - 1):
        return int(n_per_proc)
    else:
        return int(node_count % n_per_proc)


def broad_func(node_count, adj_matrix, inputs, rank, size, replication, row_groups, col_groups, ctx, comm=None, stream_handle=None):
    assert size % (replication ** 2) == 0

    n_per_proc = math.ceil(float(node_count) / (size // replication))
    proc_node_count = row_num(
        node_count, rank//replication, size // replication)

    z_loc = ndarray.empty((proc_node_count, inputs.shape[1]), ctx=ctx)
    tmp = ndarray.empty((proc_node_count, inputs.shape[1]), ctx=ctx)
    inputs_recv = ndarray.empty((int(n_per_proc), inputs.shape[1]), ctx=ctx)

    rank_c = rank // replication
    rank_col = rank % replication

    stages = size // (replication ** 2)
    node_count_col = stages * n_per_proc
    if rank_col == replication - 1:
        stages = (size // replication) - (replication - 1) * stages
        node_count_col = node_count - (replication - 1) * node_count_col

    start_pos = list(range(0, int(node_count_col), int(n_per_proc)))
    end_pos = start_pos[1:]+[int(node_count_col)]

    for i in range(stages):
        q = (rank_col * (size // (replication ** 2)) + i) * \
            replication + rank_col
        q_c = q // replication

        if q_c == size // replication - 1:
            inputs_recv = ndarray.empty((row_num(
                node_count, size//replication - 1, size//replication), inputs.shape[1]), ctx=ctx)
        if q == rank:
            inputs.copyto(inputs_recv)

        from ..communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t
        if replication > 1:
            col_groups[rank_col].dlarrayBroadcast(
                inputs_recv, inputs_recv, ncclDataType_t.ncclFloat32, q)
        else:
            comm.dlarrayBroadcast(inputs_recv, inputs_recv,
                                  ncclDataType_t.ncclFloat32, q)

        from ..gpu_links import CuSparse_Csrmm, matrix_elementwise_add
        CuSparse_Csrmm(adj_matrix, False, inputs_recv, False, tmp,
                       stream=stream_handle, start_pos=int(start_pos[i]), end_pos=int(end_pos[i]))
        matrix_elementwise_add(z_loc, tmp, z_loc, stream_handle)

    if replication > 1:
        row_groups[rank_c].dlarrayNcclAllReduce(
            z_loc, z_loc, ncclDataType_t.ncclFloat32, reduceop=ncclRedOp_t.ncclSum)

    return z_loc


class DistGCN_15dOp(Op):
    def __init__(self, node_A, node_B, node_C, node_Count_Self, node_Count_All, size, replication, device_id, comm, comm_groups=[None, None], need_W=True):
        super().__init__(DistGCN_15dOp, [
            node_A, node_B, node_C], ctx=ndarray.gpu(device_id))
        self.need_W = need_W
        self.node_Count_Self = node_Count_Self
        self.node_Count_All = node_Count_All
        self.replication = replication
        self.size = size
        self.comm = comm
        self.comm_groups = comm_groups
        self.device_id = device_id

    def compute(self, input_vals, output_val, stream_handle=None):
        adj_matrix = input_vals[0]
        inputs_H = input_vals[1]
        weight = input_vals[2]
        node_count = self.node_Count_All
        comm = self.comm
        rank = comm.localRank.value
        ctx = ndarray.gpu(self.device_id)

        if weight.shape[1] < inputs_H.shape[1]:
            HW = ndarray.empty((inputs_H.shape[0], weight.shape[1]), ctx=ctx)
            if (self.need_W == True):
                from ..gpu_links import matrix_multiply
                matrix_multiply(inputs_H, False, weight,
                                False, HW, stream_handle)
            else:
                HW = inputs_H
            z = broad_func(node_count, adj_matrix, HW, rank, self.size, self.replication,
                           row_groups=self.comm_groups[0], col_groups=self.comm_groups[1], ctx=ctx, comm=comm, stream_handle=stream_handle)
            z.copyto(output_val)
        else:
            AH = broad_func(node_count, adj_matrix, inputs_H, rank, self.size, self.replication,
                            row_groups=self.comm_groups[0], col_groups=self.comm_groups[1], ctx=ctx, comm=comm, stream_handle=stream_handle)
            z = ndarray.empty((AH.shape[0], weight.shape[1]), ctx=ctx)
            if (self.need_W == True):
                from ..gpu_links import matrix_multiply
                matrix_multiply(AH, False, weight, False, z, stream_handle)
            else:
                z = AH
            z.copyto(output_val)

    def gradient(self, output_grad):
        adj_matrix = self.inputs[0]
        inputs_H = self.inputs[1]
        weight = self.inputs[2]
        node_Count_Self = self.node_Count_Self
        node_Count_All = self.node_Count_All
        comm = self.comm
        rank = comm.localRank.value
        ag = distgcn_15d_op(adj_matrix, output_grad, weight, node_Count_Self, node_Count_All,
                            self.size, self.replication, self.device_id, comm, self.comm_groups, need_W=False)

        from . import matmul_op
        grad_A = None
        grad_H = matmul_op(ag, weight, trans_B=True)
        grad_weight = matmul_op(inputs_H, ag, trans_A=True)
        from . import groupallreduceCommunicate_op
        if self.replication > 1:
            weight_groups = self.comm_groups[1]
            if len(self.comm_groups) == 3:
                weight_groups = self.comm_groups[2]
            grad_W = groupallreduceCommunicate_op(
                grad_weight, weight_groups[rank % self.replication])
        else:
            grad_W = groupallreduceCommunicate_op(grad_weight, comm)
        return [grad_A, grad_H, grad_W]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        H = input_shapes[1]
        W = input_shapes[2]
        shape_H = H[1]
        shape_W = W[1]
        if (self.need_W == True):
            return (self.node_Count_Self, shape_W)
        else:
            return (self.node_Count_Self, shape_H)


def distgcn_15d_op(node_A, node_B, node_C, node_Count_Self, node_Count_All, size, replication, device_id, comm, comm_groups=[None, None], need_W=True):
    return DistGCN_15dOp(node_A, node_B, node_C, node_Count_Self, node_Count_All, size, replication, device_id, comm, comm_groups, need_W=need_W)
