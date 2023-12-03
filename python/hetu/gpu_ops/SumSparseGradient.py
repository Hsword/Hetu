from __future__ import absolute_import
from .Node import Op
from ..cpu_links import matrix_elementwise_add as cpu_matrix_elementwise_add, \
    sparse_add_to_dense as cpu_sparse_add_to_dense
from ..gpu_links import sparse_add_to_dense, array_set,\
    matrix_elementwise_add_simple
import numpy as np


class SumSparseGradientOp(Op):
    def __init__(self, dense_shape, *pairs_or_denses, dtype=np.float32, ctx=None):
        self.dense_shape = dense_shape
        assert len(self.dense_shape) == 2
        self.is_sparse = []
        inputs = []
        for item in pairs_or_denses:
            if isinstance(item, tuple):
                inputs.extend(item)
                self.is_sparse.append(True)
            else:
                inputs.append(item)
                self.is_sparse.append(False)
        super().__init__(SumSparseGradientOp, inputs, ctx)
        self.dtype = dtype

    def compute(self, input_vals, output_val, stream_handle=None):
        cur_ind = 0
        if self.on_cpu:
            for sp in self.is_sparse:
                if sp:
                    cpu_sparse_add_to_dense(
                        input_vals[cur_ind], input_vals[cur_ind+1], output_val)
                    cur_ind += 2
                else:
                    cpu_matrix_elementwise_add(
                        input_vals[cur_ind], output_val, output_val)
                    cur_ind += 1
        else:
            array_set(output_val, 0, stream_handle)
            for sp in self.is_sparse:
                if sp:
                    sparse_add_to_dense(
                        input_vals[cur_ind], input_vals[cur_ind+1], output_val, stream_handle)
                    cur_ind += 2
                else:
                    matrix_elementwise_add_simple(
                        input_vals[cur_ind], output_val, output_val, stream_handle)
                    cur_ind += 1
        assert cur_ind == len(input_vals)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return self.dense_shape


def sum_sparse_gradient_op(dense_shape, *pairs_or_denses, dtype=np.float32, ctx=None):
    return SumSparseGradientOp(dense_shape, *pairs_or_denses, dtype=dtype, ctx=ctx)
