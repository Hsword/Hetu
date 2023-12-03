from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import sparse_set


class SparseSetOp(Op):
    def __init__(self, table, ind, data, ctx=None):
        super().__init__(SparseSetOp, [table, ind, data], ctx)
        assert table.dtype == ind.dtype == data.dtype == np.int32

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            sparse_set(input_vals[0], input_vals[1],
                       input_vals[2], stream_handle)

    def gradient(self, output_grad):
        return [None, None, None]

    def infer_shape(self, input_shapes):
        return None


def sparse_set_op(table, ind, data, ctx=None):
    return SparseSetOp(table, ind, data, ctx=ctx)
