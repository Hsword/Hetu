from __future__ import absolute_import
from .Node import Op
import numpy as np
from .._base import DNNL_LIB
from ..gpu_links import clamp


class ClampOp(Op):
    def __init__(self, node_A, node_B=None, node_C=None, mmin=None, mmax=None, ctx=None):
        if (node_B and node_C):
            super().__init__(ClampOp, [node_A, node_B, node_C], ctx)
        elif (node_B and not node_C):
            super().__init__(ClampOp, [node_A, node_B], ctx)
        elif (not node_B and node_C):
            super().__init__(ClampOp, [node_A, node_C], ctx)
        else:
            super().__init__(ClampOp, [node_A], ctx)
        self.mmin = mmin
        self.mmax = mmax
        self.min_mat = True if node_B else False
        self.max_mat = True if node_C else False

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            assert False
        else:
            if (self.min_mat and self.max_mat):
                clamp(input_vals[0], output_val, self.mmin, self.mmax,
                      input_vals[1], input_vals[2], stream_handle)
            elif (self.min_mat and not self.max_mat):
                clamp(input_vals[0], output_val, self.mmin,
                      self.mmax, input_vals[1], None, stream_handle)
            elif (not self.min_mat and self.max_mat):
                clamp(input_vals[0], output_val, self.mmin,
                      self.mmax, None, input_vals[1], stream_handle)
            elif (not self.min_mat and not self.max_mat):
                clamp(input_vals[0], output_val, self.mmin,
                      self.mmax, None, None, stream_handle)

    def gradient(self, output_grad):
        raise NotImplemented

    def infer_shape(self, input_shapes):
        return input_shapes[0]


def clamp_op(node_A, min_mat=None, max_mat=None, min=None, max=None, ctx=None):
    """Make a new instance of ClampOp and call the instance.

    Parameters:
    ----
    node_A : Node
        Input node.
    min_mat : Node
        Input node.
    max_mat : Node
        Input node.           
    min : Scalar value
    max : Scalar value    

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ClampOp(node_A, min_mat, max_mat, min, max, ctx=ctx)
