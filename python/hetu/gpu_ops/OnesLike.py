from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import array_set as cpu_array_set
from ..gpu_links import array_set


class OnesLikeOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(OnesLikeOp, [node_A], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['cpu_ArraySet']:
                cpu_array_set(output_val, 1)
            else:
                output_val[:] = np.ones(input_vals[0].shape)
        else:
            array_set(output_val, 1, stream_handle)

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def oneslike_op(node, ctx=None):
    """Creates a node that represents np.ones(node_A.shape).

    Parameters:
    ----
    node : Node
        The Node to pad with 1.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return OnesLikeOp(node, ctx=ctx)
