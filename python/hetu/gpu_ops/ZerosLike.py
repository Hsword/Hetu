from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import array_set
from ..cpu_links import array_set as cpu_array_set


class ZerosLikeOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(ZerosLikeOp, [node_A], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['cpu_ArraySet']:
                cpu_array_set(output_val, 0)
            else:
                output_val[:] = np.zeros(input_vals[0].asnumpy().shape)
        else:
            array_set(output_val, 0, stream_handle)

    def gradient(self, output_grad):
        return [zeroslike_op(self.inputs[0], ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def zeroslike_op(node, ctx=None):
    """Creates a node that represents np.zeros(node_A.shape).

    Parameters:
    ----
    node : Node
        The Node to pad with 0.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ZerosLikeOp(node, ctx=ctx)
