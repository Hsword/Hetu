from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import reduce_sum_axis_zero as cpu_reduce_sum_axis_zero
from ..gpu_links import reduce_sum_axis_zero


class ReduceSumAxisZeroOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(ReduceSumAxisZeroOp, [node_A], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:

            if DNNL_LIB['cpu_ReduceSumAxisZero']:
                cpu_reduce_sum_axis_zero(input_vals[0], output_val)
            else:
                output_val[:] = np.sum(input_vals[0].asnumpy(), axis=0)
        else:
            reduce_sum_axis_zero(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        from .Broadcast import broadcastto_op
        return [broadcastto_op(output_grad, self.inputs[0], ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        """summation reduction axis = 0
        e.g. (3,4,5)->(4,5)
        for vector, simpler to do (3,)->(1,)
        """
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        if len(input_shape) == 1:
            return (1,)
        else:
            return input_shape[1:]


def reducesumaxiszero_op(node, ctx=None):
    """Creates a node that represents np.sum(node_A, axis=0).

    Parameters:
    ----
    node : Node
        The Node needed to be summed.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ReduceSumAxisZeroOp(node, ctx=ctx)
