from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import conv2d_reduce_sum


class Conv2d_ReduceSumOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(Conv2d_ReduceSumOp, [node_A], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.sum(input_vals[0].asnumpy(), axis=(0, 2, 3))
        else:
            conv2d_reduce_sum(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        from .Conv2dBroadcast import conv2d_broadcastto_op
        return [conv2d_broadcastto_op(output_grad, self.inputs[0], ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        """summation reduction axis = 0
        e.g. (3,4,5)->(4,5)
        for vector, simpler to do (3,)->(1,)
        """
        assert len(input_shapes) == 1
        channels = input_shapes[0][1]
        return (channels,)


def conv2d_reducesum_op(node, ctx=None):
    """Creates a node that represents np.sum(node_A, axis=0). 
    Only support common-case axis=0 reduction for simplicity of gradient.

    Parameters:
    ----
    node : Node
        The Node needed to be summed.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Conv2d_ReduceSumOp(node, ctx=ctx)
