from __future__ import absolute_import
import numpy as np
from .Node import Op
from .Conv2dReduceSum import conv2d_reducesum_op
from .ZerosLike import zeroslike_op
from ..gpu_links import broadcast_to


class Conv2d_BroadcastToOp(Op):
    def __init__(self, node_A, node_B, ctx=None):
        super().__init__(Conv2d_BroadcastToOp, [node_A, node_B], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            shapeW = input_vals[1].shape
            shapeW = list(shapeW)
            tmp = shapeW[1]
            shapeW[1] = shapeW[3]
            shapeW[3] = tmp
            output_val[:] = np.broadcast_to(
                input_vals[0].asnumpy(), input_vals[1].asnumpy().shape).swapaxes(1, 3)
        else:
            broadcast_to(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):

        grad_A = conv2d_reducesum_op(output_grad, ctx=self.raw_ctx)
        return [grad_A, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[1]


def conv2d_broadcastto_op(node_A, node_B, ctx=None):
    """Creates a node that represents np.broadcast_to(node_A, node_B.shape).

    Parameters:
    ----
    node_a : Node
        The Node to be bcast.
    node_b : Node
        Another Node with the target shape.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Conv2d_BroadcastToOp(node_A, node_B, ctx=ctx)
