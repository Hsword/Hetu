from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import one_hot


class OneHotOp(Op):
    def __init__(self, node_A, num_classes, ctx=None):
        super().__init__(OneHotOp, [node_A], ctx)
        self.num_classes = num_classes

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            inputs = input_vals[0].asnumpy().astype(np.int)
            res = np.eye(self.num_classes)[inputs.reshape(-1)]
            output_val[:] = res.reshape(
                list(inputs.shape) + [self.num_classes]).astype(np.float32)
        else:
            one_hot(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return tuple(list(input_shapes[0]) + [self.num_classes])


def one_hot_op(node, num_classes, ctx=None):
    """Creates a node that represents one hot.

    Parameters:
    ----
    node : Node
        The input Node.
    num_classes: int
        Number of classes.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return OneHotOp(node, num_classes, ctx=ctx)
