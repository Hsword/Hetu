from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import roll


class RollOp(Op):
    def __init__(self, node_A, shift=0, axis=None, ctx=None):
        super().__init__(RollOp, [node_A], ctx)
        self.shift = shift if (isinstance(shift, list)) else list([shift])
        self.axis = None
        if (axis != None):
            self.shift = axis if (isinstance(axis, list)) else list([axis])

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.roll(
                input_vals[0].asnumpy(), self.shift, self.axis)
        else:
            roll(input_vals[0], output_val,
                 self.shift, self.axis, stream_handle)

    def gradient(self, output_grad):
        shift = []
        for s in self.shift:
            shift.append(-s)
        return [roll_op(output_grad, shift, self.axis, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert (len(input_shapes) == 1)
        return input_shapes[0]


def roll_op(node, shift=0, axis=None, ctx=None):
    """Roll input along the given axes.

    Parameters:
    ----
    node : Node
        Input node.
    shift : Int or List
    axis : Int or List    

    Returns:
    ----
    A new Node instance created by Op.

    """
    return RollOp(node, shift, axis, ctx=ctx)
