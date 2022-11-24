from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import outer


class OuterOp(Op):
    def __init__(self, input, vector, ctx=None):
        super().__init__(OuterOp, [input, vector], ctx)
        
    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.outer(input_vals[0].asnumpy(), input_vals[1].asnumpy())
        else:
            outer(input_vals[0], input_vals[1], output_val, stream_handle)
            
    def gradient(self, output_grad):
        return [None, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert len(input_shapes[0]) == 1
        assert len(input_shapes[1]) == 1
        return (input_shapes[0][0], input_shapes[1][0])


def outer_op(input, vector, ctx=None):
    """Takes the outer product of input and vector.

    Parameters:
    ----
    node : Node
        Input variable.
    vector : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return OuterOp(input, vector, ctx=ctx)


