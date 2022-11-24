from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import eye


class EyeOp(Op):
    def __init__(self, n, ctx=None):
        super().__init__(EyeOp, [], ctx)
        assert isinstance(n, int)
        self.n = n
        
    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.eye(self.n)
        else:
            eye(output_val, stream_handle)
            
    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 0
        return (self.n, self.n)


def eye_op(n, ctx=None):
    """Make a new instance of EyeOp and call the instance.

    Parameters:
    ----
    n : Int
        The number of rows.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return EyeOp(n, ctx=ctx)

