from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import matrix_power


class PowerOp(Op):
    def __init__(self, node, p, ctx=None):
        super().__init__(PowerOp, [node], ctx)
        self.p = p

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.power(input_vals[0].asnumpy(), self.p)
        else:
            matrix_power(input_vals[0], output_val, self.p, stream_handle)

    def gradient(self, output_grad):
        # TODO: handle gradients for for2back/back2for maps
        from .MultiplyConst import mul_byconst_op
        from .MultiplyElewise import mul_op
        if self.p == 0:
            return [None]
        elif self.p == 1:
            return [output_grad]
        elif self.p == 2:
            return [mul_byconst_op(mul_op(output_grad, self.inputs[0], ctx=self.raw_ctx), 2, ctx=self.raw_ctx)]
        else:
            return [mul_byconst_op(mul_op(power_op(self.inputs[0], self.p-1, ctx=self.raw_ctx), output_grad, ctx=self.raw_ctx), self.p, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def power_op(node, p, ctx=None):
    """Calculate the power of a matrix elementwisely.

    Parameters:
    ----
    node : Node
        Input variable.
    pow : float
        The power to be calculated.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return PowerOp(node, p, ctx=ctx)
