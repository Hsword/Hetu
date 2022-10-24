from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import uniform_init as cpu_uniform_init
from ..gpu_links import uniform_init


class RandOp(Op):
    def __init__(self, size, ctx=None):
        super().__init__(RandOp, [], ctx)
        self.size = size
        self.seed = 0

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['cpu_UniformInit']:
                cpu_uniform_init(output_val, 0.0, 1.0, self.seed)
            else:
                output_val[:] = np_rand.uniform(
                    0.0, 1.0, size=output_val.shape).astype(np.float32)
        else:
            uniform_init(output_val, 0.0, 1.0, self.seed, stream_handle)

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 0
        return self.size


def rand_op(size, ctx=None):
    """Rand init.

    Parameters:
    ----
    size : List
        Output size.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return RandOp(size, ctx=ctx)
