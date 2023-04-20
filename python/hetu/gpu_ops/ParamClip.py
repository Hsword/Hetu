from __future__ import absolute_import
from .Node import Op
from ..gpu_links import param_clip_func
import numpy as np


class ParamClipOp(Op):
    def __init__(self, param, control, min_value, max_value, ctx=None):
        super().__init__(ParamClipOp, [param, control], ctx)
        self.min_value = min_value
        self.max_value = max_value

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            values = input_vals[0].asnumpy()
            values = np.clip(values, self.min_value, self.max_value)
            input_vals[0][:] = values
        else:
            param_clip_func(input_vals[0], self.min_value,
                            self.max_value, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return None


def param_clip_op(param, control, min_value, max_value, ctx=None):
    return ParamClipOp(param, control, min_value, max_value, ctx=ctx)
