from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import argmax_partial


# designed for MGQEmbedding
class ArgmaxPartialOp(Op):
    def __init__(self, node, use_full, topk, dim, ctx=None):
        assert dim > 0
        super().__init__(ArgmaxPartialOp, [node, use_full], ctx)
        self.topk = topk
        self.dim = dim
        assert use_full.dtype == np.int32
        self.dtype = np.int32

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            inputs = input_vals[0].asnumpy()
            full_mask = input_vals[1].asnumpy()
            inputs[(full_mask == 0,) + (slice(None),) * (self.dim-1) +
                   (slice(self.topk, None),)] = float('-inf')
            output_val[:] = np.argmax(inputs, axis=self.dim)
        else:
            argmax_partial(input_vals[0], input_vals[1],
                           output_val, self.dim, self.topk, stream_handle)

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert len(
            input_shapes[1]) == 1 and input_shapes[0][0] == input_shapes[1][0]
        if len(input_shapes[0]) == 1:
            return (1,)
        output_shapes = []
        for dim, value in enumerate(input_shapes[0]):
            if dim == self.dim:
                assert self.topk < value
                continue
            output_shapes.append(value)
        return tuple(output_shapes)


def argmax_partial_op(node, use_full, topk, dim, ctx=None):
    return ArgmaxPartialOp(node, use_full, topk, dim=dim, ctx=ctx)
