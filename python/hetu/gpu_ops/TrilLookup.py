from __future__ import absolute_import
from .Node import Op
import numpy as np
from ..gpu_links import tril_lookup, tril_lookup_gradient


class TrilLookupOp(Op):
    def __init__(self, array, offset=0, ctx=None):
        super().__init__(TrilLookupOp, [array], ctx)
        self.offset = offset

    def compute(self, input_vals, output_val, stream_handle=None):
        input_val = input_vals[0]
        if self.on_cpu:
            input_val = input_val.asnumpy()
            ori_dim = input_val.shape[-1]
            tril_indices = np.tril_indices(ori_dim, k=self.offset)
            output_val[:] = input_val[..., tril_indices[0], tril_indices[1]]
        else:
            tril_lookup(input_val, output_val, self.offset, stream_handle)

    def gradient(self, output_grad):
        return [tril_lookup_gradient_op(output_grad, self.offset, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        assert len(input_shape) >= 2 and input_shape[-1] == input_shape[-2]
        ori_dim = input_shape[-1]
        border = ori_dim - 1
        assert -border <= self.offset <= border
        cur_dim = ori_dim * (ori_dim + 1) // 2
        offset = self.offset
        if offset > 0:
            size = border
            for _ in range(offset):
                cur_dim += size
                size -= 1
        elif offset < 0:
            size = ori_dim
            for _ in range(-offset):
                cur_dim -= size
                size -= 1
        output_shape = input_shape[:-2] + (cur_dim,)
        return tuple(output_shape)


class TrilLookupGradientOp(Op):
    def __init__(self, array, offset=0, ctx=None):
        super().__init__(TrilLookupGradientOp, [array], ctx)
        self.offset = offset

    def compute(self, input_vals, output_val, stream_handle=None):
        input_val = input_vals[0]
        if self.on_cpu:
            input_val = input_val.asnumpy()
            prev_dim = output_val.shape[-1]
            tril_indices = np.tril_indices(prev_dim, k=self.offset)
            result = np.zeros(output_val.shape, dtype=np.float32)
            result[..., tril_indices[0], tril_indices[1]] = input_val
            output_val[:] = result
        else:
            tril_lookup_gradient(input_val, output_val,
                                 self.offset, stream_handle)

    def gradient(self, output_grad):
        return [tril_lookup_op(output_grad, self.offset, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        assert len(input_shape) >= 1
        ori_dim = input_shape[-1]
        cur_dim = self.get_prev_dim(ori_dim)
        output_shape = input_shape[:-1] + (cur_dim, cur_dim, )
        return tuple(output_shape)

    def get_prev_dim(self, x):
        if self.offset > 0:
            res = round(
                np.sqrt(2 * self.offset * (self.offset + 1) + 2 * x + 0.25) - 0.5 - self.offset)
        else:
            res = round(
                np.sqrt(2 * x + 0.25) - 0.5 - self.offset)
        return res


def tril_lookup_op(array, offset=0, ctx=None):
    # given tensors with shape (..., n, n), output (..., n(n+1)/2)
    # return the lower triangle part of a tensor
    # combination of tril_indices and getitem
    return TrilLookupOp(array, offset=offset, ctx=ctx)


def tril_lookup_gradient_op(array, offset=0, ctx=None):
    return TrilLookupGradientOp(array, offset=offset, ctx=ctx)
