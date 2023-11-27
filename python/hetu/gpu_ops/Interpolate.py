from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import array_set, bicubic_interpolate, bicubic_interpolate_gradient


class InterpolateOp(Op):
    def __init__(self, input, size=None, scale_factor=None, mode='bicubic', align_corners=False, ctx=None):
        super().__init__(InterpolateOp, [input], ctx)
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        if (self.mode != 'bicubic'):
            assert False

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            bicubic_interpolate(
                input_vals[0], output_val, self.align_corners, stream_handle)

    def gradient(self, output_grad):
        return [interpolate_grad_op(output_grad, self.inputs[0], self.mode, self.align_corners, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        N, C, H, W = input_shapes[0]
        if self.size != None:
            assert len(self.size) == 2
            return (N, C) + self.size

        if isinstance(self.scale_factor, tuple):
            assert len(self.scale_factor) == 2
            return (N, C, int(H*self.scale_factor[0]), int(W*self.scale_factor[0]))
        else:
            return (N, C, int(H*self.scale_factor), int(W*self.scale_factor))


class InterpolateGradOp(Op):
    def __init__(self, grad, input, mode='bicubic', align_corners=False, ctx=None):
        super().__init__(InterpolateGradOp, [grad, input], ctx)
        self.mode = mode
        self.align_corners = align_corners
        if (self.mode != 'bicubic'):
            assert False

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            array_set(output_val, 0, stream_handle)
            bicubic_interpolate_gradient(
                input_vals[0], output_val, self.align_corners, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[1]


def interpolate_op(input, size=None, scale_factor=None, mode='bicubic', align_corners=False, ctx=None):
    """Down/up samples the input to either the given size or the given scale_factor.

    Parameters:
    ----
    input : Node
        Input node.
    Size : List
        Output size.
    scale_factor : float
        Multiplier for spatial size.
    mode : str
        Only support bicubic
    align_corners : bool

    Returns:
    ----
    A new Node instance created by Op.

    """
    return InterpolateOp(input, size, scale_factor, mode, align_corners, ctx=ctx)


def interpolate_grad_op(grad, input, mode='bicubic', align_corners=False, ctx=None):
    """Make a new instance of InterpolateGradOp and call the instance.

    Parameters:
    ----
    grad : Node
        Grad node.
    input : Node
        Input node.
    size : List
        Output size.
    scale_factor : float
        Multiplier for spatial size.
    mode : str
        Only support bicubic
    align_corners : bool

    Returns:
    ----
    A new Node instance created by Op.

    """
    return InterpolateGradOp(grad, input, mode, align_corners, ctx=ctx)