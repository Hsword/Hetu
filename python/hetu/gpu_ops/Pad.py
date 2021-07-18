from __future__ import absolute_import
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import pad as cpu_pad
from ..cpu_links import pad_gradient as cpu_pad_gradient
from ..gpu_links import pad
from ..gpu_links import pad_gradient


class PadOp(Op):
    def __init__(self, node_A, paddings, mode="CONSTANT", constant_values=0, ctx=None):
        """Creates a node that represents np.sum(node_A, axis=0).
        Only support common-case axis=0 reduction for simplicity of gradient.
        """
        super().__init__(PadOp, [node_A], ctx)
        self.paddings = paddings
        self.mode = mode
        self.constant_values = constant_values

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['cpu_Pad']:
                cpu_pad(input_vals[0], output_val,
                        self.paddings, self.mode, constant_values=0)
            else:
                output_val[:] = pad_np(input_vals[0].asnumpy(
                ), self.paddings, self.mode, constant_values=0)
        else:
            pad(input_vals[0], output_val, self.paddings,
                self.mode, self.constant_values, stream_handle)

    def gradient(self, output_grad):
        return [pad_gradient_op(output_grad, self.paddings, self.mode, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        out_shape = list(input_shapes[0])
        pad_len = len(self.paddings)
        for i in range(4):
            if(i - (4 - pad_len) >= 0):
                out_shape[i] = out_shape[i] + self.paddings[i -
                                                            (4 - pad_len)][0] + self.paddings[i - (4 - pad_len)][1]
        return tuple(out_shape)


class Pad_GradientOp(Op):
    def __init__(self, node_A, paddings, mode="CONSTANT", ctx=None):
        """Creates a node that represents np.sum(node_A, axis=0).
        Only support common-case axis=0 reduction for simplicity of gradient.
        """
        super().__init__(Pad_GradientOp, [node_A], ctx)
        self.paddings = paddings
        self.mode = mode

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['cpu_Pad_Gradient']:
                cpu_pad_gradient(
                    input_vals[0], output_val, self.paddings, self.mode)
            else:
                output_val[:] = pad_np_gradient(
                    input_vals[0].asnumpy(), self.paddings)
        else:
            pad_gradient(input_vals[0], output_val,
                         self.paddings, self.mode, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        out_shape = list(input_shapes[0])
        pad_len = len(self.paddings)
        for i in range(4):
            if(i - (4 - pad_len) >= 0):
                out_shape[i] = out_shape[i] - self.paddings[i -
                                                            (4 - pad_len)][0] - self.paddings[i - (4 - pad_len)][1]
        return tuple(out_shape)


def pad_op(node_A, paddings, mode="CONSTANT", constant_values=0, ctx=None):
    """Pad an input variable.

    Parameters:
    ----
    node_A : Node
        The Node to be padded.
    paddings : Node
        padding edge
    mode :
        CONSTANT/REFLECT/SYMMETRIC
    constant_values: scalar value
        padding values

    Returns:
    ----
    A new Node instance created by Op.

    """
    return PadOp(node_A, paddings, mode, constant_values, ctx=ctx)


def pad_gradient_op(node_A, paddings, mode="CONSTANT", ctx=None):
    """Gradient node of pad operation.

    Parameters:
    ----
    node_A : Node
        The Node to be padded.
    paddings : Node
        padding edge
    mode :
        CONSTANT/REFLECT/SYMMETRIC

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Pad_GradientOp(node_A, paddings, mode, ctx=ctx)


def pad_np(node_A, paddings, mode="constant", constant_values=0):
    import numpy as np
    return np.pad(node_A, paddings, mode=mode.lower(), constant_values=(constant_values, constant_values))


def pad_np_gradient(grad, paddings):
    slices = []
    for c in paddings:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return grad[tuple(slices)]
