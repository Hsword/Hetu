from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import repeat, repeat_gradient


class RepeatOp(Op):
    def __init__(self, node_A, reps, ctx=None):
        super().__init__(RepeatOp, [node_A], ctx)
        self.reps = reps

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            inputs = input_vals[0].asnumpy()
            output_val[:] = np.tile(inputs, self.reps)
        else:
            repeat(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [repeat_gradient_op(self.inputs[0], output_grad, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        ndim = len(input_shapes[0])
        nrep = 1 if isinstance(self.reps, int) else len(self.reps)

        input_shape = input_shapes[0]
        reps = self.reps
        output_shape = []
        if ndim < nrep:
            input_shape = [1 for i in range(nrep)]
            for i in range(ndim):
                input_shape[nrep-dim+i] = input_shapes[0][i]

        elif ndim > nrep:
            reps = [1 for i in range(ndim)]
            for i in range(nrep):
                reps[ndim-nrep+i] = self.reps[i]
        for i in range(len(input_shape)):
            output_shape.append(input_shape[i]*reps[i])
        return tuple(output_shape)


class Repeat_GradientOp(Op):
    def __init__(self, node_input, node_grad, ctx=None):
        super().__init__(Repeat_GradientOp, [node_input, node_grad], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            repeat_gradient(input_vals[1], output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[0]


def repeat_op(node, reps, ctx=None):
    """Returns the repeated input.

    Parameters:
    ----
    node : Node
        Input node.
    reps : Int or List
        Dim to repeat.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return RepeatOp(node, reps, ctx=ctx)


def repeat_gradient_op(node_input, node_grad, ctx=None):
    """Returns the gradient of repeat op.

    Parameters:
    ----
    node : Node
        Input node.
    node_grad : Node
        Grad node.        

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Repeat_GradientOp(node_input, node_grad, ctx=ctx)
