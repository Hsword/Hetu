from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import binary_cross_entropy_with_logits
from ..gpu_links import binary_cross_entropy_with_logits_gradient


class BinaryCrossEntropyWithLogitsOp(Op):
    def __init__(self, inputs, targets, ctx=None):
        super().__init__(BinaryCrossEntropyWithLogitsOp,
                         [inputs, targets], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            y = input_vals[0].asnumpy()
            y_ = input_vals[1].asnumpy()
            max_val = np.maximum(-y, 0)
            output_val[:] = (1 - y_) * y + max_val + \
                np.log(np.exp(-max_val) + np.exp(-y - max_val))
        else:
            binary_cross_entropy_with_logits(
                input_vals[0], input_vals[1], output_val, stream_handle)

    def gradient(self, output_grad):

        grad_A = binarycrossentropywithlogits_gradient_op(
            self.inputs[0], self.inputs[1], output_grad, ctx=self.raw_ctx)
        grad_B = None
        return [grad_A, grad_B]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]


class BinaryCrossEntropyWithLogitsGradientOp(Op):
    def __init__(self, prediction, label, output_grad_node, ctx=None):
        super().__init__(BinaryCrossEntropyWithLogitsGradientOp, [
            prediction, label, output_grad_node], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            y = input_vals[0].asnumpy()
            y_ = input_vals[1].asnumpy()
            output_grad = input_vals[2].asnumpy()
            output_val[:] = output_grad * (1 / (1 + np.exp(-y)) - y_)
        else:
            binary_cross_entropy_with_logits_gradient(
                input_vals[0], input_vals[1], input_vals[2], output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        assert input_shapes[0] == input_shapes[1] == input_shapes[2]
        return input_shapes[0]


def binarycrossentropywithlogits_op(inputs, targets, ctx=None):
    """Computes binary cross entropy loss for logits.

    Parameters:
    ----
    inputs : Node
        Input logits.
    targets : Node
        Labels.

    Returns:
    ----
    A new Node instance created by Op.

    """

    return BinaryCrossEntropyWithLogitsOp(inputs, targets, ctx=ctx)


def binarycrossentropywithlogits_gradient_op(node_A, node_B, node_C, ctx=None):

    return BinaryCrossEntropyWithLogitsGradientOp(node_A, node_B, node_C, ctx=ctx)
