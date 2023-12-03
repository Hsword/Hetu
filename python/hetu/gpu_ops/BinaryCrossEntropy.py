from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import binary_cross_entropy
from ..gpu_links import binary_cross_entropy_gradient


class BinaryCrossEntropyOp(Op):
    def __init__(self, prediction, label, ctx=None):
        super().__init__(BinaryCrossEntropyOp, [prediction, label], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            def make_valid(arr, val=-100):
                arr[np.isnan(arr)] = val
                arr[np.isinf(arr)] = val
                arr = np.maximum(arr, val)
                return arr
            epsilon = 1e-12
            y = input_vals[0].asnumpy()
            y_ = input_vals[1].asnumpy()
            # output_val[:] = -y_ * \
            #     make_valid(np.log(y)) - (1 - y_) * make_valid(np.log(1 - y))
            
            output_val[:] = -y_ * \
                np.log(y + epsilon) - (1 - y_) * np.log(1 - y + epsilon)
        else:
            binary_cross_entropy(
                input_vals[0], input_vals[1], output_val, stream_handle)

    def gradient(self, output_grad):

        grad_A = binarycrossentropy_gradient_op(
            self.inputs[0], self.inputs[1], output_grad, ctx=self.raw_ctx)
        grad_B = None
        return [grad_A, grad_B]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert len(input_shapes[0]) >= 2
        return input_shapes[0]


class BinaryCrossEntropyGradientOp(Op):
    def __init__(self, prediction, label, output_grad_node, ctx=None):
        super().__init__(BinaryCrossEntropyGradientOp, [
            prediction, label, output_grad_node], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            y = input_vals[0].asnumpy()
            y_ = input_vals[1].asnumpy()
            output_grad = input_vals[2].asnumpy()
            output_val[:] = (- y_/y + (1 - y_)/(1-y))*output_grad
        else:
            binary_cross_entropy_gradient(
                input_vals[0], input_vals[1], input_vals[2], output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[0]


def binarycrossentropy_op(node_A, node_B, ctx=None):
    """Computes cross entropy loss for pre-softmax activations.

    Parameters:
    ----
    node_A : Node
        Predicted probability.
    node_B : Node
        Labels.

    Returns:
    ----
    A new Node instance created by Op.

    """

    return BinaryCrossEntropyOp(node_A, node_B, ctx=ctx)


def binarycrossentropy_gradient_op(node_A, node_B, node_C, ctx=None):

    return BinaryCrossEntropyGradientOp(node_A, node_B, node_C, ctx=ctx)
