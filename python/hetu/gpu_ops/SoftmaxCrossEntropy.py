from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import softmax_crossentropy
from .Softmax import softmax_func
from ..gpu_links import CuDNN_softmax_cross_entropy
from ..gpu_links import softmax_cross_entropy
from ..gpu_links import CuDNN_softmax_cross_entropy_gradient
from ..gpu_links import softmax_cross_entropy_gradient


class SoftmaxCrossEntropyOp(Op):
    def __init__(self, node_A, node_B, use_cudnn=True, ctx=None):
        super().__init__(SoftmaxCrossEntropyOp, [node_A, node_B], ctx)
        self.use_cudnn = use_cudnn

    def compute(self, input_vals, output_val, stream_handle=None):
        y = input_vals[0]
        y_ = input_vals[1]
        if self.on_cpu:
            if DNNL_LIB['DnnlSoftmaxCrossEntropy']:
                softmax_crossentropy(y, y_, output_val)
            else:
                softmax = softmax_func(y.asnumpy())
                output_val[:] = -np.sum(y_.asnumpy() * np.log(softmax), axis=1)
        else:
            if self.use_cudnn:
                CuDNN_softmax_cross_entropy(y, y_, output_val, stream_handle)
            else:
                softmax_cross_entropy(y, y_, output_val, stream_handle)

    def gradient(self, output_grad):
        from .Softmax import softmax_op
        grad_A = softmaxcrossentropy_gradient_op(
            self.inputs[0], self.inputs[1], output_grad, use_cudnn=self.use_cudnn, ctx=self.raw_ctx)
        return [grad_A, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert len(input_shapes[0]) >= 2
        return input_shapes[0][:-1]


class SoftmaxCrossEntropyGradientOp(Op):
    def __init__(self, node_A, node_B, node_C, use_cudnn=True, ctx=None):
        super().__init__(SoftmaxCrossEntropyGradientOp,
                         [node_A, node_B, node_C], ctx)
        self.use_cudnn = use_cudnn

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlSoftmaxCrossEntropy_Gradient']:
                print('No support for DnnlSoftmaxCrossEntropy_gradient')
            else:
                output_val[:] = (softmax_func(input_vals[0].asnumpy(
                )) + -1 * input_vals[1].asnumpy()) * np.expand_dims(input_vals[2].asnumpy(), -1)
        else:
            if self.use_cudnn:
                CuDNN_softmax_cross_entropy_gradient(
                    input_vals[2], input_vals[0], input_vals[1], output_val, stream_handle)
            else:
                softmax_cross_entropy_gradient(
                    input_vals[0], input_vals[1], input_vals[2], output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[0]


def softmaxcrossentropy_op(node_A, node_B, use_cudnn=True, ctx=None):
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

    return SoftmaxCrossEntropyOp(node_A, node_B, use_cudnn=use_cudnn, ctx=ctx)


def softmaxcrossentropy_gradient_op(node_A, node_B, node_C, use_cudnn=True, ctx=None):
    return SoftmaxCrossEntropyGradientOp(node_A, node_B, node_C, use_cudnn=use_cudnn, ctx=ctx)
