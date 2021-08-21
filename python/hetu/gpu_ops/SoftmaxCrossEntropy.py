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

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        if deduce_order:
            if input_statuses[0].valid_all():
                order = list(input_statuses[0].order)
                order.remove(len(order) - 2)
                status.set_order(tuple(order))
            elif input_statuses[1].valid_all():
                order = list(input_statuses[1].order)
                order.remove(len(order) - 2)
                status.set_order(tuple(order))
        else:
            if input_statuses[0].valid_state():
                state, duplicate = input_statuses[0].get()
                assert state[-1] == 1
                status.set_state(state[:-1], duplicate)
            elif input_statuses[1].valid_state():
                state, duplicate = input_statuses[1].get()
                assert state[-1] == 1
                status.set_state(state[:-1], duplicate)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        if deduce_order:
            pass
        else:
            if status.valid_state():
                state, duplicate = status.get()
                state += (1,)
                input_statuses[0].set_state(state, duplicate)
                input_statuses[1].set_state(state, duplicate)


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

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        if deduce_order:
            if input_statuses[0].valid_all():
                status.copy_order_from(input_statuses[0])
            elif input_statuses[1].valid_all():
                status.copy_order_from(input_statuses[1])
        else:
            if input_statuses[0].valid_state():
                status.copy_state_from(input_statuses[0])
            elif input_statuses[1].valid_state():
                status.copy_state_from(input_statuses[1])

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        if deduce_order:
            if status.valid_all():
                order = status.order
                input_statuses[0].set_order(order)
                input_statuses[1].set_order(order)
                order = list(order)
                order.remove(len(order) - 2)
                input_statuses[2].set_order(tuple(order))
        else:
            if status.valid_state():
                input_statuses[0].copy_state_from(status)
                input_statuses[1].copy_state_from(status)
                state, duplicate = status.get()
                assert state[-1] == 1
                input_statuses[2].set_state(state[:-1], duplicate)


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
