from __future__ import absolute_import
from .Node import Op
import numpy as np
from ..gpu_links import concatenate
from ..gpu_links import concatenate_gradient


class ConcatenateOp(Op):
    def __init__(self, node_list, axis=0, ctx=None):
        super().__init__(ConcatenateOp, list(node_list), ctx)
        self.axis = axis

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.concatenate(
                [val.asnumpy() for val in input_vals], self.axis)
        else:
            concatenate(input_vals, output_val, self.axis, stream_handle)

    def gradient(self, output_grad):
        self.grad_nodes = [concatenate_gradient_op(
            output_grad, node, self.axis, ctx=self.raw_ctx) for node in self.inputs]
        return self.grad_nodes

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == len(self.inputs)
        deduce_grads = hasattr(self, 'grad_nodes')
        out_shape = list(input_shapes[0])
        if self.axis < 0:
            self.axis = self.axis % len(out_shape)
            if deduce_grads:
                for n in self.grad_nodes:
                    n.axis = self.axis
        out_dim = out_shape[self.axis]
        ind = 0
        if deduce_grads:
            self.grad_nodes[ind].offset = 0
            ind += 1
        for shape in input_shapes[1:]:
            assert len(shape) == len(out_shape)
            for i, dim in enumerate(shape):
                if i != self.axis:
                    assert dim == out_shape[i]
                else:
                    if deduce_grads:
                        self.grad_nodes[ind].offset = out_dim
                        ind += 1
                    out_dim += dim
        out_shape[self.axis] = out_dim
        return tuple(out_shape)


class Concatenate_gradientOP(Op):
    def __init__(self, grad_node, input_node, axis, ctx=None):
        super().__init__(Concatenate_gradientOP, [grad_node, input_node], ctx)
        self.axis = axis
        self.offset = None  # determined in forward node's infer_shape

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            width = output_val.shape[self.axis]
            slices = tuple(slice(None) for _ in range(self.axis)) + \
                (slice(self.offset, self.offset+width),)
            output_val[:] = input_vals[0].asnumpy()[slices]
        else:
            concatenate_gradient(input_vals[0], output_val,
                                 self.axis, self.offset, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert self.offset is not None
        assert len(input_shapes) == 2
        return input_shapes[1]


def concatenate_op(node_list, axis=0, ctx=None):
    return ConcatenateOp(node_list, axis, ctx=ctx)


def concatenate_gradient_op(grad_node, input_node, axis, ctx=None):
    return Concatenate_gradientOP(grad_node, input_node, axis, ctx=ctx)
