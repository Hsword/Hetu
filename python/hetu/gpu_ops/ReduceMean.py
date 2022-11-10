from __future__ import absolute_import
from functools import partial
import numpy as np
from .Node import Op
from ..gpu_links import reduce_mean


class ReduceMeanOp(Op):
    def __init__(self, node_A, axes, keepdims=False, ctx=None):
        super().__init__(ReduceMeanOp, [node_A], ctx)
        self.grad_node = None
        if axes is not None:
            if isinstance(axes, int):
                axes = [axes]
            self.axes = list(axes)
            assert all(map(lambda x: isinstance(x, int), self.axes))
        if keepdims is not None:
            if keepdims is True or keepdims is False:
                self.keepdims = [keepdims] * len(self.axes)
            else:
                keepdims = list(keepdims)
                assert len(keepdims) == len(self.axes)
                assert all(map(lambda x: isinstance(x, bool), keepdims))
                self.keepdims = keepdims

    def compute(self, input_vals, output_val, stream_handle=None):
        assert self.axes is not None and self.keepdims is not None
        if self.naive_copy:
            input_vals[0].copyto(output_val)
        else:
            if self.on_cpu:
                if all(self.keepdims) or not any(self.keepdims):
                    output_val[:] = np.mean(input_vals[0].asnumpy(), axis=tuple(
                        self.axes), keepdims=self.keepdims[0])
                else:
                    temp = input_vals[0].asnumpy()
                    for i in range(len(self.keepdims))[::-1]:
                        temp = np.mean(
                            temp, self.axes[i], keepdims=self.keepdims[i])
                    output_val[:] = temp
            else:
                reduce_mean(input_vals[0], output_val, self.axes, stream_handle)

    def gradient(self, output_grad):
        self.grad_set = False
        from .MultiplyConst import mul_byconst_op
        from .BroadcastShape import broadcast_shape_op
        # Here we don't know how to calculate gradient since we don't have shape information
        # The const is determined in infer_shape phase.
        self.grad_node = mul_byconst_op(broadcast_shape_op(
            output_grad, None, None, ctx=self.raw_ctx), None, ctx=self.raw_ctx)
        return [self.grad_node]

    def infer_shape(self, input_shapes):
        assert self.axes is not None and self.keepdims is not None
        assert len(input_shapes) == 1
        input_shape = list(input_shapes[0])
        mean_multiplier = 1
        for i in range(len(self.axes)):
            if self.axes[i] < 0:
                self.axes[i] += len(input_shape)
            assert 0 <= self.axes[i] < len(input_shape)
            mean_multiplier *= input_shape[self.axes[i]]
            input_shape[self.axes[i]] = 1 if self.keepdims[i] else 0
        if self.grad_node is not None:
            self.grad_node.const_attr = 1.0 / mean_multiplier
            self.grad_node.inputs[0].target_shape = tuple(input_shapes[0])
            add_axes = []
            for i in range(len(self.axes)):
                if not self.keepdims[i]:
                    add_axes.append(self.axes[i])
            self.grad_node.inputs[0].add_axes = add_axes
        input_shape = [x for x in input_shape if x > 0]
        if input_shape == []:
            result = (1,)
        else:
            result = tuple(input_shape)
        from_size = np.prod(input_shapes[0], dtype=int)
        to_size = np.prod(result, dtype=int)
        self.naive_copy = (from_size == to_size)
        return result

    def reset_status(self):
        self.grad_set = False


def reduce_mean_op(node, axes, keepdims=False, ctx=None):
    """Creates a node that represents np.mean(node_A, axis, keepdims).

    Parameters:
    ----
    node : Node
        The Node needed to be averaged.
    axes : int or list
        The axis/axes needed to be averaged.
    keepdims: bool or list
        Whether to keep the dimension(s).

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ReduceMeanOp(node, axes, keepdims, ctx=ctx)
