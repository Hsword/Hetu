from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import reduce_min


class ReduceMinOp(Op):
    def __init__(self, node_A, axes, keepdims=False, ctx=None):
        super().__init__(ReduceMinOp, [node_A], ctx)
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
        if self.naive_copy:
            input_vals[0].copyto(output_val)
        else:
            if self.on_cpu:
                if all(self.keepdims) or not any(self.keepdims):
                    output_val[:] = np.amin(input_vals[0].asnumpy(), axis=tuple(
                        self.axes), keepdims=self.keepdims[0])
                else:
                    temp = input_vals[0].asnumpy()
                    for i in range(len(self.keepdims))[::-1]:
                        temp = np.amin(
                            temp, self.axes[i], keepdims=self.keepdims[i])
                    output_val[:] = temp
            else:
                reduce_min(input_vals[0], output_val, self.axes, stream_handle)

    def gradient(self, output_grad):
        # TODO:
        # 1. implement real gradient
        # 2. implement handler for None output_grad
        # 3. implement ReduceBaseOp to contain all reduce ops
        return [None]

    def infer_shape(self, input_shapes):
        assert self.axes is not None and self.keepdims is not None
        assert len(input_shapes) == 1
        input_shape = list(input_shapes[0])
        for i in range(len(self.axes)):
            if self.axes[i] < 0:
                self.axes[i] += len(input_shape)
            assert 0 <= self.axes[i] < len(input_shape)
            input_shape[self.axes[i]] = 1 if self.keepdims[i] else 0
        input_shape = [x for x in input_shape if x > 0]
        if input_shape == []:
            result = (1,)
        else:
            result = tuple(input_shape)
        from_size = np.prod(input_shapes[0], dtype=int)
        to_size = np.prod(result, dtype=int)
        self.naive_copy = (from_size == to_size)
        return result


def reduce_min_op(node, axes, keepdims=False, ctx=None):
    """Creates a node that represents np.amin(node_A, axis, keepdims).

    Parameters:
    ----
    node : Node
        The Node needed to be multiplied.
    axes : int or list
        The axis/axes needed to be multiplied.
    keepdims: bool or list
        Whether to keep the dimension(s).

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ReduceMinOp(node, axes, keepdims, ctx=ctx)
