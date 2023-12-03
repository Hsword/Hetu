from __future__ import absolute_import
import numpy as np
from .Node import Op
from .ReduceSum import reduce_sum_op
from .Reshape import array_reshape_op
from ..gpu_links import broadcast_shape_simple
from .. import ndarray


class TileOp(Op):
    def __init__(self, node, reps, ctx=None):
        super().__init__(TileOp, [node], ctx)
        self.reps = tuple(reps)
        self.grad_node = None
        self.middle_result = None

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.tile(input_vals[0].asnumpy(), self.reps)
        else:
            broadcast_shape_simple(
                input_vals[0], output_val, self.out_strides, self.in_dims, stream_handle)

    def gradient(self, output_grad):
        self.middle_result = array_reshape_op(output_grad, None)
        self.grad_node = reduce_sum_op(
            self.middle_result, None, None, ctx=self.raw_ctx)
        return [self.grad_node]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        new_shape = list(input_shape)
        ndim = len(input_shape)
        nrep = len(self.reps)
        diff = nrep - ndim
        if diff < 0:
            self.reps = (1,) * (-diff) + self.reps
        elif diff > 0:
            new_shape = [1, ] * (diff) + new_shape
        self.as_shape = list(new_shape)
        output_ndim = len(new_shape)
        in_dims = list(new_shape)
        labels = [False for _ in range(output_ndim)]
        for i in range(diff):
            labels[i] = True
        for i, r in list(enumerate(self.reps))[::-1]:
            new_shape[i] *= r
            if r > 1:
                if i < diff:
                    self.as_shape[i] = r
                else:
                    self.as_shape.insert(i, r)
                    in_dims.insert(i, 1)
                    labels.insert(i, True)
        output_shape = tuple(new_shape)
        if self.grad_node is not None:
            add_axes = []
            for i, label in enumerate(labels):
                if label:
                    add_axes.append(i)
            self.grad_node.axes = tuple(add_axes)
            self.grad_node.keepdims = [False] * len(add_axes)
        # here we save the output strides and input dimensions for GPU computation
        if self.on_gpu:
            out_strides = [0 for _ in range(len(self.as_shape))]
            temp_size = 1
            for i in range(len(self.as_shape) - 1, -1, -1):
                out_strides[i] = temp_size
                temp_size *= self.as_shape[i]
            self.out_strides = ndarray.array(
                out_strides, self.ctx, dtype=np.int32)
            self.in_dims = ndarray.array(in_dims, self.ctx, dtype=np.int32)
        if self.middle_result is not None:
            self.middle_result.output_shape = self.as_shape
        return output_shape


def tile_op(node, reps, ctx=None):
    """Creates a node that represents np.tile(node, reps).

    Parameters:
    ----
    node : Node
        The Node to be broadcast.
    reps : tuple
        Target repetitions.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return TileOp(node, reps, ctx=ctx)
