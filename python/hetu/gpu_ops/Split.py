from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import matrix_slice_simple
from ..gpu_links import matrix_slice_gradient_simple
from .. import ndarray


class SplitOp(Op):
    def __init__(self, node_A, axes, indices, splits, ctx=None):
        super().__init__(SplitOp, [node_A], ctx)
        self.axes = axes
        self.indices = indices
        self.splits = splits
        self.grad_node = None
        assert len(self.axes) == len(self.splits)
        assert all([x >= 0 for x in axes])
        assert all([x >= 1 for x in splits])
        assert all([x >= 0 and x < splits[i] for i, x in enumerate(indices)])

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            index = tuple([slice(i, i+j)
                           for i, j in zip(self.begin_pos, self.output_shape)])
            output_val[:] = input_vals[0].asnumpy()[index]
        else:
            # matrix_slice(input_vals[0], output_val, self.begin_pos, stream_handle)
            matrix_slice_simple(
                input_vals[0], output_val, self.gpu_buffer, stream_handle)

    def gradient(self, output_grad):
        self.grad_node = split_gradient_op(
            output_grad, self.axes, self.indices, self.splits, ctx=self.raw_ctx)
        return [self.grad_node]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        ori_shape = list(input_shapes[0])
        self.begin_pos = [0 for _ in ori_shape]
        self.output_shape = [x for x in ori_shape]
        for axe, ind, spl in zip(self.axes, self.indices, self.splits):
            part_size = ori_shape[axe] // spl
            self.begin_pos[axe] = ind * part_size
            self.output_shape[axe] = part_size if ind != spl - \
                1 else ori_shape[axe] - self.begin_pos[axe]

        if self.grad_node is not None:
            self.grad_node.begin_pos = self.begin_pos
            self.grad_node.output_shape = ori_shape

        # here we save the information on device for GPU computation
        if self.on_gpu:
            ndim = len(ori_shape)
            gpu_buf = [0 for _ in range(3 * ndim)]
            for i in range(ndim):
                gpu_buf[i] = self.begin_pos[i]
                gpu_buf[ndim + i] = ori_shape[i]
                gpu_buf[2 * ndim + i] = self.output_shape[i]
            self.gpu_buffer = ndarray.array(
                gpu_buf, self.ctx, dtype=np.uintc)
        return self.output_shape

    def naive_infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        ori_shape = list(input_shapes[0])
        self.begin_pos = [0 for _ in ori_shape]
        self.output_shape = [x for x in ori_shape]
        for axe, ind, spl in zip(self.axes, self.indices, self.splits):
            part_size = ori_shape[axe] // spl
            self.begin_pos[axe] = ind * part_size
            self.output_shape[axe] = part_size if ind != spl - \
                1 else ori_shape[axe] - self.begin_pos[axe]

        if self.grad_node is not None:
            self.grad_node.begin_pos = self.begin_pos
            self.grad_node.output_shape = ori_shape
        return self.output_shape


class SplitGradientOp(Op):
    def __init__(self, node_A, axes, indices, splits, ctx=None):
        super().__init__(SplitGradientOp, [node_A], ctx)
        self.axes = axes
        self.indices = indices
        self.splits = splits
        self.begin_pos = None
        self.output_shape = None
        assert len(self.axes) == len(self.splits)
        assert all([x >= 0 for x in axes])
        assert all([x >= 1 for x in splits])
        assert all([x >= 0 and x < splits[i] for i, x in enumerate(indices)])

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.zeros(self.output_shape, dtype=np.float32)
            index = tuple([slice(i, i+j)
                           for i, j in zip(self.begin_pos, self.ori_shape)])
            output_val[index] = input_vals[0]
        else:
            # matrix_slice_gradient(input_vals[0], output_val, self.begin_pos, stream_handle)
            matrix_slice_gradient_simple(
                input_vals[0], output_val, self.gpu_buffer, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert self.output_shape != None and self.begin_pos != None
        assert len(input_shapes) == 1
        ori_shape = list(input_shapes[0])
        for i in range(len(ori_shape)):
            assert self.begin_pos[i] + ori_shape[i] <= self.output_shape[i]
        self.ori_shape = tuple(ori_shape)

        # here we save the information on device for GPU computation
        if self.on_gpu:
            ndim = len(ori_shape)
            gpu_buf = [0 for _ in range(3 * ndim)]
            for i in range(ndim):
                gpu_buf[i] = self.begin_pos[i]
                gpu_buf[ndim + i] = ori_shape[i]
                gpu_buf[2 * ndim + i] = self.output_shape[i]
            self.gpu_buffer = ndarray.array(
                gpu_buf, self.ctx, dtype=np.uintc)
        return self.output_shape


def split_op(node, axes, indices, splits, ctx=None):
    return SplitOp(node, axes, indices, splits, ctx=ctx)


def split_gradient_op(node, axes, indices, splits, ctx=None):
    return SplitGradientOp(node, axes, indices, splits, ctx=ctx)
