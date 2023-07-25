from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import matrix_slice_simple
from ..gpu_links import matrix_slice_gradient_simple
from .. import ndarray


class SliceOp(Op):
    def __init__(self, node_A, begin_pos, output_shape, end_pos, ctx=None):
        super().__init__(SliceOp, [node_A], ctx)
        self.begin_pos = list(begin_pos)
        self.grad_node = None
        if output_shape is None:
            self.ori_output_shape = list()
            self.output_shape = list ()
            self.end_pos = list(end_pos)
            assert len(self.begin_pos) == len(self.end_pos)
        else:
            self.ori_output_shape = list(output_shape)
            self.output_shape = list(output_shape)
            self.end_pos = None
            assert len(self.begin_pos) == len(self.ori_output_shape)

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
        self.grad_node = slice_gradient_op(
            output_grad, self.begin_pos, None, ctx=self.raw_ctx)
        return [self.grad_node]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        ori_shape = list(input_shapes[0])

        for i in range(len(self.begin_pos)):
            if self.begin_pos[i] < 0:
                self.begin_pos[i] += ori_shape[i]
            assert self.begin_pos[i] >= 0
        
        if self.end_pos is not None:
            assert len(self.begin_pos) == len(self.end_pos)
            assert self.ori_output_shape is None
            for i in range(len(self.end_pos)):
                if self.end_pos[i] < 0:
                    self.end_pos[i] += ori_shape[i]
                assert self.end_pos[i] >= 0
            
            for i in range(len(ori_shape)):
                if i < len(self.end_pos):
                    self.output_shape.append(self.end_pos[i] - self.begin_pos[i])
                else:
                    self.output_shape.append(ori_shape[i])
                    self.begin_pos.append(0)
                assert self.output_shape[i] > 0 and self.output_shape[i] <= ori_shape[i]

        else:
            assert len(self.begin_pos) == len(self.ori_output_shape)
            assert self.end_pos is None
            for i in range(len(ori_shape)):
                if i < len(self.ori_output_shape):
                    if self.ori_output_shape[i] == -1:
                        self.output_shape[i] = ori_shape[i] - self.begin_pos[i]
                else:
                    self.output_shape.append(ori_shape[i])
                    self.begin_pos.append(0)
                assert self.output_shape[i] > 0 and self.output_shape[i] <= ori_shape[i]

        self.ori_shape = tuple(ori_shape)
        if self.grad_node is not None:
            self.grad_node.output_shape = self.ori_shape
            self.grad_node.begin_pos = self.begin_pos

        # here we save the information on device for GPU computation
        if self.on_gpu:
            ndim = len(ori_shape)
            gpu_buf = [0 for _ in range(3 * ndim)]
            for i in range(ndim):
                gpu_buf[i] = self.begin_pos[i]
                gpu_buf[ndim + i] = ori_shape[i]
                gpu_buf[2 * ndim + i] = self.output_shape[i]
            self.gpu_buffer = ndarray.array(
                gpu_buf, self.ctx, data_type=np.uintc)
        return tuple(self.output_shape)

    def naive_infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        ori_shape = list(input_shapes[0])
        assert len(ori_shape) == len(self.begin_pos)
        for i in range(len(ori_shape)):
            if self.ori_output_shape[i] == -1:
                self.output_shape[i] = ori_shape[i] - self.begin_pos[i]
            assert self.output_shape[i] > 0
            assert self.begin_pos[i] + self.output_shape[i] <= ori_shape[i]
        self.ori_shape = tuple(ori_shape)
        if self.grad_node is not None:
            self.grad_node.output_shape = self.ori_shape
            assert len(self.ori_shape) == len(self.grad_node.begin_pos)
        return tuple(self.output_shape)


class SliceGradientOp(Op):
    def __init__(self, node_A, begin_pos, output_shape, ctx=None):
        super().__init__(SliceGradientOp, [node_A], ctx)
        self.begin_pos = list(begin_pos)
        self.output_shape = None
        if output_shape != None:
            self.output_shape = tuple(output_shape)
            assert len(self.begin_pos) == len(self.output_shape)

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
        assert self.output_shape != None
        assert len(input_shapes) == 1
        ori_shape = list(input_shapes[0])
        assert len(ori_shape) == len(self.begin_pos)
        for i in range(len(self.begin_pos)):
            if self.begin_pos[i] < 0:
                self.begin_pos[i] += ori_shape[i]
            assert self.begin_pos[i] >= 0

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
                gpu_buf, self.ctx, data_type=np.uintc)
        return tuple(self.output_shape)

    def naive_infer_shape(self, input_shapes):
        assert self.output_shape != None
        assert len(input_shapes) == 1
        ori_shape = list(input_shapes[0])
        assert len(ori_shape) == len(self.begin_pos)
        for i in range(len(ori_shape)):
            assert self.begin_pos[i] + ori_shape[i] <= self.output_shape[i]
        self.ori_shape = tuple(ori_shape)

        return tuple(self.output_shape)


def slice_op(node, begin, size=None, end=None, ctx=None):
    """Creates a node that represents tf.slice(node, begin, size).

    Parameters:
    ----
    node : Node
        The Node needed to be summed.
    begin: tuple
        The beginning position of slice operation.
    size: tuple
        The shape(size) of output tensor.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return SliceOp(node, begin, size, end, ctx=ctx)


def slice_gradient_op(node, begin, size=None, ctx=None):
    """Creates a node that represents the gradient of tf.slice.

    Parameters:
    ----
    node : Node
        The Node needed to be summed.
    begin: tuple
        The beginning position of slice operation.
    size: tuple
        The shape(size) of output tensor.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return SliceGradientOp(node, begin, size, ctx=ctx)
