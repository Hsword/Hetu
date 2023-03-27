from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from .._base import DNNL_LIB
from ..cpu_links import matrix_elementwise_multiply as\
    cpu_matrix_elementwise_multiply
from ..cpu_links import matrix_elementwise_multiply_by_const as\
    cpu_matrix_elementwise_multiply_by_const
from ..gpu_links import matrix_elementwise_multiply_by_const,\
    array_set,\
    matrix_elementwise_multiply_simple,\
    matrix_elementwise_multiply_lazy
import numpy as np


class MulOp(Op):
    def __init__(self, node_A, node_B, ctx=None):
        super().__init__(MulOp, [node_A, node_B], ctx)
        self.lazy_execution = True
        self.compute_to_be_config = False
        self.grad_node_A = None
        self.grad_node_B = None

    def _compute_on_cpu_simple(self, input_vals, output_val, stream_handle=None):
        assert self.on_cpu
        if DNNL_LIB['DnnlMatrixElementwisemultiply'] and input_vals[0].shape == input_vals[1].shape:
            cpu_matrix_elementwise_multiply(
                input_vals[0], input_vals[1], output_val)
        elif DNNL_LIB['DnnlMatrixElementwisemultiplyByConst'] and (input_vals[1].shape == (1,) or input_vals[0].shape == (1,)):
            if input_vals[1].shape == (1,):
                const_val = input_vals[1].asnumpy()[0]
                cpu_matrix_elementwise_multiply_by_const(
                    input_vals[0], const_val, output_val)
            elif input_vals[0].shape == (1,):
                const_val = input_vals[0].asnumpy()[0]
                cpu_matrix_elementwise_multiply_by_const(
                    input_vals[1], const_val, output_val)
        else:
            output_val[:] = input_vals[0].asnumpy() * input_vals[1].asnumpy()

    def _compute_on_gpu_multiply_const(self, input_vals, output_val, stream_handle=None):
        assert self.on_gpu
        if input_vals[1].shape == (1,):
            const_val = input_vals[1].asnumpy()[0]
            matrix_elementwise_multiply_by_const(
                input_vals[0], const_val, output_val, stream_handle)
        elif input_vals[0].shape == (1,):
            const_val = input_vals[0].asnumpy()[0]
            matrix_elementwise_multiply_by_const(
                input_vals[1], const_val, output_val, stream_handle)
        else:
            assert False

    def _compute_on_gpu_simple(self, input_vals, output_val, stream_handle=None):
        assert self.on_gpu
        matrix_elementwise_multiply_simple(
            input_vals[0], input_vals[1], output_val, stream_handle)

    def _compute_on_gpu_lazy(self, input_vals, output_val, stream_handle=None):
        assert self.on_gpu
        self._reset_gpu_buffer(input_vals[0], input_vals[1], output_val)
        matrix_elementwise_multiply_lazy(
            input_vals[0], input_vals[1], output_val, self.gpu_buffer, stream_handle)

    def _compute_on_gpu_broadcast_to_0(self, input_vals, output_val, stream_handle=None):
        assert self.on_gpu
        input_vals[1].broadcast_to(input_vals[0].shape, self.middle_result)
        self._reset_gpu_buffer(input_vals[0], self.middle_result, output_val)
        matrix_elementwise_multiply_lazy(
            input_vals[0], self.middle_result, output_val, self.gpu_buffer, stream_handle)

    def _compute_on_gpu_broadcast_to_1(self, input_vals, output_val, stream_handle=None):
        assert self.on_gpu
        input_vals[0].broadcast_to(input_vals[1].shape, self.middle_result)
        self._reset_gpu_buffer(self.middle_result, input_vals[1], output_val)
        matrix_elementwise_multiply_lazy(
            self.middle_result, input_vals[1], output_val, self.gpu_buffer, stream_handle)

    def _reset_gpu_buffer(self, input_val1, input_val2, output_val):
        if self.check_reset:
            strides = list(input_val1.stride) + \
                list(input_val2.stride) + list(output_val.stride)
            self.gpu_buffer = ndarray.array(
                strides, self.ctx, data_type=np.uintc)
            self.check_reset = False

    def gradient(self, output_grad):
        from .ReduceSum import reduce_sum_op
        self.grad_node_A = reduce_sum_op(mul_op(self.inputs[1], output_grad, ctx=self.raw_ctx), None, None, ctx=self.raw_ctx)
        self.grad_node_B = reduce_sum_op(mul_op(self.inputs[0], output_grad, ctx=self.raw_ctx), None, None, ctx=self.raw_ctx)        
        return [self.grad_node_A, self.grad_node_B]

    def infer_shape(self, input_shapes):      
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        assert len(input_shapes) == 2
        no_broadcast = input_shapes[0] == input_shapes[1]
        has_const = input_shapes[0] == (1,) or input_shapes[1] == (1,)
        if no_broadcast:
            output = input_shapes[0]
        elif not has_const:
            first_size = np.prod(input_shapes[0])
            second_size = np.prod(input_shapes[1])
            diff = abs(len(input_shapes[0])-len(input_shapes[1]))
            axes = list(range(diff))
            keepdims = [False] * diff
            if first_size > second_size:
                long_shapes = input_shapes[0]
                short_shapes = input_shapes[1]
                first_long = True
            else:
                long_shapes = input_shapes[1]
                short_shapes = input_shapes[0]
                first_long = False
            for i in range(len(short_shapes)):
                if short_shapes[i] != 1 and short_shapes[i] != long_shapes[len(long_shapes)-len(short_shapes) + i]:
                    assert False, "can't multiply variables of shapes  " + \
                        str(input_shapes[0])+str(input_shapes[1])
                if short_shapes[i] == 1 and long_shapes[i + diff] > 1:
                    axes.append(i + diff)
                    keepdims.append(True)
            if first_long:
                if self.grad_node_B is not None:
                    self.grad_node_B.axes = axes
                    self.grad_node_B.keepdims = keepdims
            else:
                 if self.grad_node_A is not None:
                    self.grad_node_A.axes = axes
                    self.grad_node_A.keepdims = keepdims               
            output = long_shapes
        if self.compute_to_be_config:
            if has_const:
                self.compute = self._compute_on_gpu_multiply_const
            elif no_broadcast:
                if self.inputs[0].inplace or self.inputs[1].inplace:
                    self.compute = self._compute_on_gpu_lazy
                    self.check_reset = True
                else:
                    self.compute = self._compute_on_gpu_simple
            else:
                self.middle_result = ndarray.NDArray(None)
                if first_long:
                    self.compute = self._compute_on_gpu_broadcast_to_0
                else:
                    self.compute = self._compute_on_gpu_broadcast_to_1
                self.check_reset = True
        return output

    def forward_hook(self, config):
        super().forward_hook(config)
        if self.on_cpu:
            self.compute = self._compute_on_cpu_simple
        else:
            # determine in infer_shape
            self.compute_to_be_config = True
            self.check_reset = False


def mul_op(node_A, node_B, ctx=None):
    """Make a new instance of matrixs elementwise multiplication and call the instance.

    Parameters:
    ----
    node_a : Node
        The Node to be multiplied.
    node_b : Node
        Another Node to be multiplied.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return MulOp(node_A, node_B, ctx=ctx)
