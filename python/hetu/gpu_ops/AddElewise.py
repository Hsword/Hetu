from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from .._base import DNNL_LIB
from ..cpu_links import matrix_elementwise_add as\
    cpu_matrix_elementwise_add
from ..cpu_links import matrix_elementwise_add_by_const as\
    cpu_matrix_elementwise_add_by_const
from ..gpu_links import matrix_elementwise_add_by_const,\
    indexedslice_oneside_add,\
    array_set,\
    matrix_elementwise_add_simple,\
    matrix_elementwise_add_lazy
import numpy as np


class AddOp(Op):
    def __init__(self, node_A, node_B, ctx=None):
        super().__init__(AddOp, [node_A, node_B], ctx)
        self.lazy_execution = True
        self.compute_to_be_config = False
        assert node_A.dtype == node_B.dtype
        self.dtype = node_A.dtype

    def _compute_with_index(self, input_vals, output_val, stream_handle=None):
        def cpu_oneside_add(sparse, dense):
            dense[sparse.indices.asnumpy().astype(
                np.int)] += sparse.values.asnumpy()
        first_indexed = isinstance(input_vals[0], ndarray.IndexedSlices)
        second_indexed = isinstance(input_vals[1], ndarray.IndexedSlices)
        if self.on_cpu:
            if first_indexed and not second_indexed:
                cpu_output = input_vals[1].numpy()
                cpu_oneside_add(input_vals[0], cpu_output)
                output_val[:] = cpu_output
            elif not first_indexed and second_indexed:
                cpu_output = input_vals[0].numpy()
                cpu_oneside_add(input_vals[1], cpu_output)
                output_val[:] = cpu_output
            elif first_indexed and second_indexed:
                cpu_output = np.zeros(output_val.shape).astype(np.float32)
                cpu_oneside_add(input_vals[0], cpu_output)
                cpu_oneside_add(input_vals[1], cpu_output)
                output_val[:] = cpu_output
            else:
                assert False
        else:
            if first_indexed and not second_indexed:
                input_vals[1].copyto(output_val)
                indexedslice_oneside_add(
                    input_vals[0], output_val, stream_handle)
            elif not first_indexed and second_indexed:
                input_vals[0].copyto(output_val)
                indexedslice_oneside_add(
                    input_vals[1], output_val, stream_handle)
            elif first_indexed and second_indexed:
                array_set(output_val, 0, stream_handle)
                indexedslice_oneside_add(
                    input_vals[0], output_val, stream_handle)
                indexedslice_oneside_add(
                    input_vals[1], output_val, stream_handle)
            else:
                assert False

    def _compute_on_cpu_simple(self, input_vals, output_val, stream_handle=None):
        assert self.on_cpu
        if DNNL_LIB['DnnlMatrixElementwiseAdd'] and input_vals[0].shape == input_vals[1].shape:
            cpu_matrix_elementwise_add(
                input_vals[0], input_vals[1], output_val)
        elif DNNL_LIB['DnnlMatrixElementwiseAddByConst'] and (input_vals[1].shape == (1,) or input_vals[0].shape == (1,)):
            if input_vals[1].shape == (1,):
                const_val = input_vals[1].asnumpy()[0]
                cpu_matrix_elementwise_add_by_const(
                    input_vals[0], const_val, output_val)
            elif input_vals[0].shape == (1,):
                const_val = input_vals[0].asnumpy()[0]
                cpu_matrix_elementwise_add_by_const(
                    input_vals[1], const_val, output_val)
        else:
            # output_val[:] allows modify in-place
            output_val[:] = input_vals[0].asnumpy() + input_vals[1].asnumpy()

    def _compute_on_gpu_add_const(self, input_vals, output_val, stream_handle=None):
        assert self.on_gpu
        if input_vals[1].shape == (1,):
            const_val = input_vals[1].asnumpy()[0]
            matrix_elementwise_add_by_const(
                input_vals[0], const_val, output_val, stream_handle)
        elif input_vals[0].shape == (1,):
            const_val = input_vals[0].asnumpy()[0]
            matrix_elementwise_add_by_const(
                input_vals[1], const_val, output_val, stream_handle)
        else:
            assert False

    def _compute_on_gpu_simple(self, input_vals, output_val, stream_handle=None):
        assert self.on_gpu
        matrix_elementwise_add_simple(
            input_vals[0], input_vals[1], output_val, stream_handle)

    def _compute_on_gpu_lazy(self, input_vals, output_val, stream_handle=None):
        assert self.on_gpu
        self._reset_gpu_buffer(input_vals[0], input_vals[1], output_val)
        matrix_elementwise_add_lazy(
            input_vals[0], input_vals[1], output_val, self.gpu_buffer, stream_handle)

    def _compute_on_gpu_broadcast_to_0(self, input_vals, output_val, stream_handle=None):
        assert self.on_gpu
        input_vals[1].broadcast_to(input_vals[0].shape, self.middle_result)
        self._reset_gpu_buffer(input_vals[0], self.middle_result, output_val)
        matrix_elementwise_add_lazy(
            input_vals[0], self.middle_result, output_val, self.gpu_buffer, stream_handle)

    def _compute_on_gpu_broadcast_to_1(self, input_vals, output_val, stream_handle=None):
        assert self.on_gpu
        input_vals[0].broadcast_to(input_vals[1].shape, self.middle_result)
        self._reset_gpu_buffer(self.middle_result, input_vals[1], output_val)
        matrix_elementwise_add_lazy(
            self.middle_result, input_vals[1], output_val, self.gpu_buffer, stream_handle)

    def _reset_gpu_buffer(self, input_val1, input_val2, output_val):
        if self.check_reset:
            strides = list(input_val1.stride) + \
                list(input_val2.stride) + list(output_val.stride)
            self.gpu_buffer = ndarray.array(
                strides, self.ctx, dtype=np.uintc)
            self.check_reset = False

    def gradient(self, output_grad):
        return [output_grad, output_grad]

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
                    assert False, "can't add variables of shapes  " + \
                        str(input_shapes[0])+str(input_shapes[1])
            output = long_shapes
        else:
            assert False, "Shapes not valid; got {} in {} with inputs {}".format(
                input_shapes, self, self.inputs)
        if self.compute_to_be_config:
            if has_const:
                self.compute = self._compute_on_gpu_add_const
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

        if self.inputs[0].use_indexed_slices or \
                self.inputs[1].use_indexed_slices:
            self.compute = self._compute_with_index
        elif self.on_cpu:
            self.compute = self._compute_on_cpu_simple
        else:
            # determine in infer_shape
            self.compute_to_be_config = True
            self.check_reset = False


def add_op(node_A, node_B, ctx=None):
    """Make a new instance of Node Addition and call the instance.

    Parameters:
    ----
    node_A : Node
        The Node to be added.
    node_B : Node
        Another Node to be added.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return AddOp(node_A, node_B, ctx=ctx)
