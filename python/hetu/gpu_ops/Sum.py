from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from .._base import DNNL_LIB
from ..gpu_links import matrix_elementwise_add_by_const,\
    indexedslice_oneside_add,\
    array_set,\
    matrix_elementwise_add_simple,\
    matrix_elementwise_add_lazy
import numpy as np


class SumOp(Op):
    def __init__(self, node_list, ctx=None):
        super().__init__(SumOp, list(node_list), ctx)
        self.lazy_execution = True
        self.compute_to_be_config = False

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            cpu_output = np.zeros(output_val.shape).astype(np.float32)
            for val, callback in zip(input_vals, self.callbacks):
                callback(val, cpu_output)
            output_val[:] = cpu_output
        else:
            array_set(output_val, 0, stream_handle)
            for ind, (val, callback) in enumerate(zip(input_vals, self.callbacks)):
                callback(val, output_val, ind, stream_handle)

    def _simple_cpu_callback(self, input_val, output_val):
        output_val += input_val.asnumpy()

    def _indexed_cpu_callback(self, input_val, output_val):
        input_val.cpu_deduplicate()
        output_val[input_val.indices.asnumpy().astype(
            np.int)] += input_val.values.asnumpy()
        input_val.free_deduplicate()

    def _simple_gpu_callback(self, input_val, output_val, ind, stream_handle):
        matrix_elementwise_add_simple(
            output_val, input_val, output_val, stream_handle)

    def _const_gpu_callback(self, input_val, output_val, ind, stream_handle):
        const_val = input_val.asnumpy()[0]
        matrix_elementwise_add_by_const(
            output_val, const_val, output_val, stream_handle)

    def _indexed_gpu_callback(self, input_val, output_val, ind, stream_handle):
        indexedslice_oneside_add(
            input_val, output_val, stream_handle)

    def _lazy_gpu_callback(self, input_val, output_val, ind, stream_handle):
        self._reset_gpu_buffer(
            ind, input_val, output_val)
        matrix_elementwise_add_lazy(
            output_val, input_val, output_val, self.gpu_buffers[ind], stream_handle)

    def _broadcast_gpu_callback(self, input_val, output_val, ind, stream_handle):
        input_val.broadcast_to(
            output_val.shape, self.middle_results[ind])
        self._reset_gpu_buffer(
            ind, self.middle_results[ind], output_val)
        matrix_elementwise_add_lazy(
            output_val, self.middle_results[ind], output_val, self.gpu_buffers[ind], stream_handle)

    def _reset_gpu_buffer(self, ind, input_val, output_val):
        if self.check_reset[ind]:
            strides = list(output_val.stride) + \
                list(input_val.stride) + list(output_val.stride)
            self.gpu_buffers[ind] = ndarray.array(
                strides, self.ctx, data_type=np.uintc)
            self.check_reset[ind] = False

    def gradient(self, output_grad):
        return [output_grad for _ in self.inputs]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == len(self.inputs)
        result_shape = tuple(input_shapes[0])
        for shape in input_shapes[1:]:
            shape = tuple(shape)
            if shape != result_shape:
                if result_shape == (1,):
                    result_shape = shape
                elif shape != (1,):
                    # here needs broadcast
                    resind = len(result_shape) - 1
                    curind = len(shape) - 1
                    temp_shape = []
                    while resind >= 0 and curind >= 0:
                        temp_shape.insert(
                            0, max(result_shape[resind], shape[curind]))
                        resind -= 1
                        curind -= 1
                    while resind >= 0:
                        temp_shape.insert(0, result_shape[resind])
                        resind -= 1
                    while curind >= 0:
                        temp_shape.insert(0, shape[curind])
                        curind -= 1
                    result_shape = tuple(temp_shape)
        if hasattr(self, 'need_deduce'):
            for ind, shape in enumerate(input_shapes):
                if self.need_deduce[ind]:
                    if shape == (1,):
                        self.callbacks[ind] = self._const_gpu_callback
                    elif shape != result_shape:
                        self.callbacks[ind] = self._broadcast_gpu_callback
                        self.check_reset[ind] = True
                        self.middle_results[ind] = ndarray.NDArray(None)
                    elif self.inputs[ind].inplace:
                        self.callbacks[ind] = self._lazy_gpu_callback
                        self.check_reset[ind] = True
                    else:
                        self.callbacks[ind] = self._simple_gpu_callback
        else:
            # only in profile in FlexFlow strategy
            self.callbacks = [self._simple_gpu_callback for _ in self.inputs]
        return result_shape

    def forward_hook(self, config):
        super().forward_hook(config)
        self.callbacks = [None for _ in self.inputs]
        self.check_reset = [False for _ in self.inputs]
        self.gpu_buffers = [None for _ in self.inputs]
        self.middle_results = [None for _ in self.inputs]
        self.need_deduce = [False for _ in self.inputs]
        for ind, node in enumerate(self.inputs):
            if node.use_indexed_slices:
                self.callbacks[ind] = self._indexed_cpu_callback if self.on_cpu else self._indexed_gpu_callback
            elif self.on_cpu:
                self.callbacks[ind] = self._simple_cpu_callback
            else:
                self.need_deduce[ind] = True


def sum_op(node_list, ctx=None):
    return SumOp(node_list, ctx=ctx)
