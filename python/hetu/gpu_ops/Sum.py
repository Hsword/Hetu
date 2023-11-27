from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from .._base import DNNL_LIB
from ..gpu_links import matrix_elementwise_add_by_const,\
    indexedslice_oneside_add,\
    array_set,\
    matrix_elementwise_add_simple,\
    matrix_elementwise_add_lazy,\
    concatenate, array_reshape
import numpy as np
from .DataTransfer import DataD2HSparseOp, DataH2DSparseOp
from .EmbeddingLookUp import EmbeddingLookUp_Gradient


class SumOp(Op):
    def __init__(self, node_list, ctx=None):
        super().__init__(SumOp, list(node_list), ctx)
        self.lazy_execution = True
        self.compute_to_be_config = False
        dtype = node_list[0].dtype
        for n in node_list[1:]:
            assert dtype == n.dtype
        self.dtype = dtype

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
        output_val[input_val.indices.asnumpy().astype(
            np.int)] += input_val.values.asnumpy()

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
                strides, self.ctx, dtype=np.uintc)
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


class SparseSumOp(Op):
    def __init__(self, node_list, ctx=None):
        super().__init__(SparseSumOp, list(node_list), ctx)
        self.lazy_execution = True
        self.compute_to_be_config = False
        self.use_indexed_slices = True

    def compute(self, input_vals, output_val, stream_handle=None):
        assert isinstance(output_val, ndarray.IndexedSlices)
        merged_size = 0
        indices_list, value_list = [], []
        embed_dim, ctx = input_vals[0].values.shape[-1], input_vals[0].values.ctx
        for input_val in input_vals:
            if self.on_cpu:
                indices_list.append(input_val.indices.asnumpy().reshape(-1))
                value_list.append(
                    input_val.values.asnumpy().reshape(-1, embed_dim))
            else:
                indices_size = 1
                for shape in input_val.indices.shape:
                    indices_size *= shape
                reshaped_indices = ndarray.empty((indices_size, ), ctx=ctx)
                reshaped_values = ndarray.empty(
                    (indices_size, embed_dim), ctx=ctx)
                array_reshape(input_val.indices, reshaped_indices)
                array_reshape(input_val.values, reshaped_values)
                indices_list.append(reshaped_indices)
                value_list.append(reshaped_values)
                merged_size += indices_size

        if self.on_cpu:
            output_indices = ndarray.array(
                np.concatenate(indices_list), ctx=ctx)
            output_values = ndarray.array(
                np.concatenate(indices_list), ctx=ctx)
            output_val.update(output_indices, output_values,
                              input_vals[0].dense_shape)
            output_val.cpu_deduplicate()
        else:
            output_indices = ndarray.empty((merged_size, ), ctx=ctx)
            output_values = ndarray.empty((merged_size, embed_dim), ctx=ctx)
            concatenate(indices_list, output_indices)
            concatenate(value_list, output_values, axis=0)
            output_val.update(output_indices, output_values,
                              input_vals[0].dense_shape)
            output_val.deduplicate(stream_handle)

    def gradient(self, output_grad):
        return [output_grad for _ in self.inputs]

    def infer_shape(self, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
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
        return result_shape

    def forward_hook(self, config):
        super().forward_hook(config)
        for node in self.inputs:
            assert isinstance(node, (EmbeddingLookUp_Gradient,
                                     DataD2HSparseOp, DataH2DSparseOp))


def sum_op(node_list, ctx=None, sparse=False):
    if sparse:
        return SparseSumOp(node_list, ctx=ctx)
    return SumOp(node_list, ctx=ctx)
