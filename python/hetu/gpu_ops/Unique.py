from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..ndarray import empty
from ..cpu_links import unique_indices as cpu_unique_indices, \
    deduplicate_lookup as cpu_deduplicate_lookup, \
    deduplicate_grad as cpu_deduplicate_grad
from ..gpu_links import unique_indices, get_unique_workspace_size, \
    deduplicate_lookup, deduplicate_grad


class UniqueIndicesOp(Op):
    def __init__(self, indices, ctx=None):
        assert indices.dtype == np.int32
        super().__init__(UniqueIndicesOp, [indices], ctx)
        self.id_offsets = None
        self.dtype = np.int32

    def compute(self, input_vals, output_val, stream_handle=None):
        assert self.id_offsets is not None
        if self.on_cpu:
            cpu_unique_indices(input_vals[0], output_val, self.id_offsets)
        else:
            unique_indices(input_vals[0], output_val, self.id_offsets, self.dedup_args['sp'],
                           self.dedup_args['size'], self.dedup_args['eb'], stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        ind_size = np.prod(input_shapes[0]).item()
        if self.on_gpu:
            ws_size = get_unique_workspace_size(ind_size)
            all_ws_size = (ws_size + 3) // 4
            self.dedup_args = {
                'sp': empty((all_ws_size, ), ctx=self.ctx),
                'size': ws_size,
                'eb': 32,
            }
        else:
            self.dedup_args = {}
        return input_shapes[0]


def unique_indices_op(indices, ctx=None):
    return UniqueIndicesOp(indices, ctx=ctx)


class UniqueIndicesOffsetsOp(Op):
    def __init__(self, unique, ctx=None):
        super().__init__(UniqueIndicesOffsetsOp,
                         [unique], ctx)
        assert unique.dtype == np.int32
        self.dtype = np.int32

    def compute(self, input_vals, output_val, stream_handle=None):
        assert False, 'In memory plan we already set the result array; should not call the compute.'

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        ind_size = np.prod(input_shapes[0]).item()
        return (2 * ind_size + 2, )

    def pass_grad_array(self, array):
        self.inputs[0].id_offsets = array


def unique_indices_offsets_op(unique, ctx=None):
    return UniqueIndicesOffsetsOp(unique, ctx=ctx)


# following ops is not in use; use embedding lookup gradient ops instead
class DedupLookupOp(Op):
    def __init__(self, lookup, idoffsets, ctx=None):
        super().__init__(DedupLookupOp, [lookup, idoffsets], ctx)
        assert lookup.dtype == np.float32
        assert idoffsets.dtype == np.int32

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_gpu:
            deduplicate_lookup(
                input_vals[0], input_vals[1], output_val, stream_handle)
        else:
            cpu_deduplicate_lookup(
                input_vals[0], input_vals[1], output_val)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert len(input_shapes[1]) == 1
        return input_shapes[0]


def deduplicate_lookup_op(lookup, idoffsets, ctx=None):
    return DedupLookupOp(lookup, idoffsets, ctx=ctx)


class DedupGradOp(Op):
    def __init__(self, grad, idoffsets, ctx=None):
        super().__init__(DedupGradOp, [grad, idoffsets], ctx)
        assert grad.dtype == np.float32
        assert idoffsets.dtype == np.int32

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_gpu:
            deduplicate_grad(
                input_vals[0], input_vals[1], output_val, stream_handle)
        else:
            cpu_deduplicate_grad(
                input_vals[0], input_vals[1], output_val)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert len(input_shapes[1]) == 1
        return input_shapes[0]


def deduplicate_grad_op(grad, idoffsets, ctx=None):
    return DedupGradOp(grad, idoffsets, ctx=ctx)
