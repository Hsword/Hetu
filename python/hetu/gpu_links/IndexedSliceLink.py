from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def sparse_add_to_dense(indices, values, output, stream=None):
    assert isinstance(indices, _nd.NDArray)
    assert isinstance(values, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.IndexedSlicesOneSideAdd(
        indices.handle, values.handle, output.handle, stream.handle if stream else None)


def indexedslice_oneside_add(indslice, output, stream=None):
    assert isinstance(indslice.indices, _nd.NDArray)
    assert isinstance(indslice.values, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.IndexedSlicesOneSideAdd(indslice.indices.handle, indslice.values.handle, output.handle,
                                 stream.handle if stream else None)


def reduce_indexedslice(in_ind, in_val, out_ind, out_val, workspace, storage_size, end_bit, stream=None):
    assert isinstance(in_ind, _nd.NDArray)
    assert isinstance(in_val, _nd.NDArray)
    assert isinstance(out_ind, _nd.NDArray)
    assert isinstance(out_val, _nd.NDArray)
    assert isinstance(workspace, _nd.NDArray)
    _LIB.DLGpuReduceIndexedSlice(in_ind.handle, in_val.handle, out_ind.handle, out_val.handle,
                                 workspace.handle, ctypes.c_size_t(storage_size), ctypes.c_int(end_bit), stream.handle if stream else None)


def reduce_indexedslice_get_workspace_size(ind_size):
    size = ctypes.POINTER(ctypes.c_size_t)(ctypes.c_size_t(0))
    _LIB.DLGpuReduceIndexedSliceGetWorkspaceSize(
        ctypes.c_size_t(ind_size), size)
    return size.contents.value


def reduce_indexedslice_with_embedding(in_ind, in_val, in_par, out_ind, out_val, out_par, workspace, storage_size, end_bit, stream=None):
    assert isinstance(in_ind, _nd.NDArray)
    assert isinstance(in_val, _nd.NDArray)
    assert isinstance(out_ind, _nd.NDArray)
    assert isinstance(out_val, _nd.NDArray)
    assert isinstance(workspace, _nd.NDArray)
    assert isinstance(in_par, _nd.NDArray)
    assert isinstance(out_par, _nd.NDArray)
    _LIB.DLGpuReduceIndexedSliceWithEmbedding(in_ind.handle, in_val.handle, in_par.handle,  out_ind.handle, out_val.handle, out_par.handle,
                                              workspace.handle, ctypes.c_size_t(storage_size), ctypes.c_int(end_bit), stream.handle if stream else None)
