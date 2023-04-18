from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def unique_indices(indices, output, idoffsets, workspace, storage_size, end_bit, stream=None):
    assert isinstance(indices, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    assert isinstance(idoffsets, _nd.NDArray)
    assert isinstance(workspace, _nd.NDArray)
    _LIB.DLGpuUniqueIndices(indices.handle, output.handle, idoffsets.handle, workspace.handle, ctypes.c_size_t(
        storage_size), ctypes.c_int(end_bit), stream.handle if stream else None)


def get_unique_workspace_size(ind_size):
    size = ctypes.POINTER(ctypes.c_size_t)(ctypes.c_size_t(0))
    _LIB.DLGpuGetUniqueWorkspaceSize(ctypes.c_size_t(ind_size), size)
    return size.contents.value


def deduplicate_lookup(lookups, idoffsets, output, stream):
    assert isinstance(lookups, _nd.NDArray)
    assert isinstance(idoffsets, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuDedupLookup(lookups.handle, idoffsets.handle,
                          output.handle, stream.handle if stream else None)


def deduplicate_grad(grad, idoffsets, output, stream):
    assert isinstance(grad, _nd.NDArray)
    assert isinstance(idoffsets, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuDedupGrad(grad.handle, idoffsets.handle,
                        output.handle, stream.handle if stream else None)
