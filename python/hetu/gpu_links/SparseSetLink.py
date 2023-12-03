from __future__ import absolute_import

from .._base import _LIB
from .. import ndarray as _nd


def sparse_set(table, indices, data, stream=None):
    assert isinstance(table, _nd.NDArray)
    assert isinstance(indices, _nd.NDArray)
    assert isinstance(data, _nd.NDArray)
    _LIB.DLGpuSparseSet(table.handle, indices.handle,
                        data.handle, stream.handle if stream else None)
