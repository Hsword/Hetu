from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd

def scatter(target_mat, dim, index_mat, src_mat, stream=None):
    assert isinstance(target_mat, _nd.NDArray);
    assert isinstance(index_mat, _nd.NDArray);
    assert isinstance(src_mat, _nd.NDArray);

    _LIB.DLGpuScatter(
            target_mat.handle, dim, index_mat.handle, src_mat.handle, stream.handle if stream else None)
