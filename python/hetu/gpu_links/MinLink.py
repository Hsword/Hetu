from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def min(in_mat, out_mat_val, dim, stream=None):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(out_mat_val, _nd.NDArray)
    _LIB.DLGpuMin(in_mat.handle, out_mat_val.handle,
                  dim, stream.handle if stream else None)


def min_mat(matA, matB, out_mat, stream=None):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    _LIB.DLGpuMinMat(matA.handle, matB.handle, out_mat.handle,
                     stream.handle if stream else None)
