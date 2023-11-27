from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def slice_by_matrix(in_arr, index1, index2, out_arr, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(index1, _nd.NDArray)
    assert isinstance(index2, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuSliceByMatrix(in_arr.handle, index1.handle, index2.handle,
                            out_arr.handle, stream.handle if stream else None)


def slice_by_matrix_gradient(in_arr, index1, index2, out_arr, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(index1, _nd.NDArray)
    assert isinstance(index2, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)

    _LIB.DLGpuSliceByMatrixGradient(
        in_arr.handle, index1.handle, index2.handle, out_arr.handle, stream.handle if stream else None)
