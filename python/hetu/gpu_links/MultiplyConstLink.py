from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def matrix_elementwise_multiply_by_const(in_mat, val, out_mat, stream=None):

    assert isinstance(in_mat, (_nd.NDArray, _nd.IndexedSlices))
    assert isinstance(out_mat, (_nd.NDArray, _nd.IndexedSlices))

    if isinstance(in_mat, _nd.NDArray):
        _LIB.DLGpuMatrixMultiplyByConst(
            in_mat.handle, ctypes.c_float(val), out_mat.handle, stream.handle if stream else None)
    else:
        # isinstance(in_mat, _nd.IndexedSlices)
        _LIB.DLGpuMatrixMultiplyByConst(
            in_mat.values.handle, ctypes.c_float(val), out_mat.values.handle, stream.handle if stream else None)
