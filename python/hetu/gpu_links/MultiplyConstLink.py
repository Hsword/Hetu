from __future__ import absolute_import

import ctypes
import numpy as np
from .._base import _LIB
from .. import ndarray as _nd


def matrix_elementwise_multiply_by_const(in_mat, val, out_mat, stream=None):

    assert isinstance(in_mat, (_nd.NDArray, _nd.IndexedSlices))
    assert isinstance(out_mat, (_nd.NDArray, _nd.IndexedSlices))

    if in_mat.dtype == np.float32:
        cval = ctypes.c_float(val)
        func = _LIB.DLGpuMatrixMultiplyByConst
    elif in_mat.dtype == np.int32:
        cval = ctypes.c_int(val)
        func = _LIB.DLGpuMatrixMultiplyByConstInt

    if isinstance(in_mat, _nd.NDArray):
        func(
            in_mat.handle, cval, out_mat.handle, stream.handle if stream else None)
    else:
        # isinstance(in_mat, _nd.IndexedSlices)
        func(
            in_mat.values.handle, cval, out_mat.values.handle, stream.handle if stream else None)
