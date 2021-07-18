from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def matrix_elementwise_divide_const(val, in_mat, out_mat, stream=None):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    _LIB.DLGpuMatrixDivConst(
        ctypes.c_float(val), in_mat.handle, out_mat.handle, stream.handle if stream else None)
