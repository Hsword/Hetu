from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def baddbmm(input_mat, matA, matB, matC, alpha, beta, stream=None):
    assert isinstance(input_mat, _nd.NDArray)
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DLGpuBaddbmm(input_mat.handle, matA.handle, matB.handle, ctypes.c_float(
        alpha), ctypes.c_float(beta), matC.handle, stream.handle if stream else None)
