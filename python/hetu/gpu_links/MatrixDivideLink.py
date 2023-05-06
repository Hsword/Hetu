from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def matrix_elementwise_divide(matA, matB, matC, stream=None):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DLGpuMatrixElementwiseDivide(
        matA.handle, matB.handle, matC.handle, stream.handle if stream else None)


def matrix_elementwise_divide_handle_zero(matA, matB, matC, stream=None):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DLGpuMatrixElementwiseDivideHandleZero(
        matA.handle, matB.handle, matC.handle, stream.handle if stream else None)

