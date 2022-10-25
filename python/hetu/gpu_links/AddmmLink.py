from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def addmm(input_mat, matA, matB, matC, alpha, beta, stream=None):
    assert isinstance(input_mat, _nd.NDArray)
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DLGpuAddmm(input_mat.handle, matA.handle, matB.handle, ctypes.c_float(
        alpha), ctypes.c_float(beta), matC.handle, stream.handle if stream else None)


def addmm_gradient(input_mat, output_mat, axis, beta, stream=None):
    assert isinstance(input_mat, _nd.NDArray)
    assert isinstance(output_mat, _nd.NDArray)
    _LIB.DLGpuAddmmGradient(input_mat.handle, output_mat.handle, ctypes.c_int(
        axis), ctypes.c_float(beta), stream.handle if stream else None)
