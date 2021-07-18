from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def matrix_elementwise_add(matA, matB, matC, lazy_input=False, stream=None):
    # deprecated
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DLGpuMatrixElementwiseAdd(matA.handle, matB.handle, matC.handle, ctypes.c_bool(
        lazy_input), stream.handle if stream else None)


def matrix_elementwise_add_simple(matA, matB, matC, stream=None):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DLGpuMatrixElementwiseAddSimple(
        matA.handle, matB.handle, matC.handle, stream.handle if stream else None)


def matrix_elementwise_add_lazy(matA, matB, matC, gpu_buf, stream=None):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    assert isinstance(gpu_buf, _nd.NDArray)
    _LIB.DLGpuMatrixElementwiseAddLazy(
        matA.handle, matB.handle, matC.handle, gpu_buf.handle, stream.handle if stream else None)
