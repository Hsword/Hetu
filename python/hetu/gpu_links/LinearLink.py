from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def matmul_with_bias(matA, transA, matB, transB, bias, matC, stream=None):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(bias, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DLGpuLinear(
        matA.handle, transA, matB.handle, transB, bias.handle, matC.handle, stream.handle if stream else None)
