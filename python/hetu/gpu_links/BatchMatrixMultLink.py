from __future__ import absolute_import

from .._base import _LIB
from .. import ndarray as _nd


def batch_matrix_multiply(matA, transA, matB, transB, matC, stream=None):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DLGpuBatchMatrixMultiply(
        matA.handle, transA, matB.handle, transB, matC.handle, stream.handle if stream else None)
