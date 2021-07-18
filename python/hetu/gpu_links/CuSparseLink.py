from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def CuSparse_Csrmv(mat, trans, in_arr, out_arr, stream=None):
    assert isinstance(mat, _nd.ND_Sparse_Array)
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.CuSparse_DLGpuCsrmv(mat.data.handle, mat.row.handle, mat.col.handle, mat.nrow,
                             mat.ncol, trans, in_arr.handle, out_arr.handle, stream.handle if stream else None)


def CuSparse_Csrmm(matA, transA, matB, transB, matC, stream=None, start_pos=-1, end_pos=-1):
    assert isinstance(matA, _nd.ND_Sparse_Array)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.CuSparse_DLGpuCsrmm(matA.data.handle, matA.row.handle, matA.col.handle, matA.nrow, matA.ncol,
                             transA, matB.handle, transB, matC.handle, start_pos, end_pos, stream.handle if stream else None)
