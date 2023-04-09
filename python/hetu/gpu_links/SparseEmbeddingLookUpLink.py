from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def sparse_embedding_lookup(in_mat, ids, out_mat, stream=None):
    assert isinstance(in_mat, _nd.ND_Sparse_Array)
    assert isinstance(ids, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    if in_mat.form == 'csr':
        _LIB.DLGpuCSREmbeddingLookUp(in_mat.data.handle, in_mat.row.handle, in_mat.col.handle,
                                     ids.handle, out_mat.handle, stream.handle if stream else None)
    else:
        _LIB.DLGpuCOOEmbeddingLookUp(in_mat.data.handle, in_mat.row.handle, in_mat.col.handle,
                                     ids.handle, out_mat.handle, stream.handle if stream else None)
