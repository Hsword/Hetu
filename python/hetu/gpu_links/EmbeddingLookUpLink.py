from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def embedding_lookup(in_mat, ids, out_mat, stream=None):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(ids, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    _LIB.DLGpuEmbeddingLookUp(
        in_mat.handle, ids.handle, out_mat.handle, stream.handle if stream else None)


def embedding_lookup_gradient(grad_out, ids, grad_in, stream=None):
    assert isinstance(grad_out, _nd.NDArray)
    assert isinstance(ids, _nd.NDArray)
    assert isinstance(grad_in, _nd.NDArray)
    _LIB.DLGpuEmbeddingLookUp_Gradient(
        grad_out.handle, ids.handle, grad_in.handle, stream.handle if stream else None)
