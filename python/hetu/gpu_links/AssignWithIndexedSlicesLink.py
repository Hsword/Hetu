from __future__ import absolute_import
import ctypes

from .._base import _LIB
from .. import ndarray as _nd


def assign_embedding_with_indexedslices(embedding, unique, newparam, stream=None):
    assert isinstance(embedding, _nd.NDArray)
    assert isinstance(unique, _nd.NDArray)
    assert isinstance(newparam, _nd.NDArray)
    _LIB.DLGpuAssignWithIndexedSlices(
        embedding.handle, unique.handle, newparam.handle, stream.handle if stream else None)


def assign_quantized_embedding_unified(embedding, unique, newparam, scale, minele, digit, stream=None):
    assert isinstance(embedding, _nd.NDArray)
    assert isinstance(unique, _nd.NDArray)
    assert isinstance(newparam, _nd.NDArray)
    _LIB.DLGpuAssignQuantizedEmbeddingUnified(
        embedding.handle, unique.handle, newparam.handle,
        ctypes.c_float(scale), ctypes.c_float(minele), ctypes.c_int(digit),
        ctypes.c_bool(True), stream.handle if stream else None)


def assign_quantized_embedding(embedding, unique, newparam, qparam, digit, stream=None):
    assert isinstance(embedding, _nd.NDArray)
    assert isinstance(unique, _nd.NDArray)
    assert isinstance(newparam, _nd.NDArray)
    assert isinstance(qparam, _nd.NDArray)
    _LIB.DLGpuAssignQuantizedEmbedding(
        embedding.handle, unique.handle, newparam.handle,
        qparam.handle, ctypes.c_int(digit), ctypes.c_bool(True),
        stream.handle if stream else None)
