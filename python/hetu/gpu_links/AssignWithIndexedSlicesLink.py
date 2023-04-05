from __future__ import absolute_import
import ctypes

from .._base import _LIB
from .. import ndarray as _nd


def assign_embedding_with_indexedslices(embedding, newparam, stream=None):
    assert isinstance(embedding, _nd.NDArray)
    assert isinstance(newparam, _nd.IndexedSlices)
    assert isinstance(newparam.indices, _nd.NDArray)
    assert isinstance(newparam.values, _nd.NDArray)
    _LIB.DLGpuAssignWithIndexedSlices(
        embedding.handle, newparam.indices.handle, newparam.values.handle, stream.handle if stream else None)


def assign_quantized_embedding_unified(embedding, newparam, scale, minele, digit, stream=None):
    assert isinstance(embedding, _nd.NDArray)
    assert isinstance(newparam, _nd.IndexedSlices)
    assert isinstance(newparam.indices, _nd.NDArray)
    assert isinstance(newparam.values, _nd.NDArray)
    _LIB.DLGpuAssignQuantizedEmbeddingUnified(
        embedding.handle, newparam.indices.handle, newparam.values.handle,
        ctypes.c_float(scale), ctypes.c_float(minele), ctypes.c_int(digit),
        ctypes.c_bool(True), stream.handle if stream else None)


def assign_quantized_embedding(embedding, newparam, qparam, digit, stream=None):
    assert isinstance(embedding, _nd.NDArray)
    assert isinstance(newparam, _nd.IndexedSlices)
    assert isinstance(newparam.indices, _nd.NDArray)
    assert isinstance(newparam.values, _nd.NDArray)
    assert isinstance(qparam, _nd.NDArray)
    _LIB.DLGpuAssignQuantizedEmbedding(
        embedding.handle, newparam.indices.handle, newparam.values.handle,
        qparam.handle, ctypes.c_int(digit), ctypes.c_bool(True),
        stream.handle if stream else None)
