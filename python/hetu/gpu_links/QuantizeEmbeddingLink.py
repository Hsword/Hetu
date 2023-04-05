
from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def embedding_prepack(in_arr, out_arr, qparams, digit, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    assert isinstance(qparams, _nd.NDArray)
    _LIB.DLGpuPrepackEmbedding(in_arr.handle, out_arr.handle, qparams.handle, ctypes.c_int(
        digit), stream.handle if stream else None)


def unified_quantized_embedding_lookup(in_arr, ind_arr, out_arr, digit, scale, minele, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(ind_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuUnifiedQuantizedEmbeddingLookup(in_arr.handle, ind_arr.handle, out_arr.handle, ctypes.c_int(
        digit), ctypes.c_float(scale), ctypes.c_float(minele), stream.handle if stream else None)


def quantized_embedding_lookup(in_arr, ind_arr, out_arr, qparams, digit, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(ind_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    assert isinstance(qparams, _nd.NDArray)
    _LIB.DLGpuQuantizedEmbeddingLookup(in_arr.handle, ind_arr.handle, out_arr.handle,
                                       qparams.handle, ctypes.c_int(digit), stream.handle if stream else None)
