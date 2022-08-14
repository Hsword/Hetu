
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


def quantized_embedding_lookup(in_arr, ind_arr, out_arr, qparams, digit, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(ind_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    assert isinstance(qparams, _nd.NDArray)
    _LIB.DLGpuQuantizedEmbeddingLookup(in_arr.handle, ind_arr.handle, out_arr.handle,
                                       qparams.handle, ctypes.c_int(digit), stream.handle if stream else None)


def update_quantized_embedding(grad_arr, ind_arr, embed, qparams, lookup, digit, stream=None):
    assert isinstance(grad_arr, _nd.NDArray)
    assert isinstance(ind_arr, _nd.NDArray)
    assert isinstance(embed, _nd.NDArray)
    assert isinstance(qparams, _nd.NDArray)
    assert isinstance(lookup, _nd.NDArray)
    _LIB.DLGpuSGDUpdateQuantizedEmbedding(grad_arr.handle, ind_arr.handle, embed.handle,
                                       qparams.handle, lookup.handle, ctypes.c_int(digit), stream.handle if stream else None)
