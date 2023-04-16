from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def argmax(in_mat, out_mat, dim, stream=None):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    _LIB.DLGpuArgmax(
        in_mat.handle, out_mat.handle, ctypes.c_int(dim), stream.handle if stream else None)


def argmax_partial(in_mat, full_mask, out_mat, dim, topk, stream=None):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(full_mask, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    _LIB.DLGpuArgmaxPartial(
        in_mat.handle, full_mask.handle, out_mat.handle, ctypes.c_int(dim), ctypes.c_int(topk), stream.handle if stream else None)
