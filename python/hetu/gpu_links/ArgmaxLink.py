from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def argmax(in_mat, out_mat, dim, stream=None):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    _LIB.DLGpuArgmax(
        in_mat.handle, out_mat.handle, ctypes.c_int(dim), stream.handle if stream else None)
