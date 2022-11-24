
from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def eye(out_arr, stream=None):
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuEye(out_arr.handle, stream.handle if stream else None)
