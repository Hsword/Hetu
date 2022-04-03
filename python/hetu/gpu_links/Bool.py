from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def bool(input, output, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuBool(
        input.handle, output.handle, stream.handle if stream else None)
