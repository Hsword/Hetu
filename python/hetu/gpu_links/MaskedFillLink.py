from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def masked_fill(input, mask, output, val, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(mask, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuMaskedFill(input.handle, mask.handle, ctypes.c_float(
        val), output.handle, stream.handle if stream else None)
