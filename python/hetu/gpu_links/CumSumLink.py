from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def cumsum_with_bias(input, output, bias, dim, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuCumsumWithBias(
        input.handle, output.handle, ctypes.c_float(bias), ctypes.c_int(dim), stream.handle if stream else None)
