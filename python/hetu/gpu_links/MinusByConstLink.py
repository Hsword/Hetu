from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd

def minus_by_const(input, output, val, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuMinusByConst(input.handle, output.handle, ctypes.c_float(val), stream.handle if stream else None)
