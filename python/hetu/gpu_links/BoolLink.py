from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def bool(input, output, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuBool(
        input.handle, output.handle, stream.handle if stream else None)


def bool_val(input, output, val, cond, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuBoolVal(
        input.handle, ctypes.c_float(val), output.handle, ctypes.c_int(cond), stream.handle if stream else None)


def bool_matrix(input_A, input_B, output, cond, stream=None):
    assert isinstance(input_A, _nd.NDArray)
    assert isinstance(input_B, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuBoolMatrix(
        input_A.handle, input_B.handle, output.handle, ctypes.c_int(cond), stream.handle if stream else None)
