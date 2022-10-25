from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def argsort(input, output, index, output_index, dim, descending, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    assert isinstance(index, _nd.NDArray)
    assert isinstance(output_index, _nd.NDArray)

    _LIB.DLGpuArgsort(
        input.handle, output.handle, index.handle, output_index.handle, ctypes.c_int(dim), descending, stream.handle if stream else None)
