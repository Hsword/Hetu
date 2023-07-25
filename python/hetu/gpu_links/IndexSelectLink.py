from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def index_select(input, index, output, dim, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(index, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)

    _LIB.DLGpuIndexSelect(
        input.handle, index.handle, output.handle, ctypes.c_int(dim), stream.handle if stream else None)


def index_select_grad(grad, index, output, dim, stream=None):
    assert isinstance(grad, _nd.NDArray)
    assert isinstance(index, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)

    _LIB.DLGpuIndexSelectGrad(
        grad.handle, index.handle, output.handle, ctypes.c_int(dim), stream.handle if stream else None)
