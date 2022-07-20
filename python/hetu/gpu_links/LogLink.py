from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def log_link(input, output, eps, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuLog(input.handle, output.handle, ctypes.c_float(
        eps), stream.handle if stream else None)


def log_grad_link(output_grad, input, input_grad, eps, stream=None):
    assert isinstance(output_grad, _nd.NDArray)
    assert isinstance(input, _nd.NDArray)
    assert isinstance(input_grad, _nd.NDArray)
    _LIB.DLGpuLogGrad(output_grad.handle, input.handle, input_grad.handle,
                      ctypes.c_float(eps), stream.handle if stream else None)
