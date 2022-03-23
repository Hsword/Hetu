from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd

def nll_loss_link(input, target, output, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(target, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuNllLoss(input.handle, target.handle, output.handle, stream.handle if stream else None)

def nll_loss_grad_link(output_grad, target, input_grad, stream=None):
    assert isinstance(output_grad, _nd.NDArray)
    assert isinstance(target, _nd.NDArray)
    assert isinstance(input_grad, _nd.NDArray)

    _LIB.DLGpuNllLossGrad(output_grad.handle, target.handle, input_grad.handle, stream.handle if stream else None)
