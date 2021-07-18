from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def CuDNN_softmax_cross_entropy(y, y_, out, stream=None):
    assert isinstance(y, _nd.NDArray)
    assert isinstance(y_, _nd.NDArray)
    assert isinstance(out, _nd.NDArray)
    _LIB.CuDNN_DLGpuSoftmaxEntropy(
        y.handle, y_.handle, out.handle, stream.handle if stream else None)


def CuDNN_softmax_cross_entropy_gradient(grad_arr, y_arr, label, out_arr, stream=None):
    assert isinstance(grad_arr, _nd.NDArray)
    assert isinstance(y_arr, _nd.NDArray)
    assert isinstance(label, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.CuDNN_DLGpuSoftmaxEntropyGradient(
        grad_arr.handle, y_arr.handle, label.handle, out_arr.handle, stream.handle if stream else None)
