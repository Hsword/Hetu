from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def softmax_cross_entropy(in_arr_a, in_arr_b, out_arr, stream=None):
    assert isinstance(in_arr_a, _nd.NDArray)
    assert isinstance(in_arr_b, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuSoftmaxCrossEntropy(
        in_arr_a.handle, in_arr_b.handle, out_arr.handle, stream.handle if stream else None)


def softmax_cross_entropy_gradient(in_arr_a, in_arr_b, in_arr_c, out_arr, stream=None):
    assert isinstance(in_arr_a, _nd.NDArray)
    assert isinstance(in_arr_b, _nd.NDArray)
    assert isinstance(in_arr_c, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuSoftmaxCrossEntropy_Gradient(
        in_arr_a.handle, in_arr_b.handle, in_arr_c.handle, out_arr.handle, stream.handle if stream else None)
