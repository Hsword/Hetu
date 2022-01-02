from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def cross_entropy_sparse(y, y_, ignored_index, out, stream=None):
    assert isinstance(y, _nd.NDArray)
    assert isinstance(y_, _nd.NDArray)
    assert isinstance(out, _nd.NDArray)
    _LIB.DLGpuCrossEntropySparse(
        y.handle, y_.handle, ignored_index, out.handle, stream.handle if stream else None)


def cross_entropy_sparse_gradient(grad_arr, y_arr, label, ignored_index, out_arr, stream=None):
    assert isinstance(grad_arr, _nd.NDArray)
    assert isinstance(y_arr, _nd.NDArray)
    assert isinstance(label, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuCrossEntropySparseGradient(
        grad_arr.handle, y_arr.handle, label.handle, ignored_index, out_arr.handle, stream.handle if stream else None)
