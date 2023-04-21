from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def reduce_norm2_raw(in_arr, out_arr, offset, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    axes = list(range(len(in_arr.shape)))
    pointer_func = ctypes.c_int * len(axes)
    pointer = pointer_func(*list(axes))
    _LIB.DLGpuReduceNorm2Raw(
        in_arr.handle, out_arr.handle, pointer, ctypes.c_int(len(axes)), ctypes.c_int(offset), stream.handle if stream else None)


def all_fro_norm(grads, workspace, norm, stream=None):
    new_grads = []
    for grad in grads:
        if isinstance(grad, _nd.IndexedSlices):
            grad = grad.values
        assert isinstance(grad, _nd.NDArray)
        new_grads.append(grad)
    assert isinstance(workspace, _nd.NDArray)
    assert isinstance(norm, _nd.NDArray)
    for i, grad in enumerate(new_grads):
        reduce_norm2_raw(grad, workspace, i, stream)
    reduce_norm2_raw(workspace, norm, 0, stream)


def all_add_(tensors, others, alpha, cons=1, stream=None):
    st_handle = stream.handle if stream else None
    assert isinstance(alpha, _nd.NDArray)
    assert len(tensors) == len(others)
    for tensor, other in zip(tensors, others):
        assert isinstance(tensor, _nd.NDArray)
        if isinstance(other, _nd.IndexedSlices):
            other = other.values
        assert isinstance(other, _nd.NDArray)
        assert tensor.shape == other.shape
        _LIB.DLGpuAdd_(tensor.handle, other.handle, alpha.handle,
                       ctypes.c_float(cons), st_handle)


def div_n_mul_(tensor, alpha, cons, stream=None):
    st_handle = stream.handle if stream else None
    assert isinstance(alpha, _nd.NDArray)
    assert isinstance(tensor, _nd.NDArray)
    _LIB.DLGpuDivMul(tensor.handle, alpha.handle,
                     ctypes.c_float(cons), st_handle)
