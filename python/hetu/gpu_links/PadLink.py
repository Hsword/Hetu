from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def pad(in_arr, out_arr, paddings, mode='CONSTANT', constant_values=0, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    padding_arr = []
    for i in range(len(paddings)):
        for j in range(len(paddings[0])):
            padding_arr.append(paddings[i][j])
    pad_len = len(padding_arr)
    padding_c_arr = (ctypes.c_int * pad_len)(*padding_arr)
    f_type = 3
    if mode == 'CONSTANT':
        f_type = 0
    elif mode == 'REFLECT':
        f_type = 1
    elif mode == 'SYMMETRIC':
        f_type = 2
    assert(f_type <= 2)
    _LIB.DLGpuPad(in_arr.handle, out_arr.handle, padding_c_arr,
                  pad_len, f_type, constant_values, stream.handle if stream else None)


def pad_gradient(out_grad_arr, in_grad_arr, paddings, mode="CONSTANT", stream=None):
    assert isinstance(out_grad_arr, _nd.NDArray)
    assert isinstance(in_grad_arr, _nd.NDArray)
    padding_arr = []
    for i in range(len(paddings)):
        for j in range(len(paddings[0])):
            padding_arr.append(paddings[i][j])
    pad_len = len(padding_arr)
    padding_c_arr = (ctypes.c_int * pad_len)(*padding_arr)
    f_type = 3
    if mode == 'CONSTANT':
        f_type = 0
    elif mode == 'REFLECT':
        f_type = 1
    elif mode == 'SYMMETRIC':
        f_type = 2
    assert(f_type <= 2)
    _LIB.DLGpuPad_gradient(out_grad_arr.handle,
                           in_grad_arr.handle, padding_c_arr, pad_len, f_type, stream.handle if stream else None)
