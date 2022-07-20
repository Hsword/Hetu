from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def mod_hash(in_arr, out_arr, nembed, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuModHash(in_arr.handle, out_arr.handle, ctypes.c_int(
        nembed), stream.handle if stream else None)


def compo_hash(in_arr, out_arr, ntable, nembed, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuCompoHash(in_arr.handle, out_arr.handle, ctypes.c_int(
        ntable), ctypes.c_int(nembed), stream.handle if stream else None)


def learn_hash(in_arr, slope, bias, prime, out_arr, nbucket, normal, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(slope, _nd.NDArray)
    assert isinstance(bias, _nd.NDArray)
    assert isinstance(prime, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuLearnHash(in_arr.handle, slope.handle, bias.handle, prime.handle, out_arr.handle, ctypes.c_int(
        nbucket), ctypes.c_bool(normal), stream.handle if stream else None)
