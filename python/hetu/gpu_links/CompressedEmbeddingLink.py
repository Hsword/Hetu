from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def robe_hash(in_arr, rand_arr, out_arr, length, dim, Z, use_slot_coef, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(rand_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuRobeHash(in_arr.handle, rand_arr.handle, out_arr.handle, ctypes.c_int(length), ctypes.c_int(
        dim), ctypes.c_int(Z), ctypes.c_bool(use_slot_coef), stream.handle if stream else None)


def robe_sign(in_arr, rand_arr, out_arr, dim, use_slot_coef, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(rand_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuRobeSign(in_arr.handle, rand_arr.handle, out_arr.handle, ctypes.c_int(
        dim), ctypes.c_bool(use_slot_coef), stream.handle if stream else None)


def mod_hash(in_arr, out_arr, nembed, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuModHash(in_arr.handle, out_arr.handle, ctypes.c_int(
        nembed), stream.handle if stream else None)


def mod_hash_negative(in_arr, out_arr, nembed, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuModHashNegative(in_arr.handle, out_arr.handle, ctypes.c_int(
        nembed), stream.handle if stream else None)


def div_hash(in_arr, out_arr, nembed, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuDivHash(in_arr.handle, out_arr.handle, ctypes.c_int(
        nembed), stream.handle if stream else None)


def compo_hash(in_arr, out_arr, ntable, nembed, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuCompoHash(in_arr.handle, out_arr.handle, ctypes.c_int(
        ntable), ctypes.c_int(nembed), stream.handle if stream else None)


def learn_hash(in_arr, slope, bias, prime, out_arr, nbucket, normal, eps, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(slope, _nd.NDArray)
    assert isinstance(bias, _nd.NDArray)
    assert isinstance(prime, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuLearnHash(in_arr.handle, slope.handle, bias.handle, prime.handle, out_arr.handle, ctypes.c_int(
        nbucket), ctypes.c_bool(normal), ctypes.c_float(eps), stream.handle if stream else None)
