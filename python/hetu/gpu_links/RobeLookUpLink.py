from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def robe_lookup(in_mat, ids, x, out_mat, len, Bg, Cg, Dg, stream=None):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(ids, _nd.NDArray)
    assert isinstance(x, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    _LIB.DLGpuRobeLookUp(
        in_mat.handle, ids.handle, x.handle, out_mat.handle, ctypes.c_int(
        len), ctypes.c_int(
        Bg), ctypes.c_int(
        Cg), ctypes.c_int(
        Dg), stream.handle if stream else None)


def robe_lookup_gradient(grad_out, ids, grad_in, stream=None):
    assert(0)
    assert isinstance(grad_out, _nd.NDArray)
    assert isinstance(ids, _nd.NDArray)
    assert isinstance(grad_in, _nd.NDArray)
    _LIB.DLGpuRobeLookUp_Gradient(
        grad_out.handle, ids.handle, grad_in.handle, stream.handle if stream else None)
