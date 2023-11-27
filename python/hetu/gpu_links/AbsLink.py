from __future__ import absolute_import

from .._base import _LIB
from .. import ndarray as _nd


def abs_val(in_mat, out_mat, stream=None):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    _LIB.DLGpuAbs(in_mat.handle, out_mat.handle, stream.handle if stream else None)

def abs_gradient(in_mat, grad_mat, out_mat, stream=None):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(grad_mat, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    _LIB.DLGpuAbsGradient(grad_mat.handle, in_mat.handle, out_mat.handle, stream.handle if stream else None)
