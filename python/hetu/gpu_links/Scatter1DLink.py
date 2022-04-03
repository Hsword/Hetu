from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd

def scatter1d(input_mat, index_mat,output_mat, stream=None):
    assert isinstance(input_mat, _nd.NDArray);
    assert isinstance(index_mat, _nd.NDArray);
    assert isinstance(output_mat, _nd.NDArray);
    
    _LIB.DLGpuScatter1D(
            input_mat.handle,index_mat.handle, output_mat.handle, stream.handle if stream else None)


def scatter1d_grad(output_grad_mat, index_mat, input_grad_mat, stream=None):
    assert isinstance(output_grad_mat, _nd.NDArray)
    assert isinstance(index_mat, _nd.NDArray)
    assert isinstance(input_grad_mat, _nd.NDArray)

    _LIB.DLGpuScatter1DGrad(
            output_grad_mat.handle, index_mat.handle, input_grad_mat.handle, stream.handle if stream else None)
