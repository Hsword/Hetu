from __future__ import absolute_import

from .._base import _LIB
from ..ndarray import NDArray


def CuDNN_conv2d(in_arr_x, in_arr_f, out_arr, padding=(0, 0), stride=(1, 1), stream=None):
    assert isinstance(in_arr_x, NDArray)
    assert isinstance(in_arr_f, NDArray)
    assert isinstance(out_arr, NDArray)
    _LIB.CuDNN_DLGpuConv2d(in_arr_x.handle, in_arr_f.handle,
                           out_arr.handle, padding[0], padding[1], stride[0], stride[1], stream.handle if stream else None)


def CuDNN_conv2d_gradient_of_filter(in_arr_x, in_gradient_y, out_gradient_f, padding=(0, 0), stride=(1, 1), stream=None):
    assert isinstance(in_arr_x, NDArray)
    assert isinstance(in_gradient_y, NDArray)
    assert isinstance(out_gradient_f, NDArray)
    _LIB.CuDNN_DLGpuConv2d_Gradient_of_Filter(
        in_arr_x.handle, in_gradient_y.handle, out_gradient_f.handle, padding[0], padding[1], stride[0], stride[1], stream.handle if stream else None)


def CuDNN_conv2d_gradient_of_data(in_arr_f, in_gradient_y, out_gradient_x, padding=(0, 0), stride=(1, 1), stream=None):
    assert isinstance(in_arr_f, NDArray)
    assert isinstance(in_gradient_y, NDArray)
    assert isinstance(out_gradient_x, NDArray)
    _LIB.CuDNN_DLGpuConv2d_Gradient_of_Data(
        in_arr_f.handle, in_gradient_y.handle, out_gradient_x.handle, padding[0], padding[1], stride[0], stride[1], stream.handle if stream else None)
