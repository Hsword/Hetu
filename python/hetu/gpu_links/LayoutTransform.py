from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def layout_transform_top1(input, indices_s, location_s, output, capacity, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(indices_s, _nd.NDArray)
    assert isinstance(location_s, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuLayoutTransformTop1(
        input.handle, indices_s.handle, location_s.handle, output.handle,\
        ctypes.c_int(capacity), stream.handle if stream else None)

def layout_transform_top1_gradient(input, indice, location, output, capacity, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(indice, _nd.NDArray)
    assert isinstance(location, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuLayoutTransformTop1Gradient(input.handle, indice.handle, location.handle, output.handle, ctypes.c_int(capacity), stream.handle if stream else None)


def layout_transform_top2(input, indices_s1, indices_s2, location_s1, location_s2, output, capacity, stream=None):
    
    assert isinstance(input, _nd.NDArray)
    assert isinstance(indices_s1, _nd.NDArray)
    assert isinstance(indices_s2, _nd.NDArray)
    assert isinstance(location_s1, _nd.NDArray)
    assert isinstance(location_s2, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuLayoutTransformTop2(
        input.handle, indices_s1.handle, indices_s2.handle, location_s1.handle, location_s2.handle, output.handle, ctypes.c_int(capacity), stream.handle if stream else None)
