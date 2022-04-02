from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def ha2a_layout_transform(input, output, num_nodes, num_local_gpus, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuHA2ALayoutTransform(
        input.handle, output.handle, ctypes.c_int(num_nodes), ctypes.c_int(num_local_gpus), stream.handle if stream else None)


def ha2a_reverse_layout_transform(input, output, num_nodes, num_local_gpus, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuHA2AReverseLayoutTransform(
        input.handle, output.handle, ctypes.c_int(num_nodes), ctypes.c_int(num_local_gpus), stream.handle if stream else None)


