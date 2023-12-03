from __future__ import absolute_import

from .._base import _LIB
from .. import ndarray as _nd


def mask_func(input, mask, output, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(mask, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuMask(input.handle, mask.handle, output.handle,
                   stream.handle if stream else None)
