from __future__ import absolute_import

from ctypes import c_bool
from .._base import _LIB
from .. import ndarray as _nd


def minimum_distance_vector(lookup, key, codebook, indices, output, mode, stream=None):
    assert isinstance(lookup, _nd.NDArray)
    assert isinstance(key, _nd.NDArray)
    assert isinstance(codebook, _nd.NDArray)
    assert isinstance(indices, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    if mode == 'eu':
        cmode = True
    else:
        cmode = False
    _LIB.DLGpuMinDist(lookup.handle, key.handle, codebook.handle,
                      indices.handle, output.handle, c_bool(cmode), stream.handle if stream else None)
