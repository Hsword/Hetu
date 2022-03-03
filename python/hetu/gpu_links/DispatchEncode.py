from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def dispatch_encode_top1(input, indices_s, location_s, output, capacity, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(indices_s, _nd.NDArray)
    assert isinstance(location_s, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
#    print("input_shape:"+str(input.shape))
#    print("indices_shape:"+str(indices_s.shape))
#    print("location_shape:"+str(location_s.shape))
##    print("gates_shape:"+str(gates.shape))
#    print("output_shape:"+str(output.shape))
#    print("capacity:"+str(capacity))
    _LIB.DLGpuDispatchEncodeTop1(
        input.handle, indices_s.handle, location_s.handle, output.handle,\
        ctypes.c_int(capacity), stream.handle if stream else None)

def dispatch_encode_top1_gradient(input, indice, location, output, capacity, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(indice, _nd.NDArray)
    assert isinstance(location, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
#    print("encode_grad_input_shape:", input.shape)
#    print("ctx:", input.ctx)
#    import torch
#    input = torch.load("output_grad.pt").cpu().numpy()
#    import hetu as ht
#    input = ht.array(input, ctx=indice.ctx)
    _LIB.DLGpuDispatchEncodeTop1Gradient(input.handle, indice.handle, location.handle, output.handle, ctypes.c_int(capacity), stream.handle if stream else None)


def dispatch_encode_top2(input, indices_s1, indices_s2, location_s1, location_s2, output, capacity, stream=None):
#    print("input_shape:"+str(input.shape))
#    print("indices_1_shape:"+str(indices_s1.shape))
#    print("indices_2_shape:"+str(indices_s2.shape))
#    print("location_s1_shape:"+str(location_s1.shape))
#    print("location_s2_shape:"+str(location_s2.shape))
#    print("gates_1_shape:"+str(gate_1.shape))
#    print("gates_2_shape:"+str(gate_2.shape))
#    print("output_shape:"+str(output.shape))
#    print("capacity:"+str(capacity))

    
    assert isinstance(input, _nd.NDArray)
    assert isinstance(indices_s1, _nd.NDArray)
    assert isinstance(indices_s2, _nd.NDArray)
    assert isinstance(location_s1, _nd.NDArray)
    assert isinstance(location_s2, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuDispatchEncodeTop2(
        input.handle, indices_s1.handle, indices_s2.handle, location_s1.handle, location_s2.handle, output.handle, ctypes.c_int(capacity), stream.handle if stream else None)
"""
def dispatch_encode_top2_gradient(input, indices_s1, indices_s2, location_s1, location_s2, gates, output, capacity, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(indices_s1, _nd.NDArray)
    assert isinstance(indices_s2, _nd.NDArray)
    assert isinstance(location_s1, _nd.NDArray)
    assert isinstance(location_s2, _nd.NDArray)
    assert isinstance(gates, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuDispatch"""
