from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def dispatch_decode_top1(input, indices_s, location_s, gates, output, capacity, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(indices_s, _nd.NDArray)
    assert isinstance(location_s, _nd.NDArray)
    assert isinstance(gates, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
#    print("input_shape: ",input.shape)
#    print("indice_shape: ", indices_s.shape)
#    print("location_shaoe: ", location_s.shape)
#    print("gate_shape: ", gates.shape)
#    print("output_shape: ", output.shape)
    
    _LIB.DLGpuDispatchDecodeTop1(
        input.handle, indices_s.handle, location_s.handle, gates.handle, output.handle,\
        ctypes.c_int(capacity), stream.handle if stream else None)

def dispatch_decode_top1_gradient_gate(combined_output, expert_output, indice, location, output, capacity, stream=None):
    assert isinstance(combined_output, _nd.NDArray)
    assert isinstance(expert_output, _nd.NDArray)
    assert isinstance(indice, _nd.NDArray)
    assert isinstance(location, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
#    import hetu as ht
#    if indice.ctx == ht.gpu(0):
#        print("output_grad_shape=", combined_output.shape)
#        print("expert_output_shape=", expert_output.shape)
#        print("indice_shape=", indice.shape)
#        print("location_shape=", location.shape)
#        print("output_shape:", output.shape)

    _LIB.DLGpuDispatchDecodeTop1GradientGate(combined_output.handle, expert_output.handle, indice.handle, location.handle, output.handle, ctypes.c_int(capacity), stream.handle if stream else None)


def dispatch_decode_top1_gradient_data(input, indice, location, gate, output, capacity, stream=None):
    assert isinstance(input, _nd.NDArray)                
    assert isinstance(indice, _nd.NDArray)                
    assert isinstance(location, _nd.NDArray)                
    assert isinstance(gate, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)                
#    import torch
#    input=torch.load("combined_output.pt").cpu().numpy()
#    import hetu as ht
#    input=ht.array(input, indice.ctx)

    _LIB.DLGpuDispatchDecodeTop1GradientData(input.handle, indice.handle, location.handle, gate.handle, output.handle, ctypes.c_int(capacity), stream.handle if stream else None)




def dispatch_decode_top2(input, indices_s1, indices_s2, location_s1, location_s2, gates_1, gates_2, output, capacity, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(indices_s1, _nd.NDArray)
    assert isinstance(indices_s2, _nd.NDArray)
    assert isinstance(location_s1, _nd.NDArray)
    assert isinstance(location_s2, _nd.NDArray)
    assert isinstance(gates_1, _nd.NDArray)
    assert isinstance(gates_2, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
#    import hetu as ht
#    if input.ctx == ht.gpu(0):
#        print("input_shape=", input.shape)
##        print("indice_1_shape=", indices_s1.shape)
#        print("indice_2_shape=", indices_s2.shape)
#        print("location_1_shape=", location_s1.shape)
#        print("location_2_shape=", location_s2.shape)
##        print("gate_1_shape", gates_1.shape)
#       print("gate_2_shape", gates_2.shape)
#       print("decode_top2_output_shape:", output.shape)
    _LIB.DLGpuDispatchDecodeTop2(
        input.handle, indices_s1.handle, indices_s2.handle, location_s1.handle, location_s2.handle,\
            gates_1.handle, gates_2.handle, output.handle, ctypes.c_int(capacity), stream.handle if stream else None)

#    print("decode_fwd_finish")

def dispatch_decode_top2_gradient_data(input, indice_1, indice_2, location_1, location_2, gate_1, gate_2, output, capacity, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(indice_1, _nd.NDArray)
    assert isinstance(indice_2, _nd.NDArray)
    assert isinstance(location_1, _nd.NDArray)
    assert isinstance(location_2, _nd.NDArray)
    assert isinstance(gate_1, _nd.NDArray)
    assert isinstance(gate_2, _nd.NDArray)
    _LIB.DLGpuDispatchDecodeTop2GradientData(input.handle, indice_1.handle, indice_2.handle, location_1.handle, location_2.handle, gate_1.handle, gate_2.handle, output.handle, ctypes.c_int(capacity), stream.handle if stream else None)







