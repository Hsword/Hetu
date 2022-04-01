from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def reverse_layout_transform_top1(input, indices_s, location_s, gates, output, capacity, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(indices_s, _nd.NDArray)
    assert isinstance(location_s, _nd.NDArray)
    assert isinstance(gates, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)

    _LIB.DLGpuReverseLayoutTransformTop1(
        input.handle, indices_s.handle, location_s.handle, gates.handle, output.handle,\
        ctypes.c_int(capacity), stream.handle if stream else None)

def reverse_layout_transform_top1_gradient_gate(combined_output, expert_output, indice, location, output, capacity, stream=None):
    assert isinstance(combined_output, _nd.NDArray)
    assert isinstance(expert_output, _nd.NDArray)
    assert isinstance(indice, _nd.NDArray)
    assert isinstance(location, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)

    _LIB.DLGpuReverseLayoutTransformTop1GradientGate(combined_output.handle, expert_output.handle, indice.handle, location.handle, output.handle, ctypes.c_int(capacity), stream.handle if stream else None)


def reverse_layout_transform_top1_gradient_data(input, indice, location, gate, output, capacity, stream=None):
    assert isinstance(input, _nd.NDArray)                
    assert isinstance(indice, _nd.NDArray)                
    assert isinstance(location, _nd.NDArray)                
    assert isinstance(gate, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)                

    _LIB.DLGpuReverseLayoutTransformTop1GradientData(input.handle, indice.handle, location.handle, gate.handle, output.handle, ctypes.c_int(capacity), stream.handle if stream else None)




def reverse_layout_transform_top2(input, indices_s1, indices_s2, location_s1, location_s2, gates_1, gates_2, output, capacity, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(indices_s1, _nd.NDArray)
    assert isinstance(indices_s2, _nd.NDArray)
    assert isinstance(location_s1, _nd.NDArray)
    assert isinstance(location_s2, _nd.NDArray)
    assert isinstance(gates_1, _nd.NDArray)
    assert isinstance(gates_2, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuReverseLayoutTransformTop2(
        input.handle, indices_s1.handle, indices_s2.handle, location_s1.handle, location_s2.handle,\
            gates_1.handle, gates_2.handle, output.handle, ctypes.c_int(capacity), stream.handle if stream else None)


def reverse_layout_transform_top2_gradient_data(input, indice_1, indice_2, location_1, location_2, gate_1, gate_2, output, capacity, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(indice_1, _nd.NDArray)
    assert isinstance(indice_2, _nd.NDArray)
    assert isinstance(location_1, _nd.NDArray)
    assert isinstance(location_2, _nd.NDArray)
    assert isinstance(gate_1, _nd.NDArray)
    assert isinstance(gate_2, _nd.NDArray)
    _LIB.DLGpuReverseLayoutTransformTop2GradientData(input.handle, indice_1.handle, indice_2.handle, location_1.handle, location_2.handle, gate_1.handle, gate_2.handle, output.handle, ctypes.c_int(capacity), stream.handle if stream else None)



def reverse_layout_transform_no_gate(input, indices_s, location_s, output, capacity, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(indices_s, _nd.NDArray)
    assert isinstance(location_s, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)

    
    _LIB.DLGpuReverseLayoutTransformNoGate(input.handle, indices_s.handle, location_s.handle, output.handle,\
                                    ctypes.c_int(capacity), stream.handle if stream else None)


def reverse_layout_transform_no_gate_gradient(input, indice, location, output, capacity, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(indice, _nd.NDArray)
    assert isinstance(location, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    
    _LIB.DLGpuReverseLayoutTransformNoGateGradient(input.handle, indice.handle, location.handle, output.handle, ctypes.c_int(capacity), stream.handle if stream else None)
