from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import reverse_layout_transform_no_gate, reverse_layout_transform_no_gate_gradient


# Note: This implementation learns from the design of fast_dispatch from Tutel(https://github.com/microsoft/tutel/blob/v0.1.x/tutel/impls/fast_dispatch.py)

class ReverseLayoutTransformNoGateOp(Op):
    def __init__(self, input, indices_s, location_s, capacity, num_experts, ctx=None):
        
        input_node_list = [input, ]
        for node in indices_s:
            input_node_list.append(node)
        for node in location_s:
            input_node_list.append(node)

        super().__init__(ReverseLayoutTransformNoGateOp, input_node_list, ctx)
        self.capacity = capacity
        self.num_experts = num_experts

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else: 
            reverse_layout_transform_no_gate(input_vals[0], input_vals[1], input_vals[2], \
                                        output_val, self.capacity, stream_handle)

    def gradient(self, output_grad):
        return [reverse_layout_transform_no_gate_gradient_op(output_grad, [self.inputs[1]], [self.inputs[2]], self.capacity, self.num_experts, ctx=self.raw_ctx),]+[None,]*3

    def infer_shape(self, input_shapes):
        num_tokens = input_shapes[1][0]
        embed_dim = input_shapes[0][-1]
        return (num_tokens, embed_dim)

    def get_default_state(self, status, enforce_order):
        if enforce_order:
            super().get_default_state(status, enforce_order)
        else:
            status.set_state(None, 1)

class ReverseLayoutTransformNoGateGradientOp(Op):
    def __init__(self, input, indices_s, location_s, capacity, num_experts, ctx):
        input_node_list = [input, ]
        for node in indices_s:
            input_node_list.append(node)
        for node in location_s:
            input_node_list.append(node)
        
        super().__init__(ReverseLayoutTransformNoGateGradientOp, input_node_list, ctx)
        self.capacity = capacity
        self.num_experts = num_experts

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else: 
            reverse_layout_transform_no_gate_gradient(input_vals[0], input_vals[1], input_vals[2], output_val, self.capacity, stream_handle)
    
    def infer_shape(self, input_shapes):
        embed_dim = input_shapes[0][-1]
        
        return (int(self.num_experts*self.capacity), embed_dim)

    def gradient(self, output_grad):
        return NotImplementedError


def reverse_layout_transform_no_gate_op(input, indices_s, location_s, capacity, num_experts, ctx=None):
    """Calculate the dispatch encode.

    Parameters:
    ----
    indices_s,
    location_s,
    gates,
    capacity,
    Returns:
    ----
    A new Node instance created by Op.

    """
    return ReverseLayoutTransformNoGateOp(input, indices_s, location_s, capacity, num_experts, ctx=ctx)

def reverse_layout_transform_no_gate_gradient_op(input, indices, locations, capacity, num_experts, ctx=None):
    return ReverseLayoutTransformNoGateGradientOp(input, indices, locations, capacity, num_experts, ctx=None)

