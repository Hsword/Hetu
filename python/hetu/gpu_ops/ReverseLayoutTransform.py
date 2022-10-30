from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import reverse_layout_transform_top1, reverse_layout_transform_top2
from ..gpu_links import reverse_layout_transform_top1_gradient_data
from ..gpu_links import reverse_layout_transform_top1_gradient_gate
from ..gpu_links import reverse_layout_transform_top2_gradient_data

# Note: This implementation learns from the design of fast_dispatch from Tutel(https://github.com/microsoft/tutel/blob/v0.1.x/tutel/impls/fast_dispatch.py)

class ReverseLayoutTransformOp(Op):
    def __init__(self, input, indices_s, location_s, gates, capacity, num_experts, ctx=None):
        
        input_node_list = [input, ]
        for node in indices_s:
            input_node_list.append(node)
        for node in location_s:
            input_node_list.append(node)
        for node in gates:
            input_node_list.append(node)

        super().__init__(ReverseLayoutTransformOp, input_node_list, ctx)
        self.capacity = capacity
        self.topK = len(indices_s)
        self.num_experts = num_experts

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            if self.topK == 1:
                reverse_layout_transform_top1(input_vals[0], input_vals[1], input_vals[2], input_vals[3],\
                                        output_val, self.capacity, stream_handle)
            elif self.topK == 2:
                reverse_layout_transform_top2(input_vals[0], input_vals[1], input_vals[2], input_vals[3],\
                                        input_vals[4], input_vals[5],input_vals[6], output_val, self.capacity, stream_handle)
            else:
                raise NotImplementedError

    def gradient(self, output_grad):
        if self.topK == 1:
            return [reverse_layout_transform_gradient_data_op(output_grad, [self.inputs[1]], [self.inputs[2]], [self.inputs[3]], self.capacity, self.num_experts, ctx=self.raw_ctx),]+[None,]*2+[reverse_layout_transform_gradient_gate_op(output_grad, self.inputs[0], self.inputs[1], self.inputs[2], self.capacity, ctx=self.raw_ctx)]
        elif self.topK == 2:
            grad_data = reverse_layout_transform_gradient_data_op(output_grad, [self.inputs[1], self.inputs[2]], [self.inputs[3], self.inputs[4]], [self.inputs[5], self.inputs[6]], self.capacity,self.num_experts, ctx=self.raw_ctx)
            grad_gate_1 = reverse_layout_transform_gradient_gate_op(output_grad, self.inputs[0], self.inputs[1], self.inputs[3], self.capacity, ctx=self.raw_ctx) 
            grad_gate_2 = reverse_layout_transform_gradient_gate_op(output_grad, self.inputs[0], self.inputs[2], self.inputs[4], self.capacity, ctx=self.raw_ctx)
            return [grad_data,]+[None,]*4+[grad_gate_1, grad_gate_2,]
        else:
            assert 1 == -1

    def infer_shape(self, input_shapes):
        if self.topK in [1, 2]:
            num_tokens = input_shapes[1][0]
            embed_dim = input_shapes[0][-1]
            return (num_tokens, embed_dim)
        else:
            assert 1 == -1

    def get_default_state(self, status, enforce_order):
        if enforce_order:
            super().get_default_state(status, enforce_order)
        else:
            status.set_state(None, 1)

class ReverseLayoutTransformGradientDataOp(Op):
    def __init__(self, input, indices_s, location_s, gates, capacity, num_experts, ctx):
        input_node_list = [input, ]
        for node in indices_s:
            input_node_list.append(node)
        for node in location_s:
            input_node_list.append(node)
        for node in gates:
            input_node_list.append(node)
        
        super().__init__(ReverseLayoutTransformGradientDataOp, input_node_list, ctx)
        self.capacity = capacity
        self.num_experts = num_experts
        self.topK = len(indices_s)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            if self.topK == 1:
                reverse_layout_transform_top1_gradient_data(input_vals[0], input_vals[1], input_vals[2], input_vals[3], output_val, self.capacity, stream_handle)
            elif self.topK == 2:
                reverse_layout_transform_top2_gradient_data(input_vals[0], input_vals[1], input_vals[2], input_vals[3], input_vals[4], input_vals[5], input_vals[6], output_val, self.capacity, stream_handle)
            else:
                raise NotImplementedError
        
    
    def infer_shape(self, input_shapes):
        embed_dim = input_shapes[0][-1]
        
        return (int(self.num_experts*self.capacity), embed_dim)

    def gradient(self, output_grad):
        return NotImplementedError

class ReverseLayoutTransformGradientGateOp(Op):
    def __init__(self, combined_output, expert_output, indice, location, capacity, ctx):
        input_node_list = [combined_output, expert_output, indice, location]
        super().__init__(ReverseLayoutTransformGradientGateOp, input_node_list, ctx)
        self.capacity = capacity

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            reverse_layout_transform_top1_gradient_gate(input_vals[0], input_vals[1], input_vals[2], input_vals[3], output_val, self.capacity, stream_handle)


    def infer_shape(self, input_shapes):
        return input_shapes[-1] # shape of gate

    def gradient(self, output_grad):
        return NotImplementedError

def reverse_layout_transform_op(input, indices_s, location_s, gates, capacity, num_experts, ctx=None):
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
    return ReverseLayoutTransformOp(input, indices_s, location_s, gates, capacity, num_experts, ctx=ctx)

def reverse_layout_transform_gradient_data_op(input, indices, locations, gates, capacity, num_experts, ctx=None):
    return ReverseLayoutTransformGradientDataOp(input, indices, locations, gates, capacity, num_experts, ctx=None)

def reverse_layout_transform_gradient_gate_op(combined_output, expert_output, indices, locations, capacity, ctx=None):
    return ReverseLayoutTransformGradientGateOp(combined_output, expert_output, indices, locations, capacity, ctx=ctx)
