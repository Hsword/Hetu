from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import layout_transform_top1, layout_transform_top2
from ..gpu_links import layout_transform_top1_gradient
from .AddElewise import add_op


# Note: This implementation learns from the design of fast_dispatch from Tutel(https://github.com/microsoft/tutel/blob/v0.1.x/tutel/impls/fast_dispatch.py) 

class LayoutTransformOp(Op):
    def __init__(self, input, indices_s, location_s, capacity, total_experts, ctx=None):
        
        input_node_list = [input, ]
        for node in indices_s:
            input_node_list.append(node)
        for node in location_s:
            input_node_list.append(node)

        super().__init__(LayoutTransformOp, input_node_list, ctx)
        self.capacity = capacity
        self.topK = len(indices_s)
        self.total_experts = total_experts

    def compute(self, input_vals, output_val, stream_handle=None):
#        output_val = ht.ndarray(np.zeros(size = ), ctx = output_val.ctx)
        if self.on_cpu:
            raise NotImplementedError
        else:
            if self.topK == 1:
                layout_transform_top1(input_vals[0], input_vals[1], input_vals[2],\
                                        output_val, self.capacity, stream_handle)
            elif self.topK == 2:
                layout_transform_top2(input_vals[0], input_vals[1], input_vals[2], input_vals[3],\
                                        input_vals[4], output_val, self.capacity, stream_handle)
            else:
                raise NotImplementedError

    def gradient(self, output_grad):
        if self.topK == 1:
            return [layout_transform_gradient_op(output_grad, self.inputs[1], self.inputs[2], self.capacity, ctx=self.raw_ctx),]+[None,]*2
        elif self.topK == 2:
            result_1 = layout_transform_gradient_op(output_grad, self.inputs[1], self.inputs[3], self.capacity, ctx=self.raw_ctx)
            result_2 = layout_transform_gradient_op(output_grad, self.inputs[2], self.inputs[4], self.capacity, ctx=self.raw_ctx)
            result = add_op(result_1, result_2, ctx = self.raw_ctx)
            return [result,]+[None,]*4
        else:
            assert 1 == -1

    def infer_shape(self, input_shapes):
        if self.topK in [1, 2]:
            embed_dim = input_shapes[0][-1]
            return (int(self.capacity * self.total_experts), embed_dim)
        else:
            assert 1 == -1

    def get_default_state(self, status, enforce_order):
        if enforce_order:
            super().get_default_state(status, enforce_order)
        else:
            status.set_state(None, 1)

class LayoutTransformGradientOp(Op):
    def __init__(self, input, indice, location, capacity, ctx=None):
        input_node_list = [input, indice, location]
        super().__init__(LayoutTransformGradientOp, input_node_list, ctx)
        self.capacity = capacity

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            layout_transform_top1_gradient(input_vals[0], input_vals[1], input_vals[2], output_val, self.capacity, stream_handle)

    def infer_shape(self, input_shapes):
        model_dim = input_shapes[0][-1]
        num_tokens = input_shapes[1][0]
        return (num_tokens, model_dim)
    
    def gradient(self, output_grad):
        raise NotImplementedError


def layout_transform_op(input, indices_s, location_s, capacity, total_experts, ctx=None):
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
    return LayoutTransformOp(input, indices_s, location_s, capacity, total_experts, ctx=ctx)

def layout_transform_gradient_op(input, indice, location,capacity, ctx=None):
    return LayoutTransformGradientOp(input, indice, location, capacity, ctx=ctx)
