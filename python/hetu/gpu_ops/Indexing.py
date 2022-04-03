from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import indexing, indexing_grad

class IndexingOp(Op):
    def __init__(self, input, index, ctx=None):
        
        input_node_list = [input, index]
        super().__init__(IndexingOp, input_node_list, ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            indexing(input_vals[0], input_vals[1],output_val, stream_handle)

    def gradient(self, output_grad):
        return [indexing_grad_op(output_grad, self.inputs[1], ctx=self.raw_ctx)]+[None,]

    def infer_shape(self, input_shapes):
        return input_shapes[0]


class IndexingGradOp(Op):
        def __init__(self, input, index, ctx=None):                
            input_node_list = [input, index]
            super().__init__(IndexingGradOp, input_node_list, ctx)

        def compute(self, input_vals, output_val, stream_handle=None):        
            if self.on_cpu:                    
                raise NotImplementedError    
            else:
                indexing_grad(input_vals[0], input_vals[1],output_val, stream_handle)

        def gradient(self, output_grad):
            raise NotImplementedError
                                        
        def infer_shape(self, input_shapes):
            return input_shapes[0]

def indexing_grad_op(output_grad, index, ctx=None):
    return IndexingGradOp(output_grad, index, ctx=ctx)

def indexing_op(input_mat, index_mat, ctx=None):
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
    return IndexingOp(input_mat, index_mat, ctx=ctx)
