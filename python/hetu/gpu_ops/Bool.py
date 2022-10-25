from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import bool, bool_val, bool_matrix

class BoolOp(Op):
    def __init__(self, node, ctx=None):
        super().__init__(BoolOp, [node], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            bool(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]

    def get_default_state(self, status, enforce_order):
        if enforce_order:
            super().get_default_state(status, enforce_order)
        else:
            status.set_state(None, 1)

class BoolValOp(Op):
    def __init__(self, node, val, cond=0, ctx=None):
        super().__init__(BoolValOp, [node], ctx)
        self.val = val
        self.cond = cond

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            input = input_vals[0].asnumpy()
            if(self.cond==0):
                output_val[:] = (input==self.val).astype(np.float32)
            elif(self.cond==1):
                output_val[:] = (input<self.val).astype(np.float32)   
            elif(self.cond==2):
                output_val[:] = (input>self.val).astype(np.float32)
            elif(self.cond==3):
                output_val[:] = (input<=self.val).astype(np.float32)     
            elif(self.cond==4):
                output_val[:] = (input>=self.val).astype(np.float32)        
        else:
            bool_val(input_vals[0], output_val, self.val, self.cond, stream_handle)

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]

        
class BoolMatrixOp(Op):
    def __init__(self, node_A, node_B, cond, ctx=None):
        super().__init__(BoolMatrixOp, [node_A, node_B], ctx)
        self.cond = cond

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            input_a = input_vals[0].asnumpy()
            input_b = input_vals[1].asnumpy()
            if(self.cond==0):
                output_val[:] = (input_a==input_b).astype(np.float32)
            elif(self.cond==1):
                output_val[:] = (input_a<input_b).astype(np.float32)   
            elif(self.cond==2):
                output_val[:] = (input_a>input_b).astype(np.float32)
            elif(self.cond==3):
                output_val[:] = (input_a<=input_b).astype(np.float32)     
            elif(self.cond==4):
                output_val[:] = (input_a>=input_b).astype(np.float32)                                                
        else:
            bool_matrix(input_vals[0], input_vals[1], output_val, self.cond, stream_handle)

    def gradient(self, output_grad):
        return [None, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        if len(input_shapes[0])==len(input_shapes[1]):
            for i in range(len(input_shapes[0])):
                assert input_shapes[0][i] == input_shapes[1][i]
            return input_shapes[0]
        assert len(input_shapes[0])==1
        assert len(input_shapes[1])==2
        assert input_shapes[1][1]==1
        assert input_shapes[0][0]== input_shapes[1][0]
        return (input_shapes[0][0], input_shapes[0][0])


def bool_op(node, input=None, val=None, cond=0, ctx=None):
    """Boolean Node.

    Parameters:
    ----
    node : Node
        Input variable.
    input : Node
        Input variable.
    val : float
        Const to compare.   
    cond : Int
        Condition.
    Returns:
    ----
    A new Node instance created by Op.

    """
    if(input is None and val is None):
        return BoolOp(node, ctx=ctx)
    if(input is not None and val is None):
        return BoolMatrixOp(node, input, cond=cond, ctx=ctx)
    if(input is None and val is not None):
        return BoolValOp(node, val, cond=cond, ctx=ctx)
    assert False