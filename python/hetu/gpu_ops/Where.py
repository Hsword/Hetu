from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import where, where_const


class WhereOp(Op):
    def __init__(self, cond, node_A, node_B, ctx=None):
        super().__init__(WhereOp, [cond, node_A, node_B], ctx)
        self.check_reset = False

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.where(input_vals[0].asnumpy(
            ), input_vals[1].asnumpy(), input_vals[2].asnumpy())
        else:
            if not self.check_reset:
                where(input_vals[0], input_vals[1], input_vals[2], output_val, stream_handle)
            else:
                if input_vals[0].shape!=input_vals[1].shape and input_vals[2].shape!=input_vals[1].shape:
                    middle_result_cond = ndarray.NDArray(None)
                    middle_result_node = ndarray.NDArray(None)
                    input_vals[0].broadcast_to(input_vals[1].shape, middle_result_cond)
                    input_vals[2].broadcast_to(input_vals[1].shape, middle_result_node)
                    self._reset_gpu_buffer(middle_result_cond, input_vals[1], middle_result_node, output_val)
                    where_broadcast(middle_result_cond, input_vals[1], middle_result_node, output_val, self.gpu_buffer, stream_handle)
                elif input_vals[0].shape!=input_vals[1].shape and input_vals[2].shape==input_vals[1].shape:
                    middle_result_cond = ndarray.NDArray(None)
                    input_vals[0].broadcast_to(input_vals[1].shape, middle_result_cond)
                    self._reset_gpu_buffer(middle_result_cond, input_vals[1], input_vals[2], output_val)        
                    where_broadcast(middle_result_cond, input_vals[1], input_vals[2], output_val, self.gpu_buffer, stream_handle)           
                elif input_vals[0].shape==input_vals[1].shape and input_vals[2].shape!=input_vals[1].shape:
                    middle_result_node = ndarray.NDArray(None)
                    input_vals[2].broadcast_to(input_vals[1].shape, middle_result_node)
                    self._reset_gpu_buffer(input_vals[0], input_vals[1], middle_result_node, output_val)    
                    where_broadcast(input_vals[0], input_vals[1], middle_result_node, output_val, self.gpu_buffer, stream_handle)     
                      
    def gradient(self, output_grad):
        from .ZerosLike import zeroslike_op
        zeros = zeroslike_op(self.inputs[0], ctx=self.raw_ctx)
        grad_A = where_op(self.inputs[0], output_grad, zeros, ctx=self.raw_ctx)
        grad_B = where_op(self.inputs[0], zeros, output_grad, ctx=self.raw_ctx)
        return [None, grad_A, grad_B]

    def _reset_gpu_buffer(self, input_val1, input_val2, input_val3, output_val):
        if self.check_reset:
            strides = list(input_val1.stride) + list(input_val2.stride) + list(input_val3.stride) + list(output_val.stride)
            self.gpu_buffer = ndarray.array(
                strides, self.ctx, data_type=np.uintc)
            self.check_reset = False
            
    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        if tuple(input_shapes[0]) == tuple(input_shapes[1]) == tuple(input_shapes[2]):
            return input_shapes[0]
        
        ndim_cond = len(input_shapes[0])
        ndim_node_A = len(input_shapes[1])
        ndim_node_B = len(input_shapes[2])
        assert ndim_cond<=ndim_node_A and ndim_node_B<=ndim_node_A
        cond_shape = [1]*(ndim_node_A-ndim_cond) + list(input_shapes[0])
        node_B_shape = [1]*(ndim_node_A-ndim_node_B) + list(input_shapes[2])
        for i in range(ndim_cond):
            assert cond_shape[i]==1 or cond_shape[i]==input_shapes[1][i]
            assert node_B_shape[i]==1 or node_B_shape[i]==input_shapes[1][i]
        self.check_reset = True

        return input_shapes[1]





class WhereConstOp(Op):
    def __init__(self, cond, node_A, const_attr, ctx=None):
        super().__init__(WhereOp, [cond, node_A], ctx)
        self.const_attr = const_attr

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.where(input_vals[0].asnumpy(
            ), input_vals[1].asnumpy(), self.const_attr)
        else:
            where_const(input_vals[0], input_vals[1],
                        self.const_attr, output_val, stream_handle)

    def gradient(self, output_grad):
        grad = where_const_op(
            self.inputs[0], output_grad, 0., ctx=self.raw_ctx)
        return [None, grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert tuple(input_shapes[0]) == tuple(input_shapes[1])
        return input_shapes[0]


def where_op(cond, node_A, node_B, ctx=None):
    """Creates a node that represents np.where.

    Parameters:
    ----
    cond : Node of a condition array
    node_A : Node, output if cond
    node_B : Node, output if not cond

    Returns:
    ----
    A new Node instance created by Op.

    """
    return WhereOp(cond, node_A, node_B, ctx=ctx)


def where_const_op(cond, node_A, const_attr, ctx=None):
    """Creates a node that represents np.where.

    Parameters:
    ----
    cond : Node of a condition array
    node_A : Node, output if cond
    const_attr : float, output if not cond

    Returns:
    ----
    A new Node instance created by Op.

    """
    return WhereConstOp(cond, node_A, const_attr, ctx=ctx)
