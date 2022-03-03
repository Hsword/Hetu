from __future__ import absolute_import
import ctypes
from .Node import Op
from .. import ndarray
from .._base import DNNL_LIB
from ..cpu_links import reshape as cpu_reshape
from ..gpu_links import array_reshape


class Array_ReshapeOp(Op):
    def __init__(self, node_A, output_shape, ctx=None):
        super().__init__(Array_ReshapeOp, [node_A], ctx)
        self.output_shape = output_shape
#print(node_A.name, "??????")

    def compute(self, input_vals, output_val, stream_handle=None):

        assert(len(input_vals) == 1)
        input_size = 1
        for i in range(len(input_vals[0].shape)):
            input_size *= input_vals[0].shape[i]
        # check if there exists -1 in output_shape
        idx = -1
        cnt = 0
        output_size = 1
        output_shape = list(self.output_shape)
        for i in range(len(output_shape)):
            if(output_shape[i] == -1):
                idx = i
                cnt = cnt + 1
                assert(cnt != 2)
            output_size *= output_shape[i]

        if(idx == -1):
            assert input_size == output_size
        else:
            output_size = output_size * (-1)
            assert (input_size % output_size == 0)
            output_shape[idx] = input_size // output_size
        output_shape = tuple(output_shape)
        if self.on_cpu:
            if DNNL_LIB['cpu_Reshape']:
                cpu_reshape(input_vals[0], output_val)
            else:
                output_val[:] = input_vals[0].asnumpy().reshape(output_shape)
        else:
            if self.inplace:
                input_vals[0].reshape(output_shape, output_val)
            else:
                array_reshape(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [array_reshape_gradient_op(self.inputs[0], output_grad, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):

        assert (len(input_shapes) == 1)
        input_size = 1
        input_shape = input_shapes[0]
        for i in range(len(input_shape)):
            input_size *= input_shape[i]

        # check if there exists -1 in output_shape
        idx = -1
        cnt = 0
        output_size = 1
        output_shape = list(self.output_shape)
        for i in range(len(output_shape)):
            if(output_shape[i] == -1):
                idx = i
                cnt = cnt + 1
                assert(cnt != 2)
            output_size *= output_shape[i]
        if(idx == -1):
            assert input_size == output_size
        else:
            output_size = output_size * (-1)
            assert (input_size % output_size == 0)
            output_shape[idx] = input_size // output_size
        output_shape = tuple(output_shape)
        return output_shape

    def backward_hook(self, config):
        self.inplace = config.enable_lazy and self not in config.eval_node_list

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        # !!! NO CHECKING !!!
        assert len(input_statuses) == len(self.inputs)
        # if input_statuses[0].valid(deduce_order):
        #     input_statuses[0].check_state(1, deduce_order)
        status.copy_from(input_statuses[0], deduce_order)
        if status.valid_state() and not hasattr(self, 'processed'):
            self.processed = True
            for k, v in status.state.items():
                if self.output_shape[k] > 0:
                    self.output_shape[k] //= v

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        # !!! NO CHECKING !!!
        assert len(input_statuses) == len(self.inputs)
        # if status.valid(deduce_order):
        #     status.check_state(1, deduce_order)
        input_statuses[0].copy_from(status, deduce_order)


class Array_Reshape_GradientOp(Op):
    def __init__(self, node_in, node_out, ctx=None):
        super().__init__(Array_Reshape_GradientOp, [node_in, node_out], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        # the size of input_array
        shapeIn = input_vals[0].shape
        if self.on_cpu:
            if DNNL_LIB['cpu_Reshape']:
                cpu_reshape(input_vals[1], output_val)
            else:
                output_val[:] = input_vals[1].asnumpy().reshape(shapeIn)
        else:
            if self.inplace:
                input_vals[1].reshape(shapeIn, output_val)
            else:
                array_reshape(input_vals[1], output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[0]

    def backward_hook(self, config):
        self.inplace = config.enable_lazy and self not in config.eval_node_list

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        # !!! NO CHECKING !!!
        assert len(input_statuses) == len(self.inputs)
        for nst in input_statuses:
            # if nst.valid(deduce_order):
            #     nst.check_state(1, deduce_order)
            status.copy_from(nst, deduce_order)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        # !!! NO CHECKING !!!
        assert len(input_statuses) == len(self.inputs)
        # if status.valid(deduce_order):
        #     status.check_state(1, deduce_order)
        for nst in input_statuses:
            nst.copy_from(status, deduce_order)


def array_reshape_op(node, output_shape, ctx=None):
    """Reshapes an input array without copy.

    Parameters:
    ----
    node : Node
        Input variable.
    output_shape: tuple(int)
        Expected shape of the output array.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Array_ReshapeOp(node, output_shape, ctx=ctx)


def array_reshape_gradient_op(node_in, node_out, ctx=None):
    """Gradient of reshape operation.

    Parameters:
    ----
    node_in : Node
        Input node of reshape operation.
    node_out: Node
        Previous gradient node.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Array_Reshape_GradientOp(node_in, node_out, ctx=ctx)
