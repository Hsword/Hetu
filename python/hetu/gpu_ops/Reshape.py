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

    def deduce_states(self, states, duplicates, orders):
        # now only support: maintain batch size dimension
        assert len(states) == 1 and len(duplicates) == 1 and len(orders) == 1
        result_state = states[0]
        result_dupli = duplicates[0]
        result_order = orders[0]
        output_dim = len(self.output_shape)
        if output_dim > len(result_state):
            assert len(result_state) == 2
            assert self.output_shape[1] % result_state[1] == 0
            result_state = result_state + (1, ) * (output_dim - 2)
            mapper = {
                0: (0,),
                -1: (-1,),
                1: tuple(range(1, output_dim))
            }
            result_order = sum([mapper[x] for x in result_order], ())
        elif output_dim < len(result_state):
            assert output_dim == 2
            assert all([x == 1 for x in result_state[2:]])
            start = result_order.index(1)
            assert result_order[start:start +
                                len(result_order)-2] == tuple(range(1, len(result_order) - 1))
            result_state = result_state[:2]
            temp_order = []
            for i in result_order:
                if i in (-1, 0, 1):
                    temp_order.append(i)
            result_order = tuple(temp_order)
        return result_state, result_dupli, result_order


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

    def deduce_states(self, states, duplicates, orders):
        # now only support: maintain batch size dimension
        assert len(states) == 2 and len(duplicates) == 2 and len(orders) == 2
        return states[0], duplicates[0], orders[0]


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
