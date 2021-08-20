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
        return [array_reshape_gradient_op(self.inputs[0], output_grad, self.output_shape, ctx=self.raw_ctx)]

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

    def get_default_state(self, status, enforce_order):
        if enforce_order:
            super().get_default_state(status, enforce_order)
        elif self.raw_ctx.is_mp():
            status.set_order((-1,) + tuple(range(len(self.output_shape))))

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        deduce_states(len(self.output_shape),
                      input_statuses[0], status, deduce_order)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        if input_statuses[0].state is not None:
            deduce_states(
                len(input_statuses[0].state), status, input_statuses[0], deduce_order)


class Array_Reshape_GradientOp(Op):
    def __init__(self, node_in, node_out, input_shape, ctx=None):
        super().__init__(Array_Reshape_GradientOp, [node_in, node_out], ctx)
        self.input_shape = input_shape

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

    def get_default_state(self, status, enforce_order):
        if enforce_order:
            super().get_default_state(status, enforce_order)
        elif status.state is not None:
            output_dim = len(status.state)
            status.set_order((-1,) + tuple(range(output_dim)))

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        if deduce_order:
            if status.is_dist():
                status.set_order((-1,) + tuple(range(len(status.state))))
                status.copy_order_from(input_statuses[0])
        else:
            status.copy_state_from(input_statuses[0])

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        if deduce_order:
            input_statuses[0].copy_order_from(status)
        else:
            input_statuses[0].copy_state_from(status)
        deduce_states(len(self.input_shape), status,
                      input_statuses[1], deduce_order)


def deduce_states(output_dim, input_status, output_status, deduce_order):
    # now only support n -> 2 or 2 -> n dimension reshape
    state, duplicate, order = input_status.get_all()
    if deduce_order:
        assert state is not None
        if order is None:
            return
        if output_dim > len(state):
            assert len(state) == 2
            mapper = {
                0: (0,),
                -1: (-1,),
                1: tuple(range(1, output_dim))
            }
            result_order = sum([mapper[x] for x in order], ())
        elif output_dim < len(state):
            assert output_dim == 2
            assert all([x == 1 for x in state[2:]])
            start = order.index(1)
            assert order[start:start +
                         len(order)-2] == tuple(range(1, len(order) - 1))
            temp_order = []
            for i in order:
                if i in (-1, 0, 1):
                    temp_order.append(i)
            result_order = tuple(temp_order)
        output_status.set_order(result_order)
    elif state is not None:
        if output_dim > len(state):
            assert len(state) == 2
            result_state = state + (1, ) * (output_dim - 2)
        elif output_dim < len(state):
            assert output_dim == 2
            assert all([x == 1 for x in state[2:]])
            result_state = state[:2]
        output_status.set_state(result_state, duplicate)


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


def array_reshape_gradient_op(node_in, node_out, input_shape, ctx=None):
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
    return Array_Reshape_GradientOp(node_in, node_out, input_shape, ctx=ctx)
