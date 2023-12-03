from __future__ import absolute_import
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import reshape as cpu_reshape
from ..gpu_links import array_reshape


class ReshapeToOp(Op):
    def __init__(self, node_A, target, ctx=None):
        super().__init__(ReshapeToOp, [node_A, target], ctx)
        self.dtype = node_A.dtype

    def compute(self, input_vals, output_val, stream_handle=None):
        output_shape = output_val.shape
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
        from .Reshape import array_reshape_gradient_op
        return [array_reshape_gradient_op(self, output_grad, ctx=self.raw_ctx), None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        self.input_shape = input_shapes[0]
        return input_shapes[1]

    def backward_hook(self, config):
        self.inplace = config.enable_lazy and self not in config.eval_node_list


def reshape_to_op(node, target, ctx=None):
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
    return ReshapeToOp(node, target, ctx=ctx)
