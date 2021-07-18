from __future__ import absolute_import
from .Node import Op
import numpy as np
from .. import ndarray
from ..gpu_links import instance_normalization2d
from ..gpu_links import instance_normalization2d_gradient


class Instance_Normalization2dOp(Op):
    def __init__(self, node_in, eps=0.0000001, ctx=None):
        super().__init__(Instance_Normalization2dOp, [node_in], ctx)
        self.eps = eps
        self.save_mean = None
        self.save_var = None
        self.data_shape = None

    def compute(self, input_vals, output_val, stream_handle=None, inference=False):
        local_shape = list(input_vals[0].shape)
        assert len(local_shape) == 4
        local_shape[-1] = 1
        local_shape[-2] = 1
        local_shape = tuple(local_shape)
        if self.on_cpu:
            raise NotImplementedError
        else:
            if self.data_shape is None:
                dev_id = input_vals[0].handle.contents.ctx.device_id
                self.save_mean = ndarray.empty(
                    local_shape, ctx=ndarray.gpu(dev_id))
                self.save_var = ndarray.empty(
                    local_shape, ctx=ndarray.gpu(dev_id))
                self.data_shape = local_shape
            elif self.data_shape != local_shape:
                del self.save_mean
                del self.save_var
                dev_id = input_vals[0].handle.contents.ctx.device_id
                self.save_mean = ndarray.empty(
                    local_shape, ctx=ndarray.gpu(dev_id))
                self.save_var = ndarray.empty(
                    local_shape, ctx=ndarray.gpu(dev_id))
                self.data_shape = local_shape
            instance_normalization2d(input_vals[0], self.save_mean, self.save_var,
                                     output_val, self.eps, stream_handle)

    def gradient(self, output_grad):
        return [instance_normalization2d_gradient_op(output_grad, self.inputs[0], self, ctx=self.ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class Instance_Normalization2d_GradientOp(Op):
    def __init__(self, out_gradient, in_node, forward_node, ctx=None):
        super().__init__(Instance_Normalization2d_GradientOp,
                         [out_gradient, in_node], ctx)
        self.tmp_gradient_in_arr = None
        self.data_shape = None
        self.forward_node = forward_node

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            instance_normalization2d_gradient(input_vals[0], input_vals[1], output_val,
                                              self.forward_node.save_mean, self.forward_node.save_var,
                                              self.forward_node.eps, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[0]


def instance_normalization2d_op(node_in, eps=0.01, ctx=None):
    """Layer normalization node.

    Parameters:
    ----
    node_in : Node
        Input data.
    eps : float
        Epsilon value for numerical stability.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Instance_Normalization2dOp(node_in, eps, ctx=ctx)


def instance_normalization2d_gradient_op(out_gradient, in_node, forward_node, ctx=None):
    """Gradient node of layer normalization.

    Parameters:
    ----
    out_gradient :
        The gradient array.
    in_node : Node
        Input node of ln layer.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Instance_Normalization2d_GradientOp(out_gradient, in_node, forward_node, ctx=ctx)
