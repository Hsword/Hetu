from __future__ import absolute_import
from .Node import Op
import numpy as np
from .. import ndarray
from ..gpu_links import layer_normalization
from ..gpu_links import layer_normalization_gradient
from ..gpu_links import layer_normalization_inference


class Layer_NormalizationOp(Op):
    def __init__(self, node_in, ln_scale, ln_bias, eps=0.01, ctx=None):
        super().__init__(Layer_NormalizationOp,
                         [node_in, ln_scale, ln_bias], ctx)
        self.eps = eps
        self.save_mean = None
        self.save_var = None
        self.data_shape = None

    def compute(self, input_vals, output_val, stream_handle=None, inference=False):
        if inference:
            if self.on_cpu:
                input_vals = [n.asnumpy() for n in input_vals]
                data_type = input_vals[0].dtype
                std = np.sqrt(self.save_var + self.eps, dtype=data_type)
                centered_input = input_vals[0] - self.save_mean
                normed_input = centered_input / std

                bc_shape = [1] * len(input_vals[0].shape)
                bc_shape[-1] = input_vals[0].shape[-1]

                output_val[:] = input_vals[1].reshape(bc_shape) * normed_input + \
                    input_vals[2].reshape(bc_shape)

            else:
                layer_normalization_inference(input_vals[0], input_vals[1], input_vals[2],
                                              self.save_mean, self.save_var, output_val, self.eps, stream_handle)
        else:
            local_shape = list(input_vals[0].shape)
            local_shape[-1] = 1
            local_shape = tuple(local_shape)
            if self.on_cpu:
                input_vals = [n.asnumpy() for n in input_vals]
                data_type = input_vals[0].dtype
                if self.data_shape is None:
                    self.save_mean = np.empty(local_shape, dtype=np.float32)
                    self.save_var = np.empty(local_shape, dtype=np.float32)
                    self.data_shape = local_shape
                elif self.data_shape != local_shape:
                    del self.save_mean
                    del self.save_var
                    self.save_mean = np.empty(local_shape, dtype=np.float32)
                    self.save_var = np.empty(local_shape, dtype=np.float32)
                    self.data_shape = local_shape
                self.save_mean[:] = input_vals[0].mean(
                    axis=-1, dtype=data_type, keepdims=True)
                self.save_var[:] = input_vals[0].var(
                    axis=-1, dtype=data_type, keepdims=True)
                std = np.sqrt(self.save_var + self.eps, dtype=data_type)
                centered_input = input_vals[0] - self.save_mean
                normed_input = centered_input / std

                bc_shape = [1] * len(input_vals[0].shape)
                bc_shape[-1] = input_vals[0].shape[-1]

                output_val[:] = input_vals[1].reshape(bc_shape) * normed_input + \
                    input_vals[2].reshape(bc_shape)

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
                layer_normalization(input_vals[0], input_vals[1], input_vals[2],
                                    self.save_mean, self.save_var, output_val, self.eps, stream_handle)

    def gradient(self, output_grad):
        ln_gradient_node = layer_normalization_gradient_op(
            output_grad, self.inputs[0], self.inputs[1], self, self.eps, ctx=self.raw_ctx)
        data_gradient = layer_normalization_gradient_of_data_op(
            ln_gradient_node, self.inputs[0], ctx=self.raw_ctx)
        scale_gradient = layer_normalization_gradient_of_scale_op(
            ln_gradient_node, self.inputs[1], ctx=self.raw_ctx)
        bias_gradient = layer_normalization_gradient_of_bias_op(
            ln_gradient_node, self.inputs[2], ctx=self.raw_ctx)
        return [data_gradient, scale_gradient, bias_gradient]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        assert len(input_shapes[1]) == len(input_shapes[2]) == 1
        assert input_shapes[0][-1] == input_shapes[1][0] == input_shapes[2][0]
        return input_shapes[0]


class Layer_Normalization_GradientOp(Op):
    def __init__(self, out_gradient, in_node, ln_scale, forward_node, eps, ctx=None):
        super().__init__(Layer_Normalization_GradientOp,
                         [out_gradient, in_node, ln_scale], ctx)
        self.tmp_gradient_in_arr = None
        self.tmp_gradient_ln_bias = None
        self.tmp_gradient_ln_scale = None
        self.data_shape = None
        self.forward_node = forward_node
        self.eps = eps

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if self.tmp_gradient_ln_bias is None:
                shapeln = input_vals[2].shape
                self.data_shape = tuple(input_vals[0].shape)
                self.tmp_gradient_ln_scale = np.empty(
                    shape=shapeln, dtype=np.float32)
                self.tmp_gradient_ln_bias = np.empty(
                    shape=shapeln, dtype=np.float32)
                self.tmp_gradient_in_arr = np.empty(
                    shape=self.data_shape, dtype=np.float32)
            elif self.data_shape != tuple(input_vals[0].shape):
                self.data_shape = tuple(input_vals[0].shape)
                del self.tmp_gradient_in_arr
                self.tmp_gradient_in_arr = np.empty(
                    shape=self.data_shape, dtype=np.float32)

            red_axis = tuple(range(input_vals[0].ndim - 1))
            self.tmp_gradient_ln_bias[:] = input_vals[0].sum(red_axis)  # (X,)

            std = np.sqrt(self.forward_node.save_var + self.eps)  # (N, 1)
            x_centered = input_vals[1] - self.forward_node.save_mean  # (N, X)
            x_norm = x_centered / std  # (N, X)
            self.tmp_gradient_ln_scale[:] = (
                input_vals[0] * x_norm).sum(red_axis)  # (X,)

            last_dim = input_vals[1].shape[-1]
            dx_norm = input_vals[0] * input_vals[2].reshape(
                [1] * (input_vals[0].ndim - 1) + [-1])  # (N, X)
            dvar = (dx_norm * x_centered).sum(axis=-1, keepdims=True) * -0.5 / (
                self.forward_node.save_var + self.eps) / std  # (N, 1)
            dx_mu_1 = dx_norm / std  # (N, X)
            dx_mu_2 = dvar * 2 * x_centered / last_dim  # (N, X)
            dx_1 = dx_mu_1 + dx_mu_2  # (N, X)
            dx_2 = -1 * dx_1.sum(axis=-1, keepdims=True) / last_dim  # (N, 1)
            self.tmp_gradient_in_arr[:] = dx_1 + dx_2  # (N, X)
        else:
            if self.tmp_gradient_ln_bias is None:
                shapeln = input_vals[2].shape
                self.data_shape = tuple(input_vals[0].shape)
                self.tmp_gradient_ln_bias = ndarray.empty(
                    shape=shapeln, ctx=input_vals[0].ctx)
                self.tmp_gradient_ln_scale = ndarray.empty(
                    shape=shapeln, ctx=input_vals[0].ctx)
                self.tmp_gradient_in_arr = ndarray.empty(
                    shape=self.data_shape, ctx=input_vals[0].ctx)
            elif self.data_shape != tuple(input_vals[0].shape):
                self.data_shape = tuple(input_vals[0].shape)
                del self.tmp_gradient_in_arr
                self.tmp_gradient_in_arr = ndarray.empty(
                    shape=self.data_shape, ctx=input_vals[0].ctx)

            layer_normalization_gradient(input_vals[0], input_vals[1], input_vals[2],
                                         self.tmp_gradient_in_arr, self.tmp_gradient_ln_scale,
                                         self.tmp_gradient_ln_bias, self.forward_node.save_mean,
                                         self.forward_node.save_var, self.eps, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return (1,)


class Layer_Normalization_Gradient_of_DataOp(Op):
    def __init__(self, ln_gradient, in_arr, ctx=None):
        super().__init__(Layer_Normalization_Gradient_of_DataOp,
                         [ln_gradient, in_arr], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = self.inputs[0].tmp_gradient_in_arr
        else:
            self.inputs[0].tmp_gradient_in_arr.copyto(output_val)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[1]


class Layer_Normalization_Gradient_of_ScaleOp(Op):
    def __init__(self, ln_gradient, in_scale, ctx=None):
        super().__init__(Layer_Normalization_Gradient_of_ScaleOp,
                         [ln_gradient, in_scale], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = self.inputs[0].tmp_gradient_ln_scale
        else:
            self.inputs[0].tmp_gradient_ln_scale.copyto(output_val)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[1]


class Layer_Normalization_Gradient_of_BiasOp(Op):
    def __init__(self, ln_gradient, in_bias, ctx=None):
        super().__init__(Layer_Normalization_Gradient_of_BiasOp,
                         [ln_gradient, in_bias], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = self.inputs[0].tmp_gradient_ln_bias
        else:
            self.inputs[0].tmp_gradient_ln_bias.copyto(output_val)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[1]


def layer_normalization_op(node_in, ln_scale, ln_bias, eps=0.01, ctx=None):
    """Layer normalization node.

    Parameters:
    ----
    node_in : Node
        Input data.
    ln_scale : float
        scaling parameter
    ln_bias :
        learnable bias parameter
    eps : float
        Epsilon value for numerical stability.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Layer_NormalizationOp(node_in, ln_scale, ln_bias, eps, ctx=ctx)


def layer_normalization_gradient_op(out_gradient, in_node, ln_scale, forward_node, eps, ctx=None):
    """Gradient node of layer normalization.

    Parameters:
    ----
    out_gradient :
        The gradient array.
    in_node : Node
        Input node of ln layer.
    ln_scale :
        Scaling parameter.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Layer_Normalization_GradientOp(out_gradient, in_node, ln_scale, forward_node, eps, ctx=ctx)


def layer_normalization_gradient_of_data_op(ln_gradient, in_arr, ctx=None):
    """Gradient node of data of layer normalization.

    Parameters:
    ----
    ln_gradient :
        The gradient array.
    in_arr : Node
        Input array of ln layer.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Layer_Normalization_Gradient_of_DataOp(ln_gradient, in_arr, ctx=ctx)


def layer_normalization_gradient_of_scale_op(ln_gradient, in_scale, ctx=None):
    """Gradient node of scale parameter of layer normalization.

    Parameters:
    ----
    ln_gradient :
        The gradient array.
    in_scale :
        Scaling parameter of ln layer.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Layer_Normalization_Gradient_of_ScaleOp(ln_gradient, in_scale, ctx=ctx)


def layer_normalization_gradient_of_bias_op(ln_gradient, in_bias, ctx=None):
    """Gradient node of bias parameter of layer normalization.

    Parameters:
    ----
    ln_gradient :
        The gradient array.
    in_bias :
        Bias parameter of ln layer.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Layer_Normalization_Gradient_of_BiasOp(ln_gradient, in_bias, ctx=ctx)
