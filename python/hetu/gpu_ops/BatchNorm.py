from __future__ import absolute_import
from .Node import Op
import numpy as np
from .. import ndarray
from .._base import DNNL_LIB
from ..cpu_links import batch_norm as cpu_batch_norm
from ..cpu_links import batch_norm_inference as cpu_batch_norm_inference
from ..cpu_links import batch_norm_gradient as cpu_batch_norm_gradient
from ..gpu_links import CuDNN_Batch_Normalization
from ..gpu_links import CuDNN_Batch_Normalization_gradient
from ..gpu_links import CuDNN_Batch_Normalization_inference
import numpy as np


class Batch_NormalizationOp(Op):
    def __init__(self, node_in, bn_scale, bn_bias, momentum=0.1, eps=1e-5, ctx=None):
        super().__init__(Batch_NormalizationOp,
                         [node_in, bn_scale, bn_bias], ctx)
        self.momentum = momentum
        self.eps = eps
        # saved states
        # !!! running_mean/running_var is the estimated_mean/estimated_var used in inference (necessary!)
        # !!! save_mean/save_var is the saved forward computation result used in backward (optional!)
        self.running_mean = None
        self.running_var = None
        self.save_mean = None
        self.save_var = None

    def try_init_running_states(self, channel):
        if self.running_mean is None:
            if self.on_cpu:
                if DNNL_LIB['DnnlBatchNorm']:
                    self.running_mean = ndarray.array(
                        np.zeros((channel,), dtype=np.float32), ctx=self.ctx)
                    self.running_var = ndarray.array(
                        np.ones((channel,), dtype=np.float32), ctx=self.ctx)
                    self.save_mean = ndarray.empty(
                        (channel,), ctx=self.ctx)
                    self.save_var = ndarray.empty((channel,), ctx=self.ctx)
                else:
                    self.running_mean = np.zeros(
                        (channel,), dtype=np.float32)
                    self.running_var = np.ones(
                        (channel,), dtype=np.float32)
                    self.save_mean = np.empty((channel,), dtype=np.float32)
                    self.save_var = np.empty((channel,), dtype=np.float32)
            else:
                self.save_mean = ndarray.empty((channel,), ctx=self.ctx)
                self.save_var = ndarray.empty((channel,), ctx=self.ctx)
                self.running_mean = ndarray.array(
                    np.zeros((channel,)), ctx=self.ctx)
                self.running_var = ndarray.array(
                    np.ones((channel,)), ctx=self.ctx)

    def compute(self, input_vals, output_val, stream_handle=None, inference=False):
        self.try_init_running_states(input_vals[0].shape[1])
        if inference:
            if self.on_cpu:
                if DNNL_LIB['DnnlBatchNorm_Inference']:
                    cpu_batch_norm_inference(
                        input_vals[0], input_vals[1], input_vals[2], output_val, self.running_mean, self.running_var, self.eps)
                else:
                    output_val[:] = batchnorm_inference(input_vals[0].asnumpy(), input_vals[1].asnumpy(),
                                                        input_vals[2].asnumpy(), self.running_mean, self.running_var, self.eps)
            else:
                CuDNN_Batch_Normalization_inference(
                    input_vals[0], input_vals[1], input_vals[2], output_val, self.running_mean, self.running_var, self.eps, stream_handle)
        else:
            if self.on_cpu:
                if DNNL_LIB['DnnlBatchNorm']:
                    cpu_batch_norm(input_vals[0], input_vals[1], input_vals[2], output_val,
                                   self.running_mean, self.running_var, self.save_mean, self.save_var, self.momentum, self.eps)
                else:
                    output_val[:], self.running_mean[:], self.running_var[:], self.save_mean[:], self.save_var[:] = batchnorm_forward(
                        input_vals[0].asnumpy(), input_vals[1].asnumpy(), input_vals[2].asnumpy(), self.running_mean, self.running_var, self.momentum, self.eps)
            else:
                CuDNN_Batch_Normalization(input_vals[0], input_vals[1], input_vals[2], output_val, self.running_mean,
                                          self.running_var, self.save_mean, self.save_var, self.momentum, self.eps, stream_handle)

    def gradient(self, output_grad):

        bn_gradient_node = batch_normalization_gradient_op(
            output_grad, self.inputs[0], self.inputs[1], self, self.eps, ctx=self.raw_ctx)
        data_gradient = batch_normalization_gradient_of_data_op(
            bn_gradient_node, self.inputs[0], ctx=self.raw_ctx)
        scale_gradient = batch_normalization_gradient_of_scale_op(
            bn_gradient_node, self.inputs[1], ctx=self.raw_ctx)
        bias_gradient = batch_normalization_gradient_of_bias_op(
            bn_gradient_node, self.inputs[2], ctx=self.raw_ctx)

        return [data_gradient, scale_gradient, bias_gradient]

    def infer_shape(self, input_shapes):
        return input_shapes[0]

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == 3
        status.copy_from(input_statuses[0], deduce_order)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == 3
        input_statuses[0].copy_from(status, deduce_order)
        if deduce_order:
            if status.valid_all():
                status.check_state(2, True)
                new_order = status.combine_order((0, -1), (1, 0))
                input_statuses[1].set_order(new_order)
                input_statuses[2].set_order(new_order)
        else:
            if status.valid_state():
                status.check_state(2, False)
                new_state, new_duplicate, new_partial = status.combine_state(
                    (0, -1), (1, 0))
                input_statuses[1].set_state(
                    new_state, new_duplicate, new_partial)
                input_statuses[2].set_state(
                    new_state, new_duplicate, new_partial)

    def deduce_generated_backward_nodes_states(self, input_statuses, status, index):
        if index <= 0:
            return status
        else:
            from ..context import NodeStatus
            new_status = NodeStatus(
                dev_num=status.dev_num, partial_or_node=True)
            new_status.set_state(*status.combine_state((0, -2), (1, 0)))
            new_status.set_order(status.combine_order((0, -2), (1, 0)))
            return new_status


class Batch_Normalization_GradientOp(Op):
    def __init__(self, out_gradient, in_node, bn_scale, forward_node, eps, ctx=None):
        super().__init__(Batch_Normalization_GradientOp,
                         [out_gradient, in_node, bn_scale], ctx)
        self.tmp_gradient_in_arr = None
        self.tmp_gradient_bn_bias = None
        self.tmp_gradient_bn_scale = None
        self.forward_node = forward_node
        self.eps = eps

    def check_valid_arrs(self, shape):
        assert self.tmp_gradient_in_arr is not None
        if self.tmp_gradient_bn_scale is None:
            self.tmp_gradient_bn_scale = ndarray.empty(shape, ctx=self.ctx)
        if self.tmp_gradient_bn_bias is None:
            self.tmp_gradient_bn_bias = ndarray.empty(shape, ctx=self.ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        self.check_valid_arrs(input_vals[2].shape)
        if self.on_cpu:
            if DNNL_LIB['DnnlBatchNorm_Gradient']:
                cpu_batch_norm_gradient(input_vals[0], input_vals[1], input_vals[2],
                                        self.tmp_gradient_in_arr, self.tmp_gradient_bn_scale,
                                        self.tmp_gradient_bn_bias, self.forward_node.save_mean,
                                        self.forward_node.save_var, self.eps)
            else:
                self.tmp_gradient_in_arr[:], self.tmp_gradient_bn_scale[:], self.tmp_gradient_bn_bias[:] = batchnorm_backward(
                    input_vals[0].asnumpy(),
                    input_vals[1].asnumpy(),
                    input_vals[2].asnumpy(),
                    self.eps, self.forward_node.save_mean, self.forward_node.save_var
                )
        else:
            CuDNN_Batch_Normalization_gradient(input_vals[0], input_vals[1], input_vals[2],
                                               self.tmp_gradient_in_arr, self.tmp_gradient_bn_scale,
                                               self.tmp_gradient_bn_bias, self.forward_node.save_mean,
                                               self.forward_node.save_var, self.eps, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return None

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == 3
        status.copy_from(input_statuses[0], deduce_order)
        status.copy_from(input_statuses[1], deduce_order)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == 3
        input_statuses[0].copy_from(status, deduce_order)
        input_statuses[1].copy_from(status, deduce_order)
        if deduce_order:
            if status.valid_all():
                status.check_state(2, True)
                input_statuses[2].set_order(
                    status.combine_order((0, -1), (1, 0)))
        else:
            if status.valid_state():
                status.check_state(2, False)
                input_statuses[2].set_state(
                    *status.combine_state((0, -1), (1, 0)))


class Batch_Normalization_Gradient_of_DataOp(Op):
    def __init__(self, bn_gradient, in_arr, ctx=None):
        super().__init__(Batch_Normalization_Gradient_of_DataOp,
                         [bn_gradient, in_arr], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        assert False, 'In memory plan we already set the result array; should not call the compute.'

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[1]

    def pass_grad_array(self, array):
        self.inputs[0].tmp_gradient_in_arr = array


class Batch_Normalization_Gradient_of_ScaleOp(Op):
    def __init__(self, bn_gradient, in_scale, ctx=None):
        super().__init__(Batch_Normalization_Gradient_of_ScaleOp,
                         [bn_gradient, in_scale], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        assert False, 'In memory plan we already set the result array; should not call the compute.'

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[1]

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        src_status = input_statuses[0]
        if deduce_order:
            if src_status.valid_all():
                src_status.check_state(2, True)
                new_order = src_status.combine_order((0, -2), (1, 0))
                status.set_order(new_order)
        else:
            if src_status.valid_state():
                src_status.check_state(2, False)
                status.set_state(*src_status.combine_state((0, -2), (1, 0)))

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        pass

    def pass_grad_array(self, array):
        self.inputs[0].tmp_gradient_bn_scale = array


class Batch_Normalization_Gradient_of_BiasOp(Op):
    def __init__(self, bn_gradient, in_bias, ctx=None):
        super().__init__(Batch_Normalization_Gradient_of_BiasOp,
                         [bn_gradient, in_bias], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        assert False, 'In memory plan we already set the result array; should not call the compute.'

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[1]

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        src_status = input_statuses[0]
        if deduce_order:
            if src_status.valid_all():
                src_status.check_state(2, True)
                status.set_order(src_status.combine_order((0, -2), (1, 0)))
        else:
            if src_status.valid_state():
                src_status.check_state(2, False)
                status.set_state(*src_status.combine_state((0, -2), (1, 0)))

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        pass

    def pass_grad_array(self, array):
        self.inputs[0].tmp_gradient_bn_bias = array


def batch_normalization_op(node_in, bn_scale, bn_bias, momentum=0.1, eps=1e-5, ctx=None):
    """Batch normalization layer node.

    Parameters:
    ----
    node_in : Node
        Input data.
    bn_scale : float
        scaling parameter
    bn_bias :
        learnable bias parameter
    momentum : float
        Acting on the calculation of mean and variance, the mean and variance values in historical batch are retained.
    eps : float
        Epsilon value for numerical stability.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Batch_NormalizationOp(node_in, bn_scale, bn_bias, momentum, eps, ctx=ctx)


def batch_normalization_gradient_op(out_gradient, in_node, bn_scale, forward_node, eps, ctx=None):
    """Gradient node of batch normalization.

    Parameters:
    ----
    out_gradient :
        The gradient array.
    in_node : Node
        Input node of bn layer.
    bn_scale :
        Scaling parameter.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Batch_Normalization_GradientOp(out_gradient, in_node, bn_scale, forward_node, eps, ctx=ctx)


def batch_normalization_gradient_of_data_op(bn_gradient, in_arr, ctx=None):
    """Gradient node of data of  batch normalization.

    Parameters:
    ----
    bn_gradient :
        The gradient array.
    in_arr : Node
        Input array of bn layer.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Batch_Normalization_Gradient_of_DataOp(bn_gradient, in_arr, ctx=ctx)


def batch_normalization_gradient_of_scale_op(bn_gradient, in_scale, ctx=None):
    """Gradient node of scale parameter of batch normalization.

    Parameters:
    ----
    bn_gradient :
        The gradient array.
    in_scale :
        Scaling parameter of bn layer.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Batch_Normalization_Gradient_of_ScaleOp(bn_gradient, in_scale, ctx=ctx)


def batch_normalization_gradient_of_bias_op(bn_gradient, in_bias, ctx=None):
    """Gradient node of bias parameter of batch normalization.

    Parameters:
    ----
    bn_gradient :
        The gradient array.
    in_bias :
        Bias parameter of bn layer.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Batch_Normalization_Gradient_of_BiasOp(bn_gradient, in_bias, ctx=ctx)


def batchnorm_forward(x, bn_scale, bn_bias, running_mean, running_var, momentum=0.1, eps=1e-5):
    D = x.shape[1]
    sample_mean = x.mean(axis=(0, 2, 3), dtype=x.dtype)
    sample_var = x.var(axis=(0, 2, 3), dtype=x.dtype)
    running_mean = momentum * sample_mean + (1.0 - momentum) * running_mean
    running_var = momentum * sample_var + (1.0 - momentum) * running_var

    std = np.sqrt(sample_var.reshape(1, D, 1, 1) + eps, dtype=x.dtype)
    x_centered = x - sample_mean.reshape(1, D, 1, 1)
    x_norm = x_centered / std
    out = bn_scale.reshape(1, D, 1, 1) * x_norm + bn_bias.reshape(1, D, 1, 1)

    return out, running_mean, running_var, sample_mean, sample_var


def batchnorm_inference(x, bn_scale, bn_bias, estimated_mean, estimated_var, eps):
    D = x.shape[1]
    std = np.sqrt(estimated_var.reshape(1, D, 1, 1) + eps, dtype=x.dtype)
    x_centered = x - estimated_mean.reshape(1, D, 1, 1)
    x_norm = x_centered / std
    out = bn_scale.reshape(1, D, 1, 1) * x_norm + bn_bias.reshape(1, D, 1, 1)

    return out


def batchnorm_backward(gradient_Y, x, bn_scale, eps, save_mean, save_var):
    D = gradient_Y.shape[1]
    num_accum = gradient_Y.shape[0] * gradient_Y.shape[2] * gradient_Y.shape[3]

    if not isinstance(save_mean, np.ndarray):
        save_mean = save_mean.asnumpy()
    if not isinstance(save_var, np.ndarray):
        save_var = save_var.asnumpy()
    sample_mean = save_mean
    sample_var = save_var

    std = np.sqrt(sample_var + eps).reshape(1, D, 1, 1)
    x_centered = x - sample_mean.reshape(1, D, 1, 1)
    x_norm = x_centered / std

    dbn_scale = (gradient_Y * x_norm).sum(axis=(0, 2, 3), keepdims=True)
    dbn_bias = gradient_Y.sum(axis=(0, 2, 3), keepdims=True)

    dx_norm = gradient_Y * bn_scale.reshape(1, D, 1, 1)
    dx_centered = dx_norm / std
    dmean = -(dx_centered.sum(axis=(0, 2, 3)) + 2 / num_accum *
              x_centered.sum(axis=(0, 2, 3))).reshape(1, D, 1, 1)
    dstd = (dx_norm * x_centered * -std ** (-2)
            ).sum(axis=(0, 2, 3)).reshape(1, D, 1, 1)
    dvar = dstd / 2 / std
    dx = dx_centered + (dmean + dvar * 2 * x_centered) / num_accum

    return dx, dbn_scale, dbn_bias
