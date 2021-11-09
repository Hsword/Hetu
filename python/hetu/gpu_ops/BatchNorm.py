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
    def __init__(self, node_in, bn_scale, bn_bias, momentum=0.99, eps=0.01, ctx=None):
        super().__init__(Batch_NormalizationOp,
                         [node_in, bn_scale, bn_bias], ctx)
        self.momentum = momentum
        self.eps = eps
        self.save_mean = None
        self.save_var = None
        self.running_mean = None
        self.running_var = None

    def compute(self, input_vals, output_val, stream_handle=None, inference=False):
        if inference:
            if self.on_cpu:
                if DNNL_LIB['DnnlBatchNorm_Inference']:
                    cpu_batch_norm_inference(input_vals[0], input_vals[1], input_vals[2], output_val, self.save_mean,
                                             self.save_var, self.momentum, self.eps)
                else:
                    output_val[:] = batchnorm_inference(input_vals[0].asnumpy(), input_vals[1].asnumpy(),
                                                        input_vals[2].asnumpy(
                    ), self.save_mean, self.save_var,
                        self.eps)
            else:
                CuDNN_Batch_Normalization_inference(
                    input_vals[0], input_vals[1], input_vals[2], output_val, self.save_mean, self.save_var, self.eps, stream_handle)
        else:
            if self.on_cpu:
                if DNNL_LIB['DnnlBatchNorm']:
                    if self.save_mean is None:
                        dev_id = input_vals[0].handle.contents.ctx.device_id
                        C = input_vals[0].shape[1]
                        self.save_mean = ndarray.array(
                            np.zeros([C], dtype=np.float32), ctx=ndarray.cpu(dev_id))
                        self.save_var = ndarray.array(
                            np.zeros([C], dtype=np.float32), ctx=ndarray.cpu(dev_id))
                    cpu_batch_norm(input_vals[0], input_vals[1], input_vals[2], output_val,
                                   self.save_mean, self.save_var, self.momentum, self.eps)
                else:
                    output_val[:], self.save_mean, self.save_var = batchnorm_forward(input_vals[0].asnumpy(),
                                                                                     input_vals[1].asnumpy(
                    ),
                        input_vals[2].asnumpy(
                    ),
                        self.save_mean,
                        self.save_var, self.momentum,
                        self.eps)
            else:
                if self.save_mean == None:
                    dev_id = input_vals[0].handle.contents.ctx.device_id
                    C = input_vals[0].shape[1]
                    self.save_mean = ndarray.array(
                        np.zeros([1, C, 1, 1]), ctx=ndarray.gpu(dev_id))
                    self.save_var = ndarray.array(
                        np.zeros([1, C, 1, 1]), ctx=ndarray.gpu(dev_id))
                    self.running_mean = ndarray.array(
                        np.zeros([1, C, 1, 1]), ctx=ndarray.gpu(dev_id))
                    self.running_var = ndarray.array(
                        np.zeros([1, C, 1, 1]), ctx=ndarray.gpu(dev_id))

                CuDNN_Batch_Normalization(
                    input_vals[0], input_vals[1], input_vals[2], output_val, self.save_mean, self.save_var,
                    self.running_mean, self.running_var, self.momentum,
                    self.eps, stream_handle)

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
                new_order = deduce_order_to_reduced_tensor(status)
                input_statuses[1].set_order(new_order)
                input_statuses[2].set_order(new_order)
        else:
            if status.valid_state():
                new_state, duplicate = deduce_state_to_reduced_tensor(status)
                input_statuses[1].set_state(new_state, duplicate)
                input_statuses[2].set_state(new_state, duplicate)


class Batch_Normalization_GradientOp(Op):
    def __init__(self, out_gradient, in_node, bn_scale, forward_node, eps, ctx=None):
        super().__init__(Batch_Normalization_GradientOp,
                         [out_gradient, in_node, bn_scale], ctx)
        self.tmp_gradient_in_arr = None
        self.tmp_gradient_bn_bias = None
        self.tmp_gradient_bn_scale = None
        self.forward_node = forward_node
        self.eps = eps

    def update_mean_and_var(self, saved_mean, saved_var):
        self.saved_mean = saved_mean
        self.saved_var = saved_var

    def compute(self, input_vals, output_val, stream_handle=None):

        if self.on_cpu:
            if DNNL_LIB['DnnlBatchNorm_Gradient']:
                if self.tmp_gradient_bn_bias is None:
                    shapebn = input_vals[2].shape
                    self.tmp_gradient_bn_bias = np.zeros(
                        shape=shapebn, dtype=np.float32)
                    self.tmp_gradient_bn_scale = np.zeros(
                        shape=shapebn, dtype=np.float32)
                    self.tmp_gradient_in_arr = np.zeros(
                        shape=input_vals[1].shape, dtype=np.float32)

                cpu_batch_norm_gradient(input_vals[0], input_vals[1], input_vals[2], bn_bias, self.tmp_gradient_in_arr,
                                        self.tmp_gradient_bn_scale,
                                        self.tmp_gradient_bn_bias, self.forward_node.running_mean,
                                        self.forward_node.running_var, self.eps)
            else:
                if self.tmp_gradient_bn_bias is None:
                    typebn = input_vals[2].asnumpy().dtype
                    shapebn = input_vals[2].asnumpy().shape
                    self.tmp_gradient_bn_bias = np.zeros(
                        shape=shapebn, dtype=typebn)
                    self.tmp_gradient_bn_scale = np.zeros(
                        shape=shapebn, dtype=typebn)
                self.tmp_gradient_in_arr, self.tmp_gradient_bn_scale, self.tmp_gradient_bn_bias = batchnorm_backward(
                    input_vals[0].asnumpy(), input_vals[1].asnumpy(
                    ), input_vals[2].asnumpy(),
                    self.tmp_gradient_bn_scale, self.tmp_gradient_bn_bias,
                    self.eps, self.forward_node.save_mean, self.forward_node.save_var)
        else:
            if self.tmp_gradient_bn_bias == None:
                shapebn = input_vals[2].shape
                self.tmp_gradient_bn_scale = ndarray.empty(
                    shape=shapebn, ctx=input_vals[0].ctx)
                self.tmp_gradient_bn_bias = ndarray.empty(
                    shape=shapebn, ctx=input_vals[0].ctx)
                self.tmp_gradient_in_arr = ndarray.empty(
                    shape=input_vals[1].shape, ctx=input_vals[0].ctx)
            CuDNN_Batch_Normalization_gradient(input_vals[0], input_vals[1], input_vals[2],
                                               self.tmp_gradient_in_arr, self.tmp_gradient_bn_scale,
                                               self.tmp_gradient_bn_bias, self.forward_node.running_mean,
                                               self.forward_node.running_var, self.eps, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return (1,)

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
                new_order = deduce_order_to_reduced_tensor(status)
                input_statuses[2].set_order(new_order)
        else:
            if status.valid_state():
                new_state, duplicate = deduce_state_to_reduced_tensor(status)
                input_statuses[2].set_state(new_state, duplicate)


class Batch_Normalization_Gradient_of_DataOp(Op):
    def __init__(self, bn_gradient, in_arr, ctx=None):
        super().__init__(Batch_Normalization_Gradient_of_DataOp,
                         [bn_gradient, in_arr], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):

        if self.on_cpu:
            output_val[:] = self.inputs[0].tmp_gradient_in_arr
        else:
            self.inputs[0].tmp_gradient_in_arr.inplace_copy(output_val)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[1]


class Batch_Normalization_Gradient_of_ScaleOp(Op):
    def __init__(self, bn_gradient, in_scale, ctx=None):
        super().__init__(Batch_Normalization_Gradient_of_ScaleOp,
                         [bn_gradient, in_scale], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):

        if self.on_cpu:
            output_val[:] = self.inputs[0].tmp_gradient_bn_scale
        else:
            self.inputs[0].tmp_gradient_bn_scale.inplace_copy(output_val)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[1]

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        if deduce_order:
            if input_statuses[0].valid_all():
                new_order = deduce_order_to_reduced_tensor(input_statuses[0])
                status.set_order(new_order)
        else:
            if input_statuses[0].valid_state():
                new_state, duplicate = deduce_state_to_reduced_tensor(
                    input_statuses[0])
                status.set_state(new_state, duplicate)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        pass


class Batch_Normalization_Gradient_of_BiasOp(Op):
    def __init__(self, bn_gradient, in_bias, ctx=None):
        super().__init__(Batch_Normalization_Gradient_of_BiasOp,
                         [bn_gradient, in_bias], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):

        if self.on_cpu:
            output_val[:] = self.inputs[0].tmp_gradient_bn_bias
        else:
            self.inputs[0].tmp_gradient_bn_bias.inplace_copy(output_val)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[1]

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        if deduce_order:
            if input_statuses[0].valid_all():
                new_order = deduce_order_to_reduced_tensor(input_statuses[0])
                status.set_order(new_order)
        else:
            if input_statuses[0].valid_state():
                new_state, duplicate = deduce_state_to_reduced_tensor(
                    input_statuses[0])
                status.set_state(new_state, duplicate)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        pass


def deduce_order_to_reduced_tensor(input_status):
    input_status.check_state(2, True)
    new_order = list(input_status.order)
    if 0 in new_order:
        ind = new_order.index(0)
        if (ind > 0 and new_order[ind - 1] == -1) or (ind < len(new_order) - 1 and new_order[ind + 1] == -1):
            new_order.pop(ind)
        else:
            new_order[ind] = -1
        appeared = False
        for o in new_order:
            if o == -1:
                assert not appeared
                appeared = True
    new_order = tuple(new_order)
    return new_order


def deduce_state_to_reduced_tensor(input_status):
    input_status.check_state(2, False)
    state, duplicate = input_status.get()
    new_state = state.copy()
    if 0 in new_state:
        duplicate *= new_state.pop(0)
    return new_state, duplicate


def batch_normalization_op(node_in, bn_scale, bn_bias, momentum=0.99, eps=0.01, ctx=None):
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


def batchnorm_forward(x, bn_scale, bn_bias, save_mean, save_var, momentum=0.99, eps=0.01):
    D = x.shape[1]
    if save_mean is None:
        save_mean = np.zeros(D, dtype=x.dtype)
    if save_var is None:
        save_var = np.ones(D, dtype=x.dtype)

    sample_mean = x.mean(axis=(0, 2, 3), dtype=x.dtype)
    sample_var = x.var(axis=(0, 2, 3), dtype=x.dtype)
    save_mean = momentum * sample_mean + (1.0 - momentum) * save_mean
    save_var = momentum * sample_var + (1.0 - momentum) * save_var

    std = np.sqrt(sample_var.reshape(1, D, 1, 1) + eps, dtype=x.dtype)
    x_centered = x - sample_mean.reshape(1, D, 1, 1)
    x_norm = x_centered / std
    out = bn_scale.reshape(1, D, 1, 1) * x_norm + bn_bias.reshape(1, D, 1, 1)

    return out, save_mean, save_mean


def batchnorm_inference(x, bn_scale, bn_bias, save_mean, save_var, eps=0.01):
    D = x.shape[1]
    std = np.sqrt(save_var.reshape(1, D, 1, 1) + eps, dtype=x.dtype)
    x_centered = x - save_mean.reshape(1, D, 1, 1)
    x_norm = x_centered / std
    out = bn_scale.reshape(1, D, 1, 1) * x_norm + bn_bias.reshape(1, D, 1, 1)

    return out


def batchnorm_backward(gradient_Y, x, bn_scale, dbn_scale, dbn_bias, eps, save_mean, save_var):
    D = gradient_Y.shape[1]

    sample_mean = save_mean
    sample_var = save_var

    std = np.sqrt(sample_var.reshape(1, D, 1, 1) + eps)
    x_centered = x - sample_mean.reshape(1, D, 1, 1)
    x_norm = x_centered / std

    dbn_scale = (gradient_Y * x_norm).sum(axis=(0, 2, 3))
    dbn_bias = gradient_Y.sum(axis=(0, 2, 3))

    dx_norm = gradient_Y * bn_scale.reshape(1, D, 1, 1)
    dx_centered = dx_norm / std
    dmean = -(dx_centered.sum(axis=(0, 2, 3)) + 2 / D *
              x_centered.sum(axis=(0, 2, 3))).reshape(1, D, 1, 1)
    dstd = (dx_norm * x_centered * -std ** (-2)
            ).sum(axis=(0, 2, 3)).reshape(1, D, 1, 1)
    dvar = dstd / 2 / std
    dx = dx_centered + (dmean + dvar * 2 * x_centered) / D

    return dx, dbn_scale, dbn_bias
