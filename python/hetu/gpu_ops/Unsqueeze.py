from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import array_reshape
from .. import ndarray


class UnsqueezeOp(Op):
    def __init__(self, node_A, axis=0, ctx=None):
        super().__init__(UnsqueezeOp, [node_A], ctx)
        self.axis = axis

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.expand_dims(input_vals[0].asnumpy(), self.axis)
        else:
            array_reshape(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [squeeze_op(output_grad, self.axis, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        ori_shape = input_shapes[0]
        input_dim = len(ori_shape)
        axis = [self.axis] if isinstance(self.axis, int) else list(self.axis)
        n_axis = len(axis)
        for i in range(n_axis):
            if axis[i] < 0:
                axis[i] += (input_dim + 1)
        output_dim = input_dim + n_axis
        cnt_axis = 0
        cnt_ori = 0
        output_shape = []
        for i in range(output_dim):
            if cnt_axis < n_axis and i==axis[cnt_axis]:
                output_shape.append(1)
                cnt_axis += 1
            else:
                output_shape.append(ori_shape[cnt_ori])
                cnt_ori += 1
        self.output_shape = output_shape
        return output_shape

class SqueezeOp(Op):
    def __init__(self, node_A, axis=None, ctx=None):
        super().__init__(SqueezeOp, [node_A], ctx)
        self.axis = axis

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.squeeze(input_vals[0].asnumpy(), self.axis)
        else:
            array_reshape(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [unsqueeze_op(output_grad, self.axis, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        ori_shape = input_shapes[0]
        input_dim = len(ori_shape)

        if self.axis == None:
            output_shape = []
            for i in range(input_dim):
                if ori_shape[i]!=1:
                    output_shape.append(ori_shape[i])
            return output_shape
        else:
            axis = [self.axis] if isinstance(self.axis, int) else list(self.axis)
            n_axis = len(axis)
            for i in range(n_axis):
                if axis[i] < 0:
                    axis[i] += (input_dim + 1)

            cnt_axis = 0
            output_shape = []
            for i in range(input_dim):
                if cnt_axis < n_axis and i == axis[cnt_axis]:
                    assert ori_shape[i]==1
                    cnt_axis += 1
                else:
                    output_shape.append(ori_shape[i])
            return output_shape

def unsqueeze_op(node_A, axis=0, ctx=None):
    """Make a new instance of torch.unsqueeze and call the instance.

    Parameters:
    ----
    node_A : Node
        Input Variable.
    axis: Int or Tuple or List
        The axis to add.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return UnsqueezeOp(node_A, axis, ctx=ctx)


def squeeze_op(node_A, axis=None, ctx=None):
    """Make a new instance of torch.squeeze and call the instance.

    Parameters:
    ----
    node_A : Node
        Input Variable.
    axis: Int or Tuple or List
        The axis to delete.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return SqueezeOp(node_A, axis, ctx=ctx)