from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from .._base import DNNL_LIB
import numpy as np
#from ..cpu_links import embedding_lookup as cpu_embedding_lookup
from ..gpu_links import robe_lookup


class RobeLookUp(Op):
    def __init__(self, roar, index, len, x, ctx=None):
        super().__init__(RobeLookUp, [roar, index, x], ctx)
        self.len = len
        self.Bg = 134923849
        self.Cg = 432094230
        self.Dg = 983458345
        self.grad_node = None
    
    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if (False):
                assert(0)
            else:
                flatten_index = input_vals[1].asnumpy().reshape(-1).astype(np.int32)
                t = input_vals[0].asnumpy()
                sz = output_val.size
                if (flatten_index + sz < t.size):
                    output_val[:] = t[flatten_index:flatten_index + sz].reshape(output_val.shape)
                else:
                    output_val[:] = np.concatenate((t[flatten_index:t.size], t[0:output_val.size - (t.size - flatten_index)]), axis=0).reshape(output_val.shape)
        
        else:
            robe_lookup(input_vals[0], input_vals[1], input_vals[2], output_val, self.len, self.Bg, self.Cg, self.Dg, stream_handle)


    def gradient(self, output_grad):
        self.grad_node = robe_lookup_gradient_op(
            output_grad, self.inputs[1], self.inputs[2], None, self.Bg, self.Cg, self.Dg, ctx=self.raw_ctx)
        return [self.grad_node, None, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        if self.grad_node is not None:
           self.grad_node.robe_shape = input_shapes[0]
        assert (input_shapes[1] == input_shapes[2])
        output_shape = list(input_shapes[1])
        output_shape.append(self.len)
        return tuple(output_shape)
   


class RobeLookUp_Gradient(Op):
    def __init__(self, vectors, index, x, robe_shape, Bg, Cg, Dg, ctx=None):
        inputs = [vectors]
        if isinstance(index, Op):
            inputs.append(index)
            inputs.append(x)
            self.index = None
            self.x = None
        else:
            self.index = index
            self.x = x
        super().__init__(RobeLookUp_Gradient,
                         inputs, ctx)
        self.robe_shape = robe_shape
        self.Bg = Bg
        self.Cg = Cg
        self.Dg = Dg
        self.use_robe_slices = True

    def compute(self, input_vals, output_val, stream_handle=None):
        assert self.robe_shape
        if self.index is None:
            output_val.update(
                values=input_vals[0], indices=input_vals[1], x = input_vals[2], Bg = self.Bg, Cg = self.Cg, Dg = self.Dg, dense_shape=self.robe_shape)
        else:
            assert(0)
            output_val.update(
                values=input_vals[0], indices=self.index, x = self.x, Bg = self.Bg, Cg = self.Cg, Dg = self.Dg, dense_shape=self.robe_shape)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert self.robe_shape
        return self.robe_shape


def robe_lookup_op(roar, index, len, x, ctx=None):
    """Make a new instance of EmbeddingLookUp and call the instance.

    Parameters:
    ----
    embedding : Node
        The Node of Embedding.
    index : Node
        The index to be looked up.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return RobeLookUp(roar, index, len, x, ctx=ctx)


def robe_lookup_gradient_op(vectors, index, x, robe_shape, Bg, Cg, Dg, ctx=None):
    """Make a new instance of EmbeddingLookUp_Gradient and call the instance.

    Parameters:
    ----
    vectors : Node
        Vectors which looked up from Embedding.
    index : Node
        The index to be looked up.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return RobeLookUp_Gradient(vectors, index, x, robe_shape, Bg, Cg, Dg, ctx=ctx)
