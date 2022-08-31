from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import embedding_prepack, quantized_embedding_lookup, update_quantized_embedding
from ..ndarray import empty
from ..stream import create_event_handle


# TODO: finish this op
class QuantizedEmbeddingLookUpOp(Op):
    def __init__(self, embed, qparams, indices, digit, ctx=None):
        super().__init__(QuantizedEmbeddingLookUpOp,
                         [embed, qparams, indices], ctx)
        self.digit = digit
        assert self.digit in (8, 16)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            quantized_embedding_lookup(
                input_vals[0], input_vals[2], output_val, input_vals[1], self.digit, stream_handle)

    def gradient(self, output_grad):
        grad_node = quantized_embedding_gradient_op(
            output_grad, self.inputs[2], self.inputs[1], self, self.inputs[0], self.digit, self.raw_ctx)
        return [grad_node, None, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        output_shape = list(input_shapes[2])
        output_shape.append(input_shapes[0][1])
        return tuple(output_shape)

    def forward_hook(self, config):
        super().forward_hook(config)
        embed_var = self.inputs[0]
        qparam_var = self.inputs[1]
        ori_embed = config.placeholder_to_arr_map[embed_var]
        qparam_arr = config.placeholder_to_arr_map[qparam_var]
        if self.digit == 8:
            dtype = np.uint8
        else:
            dtype = np.uint16
        new_embed = empty(ori_embed.shape, ctx=self.ctx,
                          dtype=dtype, force32=False)
        config.placeholder_to_arr_map[embed_var] = new_embed
        embedding_prepack(ori_embed, new_embed, qparam_arr,
                          self.digit, config.comp_stream)
        config.comp_stream.sync()


def quantized_embedding_lookup_op(embed, qparams, indices, digit, ctx=None):
    return QuantizedEmbeddingLookUpOp(embed, qparams, indices, digit, ctx=ctx)


class QuantizedEmbeddingGradientOp(Op):
    def __init__(self, grad, indices, qparams, lookup, embed, digit, ctx=None):
        super().__init__(QuantizedEmbeddingGradientOp,
                         [grad, indices, qparams, lookup], ctx)
        self.embed = embed
        self.digit = digit
        assert self.digit in (8, 16)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            self.opt.optimizer.update_one(
                self.embed, input_vals[3], input_vals[0], self.opt_index, stream_handle)
            update_quantized_embedding(
                input_vals[0], input_vals[1], self.embed_arr, input_vals[2], input_vals[3], self.digit, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return None

    def forward_hook(self, config):
        from ..optimizer import OptimizerOp
        super().forward_hook(config)
        self.embed_arr = config.placeholder_to_arr_map[self.embed]


def quantized_embedding_gradient_op(grad, indices, qparams, lookup, embed, digit, ctx=None):
    return QuantizedEmbeddingGradientOp(grad, indices, qparams, lookup, embed, digit, ctx=ctx)
