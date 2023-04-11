from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import tensor_quantize, \
    quantized_embedding_lookup, \
    unified_quantized_embedding_lookup, \
    embedding_prepack
from ..ndarray import empty
from .EmbeddingLookUp import embedding_lookup_gradient_with_lookup_op, embedding_lookup_gradient_dedupgrad_op


class UnifiedQuantizedEmbeddingLookUpOp(Op):
    def __init__(self, embed, indices, scale, zero_point, digit, ctx=None):
        assert digit in (8, 16)
        super().__init__(UnifiedQuantizedEmbeddingLookUpOp,
                         [embed, indices], ctx)
        self.digit = digit
        self.scale = scale
        self.middle = zero_point
        self.minele = zero_point - 2 ** (digit - 1) * scale
        self.grad_node = None
        if self.digit == 8:
            dtype = np.uint8
        else:
            dtype = np.uint16
        embed.dtype = dtype
        embed.is_embed = True

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            unified_quantized_embedding_lookup(
                input_vals[0], input_vals[1], output_val, self.digit, self.scale, self.minele, stream_handle)

    def gradient(self, output_grad):
        self.grad_node = embedding_lookup_gradient_with_lookup_op(
            output_grad, self.inputs[1], self, None, ctx=self.raw_ctx)
        grad_node = embedding_lookup_gradient_dedupgrad_op(
            self.grad_node, output_grad, ctx=self.raw_ctx)
        return [grad_node, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        if self.grad_node is not None:
            self.grad_node.embed_shape = input_shapes[0]
        output_shape = list(input_shapes[1])
        output_shape.append(input_shapes[0][1])
        return tuple(output_shape)

    def forward_hook(self, config):
        super().forward_hook(config)
        embed_var = self.inputs[0]
        ori_embed = config.placeholder_to_arr_map[embed_var]
        dtype = embed_var.dtype
        new_embed = empty(ori_embed.shape, ctx=self.ctx,
                          dtype=dtype, force32=False)
        config.placeholder_to_arr_map[embed_var] = new_embed
        tensor_quantize(ori_embed, new_embed, self.digit, self.scale,
                        self.minele, True, config.comp_stream)
        config.comp_stream.sync()


def unified_quantized_embedding_lookup_op(embed, indices, scale, zero_point, digit, ctx=None):
    return UnifiedQuantizedEmbeddingLookUpOp(embed, indices, scale, zero_point, digit, ctx=ctx)


class QuantizedEmbeddingLookUpOp(Op):
    def __init__(self, embed, indices, qparams, digit, ctx=None):
        super().__init__(QuantizedEmbeddingLookUpOp,
                         [embed, indices, qparams], ctx)
        self.digit = digit
        self.grad_node = None
        assert self.digit in (8, 16)
        if self.digit == 8:
            dtype = np.uint8
        else:
            dtype = np.uint16
        embed.dtype = dtype
        embed.is_embed = True

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            quantized_embedding_lookup(
                input_vals[0], input_vals[1], output_val, input_vals[2], self.digit, stream_handle)

    def gradient(self, output_grad):
        self.grad_node = embedding_lookup_gradient_with_lookup_op(
            output_grad, self.inputs[1], self, None, ctx=self.raw_ctx)
        grad_node = embedding_lookup_gradient_dedupgrad_op(
            self.grad_node, output_grad, ctx=self.raw_ctx)
        return [grad_node, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        if self.grad_node is not None:
            self.grad_node.embed_shape = input_shapes[0]
        output_shape = list(input_shapes[1])
        output_shape.append(input_shapes[0][1])
        return tuple(output_shape)

    def forward_hook(self, config):
        super().forward_hook(config)
        embed_var = self.inputs[0]
        qparam_var = self.inputs[2]
        ori_embed = config.placeholder_to_arr_map[embed_var]
        qparam_arr = config.placeholder_to_arr_map[qparam_var]
        dtype = embed_var.dtype
        new_embed = empty(ori_embed.shape, ctx=self.ctx,
                          dtype=dtype, force32=False)
        config.placeholder_to_arr_map[embed_var] = new_embed
        embedding_prepack(ori_embed, new_embed, qparam_arr,
                          self.digit, config.comp_stream)
        config.comp_stream.sync()


def quantized_embedding_lookup_op(embed, indices, qparams, digit, ctx=None):
    return QuantizedEmbeddingLookUpOp(embed, indices, qparams, digit, ctx=ctx)
