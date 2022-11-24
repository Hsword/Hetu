import hetu as ht
import numpy as np
import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

from config import ViTConfig


class ViTPatchEmbeddings(object):
    def __init__(self, config, name='ViTPatchEmbeddings'):
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(
            image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(
            patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * \
            (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.hidden_size = hidden_size

        self.projection = ht.layers.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size, name=name+'.projection')

    def __call__(self, pixel_values, input_shape, interpolate_pos_encoding=False):
        batch_size, num_channels, height, width = input_shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )

        embeddings = self.projection(pixel_values)
        embeddings = ht.array_reshape_op(
            embeddings, (batch_size, self.hidden_size, -1))
        embeddings = ht.transpose_op(embeddings, (0, 2, 1))

        return embeddings


class ViTEmbeddings(object):
    def __init__(self, config, use_mask_token=False, name='ViTEmbeddings'):
        self.cls_token = ht.init.zeros(
            shape=(1, 1, config.hidden_size), name=name+'.cls_token')
        self.mask_token = ht.init.zeros(shape=(
            1, 1, config.hidden_size), name=name+'.mask_token') if use_mask_token else None
        self.patch_embeddings = ViTPatchEmbeddings(
            config, name=name+'.patch_embeddings')
        num_patches = self.patch_embeddings.num_patches
        self.num_patches = num_patches
        self.position_embeddings = ht.init.zeros(shape=(
            1, num_patches + 1, config.hidden_size), name=name+'.position_embeddings')
        self.dropout = ht.layers.DropOut(config.hidden_dropout_prob)
        self.config = config

    def interpolate_pos_encoding(self, embeddings, embeddings_shape, height, width):
        num_patches = embeddings_shape[1] - 1
        num_positions = self.num_patches
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = ht.slice_op(
            self.position_embeddings, [0, 0, 0], [-1, 1, -1])
        patch_pos_embed = ht.slice_op(
            self.position_embeddings, [0, 1, 0], [-1, -1, -1])
        dim = embeddings_shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = ht.array_reshape_op(patch_pos_embed, (1, int(
            math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim))
        patch_pos_embed = ht.transpose_op(patch_pos_embed, (0, 3, 1, 2))

        patch_pos_embed = ht.interpolate_op(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions),
                          w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = ht.transpose_op(patch_pos_embed, (0, 2, 3, 1))
        patch_pos_embed = ht.array_reshape_op(patch_pos_embed, (1, -1, dim))
        return ht.concat_op(ht.unsqueeze_op(class_pos_embed), patch_pos_embed, axis=1)

    def __call__(self, pixel_values, input_shape, bool_masked_pos=None, interpolate_pos_encoding=False):
        batch_size, num_channels, height, width = input_shape
        embeddings = self.patch_embeddings(
            pixel_values, input_shape, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = self.num_patches
            mask_tokens = ht.broadcast_shape_op(
                self.mask_token, (batch_size, seq_length, -1))
            mask = ht.unsqueeze_op(bool_masked_pos, -1)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        cls_tokens = ht.broadcast_shape_op(
            self.cls_token, (batch_size, -1, -1))
        embeddings = ht.concat_op(cls_tokens, embeddings, axis=1)

        if interpolate_pos_encoding:
            embeddings = embeddings + \
                self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + \
                ht.broadcast_shape_op(
                    self.position_embeddings, (batch_size, -1, -1))
        embeddings = self.dropout(embeddings)
        return embeddings


class ViTSelfAttention(object):
    def __init__(self, config, name='ViTSelfAttention'):
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        self.dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = ht.layers.Linear(config.hidden_size, self.all_head_size,
                                      bias=config.qkv_bias, weight_transpose=True, name=name+'.query')
        self.key = ht.layers.Linear(config.hidden_size, self.all_head_size,
                                    bias=config.qkv_bias, weight_transpose=True, name=name+'.key')
        self.value = ht.layers.Linear(config.hidden_size, self.all_head_size,
                                      bias=config.qkv_bias, weight_transpose=True, name=name+'.value')
        self.dropout = ht.layers.DropOut(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, shape):
        new_x_shape = shape[:-1] + \
            (self.num_attention_heads, self.attention_head_size)
        x = ht.array_reshape_op(x, new_x_shape)
        x = ht.transpose_op(x, (0, 2, 1, 3))
        return x

    def __call__(self, hidden_states, input_shape, head_mask=None, output_attentions=False):

        hidden_states = ht.array_reshape_op(hidden_states, [-1, self.dim])
        mixed_query_layer = self.query(hidden_states)

        shape = input_shape[:-1] + (self.all_head_size, )
        key_layer = self.transpose_for_scores(self.key(hidden_states), shape)
        value_layer = self.transpose_for_scores(
            self.value(hidden_states), shape)
        query_layer = self.transpose_for_scores(mixed_query_layer, shape)

        key_layer = ht.transpose_op(key_layer, (0, 1, 3, 2))
        attention_scores = ht.batch_matmul_op(query_layer, key_layer)
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        attention_probs = ht.softmax_op(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ht.batch_matmul_op(attention_probs, value_layer)

        context_layer = ht.transpose_op(context_layer, (0, 2, 1, 3))
        context_layer = ht.array_reshape_op(context_layer, shape)

        outputs = (context_layer, attention_probs) if output_attentions else (
            context_layer,)

        return outputs


class ViTSelfOutput(object):
    def __init__(self, config, name='ViTSelfOutput'):
        self.dense = ht.layers.Linear(
            config.hidden_size, config.hidden_size, weight_transpose=True, name=name+'.dense')
        self.dropout = ht.layers.DropOut(config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_shape):
        if (len(input_shape) > 2):
            hidden_states = ht.array_reshape_op(
                hidden_states, (-1, input_shape[-1]))
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (len(input_shape) > 2):
            hidden_states = ht.array_reshape_op(
                hidden_states, input_shape[:-1]+(-1,))
        return hidden_states


class ViTAttention(object):
    def __init__(self, config, name='ViTAttention'):
        self.attention = ViTSelfAttention(config, name=name+'.attention')
        self.output = ViTSelfOutput(config, name=name+'.output')

    def __call__(self, hidden_states, input_shape, head_mask=None, output_attentions=False):
        self_outputs = self.attention(
            hidden_states, input_shape, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], input_shape)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class ViTIntermediate(object):
    def __init__(self, config, name='ViTIntermediate'):
        self.dim = config.intermediate_size
        self.dense = ht.layers.Linear(
            config.hidden_size, config.intermediate_size, weight_transpose=True, name=name+'.dense')
        if config.hidden_act == "relu":
            self.intermediate_act_fn = ht.relu_op
        elif config.hidden_act == "gelu":
            self.intermediate_act_fn = ht.gelu_op

    def __call__(self, hidden_states, input_shape):
        if (len(input_shape) > 2):
            hidden_states = ht.array_reshape_op(
                hidden_states, (-1, input_shape[-1]))
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        if (len(input_shape) > 2):
            hidden_states = ht.array_reshape_op(
                hidden_states, input_shape[:-1]+(-1,))
        shape = input_shape[:-1]+(self.dim,)
        return hidden_states, shape


class ViTOutput(object):
    def __init__(self, config, name='ViTOutput'):
        self.dense = ht.layers.Linear(
            config.intermediate_size, config.hidden_size, weight_transpose=True, name=name+'.dense')
        self.dropout = ht.layers.DropOut(config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, input_shape):
        if (len(input_shape) > 2):
            hidden_states = ht.array_reshape_op(
                hidden_states, (-1, input_shape[-1]))
        hidden_states = self.dense(hidden_states)
        if (len(input_shape) > 2):
            hidden_states = ht.array_reshape_op(
                hidden_states, input_shape[:-1]+(-1,))
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class ViTLayer(object):
    def __init__(self, config, name='ViTLayer'):
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config, name=name+'.attention')
        self.intermediate = ViTIntermediate(config, name=name+'.intermediate')
        self.output = ViTOutput(config, name=name+'.output')
        self.layernorm_before = ht.layers.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, name=name+'.layernorm_before')
        self.layernorm_after = ht.layers.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, name=name+'.layernorm_after')

    def __call__(self, hidden_states, input_shape, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            input_shape,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output, shape = self.intermediate(layer_output, input_shape)

        layer_output = self.output(layer_output, hidden_states, shape)

        outputs = (layer_output,) + outputs

        return outputs


class ViTEncoder(object):
    def __init__(self, config, name='ViTEncoder'):
        self.config = config
        self.layer = [ViTLayer(config, name=name+'.layer.'+str(i))
                      for i in range(config.num_hidden_layers)]

    def __call__(self, hidden_states, input_shape, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i in range(len(self.layer)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_module = self.layer[i]
            layer_outputs = layer_module(
                hidden_states, input_shape, layer_head_mask, output_attentions)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        else:
            raise NotImplementedError


class ViTPooler(object):
    def __init__(self, config, name='ViTPooler'):
        self.dense = ht.layers.Linear(
            config.hidden_size, config.hidden_size, weight_transpose=True, name=name+'.dense')
        self.activation = ht.tanh_op

    def __call__(self, hidden_states, input_shape):
        first_token_tensor = ht.slice_op(hidden_states, (0, 0, 0), (-1, 1, -1))
        if (len(input_shape) > 2):
            first_token_tensor = ht.array_reshape_op(
                first_token_tensor, (-1, input_shape[-1]))
        pooled_output = self.dense(first_token_tensor)
        if (len(input_shape) > 2):
            pooled_output = ht.array_reshape_op(
                pooled_output, (input_shape[0], 1, -1))
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ViTModel(object):
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        self.config = config

        self.embeddings = ViTEmbeddings(
            config, use_mask_token=use_mask_token, name='embeddings')
        self.encoder = ViTEncoder(config, name='encoder')
        self.layernorm = ht.layers.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, name='layernorm')
        self.pooler = ViTPooler(
            config, name="pooler") if add_pooling_layer else None
        self.num_patches = self.embeddings.num_patches
        self.dim = config.hidden_size

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            assert False
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def __call__(self, pixel_values, input_shape, bool_masked_pos=None, head_mask=None, output_attentions=None, output_hidden_states=None,
                 interpolate_pos_encoding=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            pixel_values, input_shape, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )
        batch_size = input_shape[0]
        num_patches = self.num_patches
        hidden_size = self.dim

        encoder_outputs = self.encoder(
            embedding_output,
            input_shape=(batch_size, (num_patches+1), hidden_size),
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(
            sequence_output, (batch_size, (num_patches+1), hidden_size)) if self.pooler is not None else None

        head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (
            sequence_output,)
        return head_outputs
