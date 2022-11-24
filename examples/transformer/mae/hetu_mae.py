import hetu as ht
import numpy as np
import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

from config import ViTMAEConfig


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class ViTMAEPatchEmbeddings(object):
    def __init__(self, config, name='ViTMAEPatchEmbeddings'):
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

    def __call__(self, pixel_values, input_shape):
        batch_size, num_channels, height, width = input_shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        embeddings = self.projection(pixel_values)
        embeddings = ht.array_reshape_op(
            embeddings, (batch_size, self.hidden_size, -1))
        embeddings = ht.transpose_op(embeddings, (0, 2, 1))
        return embeddings


class ViTMAEEmbeddings(object):
    def __init__(self, config, name='ViTMAEEmbeddings'):
        self.cls_token = ht.init.zeros(
            shape=(1, 1, config.hidden_size), name=name+'.cls_token')
        self.patch_embeddings = ViTMAEPatchEmbeddings(
            config, name=name+'.patch_embeddings')
        num_patches = self.patch_embeddings.num_patches
        self.num_patches = num_patches
        self.dim = config.hidden_size

        pos_embed = get_2d_sincos_pos_embed(
            config.hidden_size, int(self.patch_embeddings.num_patches**0.5), add_cls_token=True
        )
        pos_embed = np.expand_dims(pos_embed, axis=0)
        self.position_embeddings = ht.Variable(
            name=name+'.position_embeddings', value=pos_embed, trainable=False)

        self.dropout = ht.layers.DropOut(config.hidden_dropout_prob)
        self.config = config

    def random_masking(self, sequence, input_shape, noise=None):
        batch_size, seq_length, dim = input_shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = np.random.rand(batch_size, seq_length)
        noise = ht.Variable(name='noise', value=noise, trainable=False)
        ids_shuffle = ht.argsort_op(noise)
        ids_restore = ht.argsort_op(ids_shuffle)
        ids_keep = ht.slice_op(ids_shuffle, (0, 0), (-1, len_keep))
        ids_keep = ht.unsqueeze_op(ids_keep, -1)
        ids_keep = ht.repeat_op(ids_keep, (1, 1, dim))

        sequence_masked = ht.gather_op(sequence, dim=1, index=ids_keep)

        mask = np.ones([batch_size, seq_length])
        mask[:, :len_keep] = 0
        mask = ht.Variable(name='mask', value=mask, trainable=False)
        mask = ht.gather_op(mask, dim=1, index=ids_restore)

        return sequence_masked, mask, ids_restore

    def __call__(self, pixel_values, input_shape, noise=None):
        batch_size, num_channels, height, width = input_shape
        embeddings = self.patch_embeddings(pixel_values, input_shape)
        position_embeddings = ht.slice_op(
            self.position_embeddings, [0, 1, 0], [-1, -1, -1])
        cls_position_embeddings = ht.slice_op(
            self.position_embeddings, [0, 0, 0], [-1, 1, -1])
        embeddings = embeddings + position_embeddings

        embeddings, mask, ids_restore = self.random_masking(
            embeddings, [batch_size, self.num_patches, self.dim], noise)

        cls_token = self.cls_token + cls_position_embeddings
        cls_tokens = ht.broadcast_shape_op(
            self.cls_token, (batch_size, -1, -1))
        embeddings = ht.concat_op(cls_tokens, embeddings, axis=1)

        return embeddings, mask, ids_restore


class ViTMAESelfAttention(object):
    def __init__(self, config, name='ViTMAESelfAttention'):
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
        attention_scores = attention_scores * \
            (1/math.sqrt(self.attention_head_size))

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


class ViTMAESelfOutput(object):
    def __init__(self, config, name='ViTMAESelfOutput'):
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


class ViTMAEAttention(object):
    def __init__(self, config, name='ViTMAEAttention'):
        self.attention = ViTMAESelfAttention(config, name=name+'.attention')
        self.output = ViTMAESelfOutput(config, name=name+'.output')

    def __call__(self, hidden_states, input_shape, head_mask=None, output_attentions=False):
        self_outputs = self.attention(
            hidden_states, input_shape, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], input_shape)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class ViTMAEIntermediate(object):
    def __init__(self, config, name='ViTMAEIntermediate'):
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


class ViTMAEOutput(object):
    def __init__(self, config, name='ViTMAEOutput'):
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


class ViTMAELayer(object):
    def __init__(self, config, name='ViTMAELayer'):
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTMAEAttention(config, name=name+'.attention')
        self.intermediate = ViTMAEIntermediate(
            config, name=name+'.intermediate')
        self.output = ViTMAEOutput(config, name=name+'.output')
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


class ViTMAEEncoder(object):
    def __init__(self, config, name='ViTMAEEncoder'):
        self.config = config
        self.layer = [ViTMAELayer(config, name=name+'.layer.'+str(i))
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


class ViTMAEModel(object):
    def __init__(self, config):
        self.config = config

        self.embeddings = ViTMAEEmbeddings(config, name='embeddings')
        self.encoder = ViTMAEEncoder(config, name='encoder')
        self.layernorm = ht.layers.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, name='layernorm')
        self.num_patches = self.embeddings.num_patches
        self.dim = config.hidden_size

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            assert False
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def __call__(self, pixel_values, input_shape, noise=None, head_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output, mask, ids_restore = self.embeddings(
            pixel_values, input_shape, noise=noise)

        batch_size = input_shape[0]
        num_patches = self.num_patches
        hidden_size = self.dim

        len_keep = int(num_patches * (1 - self.config.mask_ratio))

        encoder_outputs = self.encoder(
            embedding_output,
            input_shape=(batch_size, len_keep + 1, hidden_size),
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        return [sequence_output]
