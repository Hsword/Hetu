from typing import Any, Optional, Tuple, Union

import hetu as ht
import numpy as np
from config import CLIPConfig, CLIPVisionConfig, CLIPTextConfig

def _expand_mask(mask, input_shape, tgt_len=None):
    bsz, src_len = input_shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = ht.broadcast_shape_op(mask, (bsz, 1, tgt_len, src_len), add_axes=(1, 2))
    inverted_mask = 1.0 - expanded_mask
    inverted_mask = ht.masked_fill_op(inverted_mask, inverted_mask, np.finfo(np.float32).min)

    return inverted_mask


def contrastive_loss(logits, len):
    return ht.crossentropy_op(logits, ht.arange_op(0, len))


def clip_loss(similarity, len):
    caption_loss = contrastive_loss(similarity, len)
    image_loss = contrastive_loss(ht.transpose_op(similarity, (1,0)), len)
    return (caption_loss + image_loss) / 2.0


class CLIPVisionEmbeddings(object):
    def __init__(self, config, name='CLIPVisionEmbeddings'):       
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = ht.init.random_normal(shape=(self.embed_dim, ), name=name+'.class_embedding')

        self.patch_embedding = ht.layers.Conv2d(in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, 
                                                stride=self.patch_size, bias=False, name=name+'.patch_embedding')

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = ht.layers.Embedding(self.num_positions, self.embed_dim, name=name+'.position_embedding')
        position_ids = ht.arange_op(0, self.num_positions)
        self.position_ids = ht.array_reshape_op(position_ids, (1, -1))

    def __call__(self, pixel_values, input_shape):
        batch_size = input_shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  

        patch_embeds = ht.array_reshape_op(patch_embeds, (batch_size, self.embed_dim, -1))
        patch_embeds = ht.transpose_op(patch_embeds, (0, 2, 1))

        class_embeds = ht.broadcast_shape_op(self.class_embedding, (batch_size, 1, -1))
        embeddings = ht.concat_op(class_embeds, patch_embeds, axis=1)
        embeddings = embeddings + ht.broadcastto_op(self.position_embedding(self.position_ids), embeddings)
        return embeddings


class CLIPTextEmbeddings(object):
    def __init__(self, config, name='CLIPTextEmbeddings'):
        embed_dim = config.hidden_size

        self.token_embedding = ht.layers.Embedding(config.vocab_size, embed_dim, name=name+'.token_embedding')
        self.position_embedding = ht.layers.Embedding(config.max_position_embeddings, embed_dim, name=name+'.position_embedding')

        position_ids = ht.arange_op(0, config.max_position_embeddings)
        self.position_ids = ht.array_reshape_op(position_ids, (1, -1))

    def __call__(self, input_ids=None, input_shape=None, position_ids=None, inputs_embeds=None, inputs_embeds_shape=None):
        seq_length = input_shape[-1] if input_ids is not None else inputs_embeds_shape[-2]

        if position_ids is None:
            position_ids = ht.slice_op(self.position_ids, (0, 0), (-1, seq_length))

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + ht.broadcastto_op(position_embeddings, inputs_embeds)

        return embeddings


class CLIPAttention(object):
    def __init__(self, config, name='CLIPAttention'):
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = ht.layers.Linear(self.embed_dim, self.embed_dim, weight_transpose=True, name=name+'.k_proj')
        self.v_proj = ht.layers.Linear(self.embed_dim, self.embed_dim, weight_transpose=True, name=name+'.v_proj')
        self.q_proj = ht.layers.Linear(self.embed_dim, self.embed_dim, weight_transpose=True, name=name+'.q_proj')
        self.out_proj = ht.layers.Linear(self.embed_dim, self.embed_dim, weight_transpose=True, name=name+'.out_proj')

    def _shape(self, tensor, seq_len, bsz):
        tensor = ht.array_reshape_op(tensor, (bsz, seq_len, self.num_heads, self.head_dim))
        tensor = ht.transpose_op(tensor, (0, 2, 1, 3))
        return tensor

    def __call__(self, hidden_states, input_shape, attention_mask=None, causal_attention_mask=None, output_attentions=False):
        bsz, tgt_len, embed_dim = input_shape

        hidden_states = ht.array_reshape_op(hidden_states, (-1, embed_dim))
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)
        query_states = ht.array_reshape_op(query_states, proj_shape)
        key_states = ht.array_reshape_op(key_states, proj_shape)
        value_states = ht.array_reshape_op(value_states, proj_shape)

        src_len = tgt_len
        k = ht.transpose_op(key_states, (0, 2, 1))
        attn_weights = ht.batch_matmul_op(query_states, k)

        if causal_attention_mask is not None:
            attn_weights = ht.array_reshape_op(attn_weights, (bsz, self.num_heads, tgt_len, src_len))
            attn_weights = attn_weights + causal_attention_mask
            attn_weights = ht.array_reshape_op(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        if attention_mask is not None:
            attn_weights = ht.array_reshape_op(attn_weights, (bsz, self.num_heads, tgt_len, src_len))
            attn_weights = attn_weights + attention_mask
            attn_weights = ht.array_reshape_op(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_weights = ht.softmax_op(attn_weights)        

        if output_attentions:
            attn_weights_reshaped = ht.array_reshape_op(attn_weights, (bsz, self.num_heads, tgt_len, src_len))
            attn_weights = ht.array_reshape_op(attn_weights_reshaped, (bsz * self.num_heads, tgt_len, src_len))
        else:
            attn_weights_reshaped = None

        attn_probs = ht.dropout_op(attn_weights, 1-self.dropout)
        attn_output = ht.batch_matmul_op(attn_probs, value_states)

        attn_output = ht.array_reshape_op(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim))
        attn_output = ht.transpose_op(attn_output, (0, 2, 1, 3))

        attn_output = ht.array_reshape_op(attn_output, (-1, self.embed_dim))

        attn_output = self.out_proj(attn_output)
        attn_output = ht.array_reshape_op(attn_output, (bsz, tgt_len, -1))

        return attn_output, attn_weights_reshaped


class CLIPMLP(object):
    def __init__(self, config, name='CLIPMLP'):
        self.config = config
        if config.hidden_act == "relu":
            self.activation_fn = ht.relu_op
        elif config.hidden_act == "gelu":
            self.activation_fn = ht.gelu_op

        self.fc1 = ht.layers.Linear(config.hidden_size, config.intermediate_size, weight_transpose=True, name=name+'.fc1')
        self.fc2 = ht.layers.Linear(config.intermediate_size, config.hidden_size, weight_transpose=True, name=name+'.fc2')

    def __call__(self, hidden_states, input_shape):
        hidden_states = ht.array_reshape_op(hidden_states, (-1, input_shape[-1]))
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = ht.array_reshape_op(hidden_states, input_shape)

        return hidden_states


class CLIPEncoderLayer(object):
    def __init__(self, config, name='CLIPEncoderLayer'):
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config, name=name+'.self_attn')
        self.layer_norm1 = ht.layers.LayerNorm(self.embed_dim, name=name+'.layer_norm1')
        self.mlp = CLIPMLP(config, name=name+'.mlp')
        self.layer_norm2 = ht.layers.LayerNorm(self.embed_dim, name=name+'.layer_norm2')

    def __call__(self, hidden_states, input_shape, attention_mask, causal_attention_mask, output_attentions=False):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            input_shape=input_shape,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states, input_shape)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPEncoder(object):
    def __init__(self, config, name='CLIPEncoder'):
        self.config = config
        self.layers = [CLIPEncoderLayer(config, name=name+'.layers.'+str(i)) for i in range(config.num_hidden_layers)]

    def __call__(self, inputs_embeds, input_shape, attention_mask=None, causal_attention_mask=None,
        output_attentions=None, output_hidden_states=None, return_dict=None):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                input_shape,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        else:
            assert False


class CLIPTextTransformer(object):
    def __init__(self, config, name='CLIPTextTransformer'):
        self.config = config
        self.embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config, name=name+'.embeddings')
        self.encoder = CLIPEncoder(config, name=name+'.encoder')
        self.final_layer_norm = ht.layers.LayerNorm(self.embed_dim, name=name+'.final_layer_norm')

    def _build_causal_attention_mask(self, bsz, seq_len):
        mask = ht.full_op((bsz, seq_len, seq_len), np.finfo(np.float32).min)
        mask = ht.triu_op(mask, 1)
        mask = ht.unsqueeze_op(mask, 1)
        return mask

    def __call__(self, input_ids=None, input_shape=None, attention_mask=None, position_ids=None,
        output_attentions=None, output_hidden_states=None, return_dict=None):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        input_ids = ht.array_reshape_op(input_ids, [-1, input_shape[-1]])

        hidden_states = self.embeddings(input_ids=input_ids, input_shape=input_shape, position_ids=position_ids)

        bsz, seq_len = input_shape
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len)

        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, input_shape)

        encoder_shape = input_shape + (self.embed_dim, )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            input_shape=encoder_shape,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        index1 = ht.arange_op(0, bsz)
        index2 = ht.argmax_op(input_ids, dim=-1)
        pooled_output = ht.slice_by_matrix_op(last_hidden_state, index1, index2) 

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        else:
            assert False

class CLIPTextModel(object):
    def __init__(self, config, name='CLIPTextModel'):
        self.text_model = CLIPTextTransformer(config, name=name+'.text_model')

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def __call__(
        self,
        input_ids=None,
        input_shape=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return self.text_model(
            input_ids=input_ids,
            input_shape=input_shape,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CLIPVisionTransformer(object):
    def __init__(self, config, name='CLIPVisionTransformer'):
        self.config = config
        self.embed_dim = config.hidden_size
        self.embeddings = CLIPVisionEmbeddings(config, name=name+".embeddings")
        self.pre_layrnorm = ht.layers.LayerNorm(self.embed_dim, name=name+".pre_layrnorm")
        self.encoder = CLIPEncoder(config, name=name+".encoder")
        self.post_layernorm = ht.layers.LayerNorm(self.embed_dim, name=name+".post_layernorm")

    def __call__(
        self,
        pixel_values,
        input_shape,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values, input_shape)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_shape = (input_shape[0], self.embeddings.num_patches + 1, self.embed_dim)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            input_shape=encoder_shape,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = ht.slice_op(last_hidden_state, (0, 0, 0), (-1, 1, -1))
        pooled_output = ht.array_reshape_op(pooled_output, (input_shape[0], -1))
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        else:
            assert False

class CLIPVisionModel(object):
    def __init__(self, config, name='CLIPVisionModel'):
        self.config = config
        self.vision_model = CLIPVisionTransformer(config, name=name+'.vision_model')

    def get_input_embeddings(self):
        return self.vision_model.embeddings.patch_embedding

    def __call__(self, pixel_values, input_shape, output_attentions=None, output_hidden_states=None, return_dict=None):
        return self.vision_model(
            pixel_values=pixel_values,
            input_shape=input_shape,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CLIPModel(object):
    def __init__(self, config, name=''):
        self.config = config
        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config, name=name+'.text_model' if name else 'text_model')
        self.vision_model = CLIPVisionTransformer(vision_config, name=name+'.vision_model' if name else 'vision_model')

        self.visual_projection = ht.layers.Linear(self.vision_embed_dim, self.projection_dim, bias=False, weight_transpose=True, name=name+'.visual_projection' if name else 'visual_projection')
        self.text_projection = ht.layers.Linear(self.text_embed_dim, self.projection_dim, bias=False, weight_transpose=True, name=name+'.text_projection' if name else 'text_projection')

        self.logit_scale = ht.Variable(name=name+'.logit_scale' if name else 'logit_scale', value=np.array([self.config.logit_scale_init_value]))

    def get_text_features(self, input_ids=None, input_shape=None, attention_mask=None, position_ids=None, output_attentions=None,
        output_hidden_states=None, return_dict=None):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            input_shape=input_shape,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    def get_image_features(self, pixel_values, input_shape, output_attentions=None,
        output_hidden_states=None, return_dict=None):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            input_shape=input_shape,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  
        image_features = self.visual_projection(pooled_output)

        return image_features

    def __call__(self, input_ids, input_ids_shape, pixel_values, pixel_values_shape, attention_mask=None, position_ids=None,
        return_loss=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        text_batch_size = input_ids_shape[0]
        image_batch_size = pixel_values_shape[0]
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            input_shape=pixel_values_shape,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            input_shape=input_ids_shape,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        text_embeds_norm = ht.norm_op(text_embeds, axis=-1, p=2)
        image_embeds_norm = ht.norm_op(image_embeds, axis=-1, p=2)

        image_embeds_norm = ht.broadcastto_op(image_embeds_norm, image_embeds)
        text_embeds_norm = ht.broadcastto_op(text_embeds_norm, text_embeds)    

        image_embeds = image_embeds / image_embeds_norm
        text_embeds = text_embeds / text_embeds_norm
        
        logit_scale = ht.exp_op(self.logit_scale)
        logit_scale = ht.unsqueeze_op(logit_scale, 1)

        logits_per_text = ht.matmul_op(text_embeds, image_embeds, trans_B=True) 
        logits_per_text = logits_per_text * ht.broadcast_shape_op(logit_scale, (text_batch_size, image_batch_size))
        logits_per_image = ht.transpose_op(logits_per_text, (1,0))

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text, input_ids_shape[0])

        output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
        return ((loss,) + output) if loss is not None else output

