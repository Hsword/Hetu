import torch
from torch import nn
from torch import Tensor, device
from typing import Tuple
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../site-package')
from megatron.model.utils import init_method_normal, scaled_init_method_normal
from megatron import get_args
from megatron_layers import ParallelMLP, ParallelAttention
from megatron.model.enums import AttnMaskType, AttnType

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

class T5LayerFF_tp(nn.Module):
    def __init__(self, config, tp_group = None):
        super().__init__()
        args=get_args()
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.tp_group = tp_group.group if tp_group is not None else None
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = ParallelMLP(init_method, scaled_init_method, act_func = 'relu', bias = False, dropout_prob = config.dropout_rate, tp_group=self.tp_group)
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = ParallelMLP(init_method, scaled_init_method, act_func = 'gelu', bias = False, dropout_prob = config.dropout_rate, tp_group=self.tp_group)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states, _ = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention_tp(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, tp_group = None):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        args=get_args()
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.tp_group = tp_group.group if tp_group is not None else None
        if self.is_decoder:
            self.attention = ParallelAttention(init_method, 
                                            scaled_init_method, 
                                            attention_type=AttnType.cross_attn,
                                            attn_mask_type=AttnMaskType.causal,
                                            bias = False,
                                            tp_group=self.tp_group)
        else:
            self.attention = ParallelAttention(init_method, 
                                            scaled_init_method, 
                                            attention_type=AttnType.self_attn,
                                            attn_mask_type=AttnMaskType.padding,
                                            bias = False,
                                            tp_group=self.tp_group)

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        if self.is_decoder:
            assert(key_value_states is not None)
            attention_output, _ = self.attention(
                hidden_states,
                mask,
                encoder_output = key_value_states,
            )
        else:
            attention_output, _ = self.attention(
                hidden_states,
                mask,
            )
        outputs = (attention_output,None,None)
        return outputs


class T5Block_tp(nn.Module):
    def __init__(self, t5_block):
        super().__init__()
        self.t5_block = t5_block

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        hidden_states = hidden_states.permute(1,0,2)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.permute(1,0,2)
        layer_outputs = self.t5_block(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            layer_head_mask=layer_head_mask,
            cross_attn_layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        outputs = (layer_outputs[0].permute(1,0,2),) + layer_outputs[1:]
        return outputs

def get_extended_attention_mask_encoder(attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = 1.0 - extended_attention_mask
    extended_attention_mask = extended_attention_mask.to(dtype=torch.bool)
    return extended_attention_mask

def get_extended_attention_mask_decoder(attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
    batch_size, seq_length = input_shape
    seq_ids = torch.arange(seq_length, device=device)
    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    # in case past_key_values are used we need to add a prefix ones mask to the causal mask
    # causal and attention masks must have same type with pytorch version < 1.3
    causal_mask = causal_mask.to(attention_mask.dtype)

    if causal_mask.shape[1] < attention_mask.shape[1]:
        prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
        causal_mask = torch.cat(
            [
                torch.ones(
                    (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                ),
                causal_mask,
            ],
            axis=-1,
        )

    extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    extended_attention_mask = 1.0 - extended_attention_mask
    extended_attention_mask = extended_attention_mask.to(dtype=torch.bool)
    return extended_attention_mask

def invert_attention_mask(encoder_attention_mask: Tensor) -> Tensor:
    encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    encoder_extended_attention_mask = 1.0 - encoder_extended_attention_mask
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=torch.bool)
    return encoder_extended_attention_mask