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

class BertAttention_tp(nn.Module):
    def __init__(self, config, tp_group = None):
        super().__init__()
        args=get_args()
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.attention = ParallelAttention(init_method, 
                                        scaled_init_method, 
                                        attention_type=AttnType.self_attn,
                                        attn_mask_type=AttnMaskType.padding,
                                        tp_group = self.tp_group)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        input_tensor = hidden_states
        hidden_states, bias = self.attention(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states+bias)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertMLP_tp(nn.Module):
    def __init__(self, config, tp_group = None):
        super().__init__()
        args=get_args()
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.mlp = ParallelMLP(init_method, scaled_init_method, tp_group = self.tp_group)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        input_tensor = hidden_states
        hidden_states, bias = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states+bias)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer_tp(nn.Module):
    def __init__(self, config, tp_group = None):
        super().__init__()
        self.attention = BertAttention_tp(config, tp_group)
        self.mlp = BertMLP_tp(config, tp_group)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        hidden_states = hidden_states.permute(1, 0, 2)
        attention_output = self.attention(
            hidden_states,
            attention_mask,
        )
        layer_output = self.mlp(attention_output)
        layer_output = layer_output.permute(1, 0, 2)
        outputs = (layer_output,)
        return outputs


def get_extended_attention_mask(attention_mask, input_shape=None, device=None):
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = 1.0 - extended_attention_mask
    extended_attention_mask = extended_attention_mask.to(dtype=torch.bool)
    return extended_attention_mask
