import torch
from torch import nn
from torch import Tensor, device
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../site-package')
from megatron.model.utils import init_method_normal, scaled_init_method_normal
from megatron import get_args
from megatron_layers import ParallelMLP, ParallelAttention
from megatron.model.enums import AttnMaskType, AttnType
from transformers.modeling_outputs import BaseModelOutput

class VitAttention_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None):
        super().__init__()
        args=get_args()
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.LayerNormBefore = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.attention = ParallelAttention(init_method, 
                                        scaled_init_method, 
                                        layer_number, 
                                        attention_type=AttnType.self_attn,
                                        attn_mask_type=AttnMaskType.padding, 
                                        tp_group=self.tp_group)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.LayerNormAfter = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states, attention_mask):
        input_tensor = hidden_states
        hidden_states = self.LayerNormBefore(hidden_states)
        
        hidden_states, bias = self.attention(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states + bias)
        
        before_layernorm = hidden_states + input_tensor
        hidden_states = self.LayerNormAfter(before_layernorm)
        return hidden_states, before_layernorm

class VitMLP_tp(nn.Module):
    def __init__(self, config, tp_group=None) -> None:
        super().__init__()
        args=get_args()
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.mlp = ParallelMLP(init_method, scaled_init_method, tp_group=self.tp_group)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs):
        hidden_states, input_tensor = inputs

        hidden_states, bias = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states+bias)
        return hidden_states + input_tensor

class VitLayer_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None):
        super().__init__()
        self.attention = VitAttention_tp(config, layer_number, tp_group)
        self.mlp = VitMLP_tp(config, tp_group)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None
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

class VitEncoder_tp(nn.Module):
    def __init__(self, config, device=None, tp_groups=None):
        super().__init__()
        self.config = config
        if tp_groups is None:
            self.layer = nn.ModuleList([VitLayer_tp(config, i) for i in range(config.num_hidden_layers)])
        else:
            self.layer = nn.ModuleList([VitLayer_tp(config, i, tp_groups[i]) for i in range(config.num_hidden_layers)])
        self.mask_device = device

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        if attention_mask is None:
            dev = self.mask_device if self.mask_device is not None else hidden_states.device
            attention_mask = torch.zeros(hidden_states.shape[0], 1, 1, hidden_states.shape[1], dtype=torch.bool, device=dev)
        
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
            )
            hidden_states = layer_outputs[0]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None,
        )