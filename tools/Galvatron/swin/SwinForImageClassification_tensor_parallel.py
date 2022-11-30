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

class SwinAttention_tp(nn.Module):
    def __init__(self, config, dim, num_attention_head, tp_group=None):
        super().__init__()
        args=get_args()
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.attention = ParallelAttention(init_method, 
            scaled_init_method, 
            attention_type=AttnType.self_attn,
            attn_mask_type=AttnMaskType.padding, 
            num_attention_heads=num_attention_head,
            hidden_size=dim,
            tp_group=self.tp_group)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def forward(self, hidden_states, attention_mask, batch_size=None):
        bsz, seq_len, hs = hidden_states.shape
        hidden_states = hidden_states.permute(1, 0, 2)

        if attention_mask is None:
            attention_mask = torch.zeros((bsz, 1, seq_len, seq_len), dtype=torch.bool, device=hidden_states.device)
        else:
            attention_mask = attention_mask.repeat(batch_size, 1, 1, 1).to(hidden_states.device)

        hidden_states, bias = self.attention(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states + bias)

        hidden_states = hidden_states.permute(1, 0, 2)
        
        return hidden_states

class SwinMlp_tp(nn.Module):
    def __init__(self, config, dim, tp_group=None):
        super().__init__()
        args=get_args()
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.mlp = ParallelMLP(init_method, scaled_init_method, hidden_size=dim, tp_group=self.tp_group)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states, bias = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states+bias)
        hidden_states = hidden_states.permute(1, 0, 2)
        return hidden_states


class SwinBlock_tp(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0, tp_group=None):
        super().__init__()
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)

        self.attention = SwinAttention_tp(config, dim, num_heads, tp_group=tp_group)
        
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.intermediate = SwinMlp_tp(config, dim, tp_group=tp_group)
        
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            height, width = self.input_resolution
            img_mask = torch.zeros((1, height, width, 1))
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # win_num, ws^2
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # win_num, ws^2, ws^2
            attn_mask = attn_mask.unsqueeze(1).type(torch.bool) # win_num, 1, ws^2, ws^2
        else:
            attn_mask = None
        
        self.attention_mask = attn_mask
    
    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        height, width = self.input_resolution
        # print(hidden_states.size())
        batch_size, dim, channels = hidden_states.size()
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, channels)

        # cyclic shift
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)

        attention_output = self.attention(
            hidden_states_windows,
            self.attention_mask,
            batch_size
        )

        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        shifted_windows = window_reverse(attention_windows, self.window_size, height, width)

        # reverse cyclic shift
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        attention_windows = attention_windows.view(batch_size, height * width, channels)

        hidden_states = shortcut + attention_windows

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + layer_output
        # print('output:', layer_output.size())
        return (layer_output,)

def build_swinblock_list(config, dim, input_resolution, depth, num_heads, gen=None):
    return nn.ModuleList(
        [
            SwinBlock_tp(
                config=config,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                tp_group=gen.__next__() if gen is not None else None
            )
            for i in range(depth)
        ]
    )
        
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows

def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    batch_size = int(windows.shape[0] / (height * width / window_size / window_size))
    windows = windows.view(batch_size, height // window_size, width // window_size, window_size, window_size, -1)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, -1)
    return windows