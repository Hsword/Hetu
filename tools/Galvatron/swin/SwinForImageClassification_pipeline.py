from turtle import forward
import torch.nn as nn
import torch

class SwinEmbeddings_(nn.Module):
    def __init__(self, swin_model):
        super().__init__()
        self.embeddings = swin_model.swin.embeddings
    
    def forward(self, pixel_value):
        outputs = self.embeddings(pixel_value)
        return outputs

class SwinBlock_(nn.Module):
    def __init__(self, swin_model, layer_idx, block_idx, has_downsample=False):
        super().__init__()
        layer = swin_model.swin.encoder.layers[layer_idx]

        self.block = layer.blocks[block_idx]
        self.downsamlpe = layer.downsample if has_downsample else None
    
    def forward(self, hidden_states):
        layer_outputs = self.block(hidden_states, None, False)
        hidden_states = layer_outputs[0]
        if self.downsamlpe is not None:
            hidden_states = self.downsamlpe(hidden_states)
        return hidden_states


class SwinLayernorm_(nn.Module):
    def __init__(self, swin_model):
        super().__init__()
        self.layernorm = swin_model.swin.layernorm
    
    def forward(self, hidden_states):
        sequence_output = self.layernorm(hidden_states)
        return sequence_output


class SwinCls_(nn.Module):
    def __init__(self, swin_model):
        super().__init__()
        self.pooler = swin_model.swin.pooler
        self.classifier = swin_model.classifier
    
    def forward(self, sequence_output):

        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)
        
        logits = self.classifier(pooled_output)
        return logits
