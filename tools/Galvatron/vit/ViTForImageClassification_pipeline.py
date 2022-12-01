import torch.nn as nn
import torch

class VitEmbedding_(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.embedding = vit_model.vit.embeddings

    def forward(self, pixel_value):
        embedding_output = self.embedding(pixel_value)
        return embedding_output

class VitEncoder_(nn.Module):
    def __init__(self, vit_model, layer_idx_start, layer_idx_end):
        super().__init__()
        self.layer = vit_model.vit.encoder.layer[layer_idx_start: layer_idx_end]
    
    def forward(self, hidden_states):
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states, None, None)
            hidden_states = layer_outputs[0]

        return hidden_states

class VitLayerNorm_(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.layernorm = vit_model.vit.layernorm

    def forward(self, input):
        output = self.layernorm(input)
        return output[:, 0, :]

class VitClassification_(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.classifier = vit_model.classifier

    def forward(self, input):
        output = self.classifier(input)
        return output