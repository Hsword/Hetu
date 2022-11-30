import torch.nn as nn
import torch

class T5Embeddings_(nn.Module):
    def __init__(self, t5_model):
        super().__init__()
        self.embeddings = t5_model.shared
        self.dropout = t5_model.encoder.dropout

    def forward(self, input_ids, label, attention_mask):
        inputs_embeds = self.embeddings(input_ids)
        hidden_states = self.dropout(inputs_embeds)
        return hidden_states, label, attention_mask

class T5DecoderEmbedding_(nn.Module):
    def __init__(self, t5_model):
        super().__init__()
        self.embeddings = t5_model.shared
        self._shift_right = t5_model._shift_right
        self.dropout = t5_model.decoder.dropout

    def forward(self,  encoder_hidden_states, label, encoder_attention_mask):
        decoder_input_ids = self._shift_right(label)
        input_shape = decoder_input_ids.size()

        inputs_embeds = self.embeddings(decoder_input_ids)
        hidden_states = self.dropout(inputs_embeds)
        
        attention_mask = torch.ones(*input_shape).to(inputs_embeds.device)
        return encoder_hidden_states, encoder_attention_mask, hidden_states, attention_mask

class T5Encoder_(nn.Module):
    def __init__(self, t5_model, layer_idx, has_final_layernorm=False):
        super().__init__()
        self.dropout = t5_model.encoder.dropout
        self.block = t5_model.encoder.block[layer_idx]
        self.get_extended_attention_mask = t5_model.encoder.get_extended_attention_mask
        self.final_layernorm = t5_model.encoder.final_layer_norm if has_final_layernorm else None
    
    def forward(self, hidden_states, label, attention_mask, position_bias=None):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, attention_mask.shape, attention_mask.device)

        layer_outputs = self.block(
            hidden_states,
            attention_mask=extended_attention_mask,
            position_bias=position_bias
        )
        hidden_states = layer_outputs[0]
        
        if not self.final_layernorm is None:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = self.dropout(hidden_states)
            return hidden_states, label, attention_mask
        
        return hidden_states, label, attention_mask

class T5Decoder_(nn.Module):
    def __init__(self, t5_model, layer_idx, has_final_layernorm=False):
        super().__init__()
        self.dropout = t5_model.decoder.dropout
        self.block = t5_model.decoder.block[layer_idx]
        self.get_extended_attention_mask = t5_model.decoder.get_extended_attention_mask
        self.invert_attention_mask = t5_model.decoder.invert_attention_mask
        self.final_layernorm = t5_model.decoder.final_layer_norm if has_final_layernorm else None
    
    def forward(self, encoder_hidden_states, encoder_attention_mask, hidden_states, attention_mask, position_bias=None, encoder_decoder_position_bias=None):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, attention_mask.shape, attention_mask.device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        
        layer_outputs = self.block(
                hidden_states,
                attention_mask = extended_attention_mask,
                position_bias = position_bias,
                encoder_hidden_states = encoder_hidden_states,
                encoder_attention_mask = encoder_extended_attention_mask, 
                encoder_decoder_position_bias = encoder_decoder_position_bias
        )
        hidden_states = layer_outputs[0]
        
        if not self.final_layernorm is None:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = self.dropout(hidden_states)
            return hidden_states

        return encoder_hidden_states, encoder_attention_mask, hidden_states, attention_mask

class T5Cls_(nn.Module):
    def __init__(self, t5_model):
        super().__init__()
        self.lm_head = t5_model.lm_head
        self.tie_word_embeddings = t5_model.config.tie_word_embeddings
        self.model_dim = t5_model.model_dim
    
    def forward(self, sequence_output):
        if self.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)
        return lm_logits
