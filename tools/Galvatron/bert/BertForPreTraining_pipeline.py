import torch.nn as nn
import torch

def get_extended_attention_mask(attention_mask):
    # the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

# Use BertEmbeddings defined in bert_model
class BertEmbeddings_(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.embeddings = bert_model.bert.embeddings
    def forward(self, input_ids, token_type_ids, attention_mask):
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        return embedding_output, attention_mask

# Use BertEncoder defined in bert_model
class BertEncoder_(nn.Module):
    def __init__(self, bert_model, layer_idx_start, layer_idx_end):
        super().__init__()
        self.layer = bert_model.bert.encoder.layer[layer_idx_start:layer_idx_end]
        self.get_extended_attention_mask = get_extended_attention_mask
    def forward(self, hidden_states, attention_mask):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
            )
            hidden_states = layer_outputs[0]
        return hidden_states, attention_mask

# Use BertPooler defined in bert_model
class BertPooler_(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.pooler = bert_model.bert.pooler
    def forward(self, hidden_states, attention_mask):
        return hidden_states, self.pooler(hidden_states)

# Use BertPreTrainingHeads defined in bert_model
class BertPreTrainingHeads_(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.cls = bert_model.cls
    def forward(self, sequence_output, pooled_output):
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        return prediction_scores, seq_relationship_score