import torch.nn as nn
import torch
from galvatron.core.pipeline import PipeSequential
from galvatron.core import mixed_precision_dtype, ModelInfo

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm_parallel_residual
except ImportError:
    dropout_add_layer_norm_parallel_residual = None
    
try:
    from flash_attn.ops.rms_norm import RMSNorm, dropout_add_rms_norm
except ImportError:
    RMSNorm, dropout_add_rms_norm = None, None

try:
    from flash_attn.ops.rms_norm import dropout_add_rms_norm_parallel_residual
except ImportError:
    dropout_add_rms_norm_parallel_residual = None


class LlamaEmbeddings_(nn.Module):
    def __init__(self, model):
        super().__init__()
        model = model.transformer
        attrs = ['embeddings', 'process_group', 'sequence_parallel']
        for key in attrs:
            setattr(self, key, getattr(model, key))
    def forward(self, input_ids, position_ids=None):
        embedding_kwargs = ({'combine_batch_seqlen_dim': True}
                            if self.process_group is not None and self.sequence_parallel else {})
        hidden_states = self.embeddings(input_ids, position_ids=position_ids, **embedding_kwargs)
        # if self.parallel_block:
        #     hidden_states2 = None
        input_ids = input_ids.clone()
        return hidden_states, input_ids

class LlamaLayers_(nn.Module):
    def __init__(self, model, layer_idx_start, layer_idx_end):
        super().__init__()
        model = model.transformer
        self.layers = model.layers[layer_idx_start:layer_idx_end]
        attrs = ['prenorm', 'parallel_block', 'process_group']
        for key in attrs:
            setattr(self, key, getattr(model, key))
        
    def forward(self, hidden_states, input_ids):
        residual = None
        mixer_kwargs = ({'seqlen': hidden_states.shape[1]}
                        if self.process_group is not None and self.sequence_parallel else {})
        # if inference_params is not None:
        #     mixer_kwargs['inference_params'] = inference_params

        for layer in self.layers:
            if self.prenorm:
                if not self.parallel_block:
                    hidden_states, residual = layer(hidden_states, residual,
                                                    mixer_kwargs=mixer_kwargs)
                # else:
                #     hidden_states, hidden_states2, residual = layer(
                #         hidden_states, hidden_states2, residual, mixer_kwargs=mixer_kwargs
                #     )
            else:
                hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
        input_ids = input_ids.clone()
        return hidden_states, input_ids

class LlamaPreNorm_(nn.Module):
    def __init__(self, model):
        super().__init__()
        model = model.transformer
        attrs = ['fused_dropout_add_ln', 'drop_f', 'parallel_block', 'ln_f', 'prenorm']
        for key in attrs:
            setattr(self, key, getattr(model, key))

    def forward(self, hidden_states, input_ids):
        residual = None
        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.drop_f(hidden_states)
                if not self.parallel_block:
                    residual = (dropped + residual) if residual is not None else dropped
                # else:
                #     dropped2 = self.drop_f(hidden_states2)
                #     residual = ((residual + dropped + dropped2)
                #                 if residual is not None else dropped + dropped2)
                hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                if not self.parallel_block:
                    fused_add_norm_fn = (dropout_add_rms_norm if isinstance(self.ln_f, RMSNorm)
                                         else dropout_add_layer_norm)
                    hidden_states = fused_add_norm_fn(
                        hidden_states, residual, self.ln_f.weight, self.ln_f.bias,
                        self.drop_f.p if self.training else 0.0, self.ln_f.eps, prenorm=False,
                        residual_in_fp32=self.residual_in_fp32
                    )
                # else:
                #     fused_add_norm_fn = (dropout_add_rms_norm_parallel_residual
                #                          if isinstance(self.ln_f, RMSNorm)
                #                          else dropout_add_layer_norm_parallel_residual)
                #     hidden_states, _ = fused_add_norm_fn(
                #         hidden_states, hidden_states2, residual, self.ln_f.weight, self.ln_f.bias,
                #         None, None, self.drop_f.p if self.training else 0.0, self.ln_f.eps,
                #         prenorm=False, residual_in_fp32=self.residual_in_fp32
                #     )
        input_ids = input_ids.clone()
        return hidden_states, input_ids

class LlamaCls_(nn.Module):
    def __init__(self, model):
        super().__init__()
        attrs = ['lm_head', 'config', 'project_out']
        for key in attrs:
            setattr(self, key, getattr(model, key))

    def forward(self, hidden_states, input_ids):
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        # # During inference, we want the full logit for sampling
        # if isinstance(self.lm_head, ColumnParallelLinear) and inference_params is not None:
        #     lm_logits, _ = all_gather_raw(lm_logits, self.lm_head.process_group)
        #     lm_logits = rearrange(lm_logits, '(n b) s d -> b s (n d)', b=hidden_states.shape[0])
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        from flash_attn.losses.cross_entropy import CrossEntropyLoss
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1).long())
        return loss

def construct_sequential_model(model, config):
    model_ = PipeSequential()
    model_.add_module('embeddings', LlamaEmbeddings_(model))
    for i in range(config.num_hidden_layers):
        enc = LlamaLayers_(model, i, i + 1)
        model_.add_module('layer_%d'%i, enc)
    model_.add_module('prenorm', LlamaPreNorm_(model))
    model_.add_module('cls', LlamaCls_(model))
    return model_

class LlamaModelInfo(ModelInfo):
    def __init__(self, config, args):
        super(LlamaModelInfo, self).__init__()
        layernum_list = [config.num_hidden_layers]
        seq_len, hidden_size = config.max_position_embeddings, config.hidden_size
        mixed_precision = mixed_precision_dtype(args.mixed_precision)
        layer_shapes_list = [[[-1,seq_len,hidden_size], [-1,seq_len]]]
        layer_dtypes_list = [[mixed_precision, torch.long]]
        module_types = ['embed'] + ['gpt_dec']*config.num_hidden_layers + ['norm', 'cls']
        self.set_layernums(layernum_list)
        self.set_shapes(layer_shapes_list)
        self.set_dtypes(layer_dtypes_list)
        self.set_module_types(module_types)