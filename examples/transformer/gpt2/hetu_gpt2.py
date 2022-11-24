import hetu as ht
import numpy as np
from config import GPT2Config

class Conv1D(object):
    def __init__(self, nf, nx, name='Conv1D'):
        self.nx = nx
        self.nf = nf

        self.weight = ht.init.random_normal(shape=(nx, nf), stddev=0.2, name=name+'.weight')
        self.bias = ht.init.zeros(shape=(nf,), name=name+'.bias')

    def __call__(self, x, input_shape):
        size_out = input_shape[:-1] + (self.nf,)
        assert(input_shape[-1] == self.nx)
        x = ht.array_reshape_op(x, (-1, self.nx))
        x = ht.linear_op(x, self.weight, self.bias)
        x = ht.array_reshape_op(x, size_out)
        return x

class GPT2Attention(object):
    def __init__(self, config, name='GPT2Attention', is_cross_attention=False, layer_idx=None):
        max_positions = config.max_position_embeddings

        self.bias = np.tril(np.ones((1, 1, max_positions, max_positions)))
        self.masked_bias = np.array([-1e4])

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim, name=name+'.c_attn')
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim, name=name+'.q_attn')
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim, name=name+'.c_attn')
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim, name=name+'.c_proj')

        self.attn_dropout = ht.layers.DropOut(config.attn_pdrop)
        self.resid_dropout = ht.layers.DropOut(config.resid_pdrop)

        self.pruned_heads = set()

    def _attn(self, query, key, value, q_size, k_size, v_size, attention_mask=None, head_mask=None):
        k = ht.transpose_op(key, (0, 1, 3, 2))
        attn_weights = ht.batch_matmul_op(query, k)
        if self.scale_attn_weights:
            attn_weights = ht.mul_byconst_op(attn_weights, 1/(v_size[-1] ** 0.5))
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = ht.mul_byconst_op(attn_weights, 1/(float(self.layer_idx + 1)))
        if not self.is_cross_attention:
            query_length, key_length = q_size[-2], k_size[-2]
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = np.finfo(np.float32).min
            causal_mask = ht.Variable('causal_mask', value=causal_mask, trainable=False)
            causal_mask = ht.broadcastto_op(causal_mask, attn_weights)
            attn_weights = ht.where_const_op(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask


        attn_weights = ht.softmax_op(attn_weights)        
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = ht.batch_matmul_op(attn_weights, value)
        attn_shape = q_size[:-1] + (v_size[-1],)
        return attn_output, attn_weights, attn_shape

    def _upcast_and_reordered_attn(self, query, key, value, q_size, k_size, v_size, attention_mask=None, head_mask=None):

        bsz, num_heads, q_seq_len, dk = q_size
        _, _, k_seq_len, _ = k_size

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        q = ht.array_reshape_op(query, (-1, q_seq_len, dk))
        k = ht.transpose_op(key, (0, 1, 3, 2))
        k = ht.array_reshape_op(k, (-1, dk, k_seq_len))
        attn_weights = ht.batch_matmul_op(q, k) * scale_factor
        attn_weights = ht.array_reshape_op(attn_weights, (bsz, num_heads, q_seq_len, k_seq_len))

        if not self.is_cross_attention:
            query_length, key_length = q_seq_len, k_seq_len
            
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = np.finfo(np.float32).min
            mask_value = ht.array([mask_value], ctx=query.raw_ctx)
            causal_mask = ht.array(causal_mask, ctx=query.raw_ctx)
            attn_weights = ht.where_op(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = ht.softmax_op(attn_weights)        
        attn_weights = ht.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = ht.batch_matmul_op(attn_weights, value)
        attn_shape = (bsz, num_heads, q_seq_len, v_size[-1])
        return attn_output, attn_weights, attn_shape

    def _split_heads(self, x, num_heads, attn_head_size, shape):
        new_shape = shape[:-1] + (num_heads, attn_head_size)
        x = ht.array_reshape_op(x, new_shape)
        x = ht.transpose_op(x, (0, 2, 1, 3))
        return x, (new_shape[0], new_shape[2], new_shape[1], new_shape[3]) 

    def _merge_heads(self, x, num_heads, attn_head_size, shape):
        x = ht.transpose_op(x, (0, 2, 1, 3))
        new_shape = (shape[0], shape[2]) + (num_heads * attn_head_size,)
        x = ht.array_reshape_op(x, new_shape)
        return x, new_shape

    def __call__(self, hidden_states, layer_past, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, hidden_shape=None, encoder_hidden_shape=None, shape_past=None, use_cache=False, output_attentions=False):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states, hidden_shape)#
            kv = self.c_attn(encoder_hidden_states, encoder_hidden_shape)#
            key= ht.split_op(kv, axes=[2], indices=[0], splits=[2])
            value = ht.split_op(kv, axes=[2], indices=[1], splits=[2])
            attention_mask = encoder_attention_mask
            q_size = hidden_shape[:-1] + (self.embed_dim,)
            k_size = v_size = encoder_hidden_shape[:-1] + (self.embed_dim,)
        else:
            qkv = self.c_attn(hidden_states, hidden_shape)
            query= ht.split_op(qkv, axes=[2], indices=[0], splits=[3])            
            key= ht.split_op(qkv, axes=[2], indices=[1], splits=[3])
            value = ht.split_op(qkv, axes=[2], indices=[2], splits=[3])         
            q_size = k_size = v_size = hidden_shape[:-1] + (self.embed_dim,)
        

        query, q_size = self._split_heads(query, self.num_heads, self.head_dim, q_size)
        key, k_size = self._split_heads(key, self.num_heads, self.head_dim, k_size)
        value, v_size = self._split_heads(value, self.num_heads, self.head_dim, v_size) 
                    
        if layer_past is not None:
            past_key, past_value = layer_past
            key = ht.concat_op(past_key, key, axis=-2)
            value = ht.concat_op(past_value, value, axis=-2)
            k_size[-2] = k_size[-2] + shape_past[0][-2]
            v_size[-2] = v_size[-2] + shape_past[1][-2]

        if use_cache is True:
            present = (key, value)
        else:
            present = None
        if self.reorder_and_upcast_attn:
            attn_output, attn_weights, attn_shape = self._upcast_and_reordered_attn(query, key, value, q_size, k_size, attention_mask, head_mask)
        else:       
            attn_output, attn_weights, attn_shape = self._attn(query, key, value, q_size, k_size, v_size, attention_mask, head_mask)
        attn_output, attn_output_shape = self._merge_heads(attn_output, self.num_heads, self.head_dim, attn_shape)
        attn_output = self.c_proj(attn_output, attn_output_shape)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs, attn_output_shape 


class GPT2MLP(object):
    def __init__(self, intermediate_size, config, name='GPT2MLP'):
        embed_dim = config.hidden_size
        self.intermediate_size = intermediate_size
        self.c_fc = Conv1D(intermediate_size, embed_dim, name=name+'.c_fc')
        self.c_proj = Conv1D(embed_dim, intermediate_size, name=name+'.c_proj')
        if config.activation_function == "relu":
            self.act = ht.relu_op
        elif config.activation_function == "gelu":
            self.act = ht.gelu_op
        self.dropout = ht.layers.DropOut(config.resid_pdrop)

    def __call__(self, hidden_states, shape):
        hidden_states = self.c_fc(hidden_states, shape)
        hidden_states = self.act(hidden_states)
        shape = shape[:-1] + (self.intermediate_size,)
        hidden_states = self.c_proj(hidden_states, shape)     
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(object):
    def __init__(self, config, layer_idx=None, name='GPT2Block'):
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = ht.layers.LayerNorm(hidden_size, eps=config.layer_norm_epsilon, name=name+'.ln_1')
        self.attn = GPT2Attention(config, layer_idx=layer_idx, name=name+'.attn')
        self.ln_2 = ht.layers.LayerNorm(hidden_size, eps=config.layer_norm_epsilon, name=name+'.ln_2')       

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx, name=name+'.crossattention')
            self.ln_cross_attn = ht.layers.LayerNorm(hidden_size, eps=config.layer_norm_epsilon, name=name+'.ln_cross_attn')

        self.mlp = GPT2MLP(inner_dim, config, name=name+'.mlp')

    def __call__(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, hidden_shape=None, encoder_hidden_shape=None, shape_past=None, use_cache=False, output_attentions=False):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_outputs, attn_output_shape = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            hidden_shape=hidden_shape,
            shape_past=shape_past,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )                      
        attn_output = attn_outputs[0] 
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual
        hidden_states_shape = attn_output_shape
        if encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs, cross_attn_outputs_shape = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                hidden_shape=attn_output_shape, 
                encoder_hidden_shape=encoder_hidden_shape,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]

            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:] 
            hidden_states_shape = cross_attn_outputs_shape

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states, hidden_states_shape)
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs 


class GPT2Model(object):
    def __init__(self, config):
        self.config = config
        self.embed_dim = config.hidden_size
        self.wte = ht.layers.Embedding(config.vocab_size, self.embed_dim, name='wte')
        self.wpe = ht.layers.Embedding(config.max_position_embeddings, self.embed_dim, name='wpe')
        self.drop = ht.layers.DropOut(config.embd_pdrop)
        self.h = [GPT2Block(config, layer_idx=i, name='h.'+str(i)) for i in range(config.num_hidden_layers)]
        self.ln_f = ht.layers.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon, name='ln_f')
        
    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            assert False
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask
        
    def __call__(self, input_ids=None, input_shape=None, past_key_values=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = ht.array_reshape_op(input_ids, [-1, input_shape[-1]])
            batch_size = input_shape[0]
            sequence_length = input_shape[-1]
        elif inputs_embeds is not None:
            batch_size = input_shape[0]
            sequence_length = input_shape[-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = ht.array_reshape_op(token_type_ids, [-1, input_shape[-1]])  
        if position_ids is not None:
            position_ids = ht.array_reshape_op(position_ids, [-1, input_shape[-1]])          

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            assert False
        if position_ids is None:
            position_ids = ht.arange_op(past_length, input_shape[-1] + past_length) 
            position_ids = ht.array_reshape_op(position_ids, [-1, input_shape[-1]])


        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = ht.array_reshape_op(attention_mask, [batch_size, 1, 1, -1])
            attention_mask = (1.0 - attention_mask) * np.finfo(np.float32).min

        if self.config.add_cross_attention and encoder_hidden_states is not None:
            if encoder_attention_mask is None:
                encoder_attention_mask = ht.init.ones(shape=(batch_size, sequence_length), ctx=encoder_hidden_states.raw_ctx)
            encoder_attention_mask = ht.array_reshape_op(encoder_attention_mask, [batch_size, 1, 1, -1])
            encoder_attention_mask = (1.0 - encoder_attention_mask) * np.finfo(np.float32).min
        else:
            encoder_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (self.embed_dim,)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    hidden_shape=(batch_size, sequence_length, self.embed_dim),
                    output_attentions=output_attentions)

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = ht.array_reshape_op(hidden_states, output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return [hidden_states]
 

