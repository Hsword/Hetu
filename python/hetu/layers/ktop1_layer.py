from audioop import bias
from typing import List
from .base import BaseLayer
import hetu as ht
import numpy as np

class Expert(BaseLayer):
    def __init__(self, embed_dim, ffn_dim, dropout_rate, initializer=ht.init.GenXavierUniform(),
                 bias=False, activation = None, name="expert"):
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.keep_prob = 1 - dropout_rate
        self.bias = bias
        if isinstance(activation, str):
            if activation == 'relu':
                activation = ht.relu_op
            else:
                raise NotImplementedError
        self.activation = activation
        self.initializer = initializer
        self.name = name

    def __call__(self, x):
        ffn_weight1_var = self.initializer(
            shape=(self.embed_dim, self.ffn_dim), name=self.name+'_weight_1')
        ffn_weight2_var = self.initializer(
            shape=(self.ffn_dim, self.embed_dim), name=self.name+'_weight_2')
        x = ht.array_reshape_op(x, [-1, self.embed_dim])
        x = ht.matmul_op(x, ffn_weight1_var)
        if self.bias:
            bias_var1 = ht.init.zeros(
                shape=(1, self.out_features), name=self.name+'_bias_1')
            bias_var1 = ht.broadcastto_op(bias_var1, x)
            x = x + bias_var1
        if self.activation is not None:
            x = self.activation(x)
        x = ht.matmul_op(x, ffn_weight2_var)
        if self.bias:
            bias_var2 = ht.init.zeros(
                shape=(1, self.out_features), name=self.name+'_bias_2')
            bias_var2 = ht.broadcastto_op(bias_var2, x)
            x = x + bias_var2
        return x

class KTop1Layer(BaseLayer):
    def __init__(self, gate = None, experts = None, num_tokens = None, embed_dim = None, all2all_size = None, name='KTop1Layer',  k=None):
        self.name = name
        self.gate = gate
        self.experts = experts
        self.num_local_experts = len(experts)
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.all2all_size = all2all_size
        self.k = k

    def __call__(self, x):
        reshaped_input = ht.array_reshape_op(x, [-1, self.embed_dim]) 
        l_aux, indices_s, location_s,gates_s, capacity =self.gate(reshaped_input)
                       
        dispatch_input =  ht.layout_transform_op(reshaped_input, indices_s, location_s, capacity, self.num_local_experts*self.all2all_size)
        
                
        dispatch_input = ht.alltoall_op(dispatch_input)
        dispatch_input = ht.array_reshape_op(dispatch_input, [self.all2all_size, self.num_local_experts, -1, self.embed_dim])
            
        outputs = []
        for i in range(self.num_local_experts):
            token_i = ht.split_op(dispatch_input, axes=[1, ], indices = [i,], splits = [self.num_local_experts, ])
            output_i = self.experts[i](token_i)
            outputs.append(output_i)

        expert_output = ht.concatenate_op(outputs, axis=0)
        expert_output = ht.alltoall_op(expert_output) 
        expert_output = ht.array_reshape_op(expert_output, [-1, self.embed_dim])
        expert_output = ht.reverse_layout_transform_op(expert_output, indices_s, location_s, gates_s, capacity, self.num_local_experts * self.all2all_size)   
        return expert_output, l_aux
