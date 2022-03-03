from audioop import bias
from typing import List
from .base import BaseLayer
import hetu as ht

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
        # x = ht.dropout2d_op(x, 1 - self.keep_prob)
        x = ht.matmul_op(x, ffn_weight2_var)
        if self.bias:
            bias_var2 = ht.init.zeros(
                shape=(1, self.out_features), name=self.name+'_bias_2')
            bias_var2 = ht.broadcastto_op(bias_var2, x)
            x = x + bias_var2
        # x = ht.array_reshape_op(x, [])      
        return x

class MoELayer(BaseLayer):
    def __init__(self, gate = None, experts = None, num_tokens = None, embed_dim = None, all2all_size = None, name='MoELayer'):
        self.name = name
        self.gate = gate
        self.experts = experts
        self.num_local_experts = len(experts)
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.all2all_size = all2all_size

    def __call__(self, x, dispatch_input):
        if self.name == 'MoELayer':
            reshaped_input = ht.array_reshape_op(x, [-1, self.embed_dim])
            l_aux, indices_s, location_s, gates, capacity =self.gate(reshaped_input)
            #dispatch_input = ht.Variable("dispatch_input")
            dispatch_input =  ht.dispatch_encode_op(reshaped_input, indices_s, location_s, gates, capacity)
            # if self.all2all_size > 1:
            dispatch_input = ht.alltoall_op(dispatch_input)
            dispatch_input = ht.array_reshape_op(dispatch_input, [self.all2all_size, self.num_local_experts,\
                                                -1, self.embed_dim])
        
            outputs = []
            for i in range(self.num_local_experts):
                token_i = ht.split_op(dispatch_input, axes=[1, ], indices = [i,], splits = [self.num_local_experts, ])
                output_i = self.experts[i](token_i)
                outputs.append(output_i)

            expert_output = ht.concatenate_op(outputs, axis=0)
        
            # if self.all2all_size > 1:
            expert_output = ht.alltoall_op(expert_output)
            expert_output = ht.array_reshape_op(expert_output, [-1, self.all2all_size * self.num_local_experts, capacity, self.embed_dim])
        
#        print("??????????????????")

            #expert_output = ht.reduce_sum_op(expert_output, axes=0)
            expert_output = ht.array_reshape_op(expert_output, [-1, self.num_tokens, self.embed_dim])
            combined_output = ht.dispatch_decode_op(expert_output, indices_s, location_s, gates, capacity)
            # combined_output = ht.Variable("combined_output")
            combined_output = expert_output
            return combined_output, l_aux

        elif self.name == 'BalanceAssignmentLayer':
            
            










