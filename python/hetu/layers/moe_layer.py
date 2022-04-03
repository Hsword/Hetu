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

class MoELayer(BaseLayer):
    def __init__(self, gate = None, experts = None, num_tokens = None, embed_dim = None, all2all_size = None, name='MoELayer', device_id=None, top=None):
        self.name = name
        self.gate = gate
        self.experts = experts
        self.num_local_experts = len(experts)
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.all2all_size = all2all_size
        self.device_id=device_id
        self.top = top
        if self.name ==  'BalanceAssignmentLayer':
            arange_array_val = np.arange(self.num_tokens).astype(np.float32)
            self.arange_array = ht.Variable(value=arange_array_val, name='arange_array',trainable=False)

    def __call__(self, x):
        if self.name == 'MoELayer':
            reshaped_input = ht.array_reshape_op(x, [-1, self.embed_dim]) 
            l_aux, indices_s, location_s,gates_s, capacity =self.gate(reshaped_input)
                       
            if self.top==1:
                dispatch_input =  ht.layout_transform_op(reshaped_input, indices_s, location_s, capacity, self.num_local_experts*self.all2all_size)
            elif self.top==2:
                dispatch_input =  ht.layout_transform_op(reshaped_input, indices_s, location_s, capacity, self.num_local_experts*self.all2all_size)
            else:
                raise NotImplementedError
        
                
            dispatch_input = ht.alltoall_op(dispatch_input)
#            dispatch_input = ht.halltoall_op(dispatch_input, 2, 8)
            dispatch_input = ht.array_reshape_op(dispatch_input, [self.all2all_size, self.num_local_experts, -1, self.embed_dim])
            
            outputs = []
            for i in range(self.num_local_experts):
                token_i = ht.split_op(dispatch_input, axes=[1, ], indices = [i,], splits = [self.num_local_experts, ])
                output_i = self.experts[i](token_i)
                outputs.append(output_i)

            expert_output = ht.concatenate_op(outputs, axis=0)
            expert_output = ht.alltoall_op(expert_output) 
#            expert_output = ht.halltoall_op(expert_output, 2, 8)
            expert_output = ht.array_reshape_op(expert_output, [-1, self.embed_dim])
            expert_output = ht.reverse_layout_transform_op(expert_output, indices_s, location_s, gates_s, capacity, self.num_local_experts * self.all2all_size)   
            return expert_output, l_aux

        elif self.name == 'BalanceAssignmentLayer':
            import numpy as np
            arange_array_val = np.arange(self.num_tokens).astype(np.float32)
            arange_array = ht.Variable(value=arange_array_val, name='arange_array',trainable=False)

            reshaped_input = ht.array_reshape_op(x, [-1, self.embed_dim]) 
            indice, centroid = self.gate(reshaped_input)
            reverse_indice = ht.scatter1d_op(self.arange_array, indice)
            randperm_val = np.random.permutation(100).astype(np.float32)
            randperm = ht.Variable(value=randperm_val, name='randperm',trainable=False)
            reverse_randperm = ht.scatter1d_op(arange_array, randperm)
            
            shuffled_input = ht.indexing_op(reshaped_input, randperm)
            shuffled_input = ht.alltoall_op(shuffled_input)
            routed_input = ht.indexing_op(shuffled_input, indice)
            routed_input = ht.alltoall_op(routed_input)
            reshaped_routed_input = ht.array_reshape_op(routed_input, [self.all2all_size, self.num_local_experts,\
                                                                    -1, self.embed_dim])
            outputs = []
            for i in range(self.num_local_experts):
                token_i = ht.split_op(reshaped_routed_input, axes=[1, ], indices = [i,], splits = [self.num_local_experts, ])
                output_i = self.experts[i](token_i)
                outputs.append(output_i)

            expert_output = ht.concatenate_op(outputs, axis=0)
            
            alpha = ht.matmul_op(routed_input, centroid)
            alpha = ht.softmax_op(alpha)
            final_output_1 = ht.broadcastto_op(alpha, expert_output)
            final_output_1 = ht.mul_op(final_output_1, expert_output)
            final_output_2 = ht.addbyconst_op(alpha, -1)
            final_output_2 = ht.broadcastto_op(final_output_2, routed_input)
            final_output_2 = ht.mul_op(final_output_2, routed_input)
            
            final_output_2 = ht.mul_byconst_op(final_output_2, -1)
            final_output = ht.add_op(final_output_1, final_output_2)

            final_output = ht.indexing_op(final_output, reverse_indice)
            final_output = ht.alltoall_op(final_output)
            final_output = ht.indexing_op(final_output, reverse_randperm)
            final_output = ht.alltoall_op(final_output)
            
            combined_output = final_output
            return combined_output
