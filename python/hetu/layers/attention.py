from .base import OpLayer, OpLayerGradient
import numpy as np


class MultiHeadAttention(OpLayer):
    def __init__(self, hidden_size, num_heads, sequence_length, dropout_rate=0., causal_mask=False, ctx=None):
        super().__init__(MultiHeadAttention, ctx=ctx)
        assert hidden_size % num_heads == 0, \
            'Hidden size ({}) should be divisible by number of heads ({})!'.format(
                hidden_size, num_heads)
        assert dropout_rate >= 0 and dropout_rate < 1, \
            'Dropout rate ({}) should be in [0, 1).'.format(dropout_rate)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.attention_head_size = hidden_size // num_heads
        self.dropout_rate = dropout_rate
        self.scale = (1.0 / np.sqrt(float(self.attention_head_size)))
        self.re_shape = [-1, sequence_length,
                         num_heads, self.attention_head_size]
        self.causal_mask = causal_mask
        if self.causal_mask:
            # for GPT2
            import hetu as ht
            self.mask = ht.Variable('causal_mask', value=np.tril(np.ones((self.sequence_length, self.sequence_length)).reshape(
                1, 1, self.sequence_length, self.sequence_length)).astype(np.float32), trainable=False)
            self.masked_bias = -1e4

    def __call__(self, query, key, value, ori_attention_mask):
        '''
        inputs:
            query: [batch_size*seq_len, hidden_size]
            key: [batch_size*seq_len, hidden_size]
            value: [batch_size*seq_len, hidden_size]
            attention_mask: [batch_size, 1, 1, seq_len]
        outputs:
            context_layer: [batch_size*seq_len, hidden_size]
        '''
        import hetu as ht

        def transpose_for_scores(input_tensor):
            output_tensor = ht.array_reshape_op(input_tensor, self.re_shape)
            output_tensor = ht.transpose_op(output_tensor, [0, 2, 1, 3])
            return output_tensor

        def transpose_key_for_scores(input_tensor):
            output_tensor = ht.array_reshape_op(input_tensor, self.re_shape)
            output_tensor = ht.transpose_op(output_tensor, [0, 2, 3, 1])
            return output_tensor

        # reshape, transpose
        # [batch_size, num_heads, seq_len, head_size]
        query_layer = transpose_for_scores(query)
        # [batch_size, num_heads, head_size, seq_len]
        key_layer = transpose_key_for_scores(key)
        # [batch_size, num_heads, seq_len, head_size]
        value_layer = transpose_for_scores(value)

        # scale, get score, apply mask, softmax, dropout
        # [batch_size, num_heads, head_size, seq_len]
        key_layer_scaled = ht.mul_byconst_op(key_layer, self.scale)
        # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = ht.batch_matmul_op(query_layer, key_layer_scaled)
        if self.causal_mask:
            attention_scores = ht.where_const_op(ht.broadcastto_op(
                self.mask, attention_scores), attention_scores, self.masked_bias)
        attention_mask = ht.broadcastto_op(
            ori_attention_mask, attention_scores)
        attention_scores = ht.add_op(attention_scores, attention_mask)
        attention_probs = ht.softmax_op(attention_scores)
        if self.dropout_rate > 0.:
            attention_probs = ht.dropout_op(
                attention_probs, 1-self.dropout_rate)

        # interact with value, transpose, reshape
        # [batch_size, num_heads, seq_len, head_size]
        context_layer = ht.batch_matmul_op(attention_probs, value_layer)
        # [batch_size, seq_len, num_heads, head_size]
        context_layer = ht.transpose_op(context_layer, [0, 2, 1, 3])
        # [batch_size*seq_len, hidden_size]
        context_layer = ht.array_reshape_op(
            context_layer, [-1, self.hidden_size])
        self.make_inputs_n_output(
            [query, key, value, ori_attention_mask], context_layer)
        return self

    def gradient(self, output_grad):
        return super().gradient([
            OpLayerGradient(self, 0)(output_grad),
            OpLayerGradient(self, 1)(output_grad),
            OpLayerGradient(self, 2)(output_grad),
            None,
        ])

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        for nst in input_statuses[:3]:
            status.copy_from(nst, deduce_order)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        for nst in input_statuses[:3]:
            nst.copy_from(status, deduce_order)
        if deduce_order:
            if status.valid_all():
                input_statuses[3].set_order(status.combine_order((1, -1)))
        else:
            if status.valid_state():
                input_statuses[3].set_state(*status.combine_state((1, -1)))

    def modify_current_states(self, node_cur_state_map):
        from ..context import NodeStatus
        from ..gpu_ops.Reshape import Array_ReshapeOp
        from ..gpu_ops.BatchMatrixMult import BatchMatMulOp
        from ..gpu_ops.Transpose import TransposeOp
        from ..gpu_ops.Broadcast import BroadcastToOp
        from ..gpu_ops.ReduceSum import ReduceSumOp
        from ..gpu_ops.Variable import PlaceholderOp
        temp_status = node_cur_state_map[self]
        for grad_layer in self.grad_layers[:3]:
            assert temp_status == node_cur_state_map[grad_layer]
        state, duplicate, order = temp_status.get_all()
        dp = state.get(0, 1)
        mp = state.get(1, 1)
        dev_num = temp_status.dev_num

        # all forward nodes:
        # [Array_ReshapeOp51, TransposeOp52, Array_ReshapeOp53, TransposeOp54,
        #  MulByConstOp57, BatchMatMulOp58, BroadcastToOp59, AddOp60, SoftmaxOp61,
        #  DropoutOp62, Array_ReshapeOp55, TransposeOp56, BatchMatMulOp63,
        #  TransposeOp64, Array_ReshapeOp65]
        middle_status = NodeStatus(dev_num, {0: dp, 2: mp})
        if order is not None:
            new_order = list(order)
            if 1 in new_order:
                new_order[new_order.index(1)] = 2
            middle_status.set_order(tuple(new_order))

        for node in self.all_forward_nodes:
            if isinstance(node, Array_ReshapeOp):
                if len(node.output_shape) == 2:
                    # the last reshape op
                    node_cur_state_map[node] = temp_status
                    for bnode in self.forward2backward[node]:
                        node_cur_state_map[bnode] = middle_status
                else:
                    assert len(node.output_shape) == 4
                    # the first three reshape op
                    node_cur_state_map[node] = middle_status
                    for bnode in self.forward2backward[node]:
                        node_cur_state_map[bnode] = temp_status
                node.splits = node_cur_state_map[node].state
            elif isinstance(node, BatchMatMulOp):
                # happen to be the same with temp_status
                node_cur_state_map[node] = temp_status
                for bnode in self.forward2backward[node]:
                    node_cur_state_map[bnode] = temp_status
            elif isinstance(node, PlaceholderOp):
                cur_status = NodeStatus(dev_num, state={})
                cur_status.valid_all()
                node_cur_state_map[node] = cur_status
            elif isinstance(node, BroadcastToOp):
                node_cur_state_map[node] = temp_status
            else:
                status = NodeStatus(dev_num)
                input_statuses = [node_cur_state_map[n] for n in node.inputs]
                node.forward_deduce_states(input_statuses, status, False)
                node.forward_deduce_states(input_statuses, status, True)
                node_cur_state_map[node] = status
                for bnode in self.forward2backward[node]:
                    node_cur_state_map[bnode] = status
        for node in self.all_forward_nodes:
            if isinstance(node, TransposeOp):
                for bnode in self.forward2backward[node]:
                    if isinstance(bnode, TransposeOp):
                        status = NodeStatus(dev_num)
                        input_statuses = [node_cur_state_map[n]
                                          for n in bnode.inputs]
                        bnode.forward_deduce_states(
                            input_statuses, status, False)
                        bnode.forward_deduce_states(
                            input_statuses, status, True)
                        node_cur_state_map[bnode] = status
            elif isinstance(node, BroadcastToOp):
                for bnode in self.forward2backward[node]:
                    if isinstance(bnode, ReduceSumOp):
                        node_cur_state_map[bnode] = node_cur_state_map[self.inputs[-1]]
