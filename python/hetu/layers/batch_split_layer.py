from .base import OpLayer, OpLayerGradient


class BatchSplitOnlyLayer(OpLayer):
    def __init__(self, sequence, ctx=None):
        super().__init__(BatchSplitOnlyLayer, ctx=ctx)
        self.sequence = sequence

    def __call__(self, x):
        self.make_inputs_n_output([x], self.sequence(x))
        return self

    def gradient(self, output_grad):
        return super().gradient([
            OpLayerGradient(self, 0)(output_grad),
        ])

    def modify_current_states(self, node_cur_state_map):
        if len(self.grad_layers) > 0:
            assert node_cur_state_map[self] == node_cur_state_map[self.grad_layers[0]]
        temp_status = node_cur_state_map[self]
        temp_status.check_state(1, False)
        temp_status.check_state(1, True)
        for node in self.all_forward_nodes:
            node_cur_state_map[node] = temp_status
        for node in self.all_backward_nodes:
            node_cur_state_map[node] = temp_status


class ReserveSplitLayer(OpLayer):
    def __init__(self, sequence, ctx=None):
        super().__init__(ReserveSplitLayer, ctx=ctx)
        self.sequence = sequence

    def __call__(self, x):
        self.make_inputs_n_output([x], self.sequence(x))
        return self

    def gradient(self, output_grad):
        return super().gradient([
            OpLayerGradient(self, 0)(output_grad),
        ])

    def modify_current_states(self, node_cur_state_map):
        from ..gpu_ops.Reshape import Array_ReshapeOp
        from ..context import NodeStatus
        if len(self.grad_layers) > 0:
            assert node_cur_state_map[self] == node_cur_state_map[self.grad_layers[0]]
        temp_status = node_cur_state_map[self]
        new_status = NodeStatus(dev_num=temp_status.dev_num)
        new_status.get_combine_from(temp_status, False, (1, 3))
        new_status.get_combine_from(temp_status, True, (1, 3))
        for i, node in enumerate(self.all_forward_nodes):
            if i == 2:
                node_cur_state_map[node] = temp_status
            else:
                node_cur_state_map[node] = new_status
            if isinstance(node, Array_ReshapeOp):
                node.splits = node_cur_state_map[node].state
        for node in self.all_backward_nodes:
            if i == 2:
                node_cur_state_map[node] = temp_status
            else:
                node_cur_state_map[node] = new_status
