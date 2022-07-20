from ..gpu_ops.Node import Op


class BaseLayer(object):
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError

    def make_dataloader_func(self):
        raise NotImplementedError


class OpLayer(BaseLayer, Op):
    # this class is to simulate an Op in status deduction
    # limitation: now one instance is responsible for one op
    # TODO: make a new instance as return value of __call__ function
    # the subclass MUST implement the following 3 methods:
    # gradient(forward layer only), forward_deduce_states, backward_deduce_states
    def __init__(self, op_type=None, ctx=None):
        if op_type is None:
            op_type = OpLayer
        Op.__init__(self, op_type, [], ctx)
        self.initiated = False
        self.inputs = None
        self.output = None
        self.grad_outputs = None
        self.coarse_to_fine_target_map = None

    def get_all_inner_nodes(self):
        from ..gpu_ops.Variable import PlaceholderOp
        if self.initiated:
            return
        assert None not in [self.inputs, self.output]

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for i, n in enumerate(node.inputs):
                if n not in self.inputs:
                    dfs(n)
                else:
                    self.inputs_outputs[self.inputs.index(n)].append((node, i))
            if isinstance(node, PlaceholderOp) and node.trainable:
                self.params.append(node)
            self.all_forward_nodes.append(node)
        visited = set()
        self.all_forward_nodes = []
        self.all_backward_nodes = []
        self.grad_layers = []
        self.params = []
        # currently no more support parameters in oplayers
        # TODO: make it a flag; enable extend/shrink to modify optimizer
        assert len(self.params) == 0
        dfs(self.output)
        self.initiated = True

    def get_real_gradient(self, output_grad, index):
        if self.grad_outputs is not None:
            return self.grad_outputs[index]
        from ..gpu_ops.ReduceMean import ReduceMeanOp
        from ..gpu_ops.BatchNorm import Batch_NormalizationOp
        from ..gpu_ops.LayerNorm import Layer_NormalizationOp
        from ..gpu_ops.Division import DivConstOp, DivOp
        from ..gpu_ops.AddElewise import AddOp
        from ..gpu_ops.AddConst import AddByConstOp
        from ..gpu_ops.Sum import SumOp
        from collections import defaultdict

        def sum_node_list(node_list, ctx):
            from ..gpu_ops.Sum import sum_op
            """Custom sum func to avoid creating redundant nodes in Python sum func."""
            node_list = [n for n in node_list if n is not None]
            if node_list == []:
                return None, False
            elif len(node_list) == 1:
                return node_list[0], False
            else:
                return sum_op(node_list, ctx=ctx), True

        self.grad_input = output_grad
        node_to_output_grads_list = {self.output: [output_grad]}
        node_to_output_grad = {}
        self.forward2backward = defaultdict(list)
        # Traverse forward graph in reverse topological order
        reverse_topo_order = reversed(self.all_forward_nodes)
        for node in reverse_topo_order:
            # TODO: not support EmbeddingLookUp here!!!
            output_grad, is_new = sum_node_list(
                node_to_output_grads_list[node], node.raw_ctx)
            if is_new:
                self.forward2backward[node].append(output_grad)
            if output_grad is None:
                for n in node.inputs:
                    if n not in node_to_output_grads_list:
                        node_to_output_grads_list[n] = []
                continue
            node_to_output_grad[node] = output_grad
            input_grads_list = node.gradient(output_grad)
            # TODO: not consider following nodes in forward2backward, can be improved
            # DistGCN_15d, MatrixDot, Sigmoid, Sqrt, Where
            if input_grads_list is not None:
                if not isinstance(node, (AddOp, AddByConstOp, SumOp)):
                    self.forward2backward[node].extend([
                        n for n in input_grads_list if n is not None])
                if isinstance(node, (ReduceMeanOp, Batch_NormalizationOp, Layer_NormalizationOp)):
                    self.forward2backward[node].append(
                        input_grads_list[0].inputs[0])
                elif isinstance(node, DivConstOp):
                    temp = input_grads_list[0].inputs[0]
                    self.forward2backward[node].extend([temp, temp.inputs[0]])
                elif isinstance(node, DivOp):
                    temp = input_grads_list[1].inputs[0]
                    self.forward2backward[node].extend(
                        [input_grads_list[0].inputs[0], temp, temp.inputs[0]])
            for i in range(len(node.inputs)):
                if node.inputs[i] not in node_to_output_grads_list:
                    node_to_output_grads_list[node.inputs[i]] = []
                # Calculate partial adjoint for input nodes.
                node_to_output_grads_list[node.inputs[i]].append(
                    input_grads_list[i])
        self.grad_outputs = []
        for node in self.inputs:
            grad_node, is_new = sum_node_list(
                node_to_output_grads_list[node], node.raw_ctx)
            if is_new:
                # here we use the inputs
                # when the forward2backward is used, we assumed that inputs have already processed
                self.forward2backward[node].append(grad_node)
            self.grad_outputs.append(grad_node)
        self.grad_input_outputs = []
        self.all_backward_nodes = sum(self.forward2backward.values(), [])
        for backward_node in self.all_backward_nodes:
            if self.grad_input in backward_node.inputs:
                self.grad_input_outputs.append(
                    (backward_node, backward_node.inputs.index(self.grad_input)))
        self.param_grads = [node_to_output_grad[node] for node in self.params]
        return self.grad_outputs[index]

    def modify_current_states(self, node_cur_state_map):
        raise NotImplementedError

    def make_inputs_n_output(self, inputs, output):
        self.inputs = inputs
        self.inputs_outputs = [[] for _ in self.inputs]
        self.output = output
        if not isinstance(self, OpLayerGradient):
            self.get_all_inner_nodes()

    def reset_inputs(self, node_cur_state_map, node_tar_state_map, is_model_parallel):
        # switch to fine-grained op expression
        assert not isinstance(self, OpLayerGradient)
        # modify raw ctx
        for node in self.all_forward_nodes:
            node.raw_ctx = self.raw_ctx
        for node in self.all_backward_nodes:
            node.raw_ctx = self.raw_ctx
        if is_model_parallel:
            # it's not necessary to remove layer state maps
            # here we store the redundancy information in state maps
            # modify target states
            self.handle_target_map(node_tar_state_map)
            # modify current states, MUST be implemented by subclass
            self.modify_current_states(node_cur_state_map)

    def handle_target_map(self, node_tar_state_map):
        if self.coarse_to_fine_target_map is None:
            from collections import defaultdict
            self.coarse_to_fine_target_map = (dict(), defaultdict(list))
            first_order_map, second_order_map = self.coarse_to_fine_target_map
            for i, node in enumerate(self.inputs):
                # handle forward nodes
                if node in node_tar_state_map and self in node_tar_state_map[node]:
                    for io_node, _ in self.inputs_outputs[i]:
                        second_order_map[(node, self)].append(io_node)
                    # handle special backward nodes
                    if self.grad_outputs is not None and node in self.grad_outputs[i].inputs:
                        second_order_map[(node, self)].append(
                            self.grad_outputs[i])
                # handle backward nodes
                # here we don't consider targets between nodes within an OpLayer,
                # nor nodes between forward OpLayer and backward OpLayerGradient.
                if self.grad_outputs is not None:
                    grad_input = self.grad_input
                    grad_layer = self.grad_layers[i]
                    if grad_input in node_tar_state_map and grad_layer in node_tar_state_map[grad_input]:
                        for io_node, _ in self.grad_input_outputs:
                            second_order_map[(grad_input, grad_layer)].append(
                                io_node)
                    if grad_layer in node_tar_state_map:
                        first_order_map[grad_layer] = self.grad_outputs[i]
            if self in node_tar_state_map:
                first_order_map[self] = self.output
        first_order_map, second_order_map = self.coarse_to_fine_target_map
        # change op part
        # here we don't remove layer part, since it's not necessary
        for key, value in second_order_map.items():
            temp_target = node_tar_state_map[key[0]][key[1]]
            for vnode in value:
                node_tar_state_map[key[0]][vnode] = temp_target
        for key, value in first_order_map.items():
            node_tar_state_map[value] = node_tar_state_map[key]

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        for nst in input_statuses:
            status.copy_from(nst, deduce_order)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        for nst in input_statuses:
            nst.copy_from(status, deduce_order)

    def gradient(self, grad_layers):
        self.grad_layers = grad_layers
        return self.grad_layers

    def reset_status(self):
        for node in self.all_forward_nodes:
            node.reset_status()
        for node in self.all_backward_nodes:
            node.reset_status()


class OpLayerGradient(OpLayer):
    def __init__(self, forward_node, index, ctx=None):
        super().__init__(OpLayerGradient, ctx=ctx)
        self.forward_node = forward_node
        self.index = index

    def __call__(self, output_grad):
        self.make_inputs_n_output(
            [output_grad],
            self.forward_node.get_real_gradient(output_grad, self.index)
        )
        return self

    def reset_status(self):
        pass
