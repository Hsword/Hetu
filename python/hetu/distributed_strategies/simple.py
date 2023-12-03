from ..context import DeviceGroup, NodeStatus
from ..gpu_ops.Variable import PlaceholderOp
from .base import Strategy


class DataParallel(Strategy):
    def __init__(self, aggregate=None):
        super().__init__()
        if aggregate is None:
            aggregate = 'ps' if self.settings.enable_PS else 'allreduce'
        aggregate = aggregate.lower()
        assert aggregate in ('allreduce', 'ps', 'hybrid')
        self.aggregate = aggregate
        self.use_dispatch = False

        # TODO: check communicators; check in a method, or in executor, or in base class?
        embedding_ctxs = ['cpu:0'] if aggregate != 'allreduce' else []
        ctxs = ['cpu:0'] if aggregate == 'ps' else []
        for host, num_worker in self.settings.workers.items():
            devices = [host + ':gpu:' + str(i) for i in range(num_worker)]
            embedding_ctxs.extend(devices)
            ctxs.extend(devices)
        self.embedding_raw_ctx = DeviceGroup(embedding_ctxs)
        self.raw_ctx = DeviceGroup(ctxs)

    def set_raw_ctxs_n_states(self, graph_status, memory_pool):
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for n in node.inputs:
                dfs(n)
            if isinstance(node, PlaceholderOp) and node.trainable and not node.is_embed:
                node.raw_ctx = self.raw_ctx
            else:
                node.raw_ctx = self.embedding_raw_ctx
            node_cur_state_map[node] = NodeStatus(1, {})
        graph_status.extend_oplayers()
        visited = set()
        node_cur_state_map = graph_status.node_cur_state_map
        for node in graph_status.node_list:
            dfs(node)
        return self.raw_ctx


class ModelParallel4CNN(Strategy):
    def __init__(self):
        super().__init__()
        # only for CNN and FC layers
        ctxs = ()
        for host, num_worker in self.settings.workers.items():
            ctxs += tuple(host + ':gpu:' + str(i)
                          for i in range(num_worker))
        self.num_ctxs = len(ctxs)
        self.raw_ctx = DeviceGroup(ctxs)
        self.use_dispatch = False
        self.mp_4_lm = False

    def set_raw_ctxs_n_states(self, graph_status, memory_pool):
        from ..gpu_ops.Conv2d import Conv2dOp
        from ..gpu_ops.Conv2dAddBias import Conv2dAddBiasOp
        from ..gpu_ops.MatrixMult import MatMulOp
        from ..gpu_ops.Linear import LinearOp
        from ..gpu_ops.Sum import SumOp
        from ..gpu_ops.Concatenate import ConcatenateOp
        from ..gpu_ops.SoftmaxCrossEntropy import SoftmaxCrossEntropyOp
        from ..gpu_ops.SoftmaxCrossEntropySparse import SoftmaxCrossEntropySparseOp
        from ..gpu_ops.EmbeddingLookUp import EmbeddingLookUp
        from ..layers import MultiHeadAttention, BatchSplitOnlyLayer

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            node.raw_ctx = self.raw_ctx
            for n in node.inputs:
                dfs(n)
            if isinstance(node, (Conv2dOp, MatMulOp, Conv2dAddBiasOp, LinearOp, SumOp, ConcatenateOp)):
                node_cur_state_map[node] = NodeStatus(
                    node.raw_ctx.mp_dev_num, {1: self.num_ctxs}, partial_or_node=node)
            elif isinstance(node, (SoftmaxCrossEntropyOp, SoftmaxCrossEntropySparseOp)):
                node_cur_state_map[node] = NodeStatus(
                    node.raw_ctx.mp_dev_num, {0: self.num_ctxs})

        def dfs_lm(node):
            if node in visited:
                return
            visited.add(node)
            node.raw_ctx = self.raw_ctx
            for n in node.inputs:
                dfs_lm(n)
            if isinstance(node, (SoftmaxCrossEntropyOp, SoftmaxCrossEntropySparseOp, EmbeddingLookUp, BatchSplitOnlyLayer)):
                node_cur_state_map[node] = NodeStatus(
                    self.num_ctxs, {0: self.num_ctxs})
            elif isinstance(node, (Conv2dOp, MatMulOp, Conv2dAddBiasOp, LinearOp, SumOp, ConcatenateOp, MultiHeadAttention)):
                node_cur_state_map[node] = NodeStatus(
                    self.num_ctxs, {1: self.num_ctxs}, partial_or_node=node)
            elif isinstance(node, PlaceholderOp) and node.name == 'attention_mask':
                node_cur_state_map[node] = NodeStatus(self.num_ctxs, {})

        graph_status.assert_opt()
        node_cur_state_map = graph_status.node_cur_state_map
        visited = set()
        # add partial information for forward nodes
        if self.mp_4_lm:
            dfs_lm(graph_status.forward_node_list[0])
        else:
            dfs(graph_status.forward_node_list[0])
        graph_status.determine_by_forward_current_states(self.raw_ctx)
        return self.raw_ctx


class ModelParallel4LM(ModelParallel4CNN):
    def __init__(self):
        super().__init__()
        self.mp_4_lm = True


class OneWeirdTrick4CNN(Strategy):
    # split batch dimension in conv layers
    # split channel dimension in linear layers
    def __init__(self, feed_shapes=None):
        super().__init__()
        # only for CNN and FC layers
        ctxs = ()
        for host, num_worker in self.settings.workers.items():
            ctxs += tuple(host + ':gpu:' + str(i)
                          for i in range(num_worker))
        self.num_ctxs = len(ctxs)
        self.raw_ctx = DeviceGroup(ctxs)
        self.use_dispatch = False
        self.feed_shapes = feed_shapes

    def set_raw_ctxs_n_states(self, graph_status, memory_pool):
        from ..gpu_ops.Conv2d import Conv2dOp
        from ..gpu_ops.Conv2dAddBias import Conv2dAddBiasOp
        from ..gpu_ops.MatrixMult import MatMulOp
        from ..gpu_ops.Linear import LinearOp
        from ..gpu_ops.SoftmaxCrossEntropy import SoftmaxCrossEntropyOp
        from ..gpu_ops.SoftmaxCrossEntropySparse import SoftmaxCrossEntropySparseOp

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            node.raw_ctx = self.raw_ctx
            for n in node.inputs:
                dfs(n)
            if isinstance(node, (Conv2dOp, Conv2dAddBiasOp, SoftmaxCrossEntropyOp, SoftmaxCrossEntropySparseOp)):
                node_cur_state_map[node] = NodeStatus(
                    self.num_ctxs, {0: self.num_ctxs}, partial_or_node=node)
            elif isinstance(node, (MatMulOp, LinearOp)):
                if node_to_shape_map is not None and node_to_shape_map[node][1] % self.num_ctxs != 0:
                    dim = 0
                else:
                    dim = 1
                node_cur_state_map[node] = NodeStatus(
                    self.num_ctxs, {dim: self.num_ctxs}, partial_or_node=True)

        graph_status.assert_opt()
        if self.feed_shapes is not None:
            node_to_shape_map = self.infer_global_shapes(
                self.feed_shapes, graph_status)
        else:
            node_to_shape_map = None
        visited = set()
        node_cur_state_map = graph_status.node_cur_state_map
        # add partial information for forward nodes
        dfs(graph_status.forward_node_list[0])
        graph_status.determine_by_forward_current_states(self.raw_ctx)
        return self.raw_ctx


class MegatronLM(Strategy):
    # strategy from https://arxiv.org/abs/1909.08053
    def __init__(self, max_split=None, save_path=None):
        super().__init__(save_path)
        # only for Transformer-based Language Models
        ctxs = ()
        for host, num_worker in self.settings.workers.items():
            ctxs += tuple(host + ':gpu:' + str(i)
                          for i in range(num_worker))
        self.num_ctxs = len(ctxs)
        if max_split is not None:
            if self.num_ctxs > max_split:
                assert self.num_ctxs % max_split == 0
                new_ctxs = []
                st = 0
                en = max_split
                while en <= self.num_ctxs:
                    new_ctxs.append(ctxs[st:en])
                    st = en
                    en += max_split
                ctxs = new_ctxs
                self.num_ctxs = max_split
        self.raw_ctx = DeviceGroup(ctxs)
        self.use_dispatch = False

    def set_raw_ctxs_n_states(self, graph_status, memory_pool):
        from ..gpu_ops.Conv2d import Conv2dOp
        from ..gpu_ops.Conv2dAddBias import Conv2dAddBiasOp
        from ..gpu_ops.Linear import LinearOp
        from ..gpu_ops.MatrixMult import MatMulOp
        from ..gpu_ops.Sum import SumOp
        from ..gpu_ops.EmbeddingLookUp import EmbeddingLookUp
        from ..gpu_ops.SoftmaxCrossEntropy import SoftmaxCrossEntropyOp
        from ..gpu_ops.SoftmaxCrossEntropySparse import SoftmaxCrossEntropySparseOp
        from ..gpu_ops.Relu import ReluOp
        from ..layers import MultiHeadAttention, BatchSplitOnlyLayer, ReserveSplitLayer

        def dfs(node, need_split=False, before_attention=False):
            if node in visited:
                return
            visited.add(node)
            node.raw_ctx = self.raw_ctx
            is_linear = isinstance(node, LinearOp)
            cur_partial = False
            if isinstance(node, (LinearOp, MatMulOp)) and node.inputs[1].is_embed:
                assert self.linear_node is None
                self.partition_embedding = node.inputs[1]
                self.linear_node = node
            if isinstance(node, EmbeddingLookUp) and node.inputs[0] == self.partition_embedding:
                cur_partial = True
            if is_linear:
                in0 = node.inputs[0]
                if (isinstance(in0, MultiHeadAttention) or (isinstance(in0, ReluOp) and isinstance(in0.inputs[0], LinearOp))):
                    dfs(in0, True)
                    cur_partial = True
                else:
                    dfs(in0)
                for n in node.inputs[1:]:
                    dfs(n)
            elif isinstance(node, MultiHeadAttention):
                for n in node.inputs[:3]:
                    dfs(n, True, True)
                dfs(node.inputs[-1])
            elif isinstance(node, ReluOp) and need_split:
                dfs(node.inputs[0], True)
            else:
                for n in node.inputs:
                    dfs(n)
            if node == self.linear_node:
                node_cur_state_map[node] = NodeStatus(
                    self.num_ctxs, {1: self.num_ctxs})
            elif need_split and isinstance(node, (LinearOp, MultiHeadAttention)):
                if before_attention:
                    node_cur_state_map[node] = NodeStatus(
                        self.num_ctxs, {1: self.num_ctxs})
                else:
                    node_cur_state_map[node] = NodeStatus(
                        self.num_ctxs, {1: self.num_ctxs}, partial_or_node=node)
            elif isinstance(node, SoftmaxCrossEntropySparseOp):
                if isinstance(node.inputs[0], (LinearOp, MatMulOp)):
                    cur_input = node.inputs[0]
                elif isinstance(node.inputs[0].inputs[0], (LinearOp, MatMulOp)):
                    cur_input = node.inputs[0].inputs[0]
                if cur_input.inputs[1].is_embed:
                    node_cur_state_map[node] = NodeStatus(
                        self.num_ctxs, {}, partial_or_node=True)
                    node_cur_state_map[node].set_duplicate(1)
                else:
                    node_cur_state_map[node] = NodeStatus(
                        self.num_ctxs, {})
            elif isinstance(node, (SumOp, EmbeddingLookUp, BatchSplitOnlyLayer, Conv2dOp, Conv2dAddBiasOp, MatMulOp, LinearOp, SoftmaxCrossEntropyOp, ReserveSplitLayer)) \
                    or (isinstance(node, PlaceholderOp) and node.name == 'attention_mask'):
                node_cur_state_map[node] = NodeStatus(
                    self.num_ctxs, {}, partial_or_node=cur_partial)
                if cur_partial:
                    node_cur_state_map[node].set_duplicate(1)

        graph_status.assert_opt()
        visited = set()
        self.linear_node = None
        self.partition_embedding = None
        node_cur_state_map = graph_status.node_cur_state_map
        # add partial information for forward nodes
        dfs(graph_status.forward_node_list[0])
        if self.save_path is not None:
            self.init_node_group(graph_status)
            self.save_json({node.name: node_cur_state_map[node] for node in self.node_group}, {
                node.name: node.raw_ctx for node in self.node_group}, self.save_path)
        graph_status.determine_by_forward_current_states(self.raw_ctx)
        return self.raw_ctx
