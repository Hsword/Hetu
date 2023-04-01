import numpy as np
import pickle
from collections import defaultdict
import contextlib

from ..context import DeviceGroup, NodeStatus, DistConfig
from ..profiler import HetuSimulator
from ..ndarray import gpu, rgpu
from ..gpu_ops.Variable import PlaceholderOp
from ..layers.base import OpLayer


class Strategy(object):
    def __init__(self, save_path=None):
        # TODO: modify executor's logic to use communicators
        self.settings = DistConfig('/tmp/hetu_config.yml')
        self.use_dispatch = False
        self.save_path = save_path

    def set_raw_ctxs(self, graph_status):
        # called if use_dispatch is True
        raise NotImplementedError

    def set_raw_ctxs_n_states(self, graph_status, memory_pool):
        # called if use_dispatch is False
        raise NotImplementedError

    def infer_global_shapes(self, feed_shapes, graph_status):
        def dfs_infer_shape(node):
            if node in node_to_shape_map:
                return
            for n in node.inputs:
                dfs_infer_shape(n)
            input_shapes = [node_to_shape_map[n] for n in node.inputs]
            cur_shape = node.naive_infer_shape(input_shapes)
            node_to_shape_map[node] = cur_shape if cur_shape is None else tuple(
                cur_shape)

        node_to_shape_map = {k: v for k, v in feed_shapes.items()}
        for node in graph_status.node_list:
            dfs_infer_shape(node)
        return node_to_shape_map

    def set_overlap(self, overlap):
        self.overlap = overlap

    def init_node_group(self, graph_status):
        from ..optimizer import OptimizerOp
        from ..gpu_ops.EmbeddingLookUp import EmbeddingLookUp
        # this function forms node group

        def init_cur_node_group(node):
            if node in visited_key:
                return visited_key[node]
            key = None  # the key of node group
            is_backbone_node = self.check_backbone_node(node)
            if is_backbone_node:
                key = node
            for n in node.inputs:
                temp_key = init_cur_node_group(n)
                is_placeholder = isinstance(n, PlaceholderOp)
                if is_placeholder and n not in visited_key:
                    assert key is not None
                    visited_key[n] = key
                else:
                    if key is None:
                        key = temp_key
                    elif not is_backbone_node:
                        # for non-backbone-node, use the first key and merge two keys
                        merging[temp_key] = key
                    else:
                        if is_placeholder and isinstance(node, EmbeddingLookUp):
                            merging[temp_key] = key
                        # for non-backbone-node, use the first key
                        # TODO: add warning here
                        pass
            if isinstance(node, OpLayer):
                oplayer_nodes.add(node)
            elif not isinstance(node, PlaceholderOp) or is_backbone_node:
                visited_key[node] = key
            return key

        def reorder_node_group(node):
            # this function ensures the topo-order for computation
            if node in visited:
                return
            visited.add(node)
            for n in node.inputs:
                reorder_node_group(n)
            if not isinstance(node, OptimizerOp):
                backbone_node = self.inverse_node_group[node]
                temp_node_group[backbone_node].append(node)
                # ensure the keys of node group conform to topo order
                if node is backbone_node:
                    self.node_group[backbone_node] = temp_node_group[backbone_node]
                elif isinstance(backbone_node, OpLayer):
                    forward_group[backbone_node].discard(node)
                    if len(forward_group[backbone_node]) == 0:
                        self.node_group[backbone_node] = temp_node_group[backbone_node]

        visited_key = {}
        merging = {}
        for2back = graph_status.opt.forward2backward
        forward_loss = graph_status.forward_node_list[0]
        # here we specially handle oplayers
        # we add contents of oplayer into visited keys
        # so that every node can get the backbone node (oplayer)
        oplayer_nodes = set()
        # here user can modify the dfs function to use different backbone nodes
        init_cur_node_group(forward_loss)
        # add backward nodes into visited_keys(inverse_node_group)
        for node in list(visited_key.keys()):
            visited_key[node] = merging.get(
                visited_key[node], visited_key[node])
            backward_nodes = for2back.get(node, [])
            for bnode in backward_nodes:
                visited_key[bnode] = visited_key[node]
        for node in for2back[None]:
            visited_key[node] = visited_key[forward_loss]
        forward_group = dict()  # for reorder oplayers
        for oplayer in oplayer_nodes:
            forward_group[oplayer] = set(oplayer.all_forward_nodes)
            for node in oplayer.all_forward_nodes:
                visited_key[node] = oplayer
            for node in oplayer.all_backward_nodes:
                visited_key[node] = oplayer
        self.inverse_node_group = visited_key
        temp_node_group = defaultdict(list)
        self.node_group = dict()
        self.merging = merging
        visited = set()
        graph_status.extend_oplayers()
        for node in graph_status.node_list:
            reorder_node_group(node)
        self.coarse_topo_order = list(self.node_group.keys())
        graph_status.shrink_oplayers()
        # if self.mpi_comm.rank == 0:
        #     with open('node_group.txt', 'w') as fw:
        #         for node, group in self.node_group.items():
        #             print(node, group, file=fw, flush=True)

    def check_backbone_node(self, node):
        from ..gpu_ops.Conv2d import Conv2dOp
        from ..gpu_ops.Conv2dAddBias import Conv2dAddBiasOp
        from ..gpu_ops.MatrixMult import MatMulOp
        from ..gpu_ops.Linear import LinearOp
        from ..gpu_ops.Sum import SumOp
        from ..gpu_ops.Concatenate import ConcatenateOp
        from ..gpu_ops.SoftmaxCrossEntropy import SoftmaxCrossEntropyOp
        from ..gpu_ops.SoftmaxCrossEntropySparse import SoftmaxCrossEntropySparseOp
        from ..gpu_ops.EmbeddingLookUp import EmbeddingLookUp
        result = isinstance(node, (SoftmaxCrossEntropyOp, SoftmaxCrossEntropySparseOp, EmbeddingLookUp,
                                   Conv2dOp, Conv2dAddBiasOp, MatMulOp, LinearOp, SumOp, ConcatenateOp, OpLayer))
        result = result or isinstance(
            node, PlaceholderOp) and node.name == 'attention_mask'
        return result

    def save_contents(self, best_cur_status, best_raw_ctx, search_space=None):
        contents = []
        for node in self.node_group:
            key = node.name
            cur_rawctx = best_raw_ctx[key]
            cur_status = best_cur_status[key]
            assert cur_status.dev_num == cur_rawctx.mp_dev_num, \
                "For node {}, the context {} not conform with node status {}.".format(
                    node, cur_rawctx, cur_status)
            state, duplicate, order = cur_status.get_all()
            partial = cur_status.partial
            cur_entry = {
                'name': key,
                'status': {
                    'splits': str(state),
                    'partial': partial,
                    'duplicate': duplicate,
                    'order': str(order),
                },
                'device': cur_rawctx.full_repr(),
                'unfixed': False if search_space is None else search_space[node][0],
            }
            contents.append(cur_entry)
        return contents

    def save_json(self, best_cur_status, best_raw_ctx, save_path):
        import json
        if hasattr(self, 'search_space'):
            search_space = self.search_space
        else:
            search_space = None
        contents = self.save_contents(
            best_cur_status, best_raw_ctx, search_space=search_space)
        with open(save_path, 'w') as fw:
            json.dump(contents, fw, indent=4)

    def load_contents(self, contents):
        def eval_str(x):
            if isinstance(x, str):
                x = eval(x)
            return x
        best_cur_status = {}
        best_raw_ctx = {}
        unfixed_config = {}
        for i, node in enumerate(self.node_group):
            assert contents[i]['name'] == node.name
            status = contents[i]['status']
            splits = eval_str(status['splits'])
            duplicate = eval_str(status['duplicate'])
            partial = eval_str(status['partial'])
            if partial is None:
                partial = 1
            ctxs = DeviceGroup(eval_str(contents[i]['device']))
            assert ctxs.mp_dev_num == np.prod(list(splits.values()), dtype=int) * duplicate * partial, \
                "For node {}, the context {} not conform with node status ({}, {}({})).".format(
                    node, ctxs, splits, duplicate, partial)
            new_status = NodeStatus(
                ctxs.mp_dev_num, splits, partial_or_node=node)
            new_status.set_duplicate(duplicate)
            new_status.set_order(eval_str(status['order']))
            best_cur_status[node.name] = new_status
            best_raw_ctx[node.name] = ctxs
            unfixed_config[node.name] = contents[i]['unfixed']
        return best_cur_status, best_raw_ctx, unfixed_config

    def load_json(self, load_path):
        import json
        with open(load_path, 'r') as fr:
            contents = json.load(fr)
        return self.load_contents(contents)


class BaseSearchingStrategy(Strategy):
    def __init__(self, feed_shapes, save_path=None, load_path=None, load_init_path=None, batch_size=None, load_with_simulate=False, include_duplicate=True, pix=True):
        from itertools import combinations
        # now only consider homogeneous environment
        # the simulations now are all on the rank-0 device
        # TODO: use multiple devices to simulate
        super().__init__(save_path=save_path)
        self.debugging = False
        self.overlap = True  # overlap allreduce and computation
        self.use_dispatch = False
        self.num_ctxs = self.settings.num_workers
        self.all_devices = [rgpu(host['host'], i)
                            for host in self.settings for i in range(host['workers'])]
        self.feed_shapes = feed_shapes
        self.batch_size = batch_size
        dev_num = self.try_get_initial_dp()
        # we only use power of 2 as initial data parallel
        self.raw_ctx = DeviceGroup(
            tuple(self.all_devices[:dev_num])) if dev_num > 1 else DeviceGroup(self.all_devices[:dev_num])
        self.rank0_ctx = DeviceGroup(self.settings.chief + ':gpu:0')
        self.rank0_device = rgpu(self.settings.chief, 0)
        # save path is to save searched strategy
        # load path is to load a strategy that will straightly be used
        # load init path is to load a strategy that will be the start of the search
        self.save_path = save_path
        self.load_path = load_path
        self.load_init_path = load_init_path
        self.load_with_simulate = load_with_simulate

        # generate candidates for random sampling
        self.status_candidates = []
        left = 1
        while left <= self.num_ctxs:
            lrest = self.num_ctxs // left
            right = 1
            while right <= lrest:
                cand = {k: v for k, v in {0: left, 1: right}.items() if v != 1}
                dev_num = left * right
                if include_duplicate:
                    while dev_num <= self.num_ctxs:
                        self.status_candidates.append((dev_num, cand))
                        dev_num *= 2
                else:
                    self.status_candidates.append((dev_num, cand))
                right *= 2
            left *= 2
        self.device_candidates = {
            1: [DeviceGroup(dev) for dev in self.all_devices]}
        cur_num = 2
        while cur_num <= self.num_ctxs:
            self.device_candidates[cur_num] = [DeviceGroup(
                devs) for devs in combinations(self.all_devices, cur_num)]
            cur_num *= 2

        # initialize mpi communicator
        from ..gpu_ops.executor import wrapped_mpi_nccl_init, get_mpi_communicate
        self.nccl_comm = wrapped_mpi_nccl_init()
        self.mpi_comm = get_mpi_communicate()

        # initialize simulator
        self.simulator = HetuSimulator(
            self.feed_shapes, self.rank0_device, self.mpi_comm, self.num_ctxs, pix=pix)

    def try_get_initial_dp(self):
        if self.batch_size is None:
            # is batch size is not given, we infer it from feeding shapes
            for shape in self.feed_shapes.values():
                assert self.batch_size in (shape[0], None)
                self.batch_size = shape[0]
        cur_num = 1
        while cur_num * 2 <= self.num_ctxs:
            cur_num *= 2
        while self.batch_size % cur_num != 0:
            cur_num //= 2
        assert cur_num >= 1
        return cur_num

    @contextlib.contextmanager
    def wrapped_complete_partial_graph(self, backbone_node, status, graph_status):
        debugging = False
        if debugging:
            other_backbones = []
            for node, value in self.merging.items():
                if value is backbone_node:
                    other_backbones.append(node)
            yield graph_status.complete_partial_graph(self.node_group[backbone_node], backbone_node, other_backbones, status)
            for node in self.node_group[backbone_node]:
                node.reset_status()
        else:
            try:
                other_backbones = []
                for node, value in self.merging.items():
                    if value is backbone_node:
                        other_backbones.append(node)
                yield graph_status.complete_partial_graph(self.node_group[backbone_node], backbone_node, other_backbones, status)
            except:
                # TODO: re-design order specification!!! orders should be optional and tryable
                # optional: order can be added into search space
                # tryable: backtrack the last specified order and rollback all the changes
                yield None
            finally:
                for node in self.node_group[backbone_node]:
                    node.reset_status()

    def get_candidates(self, graph_status):
        # get candidates (search space) for each node in node_cur_state_map
        from ..gpu_ops.Concatenate import ConcatenateOp
        from ..gpu_ops.SoftmaxCrossEntropy import SoftmaxCrossEntropyOp
        from ..gpu_ops.SoftmaxCrossEntropySparse import SoftmaxCrossEntropySparseOp
        from ..gpu_ops.EmbeddingLookUp import EmbeddingLookUp
        from ..gpu_ops.LayerNorm import Layer_Normalization_GradientOp
        from ..gpu_ops.BatchNorm import Batch_Normalization_GradientOp
        from ..gpu_ops.Linear import LinearOp
        from ..gpu_ops.MatrixMult import MatMulOp
        from ..gpu_ops.Conv2d import Conv2dOp
        from ..gpu_ops.Conv2dAddBias import Conv2dAddBiasOp
        from ..layers import BatchSplitOnlyLayer

        def get_shape(node):
            if isinstance(node, OpLayer):
                return self.node_to_shape_map[node.output]
            else:
                return self.node_to_shape_map[node]

        def group_split_check(backbone_node, cand):
            sp = cand.state
            shape = get_shape(backbone_node)
            for key, value in sp.items():
                if shape[key] % value != 0:
                    return False
            with self.wrapped_complete_partial_graph(backbone_node, cand, graph_status) as local_node_cur_state_map:
                if local_node_cur_state_map is None:
                    return False
                for node, status in local_node_cur_state_map.items():
                    if isinstance(node, no_check_nodes):
                        continue
                    shape = get_shape(node)
                    for key, value in status.state.items():
                        if shape[key] % value != 0:
                            return False
                return True

        no_check_nodes = (Layer_Normalization_GradientOp,
                          Batch_Normalization_GradientOp)
        self.search_space = {}
        node_cur_state_map = graph_status.node_cur_state_map
        for node in node_cur_state_map:
            cands = []
            if isinstance(node, ConcatenateOp):
                for dn, sp in self.status_candidates:
                    cur_status = NodeStatus(dn, sp)
                    if node.axis not in sp and group_split_check(node, cur_status):
                        cands.append(cur_status)
            elif isinstance(node, (SoftmaxCrossEntropyOp, SoftmaxCrossEntropySparseOp, EmbeddingLookUp)):
                for dn, sp in self.status_candidates:
                    cur_status = NodeStatus(dn, sp, partial_or_node=node)
                    flag = False
                    if cur_status.duplicate is None:
                        flag = True
                        cur_status.set_duplicate(1)
                    if 1 not in sp and group_split_check(node, cur_status):
                        cands.append(cur_status)
                    if flag:
                        cur_status = NodeStatus(dn, sp)
                        if 1 not in sp and group_split_check(node, cur_status):
                            cands.append(cur_status)
            elif isinstance(node, BatchSplitOnlyLayer) or (isinstance(node, PlaceholderOp) and node.name == 'attention_mask'):
                for dn, sp in self.status_candidates:
                    cur_status = NodeStatus(dn, sp)
                    if 1 not in sp and group_split_check(node, cur_status):
                        cands.append(cur_status)
            else:
                for dn, sp in self.status_candidates:
                    cur_status = NodeStatus(dn, sp, partial_or_node=node)
                    if cur_status.duplicate is None:
                        if isinstance(node, (LinearOp, MatMulOp, Conv2dOp, Conv2dAddBiasOp)):
                            cur_status.set_duplicate(1)
                        else:
                            cur_status.set_partial(1)
                    if group_split_check(node, cur_status):
                        cands.append(cur_status)
            self.search_space[node] = (len(cands) > 1, cands)
        # with open('space.txt', 'w') as fw:
        #     for node, space in self.search_space.items():
        #         print(node, space, file=fw, flush=True)

    def initiator(self, node):
        # only for backbone node
        # return initial status, initial contexts
        dev_num = self.raw_ctx.mp_dev_num
        return NodeStatus(dev_num, {0: dev_num}, partial_or_node=node), self.raw_ctx

    def init_states(self, node_cur_state_map):
        # this function forms the search space
        # by specifying the candidates nodes
        # and also forms the node group (subgraph)
        # start as data parallel
        for backbone_node in self.node_group:
            status, raw_ctx = self.initiator(backbone_node)
            node_cur_state_map[backbone_node] = status
            self.set_group_raw_ctx(backbone_node, raw_ctx)

    def set_raw_ctxs_n_states(self, graph_status, memory_pool):
        # add partial information for forward nodes
        graph_status.assert_opt()
        self.init_node_group(graph_status)
        if self.mpi_comm.rank == 0:
            self.init_states(graph_status.node_cur_state_map)
            graph_status.extend_oplayers()
            self.node_to_shape_map = self.infer_global_shapes(
                self.feed_shapes, graph_status)
            graph_status.shrink_oplayers()
            self.get_candidates(graph_status)
            if self.load_path is None:
                if self.load_init_path is not None:
                    init_statuses, init_ctxs, unfixed_configs = self.load_json(
                        self.load_init_path)
                    for node in self.search_space:
                        key = node.name
                        graph_status.node_cur_state_map[node] = init_statuses[key]
                        self.set_group_raw_ctx(node, init_ctxs[key])
                        if not unfixed_configs[key]:
                            self.search_space[node] = (
                                False, [init_statuses[key]])
                from time import time
                start = time()
                best_cur_status, best_raw_ctx = self.searching(
                    graph_status, memory_pool)
                ending = time()
                print('Total searching time: {} s'.format(ending - start))
            else:
                print('Loading configuration from {} ...'.format(self.load_path))
                best_cur_status, best_raw_ctx, _ = self.load_json(
                    self.load_path)
                if self.load_with_simulate:
                    simulated_time = self.simulate_time(
                        graph_status, best_cur_status, best_raw_ctx)
                    print('The simulated time is: {:.3f}ms.'.format(
                        simulated_time))
            self.send_best_config(best_cur_status, best_raw_ctx)
        else:
            best_cur_status, best_raw_ctx = self.recv_best_config()
        del self.simulator
        if self.save_path is not None and self.mpi_comm.rank == 0:
            print('Saving configuration to {} ...'.format(self.save_path))
            self.save_json(best_cur_status, best_raw_ctx, self.save_path)
        for node in self.node_group:
            graph_status.node_cur_state_map[node] = best_cur_status[node.name]
            self.set_group_raw_ctx(node, best_raw_ctx[node.name])
        for node, value in self.merging.items():
            # for merging backbone nodes
            graph_status.node_cur_state_map[node] = best_cur_status[value.name]
        from .flexflow import FlexFlowSearching
        prune = not isinstance(self, FlexFlowSearching)
        graph_status.complete_state_map_with_partial_information(prune=prune)
        return self.raw_ctx

    def send_best_config(self, best_cur_status, best_raw_ctx):
        from ctypes import c_void_p, c_int, cast, byref
        status_bytes = pickle.dumps(best_cur_status)
        context_bytes = pickle.dumps(best_raw_ctx)
        status_length = len(status_bytes)
        context_length = len(context_bytes)
        sizes = (c_int * 2)(status_length, context_length)
        psizes = cast(sizes, c_void_p)
        signal = c_int(0)
        psignal = cast(byref(signal), c_void_p)
        self.mpi_comm.MPI_Broadcast(psignal, 4, root=0)
        self.mpi_comm.MPI_Broadcast(psizes, 8, root=0)
        self.mpi_comm.MPI_Broadcast(
            cast(status_bytes, c_void_p), status_length, root=0)
        self.mpi_comm.MPI_Broadcast(
            cast(context_bytes, c_void_p), context_length, root=0)
        # other messages for specific strategies
        from .pipeopt import PipeOptSearching
        if isinstance(self, PipeOptSearching):
            message = c_int(self.num_parts)
            self.mpi_comm.MPI_Broadcast(
                cast(byref(message), c_void_p), 4, root=0)
            self.batch_num = self.batch_num * self.num_parts // self.num_ctxs
            self.batch_size = self.batch_size // self.num_parts

    def recv_best_config(self):
        from ctypes import c_void_p, c_int, cast, string_at, c_char, byref
        from ..profiler import NCCLOP
        signal = c_int(1)
        psignal = cast(byref(signal), c_void_p)
        while signal.value > 0:
            self.mpi_comm.MPI_Broadcast(psignal, 4, root=0)
            if signal.value == 2:
                # profile allreduce
                self.simulator.profile_allreduce(0, [])
            elif signal.value == 3:
                # profile send/recv
                self.simulator.profile_sendrecv(0, [])
            elif signal.value == 4:
                # profile allgather
                self.simulator.profile_allreduce(
                    0, [], primitive=NCCLOP.AllGather)
            elif signal.value == 5:
                # profile reducescatter
                self.simulator.profile_allreduce(
                    0, [], primitive=NCCLOP.ReduceScatter)
            elif signal.value == 6:
                # profile reduce
                self.simulator.profile_allreduce(
                    0, [], primitive=NCCLOP.Reduce)
            elif signal.value == 7:
                # profile broadcast
                self.simulator.profile_allreduce(
                    0, [], primitive=NCCLOP.Broadcast)
        sizes = (c_int * 2)()
        psizes = cast(sizes, c_void_p)
        self.mpi_comm.MPI_Broadcast(psizes, 8, root=0)
        status_length, context_length = sizes[0], sizes[1]
        status_bytes = (c_char * status_length)()
        context_bytes = (c_char * context_length)()
        self.mpi_comm.MPI_Broadcast(
            cast(status_bytes, c_void_p), status_length, root=0)
        self.mpi_comm.MPI_Broadcast(
            cast(context_bytes, c_void_p), context_length, root=0)
        best_cur_status = pickle.loads(
            string_at(status_bytes, size=status_length))
        best_raw_ctx = pickle.loads(
            string_at(context_bytes, size=context_length))
        for ctx in best_raw_ctx.values():
            ctx.relocalize()
        # other messages for specific strategies
        from .pipeopt import PipeOptSearching
        if isinstance(self, PipeOptSearching):
            message = c_int()
            self.mpi_comm.MPI_Broadcast(
                cast(byref(message), c_void_p), 4, root=0)
            self.num_parts = message.value
            self.batch_num = self.batch_num * self.num_parts // self.num_ctxs
            self.batch_size = self.batch_size // self.num_parts
        return best_cur_status, best_raw_ctx

    def searching(self, graph_status, memory_pool):
        # the main function that must be implemented!
        raise NotImplementedError

    def set_group_raw_ctx(self, backbone_node, raw_ctx):
        if isinstance(backbone_node, OpLayer):
            backbone_node.raw_ctx = raw_ctx
            for oplayer in backbone_node.grad_layers:
                if oplayer is not None:
                    oplayer.raw_ctx = raw_ctx
        else:
            for node in self.node_group[backbone_node]:
                node.raw_ctx = raw_ctx

    def raw_generate_random_config(self, fixed_loss=False):
        import random
        best_cur_status = {}
        best_raw_ctx = {}
        num_nodes = len(self.search_space)
        for i, (node, spaces) in enumerate(self.search_space.items()):
            if fixed_loss and i >= num_nodes - 2:
                # make the last two nodes on main device; for Alexnet tests
                best_cur_status[node.name] = NodeStatus(1)
                best_raw_ctx[node.name] = self.rank0_ctx
            else:
                cur_node_status = random.choice(spaces[1])
                best_cur_status[node.name] = cur_node_status
                best_raw_ctx[node.name] = random.choice(
                    self.device_candidates[cur_node_status.dev_num])
        return best_cur_status, best_raw_ctx

    def raw_generate_random_config_pipedream(self):
        import random
        best_cur_status = None
        dev_splits = np.random.randint(2, size=self.num_ctxs - 1)
        dev_splits = [i for i, x in enumerate(dev_splits) if x]
        layer_splits = sorted(random.sample(
            range(len(self.search_space) - 1), len(dev_splits)))
        cur_layer_start = 0
        cur_dev_start = 0
        layers = []
        devs = []
        nodes = list(self.search_space.keys())
        status = NodeStatus(1)
        best_raw_ctx = {}
        best_cur_status = {}
        for lsp, dsp in zip(layer_splits, dev_splits):
            layers.append((cur_layer_start, lsp))
            devs.append((cur_dev_start, dsp))
            cur_dev = DeviceGroup([gpu(i)
                                   for i in range(cur_dev_start, dsp + 1)])
            for i in range(cur_layer_start, lsp + 1):
                name = nodes[i].name
                best_cur_status[name] = status
                best_raw_ctx[name] = cur_dev
            cur_layer_start = lsp + 1
            cur_dev_start = dsp + 1
        cur_dev = DeviceGroup([gpu(i)
                               for i in range(cur_dev_start, self.num_ctxs)])
        for i in range(cur_layer_start, len(self.search_space)):
            name = nodes[i].name
            best_cur_status[name] = status
            best_raw_ctx[name] = cur_dev
        return best_cur_status, best_raw_ctx

    def generate_random_config(self, eval_node_list, save_path, count=None, fixed_loss=False, pipedream=False):
        # this function is only used for tests
        # please don't use with executors
        if self.mpi_comm.rank == 0:
            from ..context import GraphStatus
            graph_status = GraphStatus(eval_node_list)
            self.init_node_group(graph_status)
            self.init_states(graph_status.node_cur_state_map)
            graph_status.extend_oplayers()
            self.node_to_shape_map = self.infer_global_shapes(
                self.feed_shapes, graph_status)
            graph_status.shrink_oplayers()
            self.get_candidates(graph_status)
            if count is None:
                if pipedream:
                    best_cur_status, best_raw_ctx = self.raw_generate_random_config_pipedream()
                else:
                    best_cur_status, best_raw_ctx = self.raw_generate_random_config(
                        fixed_loss)
                self.save_json(best_cur_status, best_raw_ctx, save_path)
            else:
                for i in range(count):
                    if pipedream:
                        best_cur_status, best_raw_ctx = self.raw_generate_random_config_pipedream()
                    else:
                        best_cur_status, best_raw_ctx = self.raw_generate_random_config(
                            fixed_loss)
                    self.save_json(best_cur_status, best_raw_ctx,
                                   save_path.format(i))

    def simulate_time(self, graph_status, best_cur_status, best_raw_ctx):
        # optional function, only used for tests
        # simulate the running time of the specific configuration
        # please don't use with executors
        raise NotImplementedError

    def init_pipeline_states(self, graph_status):
        # only for pipeline strategies (now only GPipe and PipeDream)
        from ..optimizer import OptimizerOp
        graph_status.extend_oplayers()
        self.node_status_map = dict()
        group_time = dict()
        self.cross_shape = dict()
        visited = set()
        self.group_topo = defaultdict(list)

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for n in node.inputs:
                dfs(n)
            if not isinstance(node, OptimizerOp):
                backbone_node = self.inverse_node_group[node]
                self.group_topo[backbone_node].append(node)
            else:
                params = node.optimizer.params
                opt_map_grad = defaultdict(list)
                opt_map_param = defaultdict(list)
                self.simulator.init_empty_optimizer(
                    node.optimizer, cached=True)
                for i, n in enumerate(node.inputs):
                    cur_backbone_node = self.inverse_node_group[n]
                    assert cur_backbone_node is self.inverse_node_group[params[i]]
                    opt_map_grad[cur_backbone_node].append(n)
                    opt_map_param[cur_backbone_node].append(params[i])
                for key in opt_map_param.keys():
                    new_opt = self.simulator.init_empty_optimizer(
                        node.optimizer)
                    new_opt.params = opt_map_param[key]
                    new_opt_op = OptimizerOp(opt_map_grad[key], new_opt)
                    self.group_topo[key].append(new_opt_op)

        def forward_dfs(node):
            if node in visited:
                return
            visited.add(node)
            backbone_node = self.inverse_node_group[node]
            for n in node.inputs:
                forward_dfs(n)
                input_backbone_node = self.inverse_node_group[n]
                if input_backbone_node is not backbone_node:
                    key = (input_backbone_node, backbone_node)
                    assert key not in self.cross_shape, 'Not implemented, strange topology.'
                    self.cross_shape[key] = self.node_to_shape_map[n]
            if backbone_node not in group_time:
                group_time[backbone_node] = self.get_group_compute_time(
                    self.group_topo[backbone_node])

        for node in graph_status.node_list:
            dfs(node)
        visited = set()
        for node in graph_status.forward_node_list:
            forward_dfs(node)
        self.accum_time = [0.]
        for group in self.coarse_topo_order:
            self.accum_time.append(self.accum_time[-1] + group_time[group])

        self.workers = self.simulator.nccl_profiler.workers
        graph_status.shrink_oplayers()

    def get_group_compute_time(self, group_topo):
        from ..optimizer import OptimizerOp
        from ..gpu_ops.Sum import SumOp
        # time without allreduce
        # for pipeline, no model parallel considered
        all_compute_time = 0.
        for node in group_topo:
            if isinstance(node, OptimizerOp):
                for i, n in enumerate(node.inputs):
                    if n.use_indexed_slices:
                        sparse_shape = (
                            self.node_to_shape_map[n.inputs[1]], self.node_to_shape_map[n.inputs[0]])
                    else:
                        sparse_shape = None
                    update_time = self.simulator.get_update_time(
                        self.node_to_shape_map[n], sparse_shape=sparse_shape)
                    if self.debugging:
                        with open('pipedream.txt', 'a') as fw:
                            print('update', update_time, n,
                                  node.optimizer.params[i], file=fw, flush=True)
                    all_compute_time += update_time
            else:
                if isinstance(node, SumOp):
                    input_shapes = []
                    for n in node.inputs:
                        if n.use_indexed_slices:
                            input_shapes.append(
                                (self.node_to_shape_map[n.inputs[1]], self.node_to_shape_map[n.inputs[0]]))
                        else:
                            input_shapes.append(self.node_to_shape_map[n])
                else:
                    input_shapes = [self.node_to_shape_map[n]
                                    for n in node.inputs]
                exe_time = self.simulator.get_node_time(
                    node, input_shapes, self.node_to_shape_map[node])
                if self.debugging:
                    with open('pipedream.txt', 'a') as fw:
                        print('execute', exe_time, node, file=fw, flush=True)
                all_compute_time += exe_time
        return all_compute_time

    def check_valid_topo(self):
        # ensure all hosts have the same number of workers
        self.workers = self.simulator.nccl_profiler.workers
        worker_num = None
        for _, value in self.workers.items():
            assert worker_num in (None, value)
            worker_num = value
        level_worker_num = [None, worker_num, len(self.workers)]
        return level_worker_num

    def reduce_device_candidates(self):
        # now only for optcnn
        # reduce the search space

        level_worker_num = self.check_valid_topo()
        _, num_gpu, num_nodes = level_worker_num

        chief = self.settings.chief
        hosts = [chief] + [h['host']
                           for h in self.settings if h['host'] != chief]
        indices = {1: [(0,)], 2: [], 4: [], 8: []}
        another = 1
        while another < min(8, num_gpu):
            indices[2].append((0, another))
            another *= 2
        if num_gpu >= 4:
            indices[4].append((0, 1, 2, 3))
            if num_gpu == 8:
                indices[4].append((0, 2, 4, 6))
                indices[8].append(tuple(range(8)))
        self.device_candidates.clear()

        cur_all_worker = 1
        while cur_all_worker <= self.num_ctxs:
            cur_result = []
            cur_num_nodes = 1
            cur_each_node = cur_all_worker // cur_num_nodes
            if cur_each_node > 8:
                cur_each_node = 8
                cur_num_nodes = cur_all_worker // cur_each_node
            while cur_num_nodes <= num_nodes and cur_each_node > 0:
                for ind in indices[cur_each_node]:
                    cur_result.append(DeviceGroup(tuple(rgpu(host, i)
                                                        for host in hosts[:cur_num_nodes] for i in ind)))
                cur_num_nodes *= 2
                cur_each_node = cur_all_worker // cur_num_nodes
            self.device_candidates[cur_all_worker] = cur_result
            cur_all_worker *= 2
