from .ndarray import cpu, gpu, rcpu, rgpu, DLContext, is_gpu_ctx
import contextlib
import re
import numpy as np


class DeviceGroup(object):
    def __init__(self, ctxs):
        self._contexts = self.parse_contexts(ctxs)
        self.get_servers_n_workers()
        self._is_mp = False
        self._mp_device_num = None
        for c in self._contexts:
            if isinstance(c, tuple):
                self._is_mp = True
                assert self._mp_device_num in (None, len(c)), \
                    'Now only support same model parallel in data parallel.'
                self._mp_device_num = len(c)
        if self._mp_device_num is None:
            self._mp_device_num = 1

    @classmethod
    def parse_contexts(cls, ctxs):
        if isinstance(ctxs, DeviceGroup):
            return ctxs
        if isinstance(ctxs, str):
            ctxs = re.split(';|,| +', ctxs.lower())
        if not isinstance(ctxs, list):
            ctxs = [ctxs]
        new_ctxs = []
        for c in ctxs:
            if isinstance(c, tuple):
                c = tuple([ccc for ccc in [cls.str2ctx(cc)
                                           for cc in c] if ccc is not None])
            else:
                c = cls.str2ctx(c)
            if c is not None:
                new_ctxs.append(c)
        return new_ctxs

    @classmethod
    def str2ctx(cls, c):
        if isinstance(c, str):
            c = c.lower().split(':')
            assert c[-2] in ('cpu', 'gpu'), 'Context invalid: %s' % c
            hostname = 'localhost' if len(c) == 2 else c[0]
            idx = int(c[-1])
            c = rcpu(hostname, idx) if c[-2] == 'cpu' else rgpu(hostname, idx)
        assert isinstance(c, DLContext), 'Context invalid: %s' % c
        return c

    def index(self, ctx):
        return self._contexts.index(ctx)

    def is_mp(self):
        return self._is_mp

    def __getitem__(self, key):
        return self._contexts[key]

    def __iter__(self):
        return iter(self._contexts)

    def __len__(self):
        return len(self._contexts)

    @property
    def mp_device_num(self):
        return self._mp_device_num

    @property
    def worker_num(self):
        return len(self._workers)

    @property
    def server_num(self):
        return len(self._servers)

    @property
    def workers(self):
        return self._workers

    @property
    def servers(self):
        return self._servers

    def get_servers_n_workers(self):
        self._workers = []
        self._servers = []
        for ctx in self._contexts:
            if isinstance(ctx, tuple) or is_gpu_ctx(ctx):
                self._workers.append(ctx)
            else:
                self._servers.append(ctx)

    def __repr__(self):
        result = 'DeviceGroup('
        for c in self._contexts:
            result += ('(' + ', '.join([str(cc) for cc in c]) +
                       '), ') if isinstance(c, tuple) else '%s, ' % c
        result += ')'
        return result

    def __hash__(self):
        if not hasattr(self, 'hash'):
            self.hash = hash(tuple(self._contexts))
        return self.hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_sorted(self):
        return DeviceGroup(sorted(self._contexts, key=lambda x: hash(x.hostname) + hash(x.device_id)))


class ContextStack(object):
    def __init__(self):
        self._stack = []

    def peek(self):
        return self._stack[-1] if self._stack else None

    def push(self, ctx):
        return self._stack.append(ctx)

    def pop(self):
        self._stack.pop()


_default_ctx_stack = ContextStack()


class NodeStatus(object):
    def __init__(self, state=None, dev_num=None):
        if state is not None:
            if not isinstance(state, dict):
                state = {i: v for i, v in enumerate(state) if v != 1}
            else:
                state = {i: v for i, v in sorted(state.items()) if v != 1}
        self._state = state
        self._device_num = dev_num
        if self._device_num is not None and self._state is not None:
            temp = np.prod(list(self._state.values()), dtype=int)
            assert dev_num % temp == 0
            self._duplicate = dev_num // temp
        else:
            self._duplicate = None
        self._order = None
        self._valid_state = False
        self._valid_all = False

    def get(self):
        return self._state, self._duplicate

    def get_all(self):
        return self._state, self._duplicate, self._order

    def is_dist(self):
        return self._device_num > 1

    @ property
    def state(self):
        return self._state

    @ property
    def duplicate(self):
        return self._duplicate

    @ property
    def order(self):
        return self._order

    def set_state(self, state=None, duplicate=None):
        if state is not None:
            if not isinstance(state, dict):
                state = {i: v for i, v in enumerate(state) if v != 1}
            else:
                state = {i: v for i, v in sorted(state.items()) if v != 1}
            assert self._state in (state, None)
            self._state = state
            if self._device_num is not None:
                temp = np.prod(list(self._state.values()), dtype=int)
                assert self._device_num % temp == 0
                temp_duplicate = self._device_num // temp
                assert self._duplicate in (temp_duplicate, None)
                self._duplicate = temp_duplicate
        if duplicate is not None:
            assert self._duplicate in (duplicate, None)
            self._duplicate = duplicate

    def set_order(self, order=None):
        if order is not None:
            assert self._order in (order, None)
            self._order = order

    def set_one(self):
        self._state = {}
        self._duplicate = 1
        self._order = None
        self._device_num = 1
        self._valid_state = True
        self._valid_all = True

    def copy_state_from(self, other):
        self.set_state(other.state, other.duplicate)

    def copy_order_from(self, other):
        self.set_order(other.order)

    def copy_from(self, other, copy_order):
        if copy_order:
            self.copy_order_from(other)
        else:
            self.copy_state_from(other)

    def valid_state(self):
        if self._valid_state:
            return True
        if self._state is None or self._duplicate is None:
            return False
        else:
            self._valid_state = True
            if self._device_num is None:
                self._device_num = np.prod(
                    list(self._state.values()), dtype=int) * self._duplicate
            return True

    def valid_all(self):
        if self._valid_all:
            # the one-device case has already set _valid_all to True
            return True
        if not self._valid_state or self._order is None:
            return False
        else:
            self._valid_all = True
            return True

    def valid(self, include_order):
        if include_order:
            return self.valid_all()
        else:
            return self.valid_state()

    def check_state(self, max_dim, check_order):
        # only allow dimensions lower than max_dim to split
        if check_order:
            assert all([o < max_dim for o in self.order])
        else:
            assert all([o < max_dim for o in self.state])

    def map_dev_to_index(self, global_index):
        cur_state_index = {}
        for cur_order in self._order[::-1]:
            if cur_order < 0:
                global_index //= self._duplicate
            else:
                ts = self._state[cur_order]
                cur_state_index[cur_order] = global_index % ts
                global_index //= ts
        return cur_state_index

    def get_loop_sizes(self):
        loop_sizes = [1]
        for rord in self._order[::-1]:
            temp_size = loop_sizes[0] * \
                self._duplicate if rord < 0 else loop_sizes[0] * \
                self._state[rord]
            loop_sizes.insert(0, temp_size)
        loop_sizes.pop(0)
        return loop_sizes

    def __repr__(self):
        return '(' + str(self.state) + ', ' + str(self.duplicate) + ', ' + str(self.order) + ')'


def get_current_context():
    return _default_ctx_stack.peek()


@ contextlib.contextmanager
def context(ctx):
    try:
        ctx = DeviceGroup(ctx)
        _default_ctx_stack.push(ctx)
        yield ctx
    finally:
        _default_ctx_stack.pop()


def check_worker(ctx):
    # if the context is GPU or is a tuple (which means model parallel),
    # we regard it as a worker
    return isinstance(ctx, tuple) or is_gpu_ctx(ctx)


def get_launch_config_by_traverse_nodes(node_list, default_ctx):
    node_strategy = dict()
    devices = set()
    for ctx in default_ctx:
        if isinstance(ctx, tuple):
            devices.update(ctx)
        else:
            devices.add(ctx)
    launchPS = default_ctx.server_num > 0
    launchMPI = (not launchPS) and default_ctx.worker_num > 1
    nrank = default_ctx.worker_num
    for node in node_list:
        traverse_dfs(node, node_strategy, devices, nrank)
    launchPS = launchPS or any([x == 'PS' for x in node_strategy.values()])
    launchMPI = launchMPI or any(
        [x == 'AllReduce' for x in node_strategy.values()])
    return launchMPI, launchPS, node_strategy, devices


def traverse_dfs(node, node_strategy, devices, nrank):
    if node in node_strategy:
        return
    strategy = None
    if node.raw_ctx is not None and node.raw_ctx.server_num > 0 and node.raw_ctx.worker_num > 0:
        strategy = 'PS'
    elif node.raw_ctx is not None and node.raw_ctx.worker_num > 1:
        strategy = 'AllReduce'
    node_strategy[node] = strategy
    for ctx in node.raw_ctx:
        if isinstance(ctx, tuple):
            devices.update(ctx)
        else:
            devices.add(ctx)
    local_nrank = nrank if node.raw_ctx is None else node.raw_ctx.worker_num
    assert local_nrank in (
        0, nrank), 'Number of workers not consist: (%d, %d).' % (local_nrank, nrank)
    for n in node.inputs:
        traverse_dfs(n, node_strategy, devices, nrank)


def infer_states(node_list):
    from .dataloader import DataloaderOp
    from .optimizer import OptimizerOp
    from .gpu_ops.Dispatch import DispatchOp, DispatchGradientOp
    from .gpu_ops.Variable import PlaceholderOp

    def get_node_count(node):
        if node in visited:
            return
        visited[node] = True
        nonlocal cnt
        single = False
        if not isinstance(node, (DataloaderOp, OptimizerOp, DispatchOp, DispatchGradientOp)):
            node_cur_state_map[node] = NodeStatus(
                dev_num=node.raw_ctx.mp_device_num)
            node.get_default_state(
                node_cur_state_map[node], enforce_order=False)
            cnt += 1
            single = not node.raw_ctx.is_mp()
            if single:
                node_cur_state_map[node].set_one()
        elif isinstance(node, DispatchOp):
            node_tar_state_map[node.inputs[0]] = NodeStatus(node.parts)
        elif isinstance(node, DispatchGradientOp):
            node_tar_state_map[node.inputs[0]] = NodeStatus()
        for n in node.inputs:
            get_node_count(n)
            if single and isinstance(n, (DispatchOp, DispatchGradientOp)):
                node_tar_state_map[n.inputs[0]].set_state({}, 1)

    def infer_node_states(node, infer_order):
        if node in visited:
            return
        visited[node] = True
        if isinstance(node, DataloaderOp):
            pass
        elif isinstance(node, OptimizerOp):
            for n in node.inputs:
                infer_node_states(n, infer_order)
        elif isinstance(node, DispatchOp):
            real_node = node.inputs[0]
            infer_node_states(real_node, infer_order)
            node_tar_state_map[real_node].valid(infer_order)
        elif isinstance(node, DispatchGradientOp):
            real_node = node.inputs[0]
            infer_node_states(real_node, infer_order)
            infer_node_states(node.inputs[1], infer_order)
            node_tar_state_map[real_node].copy_from(
                node_cur_state_map[node.inputs[1]], infer_order)
            node_tar_state_map[real_node].valid(infer_order)
        else:
            input_statuses = []
            if node.raw_ctx.is_mp():
                if isinstance(node, PlaceholderOp):
                    # in this case the node is initialized in mp
                    node_cur_state_map[node].copy_from(
                        node_tar_state_map[node], infer_order)
                else:
                    if infer_order:
                        nonlocal chance
                        if chance and not node_cur_state_map[node].valid_all():
                            node.get_default_state(
                                node_cur_state_map[node], enforce_order=True)
                            chance = False
                    input_statuses = []
                    for n in node.inputs:
                        if isinstance(n, (DispatchOp, DispatchGradientOp)):
                            node_status = node_tar_state_map[n.inputs[0]]
                        else:
                            node_status = node_cur_state_map[n]
                        input_statuses.append(node_status)
                    node.backward_deduce_states(
                        node_cur_state_map[node], input_statuses, deduce_order=infer_order)
                    for n in node.inputs:
                        infer_node_states(n, infer_order)
                    node.forward_deduce_states(
                        input_statuses, node_cur_state_map[node], deduce_order=infer_order)
            else:
                for n in node.inputs:
                    infer_node_states(n, infer_order)
            if node_cur_state_map[node].valid(infer_order):
                valid_nodes.add(node)

    visited = {}
    node_cur_state_map = {}  # save nodes' current state
    node_tar_state_map = {}  # save nodes' target state
    cnt = 0
    valid_nodes = set()
    for node in node_list:
        get_node_count(node)
    # first infer state and duplicate
    last_cnt = 0
    while True:
        visited = {}
        for node in node_list:
            infer_node_states(node, infer_order=False)
        valid_cnt = len(valid_nodes)
        if last_cnt == cnt:
            # here we use another loop to ensure node_tar_state_map
            break
        assert valid_cnt > last_cnt, "Not enough information for model parallel."
        last_cnt = valid_cnt
    chance = False
    last_cnt = 0
    valid_nodes = set()
    # next infer order
    while True:
        visited = {}
        for node in node_list:
            infer_node_states(node, infer_order=True)
        valid_cnt = len(valid_nodes)
        if last_cnt == cnt:
            # here we use another loop to ensure node_tar_state_map
            break
        if valid_cnt == last_cnt:
            chance = True
        last_cnt = valid_cnt

    return node_cur_state_map, node_tar_state_map


def assign_context_by_traverse_nodes(node_list, ctx, mpi_comm, p2p_stream):
    from .dataloader import DataloaderOp
    from .optimizer import OptimizerOp
    from .gpu_ops.PipelineSend import pipeline_send_op
    from .gpu_ops.PipelineReceive import pipeline_receive_op
    from .gpu_ops.Variable import PlaceholderOp
    from .gpu_ops.Dispatch import DispatchOp, DispatchGradientOp
    from .gpu_ops.Concatenate import concatenate_op
    from .gpu_ops.Split import split_op
    from .gpu_ops.Sum import sum_op
    from .gpu_ops.executor import new_group_comm

    def receiving(key, from_ctx):
        if from_ctx == ctx:
            return self_buffer[key]
        else:
            hostname = from_ctx.hostname
            target_id = from_ctx.device_id
            result = pipeline_receive_op(mpi_comm.getRankFromDevice(
                hostname, target_id), mpi_comm, stream=p2p_stream, ctx=ctx)
            layer_indices[result] = layer_id
            return result

    def sending(key, node, to_ctx):
        if ctx == to_ctx:
            self_buffer[key] = node
        else:
            hostname = to_ctx.hostname
            target_id = to_ctx.device_id
            target_rank = mpi_comm.getRankFromDevice(
                hostname, target_id)
            result = pipeline_send_op(
                node, target_rank, mpi_comm, stream=p2p_stream, ctx=ctx)
            layer_indices[result] = layer_id
            my_eval_nodes.append(result)

    def receive_model_parallel(prev_input, node):
        # assert dp_index_map[prev_input] < 0 and dp_index_map[node] >= 0
        dev_pos = dp_index_map[node]
        if not isinstance(prev_input.raw_ctx.workers[dev_pos], tuple):
            # here we receive from a node on one device dispatching to many
            # in this case current node MUST have mp_index, and the split will be handled in sending
            assert mp_index_map[node] >= 0, 'Here only support 1 to N.'
            if prev_input not in recv_src:
                recv_src[prev_input] = receiving(
                    prev_input, prev_input.raw_ctx.workers[dev_pos])
            return recv_src[prev_input]
        elif not isinstance(node.raw_ctx.workers[dev_pos], tuple):
            # here we receive from a node on multiple devices
            # in this case current node MUST NOT have mp_index, and handle the combination
            target = node_tar_state_map[prev_input]
            assert mp_index_map[node] < 0 and not target.is_dist(
            ), 'Here only support N to 1.'
            if prev_input not in recv_src:
                device_index = 0

                def make_comb(depth):
                    if depth == len(cur_order):
                        nonlocal device_index
                        result = receiving(prev_input, devices[device_index])
                        device_index += 1
                    else:
                        cur_dim = cur_order[depth]
                        if cur_dim < 0:
                            if cur_duplicate == 1:
                                result = make_comb(depth + 1)
                            else:
                                result = sum_op([make_comb(depth + 1)
                                                 for _ in range(cur_duplicate)], ctx=ctx)
                            layer_indices[result] = layer_id
                        else:
                            if cur_state[cur_dim] == 1:
                                result = make_comb(depth + 1)
                            else:
                                result = concatenate_op(
                                    [make_comb(depth + 1) for _ in range(cur_state[cur_dim])], axis=cur_dim, ctx=ctx)
                            layer_indices[result] = layer_id
                    return result
                devices = prev_input.raw_ctx.workers[dev_pos]
                cur_state, cur_duplicate, cur_order = node_cur_state_map[prev_input].get_all(
                )
                recv_src[prev_input] = make_comb(0)
                assert device_index == len(prev_input.raw_ctx.workers[dev_pos])
            return recv_src[prev_input]
        else:
            # here the prev input and the node are both in model parallel, with different states
            assert mp_index_map[node] >= 0, 'Here only support N to N.'
            if prev_input not in recv_src:
                prev_ns = node_cur_state_map[prev_input]
                target_ns = node_tar_state_map[prev_input]
                prev_state, prev_duplicate, prev_order = prev_ns.get_all()
                target_state = target_ns.state
                loop_sizes = prev_ns.get_loop_sizes()
                cur_state_index = target_ns.map_dev_to_index(
                    mp_index_map[node])
                device_index = 0

                def cross_receive(depth):
                    nonlocal device_index
                    if depth == len(prev_order):
                        res = receiving(prev_input, devices[device_index])
                        device_index += 1
                    else:
                        cur_dim = prev_order[depth]
                        if cur_dim < 0:
                            if prev_duplicate == 1:
                                res = cross_receive(depth+1)
                            else:
                                res = sum_op([cross_receive(depth+1)
                                              for _ in range(prev_duplicate)], ctx=ctx)
                            layer_indices[res] = layer_id
                        else:
                            tar_st = target_state.get(cur_dim, 1)
                            cur_st = cur_state_index.get(cur_dim, 0)
                            if prev_state[cur_dim] % tar_st == 0:
                                # at `cur_dim` dimension we need to concat some inputs
                                multiple = prev_state[cur_dim] // tar_st
                                device_index += cur_st * \
                                    multiple * loop_sizes[depth]
                                if multiple == 1:
                                    res = cross_receive(depth+1)
                                else:
                                    res = concatenate_op(
                                        [cross_receive(depth+1) for _ in range(multiple)], axis=cur_dim, ctx=ctx)
                                layer_indices[res] = layer_id
                                device_index += (tar_st - 1 - cur_st) * \
                                    multiple * loop_sizes[depth]
                            elif tar_st % prev_state[cur_dim] == 0:
                                # at `cur_dim` dimension we need to specify one input
                                multiple = tar_st // prev_state[cur_dim]
                                device_index += cur_st // multiple * \
                                    loop_sizes[depth]
                                res = cross_receive(depth+1)
                                device_index += (tar_st - 1 -
                                                 cur_st) // multiple * loop_sizes[depth]
                            else:
                                assert False, 'The dispatch state (%d, %d) at dimension %d is invalid.' % (
                                    prev_state[cur_dim], tar_st, cur_dim)
                    return res
                devices = prev_input.raw_ctx.workers[dev_pos]
                recv_src[prev_input] = cross_receive(0)
                assert device_index == len(devices)
            return recv_src[prev_input]

    def send_model_parallel(prev_input, node):
        # assert dp_index_map[prev_input] >= 0 and dp_index_map[node] < 0
        dev_pos = dp_index_map[prev_input]
        if not isinstance(prev_input.raw_ctx.workers[dev_pos], tuple):
            # here we send from a node on one device dispatching to many nodes
            # in this case current node MUST have mp_index, and the split will be handled in sending
            assert mp_index_map[prev_input] < 0, 'Here only support 1 to N.'
            device_index = 0

            devices = node.raw_ctx.workers[dev_pos]
            key = (prev_input, devices)
            if key not in send_dst:
                send_dst[key] = True

                def make_split(cur_state, depth):
                    if len(target_order) == depth:
                        nonlocal device_index
                        if target_ns.is_dist():
                            keys = list(target_state.keys())
                            indices = [cur_state[k] for k in keys]
                            splits = [target_state[k] for k in keys]
                            cur_node = split_op(
                                prev_input, keys, indices, splits, ctx=ctx)
                            layer_indices[cur_node] = layer_id
                        else:
                            cur_node = prev_input
                        sending(prev_input, cur_node, devices[device_index])
                        device_index += 1
                    else:
                        cur_dim = target_order[depth]
                        if cur_dim < 0:
                            for _ in range(target_duplicate):
                                make_split(cur_state, depth + 1)
                        else:
                            for ts in range(target_state[cur_dim]):
                                cur_state[cur_dim] = ts
                                make_split(cur_state, depth + 1)
                target_ns = node_tar_state_map[prev_input]
                target_state, target_duplicate, target_order = target_ns.get_all()
                make_split({}, 0)
                assert device_index == len(
                    node.raw_ctx.workers[dev_pos])
        elif not isinstance(node.raw_ctx.workers[dev_pos], tuple):
            # here we send from a node on multiple devices to one node
            # in this case current node MUST NOT have mp_index, and the combination will be handled in receiving
            target = node_tar_state_map[prev_input]
            assert mp_index_map[prev_input] >= 0 and not target.is_dist(
            ), 'Here only support N to 1.'
            device = node.raw_ctx.workers[dev_pos]
            key = (prev_input, device)
            if key not in send_dst:
                send_dst[key] = True
                sending(prev_input, prev_input, device)
        else:
            # here the prev input and the node are both in model parallel, with different states
            assert mp_index_map[prev_input] >= 0, 'Here only support N to N.'
            devices = node.raw_ctx.workers[dev_pos]
            key = (prev_input, devices)
            if key not in send_dst:
                send_dst[key] = True
                prev_ns = node_cur_state_map[prev_input]
                target_ns = node_tar_state_map[prev_input]
                prev_state = prev_ns.state
                target_state, target_duplicate, target_order = target_ns.get_all()
                cur_state_index = prev_ns.map_dev_to_index(
                    mp_index_map[prev_input])
                loop_sizes = target_ns.get_loop_sizes()
                device_index = 0

                def cross_send(split_cur_state, split_target_state, depth, need_split):
                    nonlocal device_index
                    if depth == len(target_order):
                        if need_split:
                            keys = list(split_target_state.keys())
                            indices = [split_cur_state[k] for k in keys]
                            splits = [split_target_state[k] for k in keys]
                            cur_node = split_op(
                                prev_input, keys, indices, splits, ctx=ctx)
                            layer_indices[cur_node] = layer_id
                        else:
                            cur_node = prev_input
                        sending(prev_input, cur_node, devices[device_index])
                        device_index += 1
                    else:
                        cur_dim = target_order[depth]
                        if cur_dim < 0:
                            for _ in range(target_duplicate):
                                cross_send(
                                    split_cur_state, split_target_state, depth+1, need_split)
                        else:
                            pre_st = prev_state.get(cur_dim, 1)
                            cur_st = cur_state_index.get(cur_dim, 0)
                            if pre_st % target_state[cur_dim] == 0:
                                # at `cur_dim` dimension we need to send one output
                                multiple = pre_st // target_state[cur_dim]
                                device_index += cur_st // multiple * \
                                    loop_sizes[depth]
                                split_cur_state[cur_dim] = 0
                                split_target_state[cur_dim] = 1
                                cross_send(split_cur_state,
                                           split_target_state, depth+1, need_split)
                                device_index += (pre_st - 1 -
                                                 cur_st) // multiple * loop_sizes[depth]
                            elif target_state[cur_dim] % pre_st == 0:
                                # at `cur_dim` dimension we need to split and send some outputs
                                multiple = target_state[cur_dim] // pre_st
                                device_index += cur_st * \
                                    multiple * loop_sizes[depth]
                                for index in range(multiple):
                                    split_cur_state[cur_dim] = index
                                    split_target_state[cur_dim] = multiple
                                    cross_send(split_cur_state,
                                               split_target_state, depth+1, True)
                                device_index += (pre_st - 1 -
                                                 cur_st) * multiple * loop_sizes[depth]
                            else:
                                assert False, 'The dispatch state (%d, %d) at dimension %d is invalid.' % (
                                    pre_st, target_state[cur_dim], cur_dim)
                cross_send({}, {}, 0, False)
                assert device_index == len(devices)

    def assign_ctx(node):
        nonlocal layer_id
        if node in dp_index_map:
            return
        mp_index_map[node] = -1
        dp_index_map[node] = -1
        if isinstance(node, DataloaderOp):
            layer_indices[node] = layer_id
            layer_id += 1
            return
        elif isinstance(node, OptimizerOp):
            nonlocal opt
            assert opt is None, 'Multiple optimizer is invalid.'
            opt = node
            for n in node.inputs:
                assign_ctx(n)
            grads = []
            original_params = node.optimizer.params
            for ind, param in enumerate(original_params):
                ori_grad = node.inputs[ind]
                if mp_index_map.get(param, -1) >= 0:
                    # consider the situation that parameter is on model parallel devices
                    real_grad = ori_grad.inputs[0]
                    assert real_grad.raw_ctx == param.raw_ctx
                    grads.append(real_grad)
                else:
                    if param in trainable_params:
                        new_grad = receive_model_parallel(ori_grad.inputs[0], param) if isinstance(
                            ori_grad, (DispatchOp, DispatchGradientOp)) else ori_grad
                        grads.append(new_grad)
                    elif isinstance(ori_grad, (DispatchOp, DispatchGradientOp)):
                        real_input = ori_grad.inputs[0]
                        my_pos = dp_index_map[real_input]
                        if my_pos >= 0:
                            send_model_parallel(ori_grad.inputs[0], param)
                layer_id += 2
            if trainable_params:
                # indices = [original_params.index(param) for param in trainable_params]
                node.optimizer.params = trainable_params
                # grads = [node.inputs[index] for index in indices]
                node.inputs = grads
                node.ctx = ctx
                my_eval_nodes.append(node)
                layer_indices[node] = layer_id
                for param in trainable_params:
                    # here handle the nodes that need allreduce
                    if param.raw_ctx.server_num == 0:
                        allreduce_devices = None
                        if mp_index_map[param] < 0 and param.raw_ctx.worker_num > 1:
                            allreduce_devices = param.raw_ctx
                        elif mp_index_map[param] >= 0:
                            target_state, target_duplicate, target_order = \
                                node_tar_state_map[param].get_all()
                            if -1 in target_order:
                                mp_index = mp_index_map[param]
                                interval = 1
                                allreduce_devices = []
                                dup_dim = target_order.index(-1)
                                for cur_order in target_order[dup_dim+1:]:
                                    interval *= target_state[cur_order]
                                macro_interval = interval * target_duplicate
                                start = mp_index - mp_index % macro_interval + mp_index % interval
                                for rc in range(param.raw_ctx.worker_num):
                                    for ind in range(start, start + interval * target_duplicate, interval):
                                        allreduce_devices.append(
                                            param.raw_ctx[rc][ind])
                                allreduce_devices = None if len(
                                    allreduce_devices) <= 1 else DeviceGroup(allreduce_devices)
                        if allreduce_devices is not None:
                            allreduce_devices = allreduce_devices.get_sorted()
                            if allreduce_devices not in comm_groups:
                                if len(allreduce_devices) == mpi_comm.nrank:
                                    comm_groups[allreduce_devices] = mpi_comm
                                else:
                                    comm_groups[allreduce_devices] = new_group_comm(
                                        allreduce_devices)
                            param_allreduce_group[param] = comm_groups[allreduce_devices]
        elif isinstance(node, DispatchOp):
            real_node = node.inputs[0]
            assign_ctx(real_node)
        elif isinstance(node, DispatchGradientOp):
            real_node = node.inputs[0]
            assign_ctx(real_node)
            assign_ctx(node.inputs[1])
        else:
            # now we only support SAME model parallel in data parallel
            # and 1 context can only appear once
            mp_index = -1
            dp_index = -1
            for i, c in enumerate(node.raw_ctx.workers):
                if isinstance(c, tuple):
                    if ctx in c:
                        mp_index = c.index(ctx)
                        dp_index = i
                elif ctx == c:
                    dp_index = i
            mp_index_map[node] = mp_index
            dp_index_map[node] = dp_index
            for i, n in enumerate(node.inputs):
                if isinstance(n, DataloaderOp):
                    if n not in dp_index_map:
                        layer_indices[n] = layer_id
                        layer_id += 1
                        dp_index_map[n] = -1
                    continue
                assign_ctx(n)
                if isinstance(n, (DispatchOp, DispatchGradientOp)):
                    real_node = n.inputs[0]
                    if isinstance(real_node, PlaceholderOp) and mp_index_map[real_node] >= 0:
                        node_status = node_tar_state_map[real_node]
                        real_node.reshape_in_mp(node_status.map_dev_to_index(
                            mp_index_map[real_node]), node_status.state)
            for i, n in enumerate(node.inputs):
                if isinstance(n, DataloaderOp):
                    if dp_index >= 0 and n in node_list and n not in my_eval_nodes:
                        my_eval_nodes.append(n)
                    continue
                # we assume that in pipeline + data parallel mode,
                # devices number of each stage is equal
                # the device in correspondent place will communicate with each other
                assert node.raw_ctx.worker_num == n.raw_ctx.worker_num, \
                    'In pipeline + data parallel, devices number of each stage should be equal!'

                if isinstance(n, (DispatchOp, DispatchGradientOp)):
                    # here in every context each device appear only once
                    # TODO: consider whether or not release the constraint above?
                    real_input = n.inputs[0]
                    if dp_index >= 0 and dp_index_map[real_input] < 0:
                        node.inputs[i] = receive_model_parallel(
                            real_input, node)
                    elif dp_index < 0 and dp_index_map[real_input] >= 0:
                        send_model_parallel(real_input, node)
                    elif dp_index >= 0 and dp_index_map[real_input] >= 0:
                        if isinstance(real_input, PlaceholderOp):
                            assert real_input.raw_ctx == node.raw_ctx
                            node.inputs[i] = real_input
                        else:
                            send_model_parallel(real_input, node)
                            node.inputs[i] = receive_model_parallel(
                                real_input, node)
                else:
                    assert mp_index == mp_index_map[n]
                    if mp_index < 0:
                        # handle receiving
                        if dp_index >= 0 and dp_index != dp_index_map[n]:
                            my_pos = dp_index
                            if n not in recv_src:
                                recv_src[n] = receiving(
                                    n, n.raw_ctx.workers[my_pos])
                            node.inputs[i] = recv_src[n]
                        # handle sending
                        if dp_index_map[n] >= 0 and dp_index != dp_index_map[n]:
                            my_pos = dp_index_map[n]
                            device = node.raw_ctx.workers[my_pos]
                            key = (n, device)
                            if key not in send_dst:
                                send_dst[key] = True
                                sending(n, n, device)
                    else:
                        # here in the same model parallel
                        assert node.raw_ctx == n.raw_ctx
                layer_id += 1

            layer_indices[node] = layer_id
            layer_id += 1

            if dp_index >= 0:
                node.ctx = ctx
                if node in node_list:
                    my_eval_nodes.append(node)
                if isinstance(node, PlaceholderOp) and node.trainable:
                    trainable_params.append(node)

    opt = None
    trainable_params = []
    comm_groups = {}
    param_allreduce_group = {}
    send_dst = {}
    recv_src = {}
    self_buffer = {}  # send and receive from self device
    mp_index_map = {}  # model parallel index
    dp_index_map = {}  # data parallel index
    layer_indices = {}
    layer_id = 0
    my_eval_nodes = []
    node_cur_state_map, node_tar_state_map = infer_states(node_list)
    for node in node_list:
        assign_ctx(node)

    return my_eval_nodes, param_allreduce_group, layer_indices
