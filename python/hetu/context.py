from .ndarray import cpu, gpu, rcpu, rgpu, DLContext, is_gpu_ctx
import contextlib
import re
import numpy as np


class DeviceGroup(object):
    def __init__(self, ctxs):
        self._contexts = self.parse_contexts(ctxs)
        self.get_servers_n_workers()

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

    def __getitem__(self, key):
        return self._contexts[key]

    def __iter__(self):
        return iter(self._contexts)

    def __len__(self):
        return len(self._contexts)

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
    def __init__(self, state, duplicate=None, order=None):
        self._state = state
        self._duplicate = duplicate
        self._order = order
        self._defaulted = False
        self.try_get_device_num()

    @ classmethod
    def from_other(cls, other):
        if other is None:
            return cls(None, None, None)
        else:
            return cls(other._state, other._duplicate, other._order)

    def get_default(self):
        # This function is VERY STRONG! Please use after any set_attr.
        self._defaulted = True
        if self._duplicate is None:
            self._duplicate = 1
        if self._order is None:
            self._order = (-1,) + tuple(range(len(self._state)))
        self.try_get_device_num()
        return self._state, self._duplicate, self._order

    def is_dist(self):
        return not (self._state is None or all([x == 1 for x in self._state]))

    @ property
    def state(self):
        return self._state

    @ property
    def duplicate(self):
        return self._duplicate

    @ property
    def order(self):
        return self._order

    def set_attr(self, duplicate, order):
        if self._defaulted:
            assert self._duplicate == duplicate
            assert self._order == order
        else:
            self._duplicate = duplicate
            self._order = order

    def map_dev_to_index(self, global_index):
        cur_state_index = self.make_empty_state()
        for cur_order in self._order[::-1]:
            if cur_order < 0:
                global_index //= self._duplicate
            else:
                ts = self._state[cur_order]
                cur_state_index[cur_order] = global_index % ts
                global_index //= ts
        return cur_state_index

    def make_empty_state(self):
        return [0 for _ in range(len(self._state))]

    def try_get_device_num(self):
        self._device_num = None if self._duplicate is None or self._state is None else np.prod(
            self._state, dtype=int) * self._duplicate

    def check_devices(self, devices):
        assert self._device_num == len(devices)

    def get_loop_sizes(self):
        loop_sizes = [1]
        for rord in self._order[::-1]:
            temp_size = loop_sizes[0] * \
                self._duplicate if rord < 0 else loop_sizes[0] * \
                self._state[rord]
            loop_sizes.insert(0, temp_size)
        loop_sizes.pop(0)
        return loop_sizes


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


def assign_context_by_traverse_nodes(node_list, ctx, mpi_comm, p2p_stream):
    from .dataloader import DataloaderOp
    from .optimizer import OptimizerOp
    from .gpu_ops.PipelineSend import pipeline_send_op
    from .gpu_ops.PipelineReceive import pipeline_receive_op
    from .gpu_ops.Variable import PlaceholderOp
    from .gpu_ops.Dispatch import DispatchOp, DispatchGradientOp
    from .gpu_ops.Concat import concat_op
    from .gpu_ops.Split import split_op
    from .gpu_ops.AddElewise import add_op
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
                        result = make_comb(depth + 1)
                        cur_dim = cur_order[depth]
                        if cur_dim < 0:
                            for _ in range(1, cur_duplicate):
                                result = add_op(result, make_comb(
                                    depth + 1), ctx=ctx)
                                layer_indices[result] = layer_id
                        else:
                            for _ in range(1, cur_state[cur_dim]):
                                result = concat_op(result, make_comb(
                                    depth + 1), axis=cur_dim, ctx=ctx)
                                layer_indices[result] = layer_id
                    return result
                devices = prev_input.raw_ctx.workers[dev_pos]
                cur_state, cur_duplicate, cur_order = node_cur_state_map[prev_input].get_default(
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
                prev_state, prev_duplicate, prev_order = prev_ns.get_default()
                target_state, target_duplicate, target_order = target_ns.get_default()
                prev_ns.check_devices(prev_input.raw_ctx[dev_pos])
                target_ns.check_devices(node.raw_ctx[dev_pos])
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
                            res = cross_receive(depth+1)
                            for _ in range(1, prev_duplicate):
                                res = add_op(
                                    res, cross_receive(depth+1), ctx=ctx)
                                layer_indices[res] = layer_id
                        else:
                            if prev_state[cur_dim] % target_state[cur_dim] == 0:
                                # at `cur_dim` dimension we need to concat some inputs
                                multiple = prev_state[cur_dim] // \
                                    target_state[cur_dim]
                                device_index += cur_state_index[cur_dim] * \
                                    multiple * loop_sizes[depth]
                                res = cross_receive(depth+1)
                                for _ in range(1, multiple):
                                    res = concat_op(
                                        res, cross_receive(depth+1), axis=cur_dim, ctx=ctx)
                                    layer_indices[res] = layer_id
                                device_index += (target_state[cur_dim] - 1 -
                                                 cur_state_index[cur_dim]) * multiple * loop_sizes[depth]
                            elif target_state[cur_dim] % prev_state[cur_dim] == 0:
                                # at `cur_dim` dimension we need to specify one input
                                multiple = target_state[cur_dim] // prev_state[cur_dim]
                                device_index += cur_state_index[cur_dim] // multiple * \
                                    loop_sizes[depth]
                                res = cross_receive(depth+1)
                                device_index += (target_state[cur_dim] - 1 -
                                                 cur_state_index[cur_dim]) // multiple * loop_sizes[depth]
                            else:
                                assert False, 'The dispatch state (%d, %d) at dimension %d is invalid.' % (
                                    prev_state[cur_dim], target_state[cur_dim], cur_dim)
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
                            cur_node = split_op(prev_input, list(range(len(target_state))), list(
                                cur_state), list(target_state), ctx=ctx)
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
                target_state, target_duplicate, target_order = target_ns.get_default()
                make_split(target_ns.make_empty_state(), 0)
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
                prev_state, prev_duplicate, prev_order = prev_ns.get_default()
                target_state, target_duplicate, target_order = target_ns.get_default()
                prev_ns.check_devices(prev_input.raw_ctx[dev_pos])
                target_ns.check_devices(node.raw_ctx[dev_pos])
                cur_state_index = prev_ns.map_dev_to_index(
                    mp_index_map[prev_input])
                loop_sizes = target_ns.get_loop_sizes()
                device_index = 0

                def cross_send(split_cur_state, split_target_state, depth, need_split):
                    nonlocal device_index
                    if depth == len(target_order):
                        if need_split:
                            cur_node = split_op(prev_input, list(range(len(split_target_state))), list(
                                split_cur_state), list(split_target_state), ctx=ctx)
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
                            if prev_state[cur_dim] % target_state[cur_dim] == 0:
                                # at `cur_dim` dimension we need to send one output
                                multiple = prev_state[cur_dim] // \
                                    target_state[cur_dim]
                                device_index += cur_state_index[cur_dim] // multiple * \
                                    loop_sizes[depth]
                                split_cur_state[cur_dim] = 0
                                split_target_state[cur_dim] = 1
                                cross_send(split_cur_state,
                                           split_target_state, depth+1, need_split)
                                device_index += (prev_state[cur_dim] - 1 -
                                                 cur_state_index[cur_dim]) // multiple * loop_sizes[depth]
                            elif target_state[cur_dim] % prev_state[cur_dim] == 0:
                                # at `cur_dim` dimension we need to split and send some outputs
                                multiple = target_state[cur_dim] // prev_state[cur_dim]
                                device_index += cur_state_index[cur_dim] * \
                                    multiple * loop_sizes[depth]
                                for index in range(multiple):
                                    split_cur_state[cur_dim] = index
                                    split_target_state[cur_dim] = multiple
                                    cross_send(split_cur_state,
                                               split_target_state, depth+1, True)
                                device_index += (prev_state[cur_dim] - 1 -
                                                 cur_state_index[cur_dim]) * multiple * loop_sizes[depth]
                            else:
                                assert False, 'The dispatch state (%d, %d) at dimension %d is invalid.' % (
                                    prev_state[cur_dim], target_state[cur_dim], cur_dim)
                split_cur_state = prev_ns.make_empty_state()
                split_target_state = prev_ns.make_empty_state()
                cross_send(split_cur_state, split_target_state, 0, False)
                assert device_index == len(devices)

    def assign_ctx(node):
        nonlocal layer_id
        if node in dp_index_map:
            return
        mp_index_map[node] = -1
        dp_index_map[node] = -1
        if isinstance(node, DataloaderOp):
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
                            allreduce_devices = []
                            target_state, target_duplicate, target_order = \
                                node_tar_state_map[param].get_default()
                            mp_index = mp_index_map[param]
                            interval = 1
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
                                comm_groups[allreduce_devices] = new_group_comm(
                                    allreduce_devices)
                            param_allreduce_group[param] = comm_groups[allreduce_devices]
        elif isinstance(node, DispatchOp):
            real_node = node.inputs[0]
            assign_ctx(real_node)
            node_tar_state_map[real_node] = NodeStatus(node.parts)
        elif isinstance(node, DispatchGradientOp):
            real_node = node.inputs[0]
            assign_ctx(real_node)
            assign_ctx(node.inputs[1])
            node_tar_state_map[real_node] = NodeStatus.from_other(
                node_cur_state_map.get(node.inputs[1], None))
        else:
            # now we only support SAME model parallel in data parallel
            # and 1 context can only appear once
            mp_index = -1
            dp_index = -1
            need_state_deduction = False
            for i, c in enumerate(node.raw_ctx.workers):
                if isinstance(c, tuple):
                    need_state_deduction = True
                    if ctx in c:
                        mp_index = c.index(ctx)
                        dp_index = i
                elif ctx == c:
                    dp_index = i
            mp_index_map[node] = mp_index
            dp_index_map[node] = dp_index
            input_states = []
            input_duplicates = []
            input_orders = []
            for i, n in enumerate(node.inputs):
                if isinstance(n, DataloaderOp):
                    continue
                assign_ctx(n)
                if isinstance(n, (DispatchOp, DispatchGradientOp)):
                    need_state_deduction = True
                    node_status = node_tar_state_map[n.inputs[0]]
                else:
                    node_status = node_cur_state_map.get(n, None)
                input_states.append(
                    None if node_status is None else node_status.state)
                input_duplicates.append(
                    None if node_status is None else node_status.duplicate)
                input_orders.append(
                    None if node_status is None else node_status.order)
            if need_state_deduction:
                node_cur_state_map[node] = NodeStatus(
                    *node.deduce_states(input_states, input_duplicates, input_orders))
                for i, n in enumerate(node.inputs):
                    if isinstance(n, (DispatchOp, DispatchGradientOp)):
                        real_node = n.inputs[0]
                        node_tar_state_map[real_node].set_attr(
                            input_duplicates[i], input_orders[i])
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
                        assert need_state_deduction
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
    node_cur_state_map = {}  # save nodes' current state
    node_tar_state_map = {}  # save nodes' target state
    layer_indices = {}
    layer_id = 0
    my_eval_nodes = []
    for node in node_list:
        assign_ctx(node)

    return my_eval_nodes, param_allreduce_group, layer_indices
