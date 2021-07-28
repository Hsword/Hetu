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


def get_current_context():
    return _default_ctx_stack.peek()


@contextlib.contextmanager
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


def dev_index_to_states(index, states, duplicate, order):
    cur_states_index = [0 for _ in range(len(states))]
    temp_index = index
    for cur_order in order[::-1]:
        if cur_order < 0:
            temp_index //= duplicate
        else:
            ts = states[cur_order]
            cur_states_index[cur_order] = temp_index % ts
            temp_index //= ts
    return cur_states_index


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

    def receive_model_parallel(prev_input, node):
        # assert dp_index_map[prev_input] < 0 and dp_index_map[node] >= 0
        dev_pos = dp_index_map[node]
        if not isinstance(prev_input.raw_ctx.workers[dev_pos], tuple):
            # here we receive from a node on one device dispatching to many
            # in this case current node MUST have mp_index, and the split will be handled in sending
            assert mp_index_map[node] >= 0, 'Here only support 1 to N.'
            hostname = prev_input.raw_ctx.workers[dev_pos].hostname
            target_id = prev_input.raw_ctx.workers[dev_pos].device_id
            if prev_input not in recv_src:
                recv_src[prev_input] = pipeline_receive_op(mpi_comm.getRankFromDevice(
                    hostname, target_id), mpi_comm, stream=p2p_stream, ctx=ctx)
            return recv_src[prev_input]
        elif not isinstance(node.raw_ctx.workers[dev_pos], tuple):
            # here we receive from a node on multiple devices
            # in this case current node MUST NOT have mp_index, and handle the combination
            target = node_tar_states_map[prev_input]
            assert mp_index_map[node] < 0 and (target is None or all(
                [ts == 1 for ts in target])), 'Here only support N to 1.'
            if prev_input not in recv_src:
                device_index = 0

                def make_comb(depth):
                    if depth == len(cur_order):
                        nonlocal device_index
                        res = pipeline_receive_op(mpi_comm.getRankFromDevice(
                            devices[device_index].hostname, devices[device_index].device_id), mpi_comm, stream=p2p_stream, ctx=ctx)
                        device_index += 1
                        return res
                    else:
                        result = make_comb(depth + 1)
                        cur_dim = cur_order[depth]
                        if cur_dim < 0:
                            for _ in range(1, cur_duplicate):
                                result = add_op(result, make_comb(
                                    depth + 1), ctx=ctx)
                        else:
                            for _ in range(1, cur_states[cur_dim]):
                                result = concat_op(result, make_comb(
                                    depth + 1), axis=cur_dim, ctx=ctx)
                        return result
                devices = prev_input.raw_ctx.workers[dev_pos]
                cur_states = node_cur_states_map[prev_input]
                cur_duplicate = node_cur_duplicate_map.get(prev_input, 1)
                cur_order = node_cur_order_map.get(prev_input, None)
                if cur_order is None:
                    cur_order = (-1,) + tuple(range(len(cur_states)))
                res = make_comb(0)
                assert device_index == len(prev_input.raw_ctx.workers[dev_pos])
                recv_src[prev_input] = res
            return recv_src[prev_input]
        else:
            # here the prev input and the node are both in model parallel, with different states
            assert mp_index_map[node] >= 0, 'Here only support N to N.'
            if prev_input not in recv_src:
                prev_states = node_cur_states_map[prev_input]
                prev_duplicate = node_cur_duplicate_map[prev_input]
                prev_order = node_cur_order_map[prev_input]
                target_states = node_tar_states_map[prev_input]
                target_duplicate = node_tar_duplicate_map[prev_input]
                target_order = node_tar_order_map[prev_input]
                assert np.prod(prev_states, dtype=int) * \
                    prev_duplicate == len(prev_input.raw_ctx[dev_pos])
                assert np.prod(target_states, dtype=int) * \
                    target_duplicate == len(node.raw_ctx[dev_pos])
                cur_states_index = dev_index_to_states(
                    mp_index_map[node], target_states, target_duplicate, target_order)
                device_index = 0
                loop_sizes = [1]
                for pord in prev_order[::-1]:
                    temp_size = loop_sizes[0] * \
                        prev_duplicate if pord < 0 else loop_sizes[0] * \
                        prev_states[pord]
                    loop_sizes.insert(0, temp_size)
                loop_sizes.pop(0)

                def cross_receive(depth):
                    nonlocal device_index
                    if depth == len(prev_order):
                        res = pipeline_receive_op(mpi_comm.getRankFromDevice(
                            devices[device_index].hostname, devices[device_index].device_id), mpi_comm, stream=p2p_stream, ctx=ctx)
                        device_index += 1
                    else:
                        cur_dim = prev_order[depth]
                        if cur_dim < 0:
                            res = cross_receive(depth+1)
                            for _ in range(1, prev_duplicate):
                                res = add_op(
                                    res, cross_receive(depth+1), ctx=ctx)
                        else:
                            if prev_states[cur_dim] % target_states[cur_dim] == 0:
                                # at `cur_dim` dimension we need to concat some inputs
                                multiple = prev_states[cur_dim] // \
                                    target_states[cur_dim]
                                device_index += cur_states_index[cur_dim] * \
                                    multiple * loop_sizes[depth]
                                res = cross_receive(depth+1)
                                for _ in range(1, multiple):
                                    res = concat_op(
                                        res, cross_receive(depth+1), axis=cur_dim, ctx=ctx)
                                device_index += (target_states[cur_dim] - 1 -
                                                 cur_states_index[cur_dim]) * multiple * loop_sizes[depth]
                            elif target_states[cur_dim] % prev_states[cur_dim] == 0:
                                # at `cur_dim` dimension we need to specify one input
                                multiple = target_states[cur_dim] // prev_states[cur_dim]
                                device_index += cur_states_index[cur_dim] // multiple * \
                                    loop_sizes[depth]
                                res = cross_receive(depth+1)
                                device_index += (target_states[cur_dim] - 1 -
                                                 cur_states_index[cur_dim]) // multiple * loop_sizes[depth]
                            else:
                                assert False, 'The dispatch states (%d, %d) at dimension %d is invalid.' % (
                                    prev_states[cur_dim], target_states[cur_dim], cur_dim)
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

                def make_split(cur_states, depth):
                    if len(target_order) == depth:
                        nonlocal device_index
                        hostname = devices[device_index].hostname
                        target_id = devices[device_index].device_id
                        cur_node = prev_input if all([x == 1 for x in target_states]) else split_op(
                            prev_input, list(range(len(target_states))), list(cur_states), list(target_states), ctx=ctx)
                        target_rank = mpi_comm.getRankFromDevice(
                            hostname, target_id)
                        my_eval_nodes.append(pipeline_send_op(
                            cur_node, target_rank, mpi_comm, stream=p2p_stream, ctx=ctx))
                        device_index += 1
                    else:
                        cur_dim = target_order[depth]
                        if cur_dim < 0:
                            for _ in range(target_duplicate):
                                make_split(cur_states, depth + 1)
                        else:
                            for ts in range(target_states[cur_dim]):
                                cur_states[cur_dim] = ts
                                make_split(cur_states, depth + 1)
                cur_states = [0 for _ in range(
                    len(node_tar_states_map[prev_input]))]
                target_states = node_tar_states_map[prev_input]
                target_duplicate = node_tar_duplicate_map[prev_input]
                target_order = node_tar_order_map[prev_input]
                make_split(cur_states, 0)
                assert device_index == len(
                    node.raw_ctx.workers[dev_pos])
        elif not isinstance(node.raw_ctx.workers[dev_pos], tuple):
            # here we send from a node on multiple devices to one node
            # in this case current node MUST NOT have mp_index, and the combination will be handled in receiving
            target = node_tar_states_map[prev_input]
            assert mp_index_map[prev_input] >= 0 and (target is None or all(
                [ts == 1 for ts in target])), 'Here only support N to 1.'
            hostname = node.raw_ctx.workers[dev_pos].hostname
            target_id = node.raw_ctx.workers[dev_pos].device_id
            key = (prev_input, target_id)
            if key not in send_dst:
                send_dst[key] = True
                my_eval_nodes.append(pipeline_send_op(prev_input, mpi_comm.getRankFromDevice(
                    hostname, target_id), mpi_comm, stream=p2p_stream, ctx=ctx))
        else:
            # here the prev input and the node are both in model parallel, with different states
            assert mp_index_map[prev_input] >= 0, 'Here only support N to N.'
            devices = node.raw_ctx.workers[dev_pos]
            key = (prev_input, devices)
            if key not in send_dst:
                send_dst[key] = True
                prev_states = node_cur_states_map[prev_input]
                prev_duplicate = node_cur_duplicate_map[prev_input]
                prev_order = node_cur_order_map[prev_input]
                target_states = node_tar_states_map[prev_input]
                target_duplicate = node_tar_duplicate_map[prev_input]
                target_order = node_tar_order_map[prev_input]
                assert np.prod(prev_states, dtype=int) * \
                    prev_duplicate == len(prev_input.raw_ctx[dev_pos])
                assert np.prod(target_states, dtype=int) * \
                    target_duplicate == len(node.raw_ctx[dev_pos])
                cur_states_index = dev_index_to_states(
                    mp_index_map[prev_input], prev_states, prev_duplicate, prev_order)
                loop_sizes = [1]
                for pord in target_order[::-1]:
                    temp_size = loop_sizes[0] * \
                        target_duplicate if pord < 0 else loop_sizes[0] * \
                        target_states[pord]
                    loop_sizes.insert(0, temp_size)
                loop_sizes.pop(0)
                device_index = 0

                def cross_send(split_cur_states, split_target_states, depth, need_split):
                    nonlocal device_index
                    if depth == len(target_order):
                        hostname = devices[device_index].hostname
                        target_id = devices[device_index].device_id
                        cur_node = prev_input if not need_split else split_op(
                            prev_input, list(range(len(split_target_states))), list(split_cur_states), list(split_target_states), ctx=ctx)
                        target_rank = mpi_comm.getRankFromDevice(
                            hostname, target_id)
                        my_eval_nodes.append(pipeline_send_op(
                            cur_node, target_rank, mpi_comm, stream=p2p_stream, ctx=ctx))
                        device_index += 1
                    else:
                        cur_dim = target_order[depth]
                        if cur_dim < 0:
                            for _ in range(target_duplicate):
                                cross_send(
                                    split_cur_states, split_target_states, depth+1, need_split)
                        else:
                            if prev_states[cur_dim] % target_states[cur_dim] == 0:
                                # at `cur_dim` dimension we need to send one output
                                multiple = prev_states[cur_dim] // \
                                    target_states[cur_dim]
                                device_index += cur_states_index[cur_dim] // multiple * \
                                    loop_sizes[depth]
                                split_cur_states[cur_dim] = 0
                                split_target_states[cur_dim] = 1
                                cross_send(split_cur_states,
                                           split_target_states, depth+1, need_split)
                                device_index += (prev_states[cur_dim] - 1 -
                                                 cur_states_index[cur_dim]) // multiple * loop_sizes[depth]
                            elif target_states[cur_dim] % prev_states[cur_dim] == 0:
                                # at `cur_dim` dimension we need to split and send some outputs
                                multiple = target_states[cur_dim] // prev_states[cur_dim]
                                device_index += cur_states_index[cur_dim] * \
                                    multiple * loop_sizes[depth]
                                for index in range(multiple):
                                    split_cur_states[cur_dim] = index
                                    split_target_states[cur_dim] = multiple
                                    cross_send(split_cur_states,
                                               split_target_states, depth+1, True)
                                device_index += (prev_states[cur_dim] - 1 -
                                                 cur_states_index[cur_dim]) * multiple * loop_sizes[depth]
                            else:
                                assert False, 'The dispatch states (%d, %d) at dimension %d is invalid.' % (
                                    prev_states[cur_dim], target_states[cur_dim], cur_dim)
                split_cur_states = [0 for _ in range(len(prev_states))]
                split_target_states = [0 for _ in range(len(prev_states))]
                cross_send(split_cur_states, split_target_states, 0, False)
                assert device_index == len(devices)

    def assign_ctx(node):
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
            if trainable_params:
                # indices = [original_params.index(param) for param in trainable_params]
                node.optimizer.params = trainable_params
                # grads = [node.inputs[index] for index in indices]
                node.inputs = grads
                node.ctx = ctx
                my_eval_nodes.append(node)
                for param in trainable_params:
                    # here handle the nodes that need allreduce
                    if param.raw_ctx.server_num == 0:
                        allreduce_devices = None
                        if mp_index_map[param] < 0 and param.raw_ctx.worker_num > 1:
                            allreduce_devices = param.raw_ctx
                        elif mp_index_map[param] >= 0:
                            allreduce_devices = []
                            target_states = node_tar_states_map[param]
                            target_duplicate = node_tar_duplicate_map[param]
                            target_order = node_tar_order_map[param]
                            mp_index = mp_index_map[param]
                            interval = 1
                            dup_dim = target_order.index(-1)
                            for cur_order in target_order[dup_dim+1:]:
                                interval *= target_states[cur_order]
                            macro_interval = interval * target_duplicate
                            start = mp_index - mp_index % macro_interval + mp_index % interval
                            for rc in range(param.raw_ctx.worker_num):
                                for ind in range(start, start + interval * target_duplicate, interval):
                                    allreduce_devices.append(
                                        param.raw_ctx[rc][ind])
                            allreduce_devices = None if len(
                                allreduce_devices) <= 1 else DeviceGroup(allreduce_devices)
                        if allreduce_devices is not None:
                            if allreduce_devices not in comm_groups:
                                comm_groups[allreduce_devices] = new_group_comm(
                                    allreduce_devices)
                            param_allreduce_group[param] = comm_groups[allreduce_devices]
        elif isinstance(node, DispatchOp):
            real_node = node.inputs[0]
            assign_ctx(real_node)
            node_tar_states_map[real_node] = node.parts
        elif isinstance(node, DispatchGradientOp):
            real_node = node.inputs[0]
            assign_ctx(real_node)
            assign_ctx(node.inputs[1])
            node_tar_states_map[real_node] = node_cur_states_map.get(
                node.inputs[1], None)
            node_tar_duplicate_map[real_node] = node_cur_duplicate_map.get(
                node.inputs[1], None)
            node_tar_order_map[real_node] = node_cur_order_map.get(
                node.inputs[1], None)
        else:
            # now we only support SAME model parallel in data parallel
            # and 1 context can only appear once
            mp_index = -1
            dp_index = -1
            need_states_deduction = False
            for i, c in enumerate(node.raw_ctx.workers):
                if isinstance(c, tuple):
                    need_states_deduction = True
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
                    need_states_deduction = True
                    input_states.append(node_tar_states_map[n.inputs[0]])
                    input_duplicates.append(
                        node_tar_duplicate_map.get(n.inputs[0], None))
                    input_orders.append(
                        node_tar_order_map.get(n.inputs[0], None))
                else:
                    input_states.append(node_cur_states_map.get(n, None))
                    input_duplicates.append(
                        node_cur_duplicate_map.get(n, None))
                    input_orders.append(node_cur_order_map.get(n, None))
            if need_states_deduction:
                node_cur_states_map[node], node_cur_duplicate_map[node], node_cur_order_map[node] = node.deduce_states(
                    input_states, input_duplicates, input_orders)
                for i, n in enumerate(node.inputs):
                    if isinstance(n, (DispatchOp, DispatchGradientOp)):
                        real_node = n.inputs[0]
                        node_tar_duplicate_map[real_node] = input_duplicates[i]
                        node_tar_order_map[real_node] = input_orders[i]
                        if isinstance(real_node, PlaceholderOp) and mp_index_map[real_node] >= 0:
                            target_states = node_tar_states_map[real_node]
                            real_node.reshape_in_mp(dev_index_to_states(
                                mp_index_map[real_node], target_states, node_tar_duplicate_map[real_node], node_tar_order_map[real_node]), target_states)

            for i, n in enumerate(node.inputs):
                if isinstance(n, DataloaderOp):
                    if dp_index >= 0 and n in node_list and n not in my_eval_nodes:
                        my_eval_nodes.append(n)
                    continue
                # we assume that in model parallel + data parallel mode,
                # devices number of each stage is equal
                # the device in correspondent place will communicate with each other
                # TODO: not support following case: context(1,5) -> context(5,1); context(1,5) -> context(3,1)
                # solution: modify following is_my_node logic to support
                # TODO: not support the case that each process has different group init numbers, since there is an AllGather in mpi_nccl_comm's init
                # solution: modify mpi_nccl_comm class, so that the MPI part only process once while nccl has several groups
                assert node.raw_ctx.worker_num == n.raw_ctx.worker_num, \
                    'In pipeline + data parallel, devices number of each stage should be equal!'

                if isinstance(n, (DispatchOp, DispatchGradientOp)):
                    # here we only allow pipeline + model parallel, which means the devices are all different
                    # TODO: release the constraint above
                    # here in every context each device appear only once
                    # TODO: consider whether or not release the constraint above?
                    # here we only allow one2n/n2one/n2n, can not change from x to y where x != 1 and y != 1 and x != y in dimension-granularity
                    # TODO: consider whether or not release the constraint above? too complex and not realistic!
                    real_input = n.inputs[0]
                    if dp_index >= 0 and dp_index_map[real_input] < 0:
                        node.inputs[i] = receive_model_parallel(
                            real_input, node)
                    elif dp_index < 0 and dp_index_map[real_input] >= 0:
                        send_model_parallel(real_input, node)
                    elif dp_index >= 0 and dp_index_map[real_input] >= 0:
                        # now only allow initialized variables
                        assert real_input.raw_ctx == node.raw_ctx
                        assert isinstance(real_input, PlaceholderOp)
                        node.inputs[i] = real_input
                else:
                    assert mp_index == mp_index_map[n]
                    if mp_index < 0:
                        # handle receiving
                        if dp_index >= 0 and dp_index != dp_index_map[n]:
                            my_pos = dp_index
                            hostname = n.raw_ctx.workers[my_pos].hostname
                            target_id = n.raw_ctx.workers[my_pos].device_id
                            if n not in recv_src:
                                recv_src[n] = pipeline_receive_op(mpi_comm.getRankFromDevice(
                                    hostname, target_id), mpi_comm, stream=p2p_stream, ctx=ctx)
                            node.inputs[i] = recv_src[n]
                        # handle sending
                        if dp_index_map[n] >= 0 and dp_index != dp_index_map[n]:
                            my_pos = dp_index_map[n]
                            hostname = node.raw_ctx.workers[my_pos].hostname
                            target_id = node.raw_ctx.workers[my_pos].device_id
                            key = (n, target_id)
                            if key not in send_dst:
                                send_dst[key] = True
                                my_eval_nodes.append(pipeline_send_op(n, mpi_comm.getRankFromDevice(
                                    hostname, target_id), mpi_comm, stream=p2p_stream, ctx=ctx))
                    else:
                        # here in the same model parallel
                        assert node.raw_ctx == n.raw_ctx
                        assert need_states_deduction

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
    mp_index_map = {}  # model parallel index
    dp_index_map = {}  # data parallel index
    node_cur_duplicate_map = {}  # save nodes' duplicate information
    node_tar_duplicate_map = {}  # save nodes' target duplicate
    node_cur_order_map = {}  # save nodes' order information
    node_tar_order_map = {}  # save nodes' target order
    node_cur_states_map = {}  # save nodes' current states
    node_tar_states_map = {}  # save nodes' target states
    my_eval_nodes = []
    for node in node_list:
        assign_ctx(node)

    return my_eval_nodes, param_allreduce_group
