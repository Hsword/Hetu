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
            assert mp_index_map[node] >= 0, 'Now only support 1 to N.'
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
                [ts == 1 for ts in target])), 'Now only support N to 1.'
            if prev_input not in recv_src:
                device_index = -1

                def make_comb(devices, cur_states, depth):
                    if depth == len(cur_states):
                        nonlocal device_index
                        device_index += 1
                        return pipeline_receive_op(mpi_comm.getRankFromDevice(devices[device_index].hostname, devices[device_index].device_id), mpi_comm, stream=p2p_stream, ctx=ctx)
                    else:
                        result = make_comb(devices, cur_states, depth + 1)
                        for _ in range(1, cur_states[depth]):
                            result = concat_op(result, make_comb(
                                devices, cur_states, depth + 1), axis=depth, ctx=ctx)
                        return result
                res = make_comb(
                    prev_input.raw_ctx.workers[dev_pos], node_cur_states_map[prev_input], 0)
                for _ in range(1, node_cur_duplicate_map.get(prev_input, 1)):
                    res = add_op(res, make_comb(
                        prev_input.raw_ctx.workers[dev_pos], node_cur_states_map[prev_input], 0), ctx=ctx)
                assert device_index + \
                    1 == len(prev_input.raw_ctx.workers[dev_pos])
                recv_src[prev_input] = res
            return recv_src[prev_input]
        else:
            pass

    def send_model_parallel(prev_input, node):
        # assert dp_index_map[prev_input] >= 0 and dp_index_map[node] < 0
        dev_pos = dp_index_map[prev_input]
        if not isinstance(prev_input.raw_ctx.workers[dev_pos], tuple):
            # here we send from a node on one device dispatching to many nodes
            # in this case current node MUST have mp_index, and the split will be handled in sending
            assert mp_index_map[prev_input] < 0, 'Now only support 1 to N.'
            device_index = 0

            def make_split(devices, target_states, cur_states, depth):
                if len(target_states) == depth:
                    nonlocal device_index
                    nonlocal loop_size
                    hostname = devices[device_index].hostname
                    target_id = devices[device_index].device_id
                    key = (prev_input, target_id)
                    if key not in send_dst:
                        cur_node = prev_input if all([x == 1 for x in target_states]) else split_op(
                            prev_input, list(range(len(target_states))), list(cur_states), list(target_states), ctx=ctx)
                        for cur_dev_id in range(device_index, len(devices), loop_size):
                            hostname = devices[cur_dev_id].hostname
                            target_id = devices[cur_dev_id].device_id
                            target_rank = mpi_comm.getRankFromDevice(
                                hostname, target_id)
                            key = (prev_input, target_id)
                            send_dst[key] = pipeline_send_op(
                                cur_node, target_rank, mpi_comm, stream=p2p_stream, ctx=ctx)
                            my_eval_nodes.append(send_dst[key])
                    device_index += 1
                else:
                    for ts in range(target_states[depth]):
                        cur_states[depth] = ts
                        make_split(devices, target_states,
                                   cur_states, depth + 1)
            cur_states = [0 for _ in range(
                len(node_tar_states_map[prev_input]))]
            loop_size = np.prod(node_tar_states_map[prev_input])
            make_split(
                node.raw_ctx.workers[dev_pos], node_tar_states_map[prev_input], cur_states, 0)
            assert device_index * \
                node_tar_duplicate_map.get(prev_input, 1) == len(
                    node.raw_ctx.workers[dev_pos])
        elif not isinstance(node.raw_ctx.workers[dev_pos], tuple):
            # here we send from a node on multiple devices to one node
            # in this case current node MUST NOT have mp_index, and the combination will be handled in receiving
            target = node_tar_states_map[prev_input]
            assert mp_index_map[prev_input] >= 0 and (target is None or all(
                [ts == 1 for ts in target])), 'Now only support N to 1.'
            hostname = node.raw_ctx.workers[dev_pos].hostname
            target_id = node.raw_ctx.workers[dev_pos].device_id
            key = (prev_input, target_id)
            if key not in send_dst:
                send_dst[key] = pipeline_send_op(prev_input, mpi_comm.getRankFromDevice(
                    hostname, target_id), mpi_comm, stream=p2p_stream, ctx=ctx)
                my_eval_nodes.append(send_dst[key])
        else:
            pass

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
                            num_device_per_worker = len(
                                param.raw_ctx[dp_index_map[param]])
                            interval = num_device_per_worker // node_tar_duplicate_map[param]
                            for rc in range(param.raw_ctx.worker_num):
                                for ind in range(mp_index_map[param] % interval, num_device_per_worker, interval):
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
            if isinstance(real_node, PlaceholderOp) and mp_index_map[real_node] >= 0:
                real_node.reshape_in_mp(mp_index_map[real_node], node.parts)
            node_tar_states_map[real_node] = node.parts
            node_tar_duplicate_map[real_node] = node.duplicate
        elif isinstance(node, DispatchGradientOp):
            real_node = node.inputs[0]
            assign_ctx(real_node)
            assign_ctx(node.inputs[1])
            if isinstance(real_node, PlaceholderOp) and mp_index_map[real_node] >= 0:
                real_node.reshape_in_mp(mp_index_map[real_node], node.parts)
            node_tar_states_map[real_node] = node_cur_states_map.get(
                node.inputs[1], None)
            node_tar_duplicate_map[real_node] = node_cur_duplicate_map.get(
                node.inputs[1], 1)
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
            for i, n in enumerate(node.inputs):
                if isinstance(n, DataloaderOp):
                    if dp_index >= 0 and n in node_list and n not in my_eval_nodes:
                        my_eval_nodes.append(n)
                    continue
                assign_ctx(n)

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
                    need_states_deduction = True
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
                                send_dst[key] = pipeline_send_op(n, mpi_comm.getRankFromDevice(
                                    hostname, target_id), mpi_comm, stream=p2p_stream, ctx=ctx)
                                my_eval_nodes.append(send_dst[key])
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
            if need_states_deduction:
                input_states = []
                input_duplicates = []
                for n in node.inputs:
                    if isinstance(n, (DispatchOp, DispatchGradientOp)):
                        input_states.append(node_tar_states_map[n.inputs[0]])
                        input_duplicates.append(
                            node_tar_duplicate_map[n.inputs[0]])
                    else:
                        input_states.append(node_cur_states_map.get(n, None))
                        input_duplicates.append(
                            node_cur_duplicate_map.get(n, 1))
                node_cur_states_map[node], node_cur_duplicate_map[node] = node.deduce_states(
                    input_states, input_duplicates)

    opt = None
    trainable_params = []
    comm_groups = {}
    param_allreduce_group = {}
    send_dst = {}
    recv_src = {}
    mp_index_map = {}  # model parallel index
    dp_index_map = {}  # data parallel index
    node_cur_duplicate_map = {}  # save nodes' duplicate information
    node_tar_duplicate_map = {}  # save nodes' target states
    node_cur_states_map = {}  # save nodes' current states
    node_tar_states_map = {}  # save nodes' target states
    my_eval_nodes = []
    for node in node_list:
        assign_ctx(node)

    return my_eval_nodes, param_allreduce_group
