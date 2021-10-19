import hetu as ht
import yaml
import socket
import psutil
import numpy as np
from random import choice, random
from collections import defaultdict

from .context import DeviceGroup, NodeStatus
from .gpu_ops.Variable import PlaceholderOp


class DistConfig(object):
    def __init__(self, file=None, num_local_servers=0, num_local_workers=1):
        if file is None:
            assert num_local_workers > 0, \
                'Please specify the configuration file or set the number of local workers.'
            self.settings = {'nodes': [{
                'host': 'localhost',
                'servers': num_local_servers,
                'workers': num_local_workers,
                'chief': True,
            }]}
        else:
            self.settings = yaml.load(
                open(file).read(), Loader=yaml.FullLoader)
        attributes = set(['host', 'servers', 'workers', 'chief'])
        hosts = []
        servers, workers = {}, {}
        chief = None
        self.chief_address = socket.gethostbyname(socket.gethostname())
        for node in self.settings['nodes']:
            assert set(node.keys(
            )) <= attributes, 'Attributes of nodes invalid, %s / %s.' % (set(node.keys()), attributes)
            hosts.append(node['host'])
            if node.get('servers', 0):
                servers[node['host']] = node['servers']
            if node.get('workers', 0):
                workers[node['host']] = node['workers']
            if node.get('chief', False):
                assert chief is None, 'There should be only one chief.'
                chief = node['host']
        assert chief, 'There should be one chief.'
        self.num_servers = sum(servers.values())
        self.num_workers = sum(workers.values())
        self.enable_PS = (self.num_servers > 0)
        self.servers = servers
        self.workers = workers
        self.chief = chief
        self.hosts = hosts
        self.chief_address = socket.gethostbyname(socket.gethostname())

    def __str__(self):
        return '\n'.join([
            'Cluster: {',
            '  Chief: %s,' % self.chief,
            '  Servers(%d): %s,' % (self.num_servers, self.servers),
            '  Workers(%d): %s,' % (self.num_workers, self.workers),
            '}',
        ])

    def save(self, path):
        with open(path, 'w') as fw:
            yaml.dump(self.settings, fw)

    def make_ps_config(self):
        port = self.get_available_port(self.chief_address)
        return {
            'DMLC_PS_ROOT_URI': self.chief_address,
            'DMLC_PS_ROOT_PORT': port,
            'DMLC_NUM_WORKER': self.num_workers,
            'DMLC_NUM_SERVER': self.num_servers,
            'DMLC_PS_VAN_TYPE': 'p3'
        }

    def get_available_port(self, localhost):
        ports = set()
        for conn in psutil.net_connections():
            la = conn.laddr
            ra = conn.raddr
            if len(la) == 2 and la.ip in (localhost, '127.0.0.1'):
                ports.add(la.port)
            if len(ra) == 2 and ra.ip in (localhost, '127.0.0.1'):
                ports.add(ra.port)
        for p in range(13100, 13200):
            if p not in ports:
                return p


class Strategy(object):
    def __init__(self):
        # TODO: modify executor's logic to use communicators
        self.settings = DistConfig('/tmp/hetu_config.yml')
        self.use_dispatch = True

    def set_raw_ctxs(self, eval_node_list):
        # called if use_dispatch is True
        raise NotImplementedError

    def set_raw_ctxs_n_states(self, eval_node_list):
        # called if use_dispatch is False
        raise NotImplementedError

    def get_forward_eval_nodes(self, eval_node_list):
        from .optimizer import OptimizerOp
        opt = None
        for node in eval_node_list:
            if isinstance(node, OptimizerOp):
                assert opt is None
                opt = node
        # only get loss to deduce forward graph
        new_eval_nodes = eval_node_list if opt is None else [
            opt.optimizer.loss]
        return new_eval_nodes, opt


class DataParallel(Strategy):
    def __init__(self, aggregate=None):
        super().__init__()
        if aggregate is None:
            aggregate = 'ps' if self.settings.enable_PS else 'allreduce'
        aggregate = aggregate.lower()
        assert aggregate in ('allreduce', 'ps', 'parallax')
        self.aggregate = aggregate

        # TODO: check communicators; check in a method, or in executor, or in base class?
        embedding_ctxs = ['cpu:0'] if aggregate != 'allreduce' else []
        ctxs = ['cpu:0'] if aggregate == 'ps' else []
        for host, num_worker in self.settings.workers.items():
            devices = [host + ':gpu:' + str(i) for i in range(num_worker)]
            embedding_ctxs.extend(devices)
            ctxs.extend(devices)
        self.embedding_raw_ctx = DeviceGroup(embedding_ctxs)
        self.raw_ctx = DeviceGroup(ctxs)

    def set_raw_ctxs(self, eval_node_list):
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
        visited = set()
        for node in eval_node_list:
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
        rank0 = self.settings.chief + ':gpu:0'
        assert rank0 in ctxs, 'This strategy requires that chief node has at least one worker.'
        self.num_ctxs = len(ctxs)
        self.rank0_ctx = DeviceGroup(rank0)
        self.raw_ctx = DeviceGroup(ctxs)
        self.use_dispatch = False

    def set_raw_ctxs(self, eval_node_list):
        # deprecated
        from .gpu_ops.Conv2d import Conv2dOp
        from .gpu_ops.Conv2dAddBias import Conv2dAddBiasOp
        from .gpu_ops.MatrixMult import MatMulOp
        from .gpu_ops.Linear import LinearOp
        from .gpu_ops.SoftmaxCrossEntropy import SoftmaxCrossEntropyOp
        from .gpu_ops.SoftmaxCrossEntropySparse import SoftmaxCrossEntropySparseOp

        def dfs(node, ctx):
            if node in visited:
                return
            visited.add(node)
            if isinstance(node, (SoftmaxCrossEntropyOp, SoftmaxCrossEntropySparseOp)):
                dfs(node.inputs[0], self.raw_ctx)
                dfs(node.inputs[1], self.rank0_ctx)
                node.inputs[0] = ht.dispatch(node.inputs[0])
            else:
                for n in node.inputs:
                    dfs(n, ctx)
                if isinstance(node, (Conv2dOp, MatMulOp, Conv2dAddBiasOp, LinearOp)):
                    split_dim = {Conv2dOp: 0, Conv2dAddBiasOp: 0,
                                 MatMulOp: 1, LinearOp: 1}[type(node)]
                    new_node_A = ht.dispatch(node.inputs[0])
                    new_node_B = ht.dispatch(
                        node.inputs[1], {split_dim: self.num_ctxs})
                    if isinstance(node, (Conv2dOp, MatMulOp)):
                        node.inputs = [new_node_A, new_node_B]
                    else:
                        new_node_C = ht.dispatch(
                            node.inputs[2], {0: self.num_ctxs})
                        node.inputs = [new_node_A, new_node_B, new_node_C]
            node.raw_ctx = ctx

        eval_nodes, opt = self.get_forward_eval_nodes(eval_node_list)
        assert opt is not None
        visited = set()
        dfs(eval_nodes[0], self.rank0_ctx)
        with ht.context(self.rank0_ctx):
            opt.re_minimize()

        return self.raw_ctx

    def set_raw_ctxs_n_states(self, eval_node_list):
        from .gpu_ops.Conv2d import Conv2dOp
        from .gpu_ops.Conv2dAddBias import Conv2dAddBiasOp
        from .gpu_ops.MatrixMult import MatMulOp
        from .gpu_ops.Linear import LinearOp
        from .gpu_ops.SoftmaxCrossEntropy import SoftmaxCrossEntropyOp
        from .gpu_ops.SoftmaxCrossEntropySparse import SoftmaxCrossEntropySparseOp
        from .context import complete_state_map_with_partial_information

        def dfs(node, ctx):
            if node in visited:
                return
            visited.add(node)
            node.raw_ctx = ctx
            if isinstance(node, (SoftmaxCrossEntropyOp, SoftmaxCrossEntropySparseOp)):
                dfs(node.inputs[0], self.raw_ctx)
                dfs(node.inputs[1], self.rank0_ctx)
                node_cur_state_map[node] = NodeStatus({}, dev_num=1)
            else:
                for n in node.inputs:
                    dfs(n, ctx)
                if isinstance(node, (Conv2dOp, MatMulOp, Conv2dAddBiasOp, LinearOp)):
                    node_cur_state_map[node] = NodeStatus(
                        {1: self.num_ctxs}, dev_num=node.raw_ctx.mp_device_num)

        eval_nodes, opt = self.get_forward_eval_nodes(eval_node_list)
        assert opt is not None
        visited = set()
        node_cur_state_map = {}
        # add partial information for forward nodes
        dfs(eval_nodes[0], self.rank0_ctx)

        # set context for backward nodes using forward nodes
        for2back = opt.optimizer.forward2backward
        for grad in for2back.pop(None):
            grad.raw_ctx = self.rank0_ctx
        for node, grads in for2back.items():
            for grad in grads:
                grad.raw_ctx = node.raw_ctx

        # infer states using partial information
        node_cur_state_map, node_tar_state_map = complete_state_map_with_partial_information(
            eval_nodes, eval_node_list, node_cur_state_map, opt.optimizer.backward2forward)
        return self.raw_ctx, node_cur_state_map, node_tar_state_map


class OneWeirdTrick4CNN(Strategy):
    # split batch dimension in conv layers
    # split channel dimension in linear layers
    def __init__(self):
        super().__init__()
        # only for CNN and FC layers
        ctxs = ()
        for host, num_worker in self.settings.workers.items():
            ctxs += tuple(host + ':gpu:' + str(i)
                          for i in range(num_worker))
        rank0 = self.settings.chief + ':gpu:0'
        assert rank0 in ctxs, 'This strategy requires that chief node has at least one worker.'
        self.num_ctxs = len(ctxs)
        self.rank0_ctx = DeviceGroup(rank0)
        self.raw_ctx = DeviceGroup(ctxs)
        self.use_dispatch = False

    def set_raw_ctxs(self, eval_node_list):
        # deprecated
        from .gpu_ops.Conv2d import Conv2dOp
        from .gpu_ops.Conv2dAddBias import Conv2dAddBiasOp
        from .gpu_ops.MatrixMult import MatMulOp
        from .gpu_ops.Linear import LinearOp
        from .gpu_ops.SoftmaxCrossEntropy import SoftmaxCrossEntropyOp
        from .gpu_ops.SoftmaxCrossEntropySparse import SoftmaxCrossEntropySparseOp
        from .dataloader import DataloaderOp

        def dfs(node, ctx):
            if node in visited:
                return
            visited.add(node)
            if isinstance(node, (SoftmaxCrossEntropyOp, SoftmaxCrossEntropySparseOp)):
                dfs(node.inputs[0], self.raw_ctx)
                dfs(node.inputs[1], self.rank0_ctx)
                node.inputs[0] = ht.dispatch(node.inputs[0])
            elif isinstance(node, (Conv2dOp, Conv2dAddBiasOp)) and isinstance(node.inputs[0], (DataloaderOp, PlaceholderOp)):
                dfs(node.inputs[0], self.raw_ctx)
                dfs(node.inputs[1], self.raw_ctx)
                if isinstance(node, Conv2dAddBiasOp):
                    dfs(node.inputs[2], self.raw_ctx)
                    node.inputs[2] = ht.dispatch(node.inputs[2])
                node.inputs[0] = ht.dispatch(
                    node.inputs[0], {0: self.num_ctxs})
                node.inputs[1] = ht.dispatch(node.inputs[1])
            else:
                for n in node.inputs:
                    dfs(n, ctx)
                if isinstance(node, (MatMulOp, LinearOp)):
                    new_node_A = ht.dispatch(node.inputs[0])
                    new_node_B = ht.dispatch(
                        node.inputs[1], {1: self.num_ctxs})
                    if isinstance(node, MatMulOp):
                        node.inputs = [new_node_A, new_node_B]
                    else:
                        new_node_C = ht.dispatch(
                            node.inputs[2], {0: self.num_ctxs})
                        node.inputs = [new_node_A, new_node_B, new_node_C]
                elif isinstance(node, (Conv2dOp, Conv2dAddBiasOp)):
                    node.inputs[1] = ht.dispatch(node.inputs[1])
                    if isinstance(node, Conv2dAddBiasOp):
                        node.inputs[2] = ht.dispatch(node.inputs[2])
            node.raw_ctx = ctx

        eval_nodes, opt = self.get_forward_eval_nodes(eval_node_list)
        assert opt is not None
        visited = set()
        dfs(eval_nodes[0], self.rank0_ctx)
        with ht.context(self.rank0_ctx):
            opt.re_minimize()

        return self.raw_ctx

    def set_raw_ctxs_n_states(self, eval_node_list):
        from .gpu_ops.Conv2d import Conv2dOp
        from .gpu_ops.Conv2dAddBias import Conv2dAddBiasOp
        from .gpu_ops.MatrixMult import MatMulOp
        from .gpu_ops.Linear import LinearOp
        from .gpu_ops.SoftmaxCrossEntropy import SoftmaxCrossEntropyOp
        from .gpu_ops.SoftmaxCrossEntropySparse import SoftmaxCrossEntropySparseOp
        from .context import complete_state_map_with_partial_information

        def dfs(node, ctx):
            if node in visited:
                return
            visited.add(node)
            node.raw_ctx = ctx
            if isinstance(node, (SoftmaxCrossEntropyOp, SoftmaxCrossEntropySparseOp)):
                dfs(node.inputs[0], self.raw_ctx)
                dfs(node.inputs[1], self.rank0_ctx)
                node_cur_state_map[node] = NodeStatus({}, dev_num=1)
            else:
                for n in node.inputs:
                    dfs(n, ctx)
                if isinstance(node, (Conv2dOp, Conv2dAddBiasOp)):
                    node_cur_state_map[node] = NodeStatus(
                        {0: self.num_ctxs}, dev_num=node.raw_ctx.mp_device_num)
                elif isinstance(node, (MatMulOp, LinearOp)):
                    node_cur_state_map[node] = NodeStatus(
                        {1: self.num_ctxs}, dev_num=node.raw_ctx.mp_device_num)

        eval_nodes, opt = self.get_forward_eval_nodes(eval_node_list)
        assert opt is not None
        visited = set()
        node_cur_state_map = {}
        # add partial information for forward nodes
        dfs(eval_nodes[0], self.rank0_ctx)

        # set context for backward nodes using forward nodes
        for2back = opt.optimizer.forward2backward
        for grad in for2back.pop(None):
            grad.raw_ctx = self.rank0_ctx
        for node, grads in for2back.items():
            for grad in grads:
                grad.raw_ctx = node.raw_ctx

        # infer states using partial information
        node_cur_state_map, node_tar_state_map = complete_state_map_with_partial_information(
            eval_nodes, eval_node_list, node_cur_state_map, opt.optimizer.backward2forward)
        return self.raw_ctx, node_cur_state_map, node_tar_state_map


class FlexFlow(Strategy):
    def __init__(self, feed_shapes, bandwidth=None, budget=-1, alpha=0.05, save_path=None, load_path=None):
        from itertools import combinations
        from collections import namedtuple
        # now only consider homogeneous environment
        # the simulations now are all on the rank-0 device
        # TODO: use multiple devices to simulate
        # now we only use flexflow strategy in one machine with multiple devices
        # TODO: extend to multiple machines, how to sample new configurations?
        super().__init__()
        self.use_dispatch = False
        self.feed_shapes = feed_shapes
        # bandwidth should contain following keys:
        # p2p: send/recv bandwidth
        # allreduce groups: 248(0, 1), 48(0, 2), 8(0, 4),
        # 8(0, 1, 2, 3), 8(0, 1, 2, 4), 8(0, 1, 4, 5), 8(0, 1, 4, 6), 8(0, 2, 4, 6)
        # 8(0, 1, 2, 3, 4, 5, 6, 7)
        if bandwidth is None:
            bandwidth = {
                'p2p': 10659721.697151,
                'allreduce': {
                    (2, 1): 4053906.914546,
                    (2, 2): 7207890.632895,
                    (2, 3): 7157173.727458,
                    (4, 1): 3301009.95906475,  # same pcie switch
                    (4, 2): 6234281.734696,  # not using same pcie switch
                    8: 3005380.3449143744,
                }
            }
        self.bandwidth = bandwidth
        self.budget = budget
        self.alpha = alpha
        assert len(self.settings.hosts) == 1, 'Not support multiple machines.'
        self.num_ctxs = self.settings.num_workers
        assert self.num_ctxs in (
            2, 4, 8), 'Number of workers should be 2, 4, 8.'
        self.all_devices = [ht.gpu(i) for i in range(self.num_ctxs)]
        self.profiler = ht.HetuProfiler([], {}, {})
        self.cached_exetime = {}
        self.cached_placeholders = []
        self.cached_optimizer = None
        self.cached_config = namedtuple(
            'Config', 'placeholder_to_arr_map')(placeholder_to_arr_map={})
        self.raw_ctx = DeviceGroup(tuple(self.all_devices))
        self.rank0_ctx = DeviceGroup(self.settings.chief + ':gpu:0')
        self.rank0_device = ht.gpu(0)
        self.save_path = save_path
        self.load_path = load_path

        # generate candidates for random sampling
        cur_num = 1
        self.split_candidates = []
        while cur_num <= self.num_ctxs:
            left = cur_num
            right = 1
            while left >= 1:
                cand = {k: v for k, v in {0: left, 1: right}.items() if v != 1}
                self.split_candidates.append(cand)
                left //= 2
                right *= 2
            cur_num *= 2
        self.device_candidates = {
            1: [DeviceGroup(dev) for dev in self.all_devices]}
        cur_num = 2
        while cur_num <= self.num_ctxs:
            self.device_candidates[cur_num] = [DeviceGroup(
                devs) for devs in combinations(self.all_devices, cur_num)]
            cur_num *= 2

    class TaskNode(object):
        def __init__(self, name, device, inputs=[], shape=-1):
            self.name = name
            self.device = device
            self.inputs = inputs
            if shape == -1:
                shape = self.inputs[0].shape
            self.shape = shape
            self.outputs = []
            for task in inputs:
                task.add_output(self)
            self.exeTime = None
            self.readyTime = 0
            self.startTime = 0
            self.endTime = 0
            self.contents = None
            self.state = 0  # 0 for incomplete, 1 for complete

        def add_output(self, out_node):
            self.outputs.append(out_node)

        def set_exetime(self, exeTime):
            self.exeTime = exeTime

        def __repr__(self):
            return self.name

        def __lt__(self, other):
            return self.readyTime < other.readyTime

    def profile_new_case(self, new_node, task, opt=False):
        new_node.ctx = self.rank0_device
        new_node.on_cpu = False
        new_node.on_gpu = True
        num_cur_ph = len(self.cached_placeholders)
        num_inputs = len(task.inputs)
        if num_cur_ph < num_inputs:
            self.cached_placeholders.extend([PlaceholderOp(
                'test_node', ctx=self.rank0_device) for _ in range(num_inputs - num_cur_ph)])
        # return 0.
        new_node.inputs = self.cached_placeholders[:len(task.inputs)]
        new_node.infer_shape([t.shape for t in task.inputs])
        node_to_arr_map = {n: ht.empty(t.shape, ctx=self.rank0_device) for n, t in zip(
            new_node.inputs, task.inputs)}
        node_to_arr_map[new_node] = ht.empty(
            task.shape, ctx=self.rank0_device) if not opt else None
        self.profiler.renew_nodes(
            [new_node], {n: t.shape for n, t in zip(new_node.inputs, task.inputs)}, node_to_arr_map)
        result = self.profiler.profile_all(
            num_iterations=10, profiler='gpu')
        return result['all']

    def make_task_node(self, node, index, shape, inputs=[]):
        from copy import copy
        node_type = type(node)
        name = '{}_{}'.format(node.name, index)
        device_group = node.raw_ctx.workers[0]
        if isinstance(device_group, tuple):
            device = device_group[index].device_id
        else:
            assert index == 0
            device = device_group.device_id
        task = self.TaskNode(name, device, inputs=inputs, shape=shape)
        if node_type == PlaceholderOp:
            # TODO: consider data feeding time ?
            task.set_exetime(0.)
            return task
        key = (node_type, task.shape) + tuple(n.shape for n in task.inputs)
        if key not in self.cached_exetime:
            new_node = copy(node)
            new_node.inputs = []
            if hasattr(new_node, 'grad_node'):
                del new_node.grad_node
            if hasattr(new_node, 'grad_nodes'):
                del new_node.grad_nodes
            if hasattr(new_node, 'forward_node'):
                del new_node.forward_node
            self.cached_exetime[key] = self.profile_new_case(
                new_node, task)
        task.set_exetime(self.cached_exetime[key])
        return task

    def make_split_task_node(self, op_index, input_task, axes, inds, splits):
        from .gpu_ops.Split import split_op, SplitOp
        name = 'split_{}'.format(op_index)
        device = input_task.device
        shape = self.get_split_shape(dict(zip(axes, splits)), input_task.shape)
        task = self.TaskNode(name, device, inputs=[input_task, ], shape=shape)
        # TODO: consider whether add indices in key
        key = (SplitOp, task.shape, input_task.shape)
        if key not in self.cached_exetime:
            new_node = split_op(
                self.cached_placeholders[0], axes, inds, splits, ctx=self.rank0_device)
            self.cached_exetime[key] = self.profile_new_case(new_node, task)
        task.set_exetime(self.cached_exetime[key])
        return task

    def make_concatenate_task_node(self, device, inputs, axis):
        from .gpu_ops.Concatenate import concatenate_op, ConcatenateOp
        name = 'concatenate_dim{}'.format(axis)
        shape = self.get_concatenate_shape(inputs, axis)
        task = self.TaskNode(name, device, inputs=inputs, shape=shape)
        key = (ConcatenateOp, task.shape) + tuple(n.shape for n in inputs)
        if key not in self.cached_exetime:
            new_node = concatenate_op(
                self.cached_placeholders, axis=axis, ctx=self.rank0_device)
            self.cached_exetime[key] = self.profile_new_case(new_node, task)
        task.set_exetime(self.cached_exetime[key])
        return task

    def make_sum_task_node(self, device, inputs):
        from .gpu_ops.Sum import sum_op, SumOp
        task = self.TaskNode('sum_duplicate', device, inputs=inputs)
        key = (SumOp, task.shape) + tuple(n.shape for n in inputs)
        if key not in self.cached_exetime:
            new_node = sum_op(self.cached_placeholders, ctx=self.rank0_device)
            self.cached_exetime[key] = self.profile_new_case(new_node, task)
        task.set_exetime(self.cached_exetime[key])
        return task

    def make_update_task_node(self, device_group, index, input_task):
        from .optimizer import OptimizerOp
        name = 'update_{}'.format(index)
        if isinstance(device_group, tuple):
            device = device_group[index].device_id
        else:
            assert index == 0
            device = device_group.device_id
        task = self.TaskNode(name, device, inputs=[input_task, ])
        key = (OptimizerOp, task.shape)
        if key not in self.cached_exetime:
            param = self.cached_placeholders[0]
            self.cached_optimizer.params = [param]
            self.cached_config.placeholder_to_arr_map[param] = ht.empty(
                input_task.shape, ctx=self.rank0_device)
            self.cached_optimizer.initiated = False
            self.cached_optimizer.initiate_states(self.cached_config)
            new_node = OptimizerOp(
                self.cached_placeholders, self.cached_optimizer)
            new_node.comm_mode = None
            self.cached_exetime[key] = self.profile_new_case(
                new_node, task, opt=True)
        task.set_exetime(self.cached_exetime[key])
        return task

    def make_comm_task_node(self, from_device, to_device, prev_task):
        name = 'comm_{}_to_{}'.format(from_device, to_device)
        device = (from_device, to_device)
        task = self.TaskNode(name, device, inputs=[prev_task, ])
        cur_band = self.bandwidth['p2p']
        if from_device // 2 == to_device // 2:
            cur_band /= 2
        cur_size = 4 * np.prod(task.shape, dtype=int)
        task.set_exetime(cur_size / cur_band)
        return task

    def make_group_comm_task_node(self, tasks):
        if len(tasks) == 0:
            return None
        name = 'group_comm'
        device = tuple(t.device for t in tasks)
        task = self.TaskNode(name, device, shape=None)
        task.inputs = list(set([ti for t in tasks for ti in t.inputs]))
        send_sizes = [0 for _ in range(self.num_ctxs // 2)]
        recv_sizes = [0 for _ in range(self.num_ctxs // 2)]
        for t in tasks:
            cur_size = np.prod(t.shape, dtype=int)
            send_sizes[t.device[0] // 2] += cur_size
            recv_sizes[t.device[1] // 2] += cur_size
        max_size = max(send_sizes + recv_sizes)
        task.set_exetime(4 * max_size / self.bandwidth['p2p'])
        task.contents = tasks
        return task

    def make_allreduce_task_nodes(self, device_group, prev_tasks, status):
        name = 'allreduce'
        state, duplicate, order = status.get_all()
        cur_size = 4 * np.prod(prev_tasks[0].shape, dtype=int)
        if duplicate == 8:
            devices = tuple(range(8))
            cur_band = self.bandwidth['allreduce'][8]
            task = self.TaskNode(name, devices, inputs=list(prev_tasks))
            task.set_exetime(cur_size / cur_band)
            allreduce_tasks = [task]
            update_tasks = [self.make_update_task_node(
                device_group, i, task) for i in range(8)]
        else:
            num_splits = status.dev_num // duplicate
            allreduce_tasks = []
            update_tasks = []
            interval = 1
            dup_dim = order.index(-1)
            for cur_order in order[dup_dim+1:]:
                interval *= state[cur_order]
            macro_interval = interval * duplicate
            for index in range(num_splits):
                start = (index // interval) * \
                    macro_interval + (index % interval)
                indices = range(start, start + macro_interval, interval)
                allreduce_devices = [
                    device_group[ind].device_id for ind in indices]
                devices = tuple(allreduce_devices)
                if duplicate == 2:
                    d0, d1 = allreduce_devices
                    distance = 0
                    while d0 != d1:
                        distance += 1
                        d0 //= 2
                        d1 //= 2
                    cur_band = self.bandwidth['allreduce'][(2, distance)]
                else:
                    assert duplicate == 4
                    allreduce_devices = [ad // 2 for ad in allreduce_devices]
                    distance = 1 + (len(np.unique(allreduce_devices)) == 4)
                    cur_band = self.bandwidth['allreduce'][(4, distance)]
                task = self.TaskNode(name, devices, inputs=[
                                     prev_tasks[ind] for ind in indices])
                task.set_exetime(cur_size / cur_band)
                allreduce_tasks.append(task)
                update_tasks.extend([self.make_update_task_node(
                    device_group, ind, task) for ind in indices])
        return allreduce_tasks, update_tasks

    def init_states(self, node_list, for2back, init_ctx):
        # init as data parallel
        from .gpu_ops.Conv2d import Conv2dOp
        from .gpu_ops.Conv2dAddBias import Conv2dAddBiasOp
        from .gpu_ops.MatrixMult import MatMulOp
        from .gpu_ops.Linear import LinearOp
        from .gpu_ops.Sum import SumOp
        from .gpu_ops.Concatenate import ConcatenateOp
        from .gpu_ops.SoftmaxCrossEntropy import SoftmaxCrossEntropyOp
        from .gpu_ops.SoftmaxCrossEntropySparse import SoftmaxCrossEntropySparseOp

        def init_cur_states(node, ctx):
            if node in visited:
                return
            visited.add(node)
            node.raw_ctx = ctx
            key = None  # the key of node group
            if isinstance(node, (SoftmaxCrossEntropyOp, SoftmaxCrossEntropySparseOp)):
                init_cur_states(node.inputs[0], self.raw_ctx)
                init_cur_states(node.inputs[1], self.rank0_ctx)
                node_cur_state_map[node] = NodeStatus({}, dev_num=1)
            else:
                backbone_node = isinstance(
                    node, (Conv2dOp, Conv2dAddBiasOp, MatMulOp, LinearOp, SumOp, ConcatenateOp))
                for n in node.inputs:
                    if backbone_node and isinstance(n, PlaceholderOp) and n not in visited:
                        node_group[node].append(n)
                    temp_key = init_cur_states(n, ctx)
                    if key is None:
                        key = temp_key
                    else:
                        assert backbone_node or temp_key in (key, None)
                if backbone_node:
                    node_cur_state_map[node] = NodeStatus(
                        {0: self.num_ctxs}, dev_num=node.raw_ctx.mp_device_num)
                    key = node
                elif key is not None:
                    node_group[key].append(node)
            return key

        visited = set()
        node_cur_state_map = {}
        node_group = defaultdict(list)
        for node in node_list:
            init_cur_states(node, init_ctx)
        for k, v in node_group.items():
            backward_nodes = for2back[k] + \
                sum([for2back.get(n, []) for n in v], [])
            node_group[k].extend(backward_nodes)
        return node_cur_state_map, node_group

    def get_split_shape(self, parts, shape):
        shape = list(shape)
        if isinstance(parts, list):
            parts = {k: v for k, v in enumerate(parts) if v != 1}
        for i, pts in parts.items():
            assert shape[i] % pts == 0
            shape[i] //= pts
        return tuple(shape)

    def get_concatenate_shape(self, inputs, dim):
        shape = list(inputs[0].shape)
        for n in inputs[1:]:
            shape[dim] += n.shape[dim]
        return tuple(shape)

    def init_task_graph(self, node_list, node_cur_state_map, node_tar_state_map):
        # TODO: transfer node_to_cur_state_map to task graph
        from copy import copy
        from .optimizer import OptimizerOp

        def init_task(node):
            if node not in node_to_task_map:
                if isinstance(node, PlaceholderOp):
                    status = node_cur_state_map[node]
                    new_shape = self.get_split_shape(
                        status.state, self.feed_shapes.get(node, node.shape))
                    cur_tasks = [self.make_task_node(
                        node, i, new_shape) for i in range(status.dev_num)]

                elif isinstance(node, OptimizerOp):
                    self.cached_optimizer = copy(node.optimizer)
                    del self.cached_optimizer.loss
                    del self.cached_optimizer.params
                    del self.cached_optimizer.backward2forward
                    del self.cached_optimizer.forward2backward
                    self.cached_placeholders[0].on_cpu = False
                    self.cached_placeholders[0].on_gpu = True
                    cur_tasks = []
                    for grad, param in zip(node.inputs, node.optimizer.params):
                        temp_tasks = init_comm_task(grad, param)
                        if node_cur_state_map[param].duplicate > 1:
                            # allreduce tasks + update tasks
                            allreduce_tasks, update_tasks = self.make_allreduce_task_nodes(
                                param.raw_ctx.workers[0], temp_tasks, node_cur_state_map[param])
                            task_topo_order.extend(allreduce_tasks)
                            cur_tasks.extend(update_tasks)
                        else:
                            # update task
                            cur_tasks.extend([self.make_update_task_node(
                                param.raw_ctx.workers[0], i, t) for i, t in enumerate(temp_tasks)])
                else:
                    inputs = {}
                    for n in node.inputs:
                        inputs[n] = init_comm_task(n, node)
                    cur_tasks = [self.make_task_node(node, i, node.naive_infer_shape(
                        [n.shape for n in ns]), inputs=list(ns)) for i, ns in enumerate(zip(*inputs.values()))]

                    # TODO: set the exetime of TaskNode
                node_to_task_map[node] = cur_tasks
                task_topo_order.extend(cur_tasks)
            return node_to_task_map[node]

        def init_comm_task(prev, node):
            prev_tasks = init_task(prev)
            deduce_mp = node in node_tar_state_map[prev] \
                and (node_cur_state_map[prev] != node_tar_state_map[prev][node] or prev.raw_ctx != node.raw_ctx) \
                and (node_cur_state_map[prev].is_dist() or node_tar_state_map[prev][node].is_dist())
            if deduce_mp:
                generated_splits = []
                generated_comb = []
                comm_tasks = []
                key = node_tar_state_map[prev][node]
                if not key in recv_src[prev]:
                    cur_tasks = []
                    if not node_cur_state_map[prev].is_dist():
                        # here is the 1 to N case
                        assert len(prev_tasks) == 1 and node_cur_state_map[node].is_dist(
                        ) and key.is_dist(), 'Here only support 1 to N.'
                        prev_ctx = prev.raw_ctx.workers[0]
                        prev_task = prev_tasks[0]

                        def make_split(cur_state, depth):
                            if len(target_order) == depth:
                                nonlocal device_index
                                keys = list(target_state.keys())
                                indices = [cur_state[k] for k in keys]
                                splits = [target_state[k] for k in keys]
                                split_task = self.make_split_task_node(
                                    device_index, prev_task, keys, indices, splits)
                                generated_splits.append(split_task)
                                if devices[device_index] != prev_ctx:
                                    res_task = self.make_comm_task_node(
                                        prev_ctx.device_id, devices[device_index].device_id, split_task)
                                    comm_tasks.append(res_task)
                                else:
                                    res_task = split_task
                                cur_tasks.append(res_task)
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
                        device_index = 0
                        devices = node.raw_ctx.workers[0]
                        target_state, target_duplicate, target_order = key.get_all()
                        make_split({}, 0)
                        assert device_index == len(devices)
                    elif not node_cur_state_map[node].is_dist():
                        # here is the N to 1 case
                        assert len(prev_tasks) > 1 and not key.is_dist(
                        ), 'Here only support N to 1.'
                        cur_ctx = node.raw_ctx.workers[0]

                        def make_comb(depth):
                            if depth == len(cur_order):
                                nonlocal device_index
                                if devices[device_index] != cur_ctx:
                                    res_task = self.make_comm_task_node(
                                        devices[device_index].device_id, cur_ctx.device_id, prev_tasks[device_index])
                                    comm_tasks.append(res_task)
                                else:
                                    res_task = prev_tasks[device_index]
                                device_index += 1
                            else:
                                cur_dim = cur_order[depth]
                                if cur_dim < 0:
                                    if cur_duplicate == 1:
                                        res_task = make_comb(depth + 1)
                                    else:
                                        # sum op task
                                        res_task = self.make_sum_task_node(
                                            cur_ctx.device_id, [make_comb(depth + 1) for _ in range(cur_duplicate)])
                                        generated_comb.append(res_task)
                                else:
                                    if cur_state[cur_dim] == 1:
                                        res_task = make_comb(depth + 1)
                                    else:
                                        # concatenate op task
                                        inputs = [make_comb(depth + 1)
                                                  for _ in range(cur_state[cur_dim])]
                                        res_task = self.make_concatenate_task_node(
                                            cur_ctx.device_id, inputs, cur_dim)
                                        generated_comb.append(res_task)
                            return res_task
                        device_index = 0
                        devices = prev.raw_ctx.workers[0]
                        cur_state, cur_duplicate, cur_order = \
                            node_cur_state_map[prev].get_all()
                        cur_tasks.append(make_comb(0))
                        assert device_index == len(devices)
                    else:
                        # here is the N to N case
                        assert len(prev_tasks) > 1 and key.is_dist(
                        ), 'Here only support N to N.'
                        prev_task = prev_tasks[0]
                        task_buffer = defaultdict(dict)
                        prev_devices = prev.raw_ctx.workers[0]
                        cur_devices = node.raw_ctx.workers[0]
                        prev_ns = node_cur_state_map[prev]
                        prev_state, prev_duplicate, prev_order = prev_ns.get_all()
                        target_state, target_duplicate, target_order = key.get_all()

                        # send first
                        def cross_send(split_cur_state, split_target_state, depth, need_split):
                            nonlocal device_index
                            if depth == len(target_order):
                                if need_split:
                                    keys = list(
                                        split_target_state.keys())
                                    indices = [split_cur_state[k]
                                               for k in keys]
                                    splits = [split_target_state[k]
                                              for k in keys]
                                    # split op
                                    res_task = self.make_split_task_node(
                                        device_index, prev_tasks[mp_index], keys, indices, splits)
                                    generated_splits.append(res_task)
                                else:
                                    res_task = prev_tasks[mp_index]
                                if prev_devices[mp_index] != cur_devices[device_index]:
                                    res_task = self.make_comm_task_node(
                                        prev_devices[mp_index].device_id, cur_devices[device_index].device_id, res_task)
                                    comm_tasks.append(res_task)
                                task_buffer[mp_index][device_index] = res_task
                                device_index += 1
                            else:
                                cur_dim = target_order[depth]
                                if cur_dim < 0:
                                    for _ in range(target_duplicate):
                                        cross_send(
                                            split_cur_state, split_target_state, depth+1, need_split)
                                else:
                                    pre_st = prev_state.get(cur_dim, 1)
                                    cur_st = cur_state_index.get(
                                        cur_dim, 0)
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
                                            multiple * \
                                            loop_sizes[depth]
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

                        for mp_index in range(prev.raw_ctx.mp_device_num):
                            cur_state_index = prev_ns.map_dev_to_index(
                                mp_index)
                            loop_sizes = key.get_loop_sizes()
                            device_index = 0
                            cross_send({}, {}, 0, False)
                            assert device_index == len(cur_devices)

                        # receive next
                        def cross_receive(depth):
                            nonlocal device_index
                            if depth == len(prev_order):
                                res_task = task_buffer[device_index][mp_index]
                                device_index += 1
                            else:
                                cur_dim = prev_order[depth]
                                if cur_dim < 0:
                                    if prev_duplicate == 1:
                                        res_task = cross_receive(depth+1)
                                    else:
                                        # sum op task
                                        res_task = self.make_sum_task_node(cur_devices[mp_index].device_id, [
                                                                           cross_receive(depth+1) for _ in range(prev_duplicate)])
                                        generated_comb.append(res_task)
                                else:
                                    tar_st = target_state.get(cur_dim, 1)
                                    cur_st = cur_state_index.get(
                                        cur_dim, 0)
                                    if prev_state[cur_dim] % tar_st == 0:
                                        # at `cur_dim` dimension we need to concat some inputs
                                        multiple = prev_state[cur_dim] // tar_st
                                        device_index += cur_st * \
                                            multiple * loop_sizes[depth]
                                        if multiple == 1:
                                            res_task = cross_receive(depth+1)
                                        else:
                                            # concatenate op task
                                            inputs = [cross_receive(
                                                depth+1) for _ in range(multiple)]
                                            res_task = self.make_concatenate_task_node(
                                                cur_devices[mp_index].device_id, inputs, cur_dim)
                                            generated_comb.append(res_task)
                                        device_index += (tar_st - 1 - cur_st) * \
                                            multiple * loop_sizes[depth]
                                    elif tar_st % prev_state[cur_dim] == 0:
                                        # at `cur_dim` dimension we need to specify one input
                                        multiple = tar_st // prev_state[cur_dim]
                                        device_index += cur_st // multiple * \
                                            loop_sizes[depth]
                                        res_task = cross_receive(depth+1)
                                        device_index += (tar_st - 1 -
                                                         cur_st) // multiple * loop_sizes[depth]
                                    else:
                                        assert False, 'The dispatch state (%d, %d) at dimension %d is invalid.' % (
                                            prev_state[cur_dim], tar_st, cur_dim)
                            return res_task

                        loop_sizes = prev_ns.get_loop_sizes()
                        for mp_index in range(node.raw_ctx.mp_device_num):
                            cur_state_index = key.map_dev_to_index(mp_index)
                            device_index = 0
                            cur_tasks.append(cross_receive(0))
                            assert device_index == len(prev_devices)

                    recv_src[prev][key] = cur_tasks
                task_topo_order.extend(generated_splits)
                group_comm_task = self.make_group_comm_task_node(comm_tasks)
                for t in comm_tasks:
                    group_comm_map[t] = group_comm_task
                if group_comm_task is not None:
                    task_topo_order.append(group_comm_task)
                task_topo_order.extend(generated_comb)
                return recv_src[prev][key]
            else:
                # check parallel + data parallel
                generated_tasks = []
                assert prev.raw_ctx.worker_num == node.raw_ctx.worker_num == 1, \
                    'In flexflow, the worker number should be 1!'
                assert prev.raw_ctx.mp_device_num == node.raw_ctx.mp_device_num
                if prev.raw_ctx.mp_device_num == 1:
                    if prev.raw_ctx.workers[0] != node.raw_ctx.workers[0]:
                        if -1 not in recv_src[prev]:
                            res_task = self.make_comm_task_node(
                                prev.raw_ctx.workers[0].device_id, node.raw_ctx.workers[0].device_id, prev_tasks[0])
                            generated_tasks.append(res_task)
                            recv_src[prev][-1] = [res_task]
                        res_task = recv_src[prev][-1][0]
                    else:
                        res_task = prev_tasks[0]
                    task_topo_order.extend(generated_tasks)
                    return [res_task]
                else:
                    # here in the same model parallel
                    assert prev.raw_ctx == node.raw_ctx
                    return prev_tasks
        node_to_task_map = {}
        group_comm_map = {}
        task_topo_order = []
        recv_src = defaultdict(dict)
        for node in node_list:
            init_task(node)
        for task in task_topo_order:
            changed = False
            for i, t in enumerate(task.inputs):
                if t in group_comm_map:
                    task.inputs[i] = group_comm_map[t]
                    changed = True
            if changed:
                task.inputs = list(set(task.inputs))
            changed = False
            for i, t in enumerate(task.outputs):
                if t in group_comm_map:
                    task.outputs[i] = group_comm_map[t]
                    changed = True
            if changed:
                task.outputs = list(set(task.outputs))
        for task in set(group_comm_map.values()):
            outputs = sum([t.outputs for t in task.contents], [])
            task.outputs = list(set(outputs))
        return task_topo_order

    def full_simulate(self, task_graph):
        last_tasks = [self.TaskNode('NULL', i, shape=())
                      for i in range(self.num_ctxs)]
        last_comm_tasks = [self.TaskNode('NULL', i, shape=())
                           for i in range(self.num_ctxs)]
        for task in task_graph:
            assert all([t.state == 1 for t in task.inputs])
            dev = task.device
            task.state = 1
            if isinstance(dev, tuple):
                # communication
                is_group = (task.name == 'group_comm')
                is_allreduce = (task.name == 'allreduce')
                startTime = task.readyTime
                if is_group:
                    for d in dev:
                        startTime = max(
                            startTime,
                            last_tasks[d[0]].endTime,
                            last_comm_tasks[d[0] // 2 * 2].endTime,
                            last_comm_tasks[d[1] // 2 * 2 + 1].endTime,
                        )
                elif is_allreduce:
                    for d in dev:
                        startTime = max(
                            startTime,
                            last_tasks[d].endTime,
                            last_comm_tasks[d // 2 * 2].endTime,
                            last_comm_tasks[d // 2 * 2 + 1].endTime,
                        )
                else:
                    startTime = max(
                        startTime,
                        last_tasks[dev[0]].endTime,
                        last_comm_tasks[dev[0] // 2 * 2].endTime,
                        last_comm_tasks[dev[1] // 2 * 2 + 1].endTime
                    )
                task.startTime = startTime
                if is_group:
                    for d in dev:
                        last_comm_tasks[d[0] // 2 * 2] = task
                        last_comm_tasks[d[1] // 2 * 2 + 1] = task
                elif is_allreduce:
                    for d in dev:
                        last_comm_tasks[d // 2 * 2] = task
                        last_comm_tasks[d // 2 * 2 + 1] = task
                else:
                    last_comm_tasks[dev[0] // 2 * 2] = task
                    last_comm_tasks[dev[1] // 2 * 2 + 1] = task
            else:
                task.startTime = max(task.readyTime, last_tasks[dev].endTime)
                last_tasks[dev] = task
            task.endTime = task.startTime + task.exeTime
            for t in task.outputs:
                t.readyTime = task.endTime
        result = 0
        for task in last_tasks:
            result = max(result, task.endTime)
        return result

    def log_task_graph(self, task_topo, log_path='task_graph.txt'):
        # only for debug
        with open(log_path, 'w') as fw:
            for task in task_topo:
                print(task, task.inputs, task.outputs, task.exeTime, task.device, task.shape,
                      task.readyTime, task.startTime, task.endTime, file=fw, flush=True)

    def set_raw_ctxs_n_states(self, eval_node_list):
        from .context import complete_state_map_with_partial_information
        from .gpu_ops.executor import wrapped_mpi_nccl_init, get_mpi_communicate
        from time import time
        import pickle
        from ctypes import c_void_p, c_int, cast, string_at, c_char

        wrapped_mpi_nccl_init()
        mpi_comm = get_mpi_communicate()
        eval_nodes, opt = self.get_forward_eval_nodes(eval_node_list)
        assert opt is not None
        # add partial information for forward nodes
        for2back = opt.optimizer.forward2backward
        node_cur_state_map, node_group = self.init_states(
            eval_nodes, for2back, self.rank0_ctx)
        all_possible_nodes = [
            k for k, v in node_cur_state_map.items() if v.is_dist()]
        # set context for backward nodes using forward nodes
        for grad in for2back.pop(None):
            grad.raw_ctx = self.rank0_ctx
        for node, grads in for2back.items():
            for grad in grads:
                grad.raw_ctx = node.raw_ctx
        if mpi_comm.rank == 0:
            if self.load_path is None:
                print('No configuration loaded. Start autotuning.')
                # infer states using partial information
                start = time()
                meta_cur_state_map = node_cur_state_map.copy()
                node_cur_state_map, node_tar_state_map = complete_state_map_with_partial_information(
                    eval_nodes, eval_node_list, node_cur_state_map, opt.optimizer.backward2forward)
                task_graph = self.init_task_graph(
                    eval_node_list, node_cur_state_map, node_tar_state_map)
                simulation_result = self.full_simulate(task_graph)
                # self.log_task_graph(task_graph)
                print('Initial data parallel configuration generated; simulation result: {:.3f}ms.'.format(
                    simulation_result))
                best_cur_status = [meta_cur_state_map[node]
                                   for node in all_possible_nodes]
                best_raw_ctx = [node.raw_ctx for node in all_possible_nodes]
                best_simulation = simulation_result
                best_emerge_time = time()
                prev_simulation = simulation_result
                cnt = 0

                ending = time()
                cnt += 1
                while (self.budget < 0 or ending - start < self.budget) and 2 * (ending - best_emerge_time) < ending - start:
                    # sample new configuration
                    changing_node = choice(all_possible_nodes)
                    new_split = choice(self.split_candidates)
                    new_raw_ctx = choice(self.device_candidates[np.prod(
                        list(new_split.values()), dtype=int)])
                    while new_split == node_cur_state_map[changing_node].state and new_raw_ctx == changing_node.raw_ctx:
                        new_split = choice(self.split_candidates)
                        new_raw_ctx = choice(self.device_candidates[np.prod(
                            list(new_split.values()), dtype=int)])
                    print('Search round {}'.format(cnt))
                    print('    Change node {} from {} in {} to {} in {}.'.format(
                        changing_node, node_cur_state_map[changing_node].state, changing_node.raw_ctx, new_split, new_raw_ctx))
                    ori_raw_ctx = changing_node.raw_ctx
                    changing_node.raw_ctx = new_raw_ctx
                    for grad in node_group[changing_node]:
                        grad.raw_ctx = new_raw_ctx
                    ori_status = meta_cur_state_map[changing_node]
                    meta_cur_state_map[changing_node] = NodeStatus(
                        new_split, dev_num=changing_node.raw_ctx.mp_device_num)
                    node_cur_state_map = meta_cur_state_map.copy()
                    node_cur_state_map, node_tar_state_map = complete_state_map_with_partial_information(
                        eval_nodes, eval_node_list, node_cur_state_map, opt.optimizer.backward2forward)
                    task_graph = self.init_task_graph(
                        eval_node_list, node_cur_state_map, node_tar_state_map)
                    new_simulation = self.full_simulate(task_graph)
                    print('    Simulation result {:.3f}ms.'.format(
                        new_simulation))
                    if new_simulation < prev_simulation:
                        # move to new strategy
                        print(
                            '    Better than last simulation; move to new configuration.')
                        if new_simulation < best_simulation:
                            best_cur_status = [meta_cur_state_map[node]
                                               for node in all_possible_nodes]
                            best_raw_ctx = [
                                node.raw_ctx for node in all_possible_nodes]
                            best_simulation = new_simulation
                            best_emerge_time = time()
                            print('    Reach the best simulation so far!')
                        prev_simulation = new_simulation
                    else:
                        # probably move to new strategy
                        threshold = np.exp(
                            self.alpha * (prev_simulation - new_simulation))
                        print('    Worse than last simulation; the probability of moving is {:.3f}.'.format(
                            threshold))
                        if random() < threshold:
                            # move to new strategy
                            print('    Move to new configuration.')
                            prev_simulation = new_simulation
                        else:
                            # not move
                            print('    Keep the last configuration.')
                            meta_cur_state_map[changing_node] = ori_status
                            changing_node.raw_ctx = ori_raw_ctx
                            for grad in node_group[changing_node]:
                                grad.raw_ctx = ori_raw_ctx

                        # TODO: delta change current states, target states, task graph, simulation result(maybe need to profile)
                    ending = time()
                    cnt += 1
                node_cur_state_map = meta_cur_state_map
            else:
                print('Loading configuration from {} ...'.format(self.load_path))
                with open(self.load_path, 'rb') as fr:
                    best_cur_status, best_raw_ctx = pickle.load(fr)
            status_bytes = pickle.dumps(best_cur_status)
            context_bytes = pickle.dumps(best_raw_ctx)
            status_length = len(status_bytes)
            context_length = len(context_bytes)
            sizes = (c_int * 2)(status_length, context_length)
            psizes = cast(sizes, c_void_p)
            mpi_comm.MPI_Broadcast(psizes, 8, root=0)
            mpi_comm.MPI_Broadcast(
                cast(status_bytes, c_void_p), status_length, root=0)
            mpi_comm.MPI_Broadcast(
                cast(context_bytes, c_void_p), context_length, root=0)
        else:
            sizes = (c_int * 2)()
            psizes = cast(sizes, c_void_p)
            mpi_comm.MPI_Broadcast(psizes, 8, root=0)
            status_length, context_length = sizes[0], sizes[1]
            status_bytes = (c_char * status_length)()
            context_bytes = (c_char * context_length)()
            mpi_comm.MPI_Broadcast(
                cast(status_bytes, c_void_p), status_length, root=0)
            mpi_comm.MPI_Broadcast(
                cast(context_bytes, c_void_p), context_length, root=0)
            best_cur_status = pickle.loads(
                string_at(status_bytes, size=status_length))
            best_raw_ctx = pickle.loads(
                string_at(context_bytes, size=context_length))
        if self.save_path is not None and mpi_comm.rank == 0:
            print('Saving configuration to {} ...'.format(self.save_path))
            with open(self.save_path, 'wb') as fw:
                pickle.dump((best_cur_status, best_raw_ctx), fw)
        for i, node in enumerate(all_possible_nodes):
            node_cur_state_map[node] = best_cur_status[i]
            node.raw_ctx = best_raw_ctx[i]
        for node, grads in node_group.items():
            for grad in grads:
                grad.raw_ctx = node.raw_ctx
        node_cur_state_map, node_tar_state_map = complete_state_map_with_partial_information(
            eval_nodes, eval_node_list, node_cur_state_map, opt.optimizer.backward2forward)
        return self.raw_ctx, node_cur_state_map, node_tar_state_map
