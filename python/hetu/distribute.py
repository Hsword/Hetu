import hetu as ht
import yaml
import socket
import psutil

from .context import DeviceGroup
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

    def set_raw_ctxs(self):
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

    def set_raw_ctxs(self, eval_node_list):
        from .gpu_ops.Conv2d import Conv2dOp
        from .gpu_ops.MatrixMult import MatMulOp
        from .gpu_ops.SoftmaxCrossEntropy import SoftmaxCrossEntropyOp
        from .gpu_ops.SoftmaxCrossEntropySparse import SoftmaxCrossEntropySparseOp
        from .gpu_ops.Broadcast import BroadcastToOp

        def dfs(node, ctx):
            if node in visited:
                return
            visited.add(node)
            if isinstance(node, (SoftmaxCrossEntropyOp, SoftmaxCrossEntropySparseOp)):
                dfs(node.inputs[0], self.raw_ctx)
                dfs(node.inputs[1], self.rank0_ctx)
                node.inputs[0] = ht.dispatch(node.inputs[0])
            elif isinstance(node, BroadcastToOp) and isinstance(node.inputs[0], PlaceholderOp):
                dfs(node.inputs[0], ctx)
                node.inputs[0] = ht.dispatch(
                    node.inputs[0], {1: self.num_ctxs})
            else:
                for n in node.inputs:
                    dfs(n, ctx)
                if isinstance(node, (Conv2dOp, MatMulOp)):
                    split_dim = {Conv2dOp: 0, MatMulOp: 1}[type(node)]
                    new_node_A = ht.dispatch(node.inputs[0])
                    new_node_B = ht.dispatch(
                        node.inputs[1], {split_dim: self.num_ctxs})
                    node.inputs = [new_node_A, new_node_B]
            node.raw_ctx = ctx

        eval_nodes, opt = self.get_forward_eval_nodes(eval_node_list)
        assert opt is not None
        visited = set()
        dfs(eval_nodes[0], self.rank0_ctx)
        with ht.context(self.rank0_ctx):
            opt.re_minimize()

        return self.raw_ctx


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

    def set_raw_ctxs(self, eval_node_list):
        from .gpu_ops.Conv2d import Conv2dOp
        from .gpu_ops.MatrixMult import MatMulOp
        from .gpu_ops.SoftmaxCrossEntropy import SoftmaxCrossEntropyOp
        from .gpu_ops.SoftmaxCrossEntropySparse import SoftmaxCrossEntropySparseOp
        from .gpu_ops.Broadcast import BroadcastToOp
        from .dataloader import DataloaderOp

        def dfs(node, ctx):
            if node in visited:
                return
            visited.add(node)
            if isinstance(node, (SoftmaxCrossEntropyOp, SoftmaxCrossEntropySparseOp)):
                dfs(node.inputs[0], self.raw_ctx)
                dfs(node.inputs[1], self.rank0_ctx)
                node.inputs[0] = ht.dispatch(node.inputs[0])
            elif isinstance(node, BroadcastToOp) and isinstance(node.inputs[0], PlaceholderOp) and isinstance(node.inputs[1], MatMulOp):
                dfs(node.inputs[0], ctx)
                node.inputs[0] = ht.dispatch(
                    node.inputs[0], {1: self.num_ctxs})
            elif isinstance(node, Conv2dOp) and isinstance(node.inputs[0], (DataloaderOp, PlaceholderOp)):
                dfs(node.inputs[0], self.raw_ctx)
                dfs(node.inputs[1], self.raw_ctx)
                node.inputs[0] = ht.dispatch(
                    node.inputs[0], {0: self.num_ctxs})
                node.inputs[1] = ht.dispatch(node.inputs[1])
            else:
                for n in node.inputs:
                    dfs(n, ctx)
                if isinstance(node, MatMulOp):
                    new_node_A = ht.dispatch(node.inputs[0])
                    new_node_B = ht.dispatch(
                        node.inputs[1], {1: self.num_ctxs})
                    node.inputs = [new_node_A, new_node_B]
                elif isinstance(node, Conv2dOp):
                    node.inputs[1] = ht.dispatch(node.inputs[1])
                elif isinstance(node, BroadcastToOp):
                    node.inputs[0] = ht.dispatch(node.inputs[0])
            node.raw_ctx = ctx

        eval_nodes, opt = self.get_forward_eval_nodes(eval_node_list)
        assert opt is not None
        visited = set()
        dfs(eval_nodes[0], self.rank0_ctx)
        with ht.context(self.rank0_ctx):
            opt.re_minimize()

        return self.raw_ctx
