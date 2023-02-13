from __future__ import annotations

from .ndarray import rcpu, rgpu, DLContext, is_gpu_ctx
import contextlib
import yaml
import socket
import psutil
import re
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple, Union, Iterator, Dict, List, Set, Any
    from .gpu_ops.Node import Op
    ContextRepr = Union[str, DLContext]
    ContextUnitRepr = Union[ContextRepr, Tuple[ContextRepr, ...]]
    ContextReprList = Union[ContextUnitRepr, List[ContextUnitRepr]]
    ContextUnit = Union[DLContext, Tuple[DLContext, ...]]


class DeviceGroup(object):
    def __init__(
        self,
        ctxs: Union[DeviceGroup, str, ContextReprList],
    ) -> None:
        self._contexts = self.parse_contexts(ctxs)
        self.get_servers_n_workers()

    @classmethod
    def parse_contexts(
        cls,
        ctxs: Union[DeviceGroup, str, ContextReprList],
    ) -> List[ContextUnit]:
        if isinstance(ctxs, DeviceGroup):
            return ctxs
        if isinstance(ctxs, str):
            ctxs = re.split(';|,| +', ctxs.lower())
        if not isinstance(ctxs, list):
            ctxs = [ctxs]
        new_ctxs = []
        for c in ctxs:
            c = cls.str2ctx(c)
            if c is not None:
                new_ctxs.append(c)
        return new_ctxs

    @classmethod
    def str2ctx(
        cls,
        c: ContextRepr,
    ) -> DLContext:
        if isinstance(c, str):
            c = c.lower().split(':')
            assert c[-2] in ('cpu', 'gpu'), 'Context invalid: %s' % c
            hostname = 'localhost' if len(c) == 2 else c[0]
            idx = int(c[-1])
            c = rcpu(hostname, idx) if c[-2] == 'cpu' else rgpu(hostname, idx)
        assert isinstance(c, DLContext), 'Context invalid: %s' % c
        return c

    def index(
        self,
        ctx: ContextUnit,
    ) -> int:
        return self._contexts.index(ctx)

    def __getitem__(self, key: int) -> ContextUnit:
        return self._workers[key]

    def __iter__(self) -> Iterator[ContextUnit]:
        return iter(self._contexts)

    def __len__(self) -> int:
        return len(self._contexts)

    def check_mp_num(self, mp_dev_num: int) -> None:
        assert mp_dev_num == self._mp_dev_num

    @property
    def is_mp(self) -> bool:
        return self._is_mp

    @property
    def mp_dev_num(self) -> int:
        return self._mp_dev_num

    @property
    def worker_num(self) -> int:
        return len(self._workers)

    @property
    def server_num(self) -> int:
        return len(self._servers)

    @property
    def workers(self) -> Tuple[ContextUnit]:
        return self._workers

    @property
    def servers(self) -> Tuple[DLContext]:
        return self._servers

    def get_servers_n_workers(self) -> None:
        workers = []
        servers = []
        for ctx in self._contexts:
            if is_gpu_ctx(ctx):
                workers.append(ctx)
            else:
                servers.append(ctx)
        self._workers: Tuple[ContextUnit] = tuple(workers)
        self._servers: Tuple[DLContext] = tuple(servers)

    def __repr__(self) -> str:
        result = 'DeviceGroup('
        for c in self._contexts:
            result += '%s, ' % c
        result += ')'
        return result

    def full_repr(self) -> str:
        return str([c.full_repr() for c in self._contexts])

    def __hash__(self) -> int:
        if not hasattr(self, 'hash'):
            self.hash = hash(tuple(self._contexts))
        return self.hash

    def __eq__(self, other: DeviceGroup) -> bool:
        return hash(self) == hash(other)

    def get_sorted(self) -> DeviceGroup:
        # this function is only for make group allreduce communicator
        # each device in self._contexts MUST NOT be tuple
        return DeviceGroup(sorted(self._contexts, key=lambda x: '{}:{}:{}'.format(x.hostname, x.device_type, x.device_id)))

    def cur_worker(self) -> ContextUnit:
        if self.dp_index:
            return None
        else:
            return self._contexts[self.dp_index]

    def get_only(self) -> DLContext:
        assert self.server_num + self.worker_num == 1
        if self.server_num == 1:
            return self.servers[0]
        else:
            res = self.workers[0]
            if isinstance(res, tuple):
                assert len(res) == 1
                res = res[0]
            return res


class ContextStack(object):
    def __init__(self) -> None:
        self._stack: List[DeviceGroup] = []

    def peek(self) -> Optional[DeviceGroup]:
        return self._stack[-1] if self._stack else None

    def push(self, ctx: DeviceGroup) -> None:
        return self._stack.append(ctx)

    def pop(self) -> DeviceGroup:
        self._stack.pop()


_default_ctx_stack = ContextStack()


def get_current_context() -> Optional[DeviceGroup]:
    return _default_ctx_stack.peek()


@ contextlib.contextmanager
def context(ctx):
    try:
        ctx = DeviceGroup(ctx)
        _default_ctx_stack.push(ctx)
        yield ctx
    finally:
        _default_ctx_stack.pop()


def get_launch_config_by_traverse_nodes(
    node_list: List[Op],
    default_ctx: Union[DLContext, DeviceGroup],
) -> Tuple[bool, bool, Dict[Op, Optional[str]], Set[DLContext], int, int]:
    def gcd(a: int, b: int) -> int:
        if a < b:
            return gcd(b, a)
        if b == 0 or a == b:
            return a
        return gcd(b, a % b)

    def traverse_dfs(node: Op) -> None:
        from .optimizer import OptimizerOp
        from .dataloader import DataloaderOp
        if node in node_strategy:
            return
        strategy = None
        cur_worker_num = 1 if node.raw_ctx is None else node.raw_ctx.worker_num
        if not isinstance(node, DataloaderOp):
            nonlocal min_worker_num
            assert cur_worker_num == min_worker_num
        if node.raw_ctx is not None and node.raw_ctx.server_num > 0 and cur_worker_num > 0:
            strategy = 'PS'
        elif node.raw_ctx is not None and cur_worker_num > 1:
            strategy = 'AllReduce'
        node_strategy[node] = strategy
        if not isinstance(node, OptimizerOp):
            for ctx in node.raw_ctx:
                if isinstance(ctx, tuple):
                    devices.update(ctx)
                else:
                    devices.add(ctx)
        for n in node.inputs:
            traverse_dfs(n)

    node_strategy = dict()
    devices = set()
    for ctx in default_ctx:
        if isinstance(ctx, tuple):
            devices.update(ctx)
        else:
            devices.add(ctx)
    min_worker_num = default_ctx.worker_num
    launchPS = default_ctx.server_num > 0
    launchMPI = (not launchPS) and min_worker_num > 1
    for node in node_list:
        traverse_dfs(node)
    launchPS = launchPS or any([x == 'PS' for x in node_strategy.values()])
    launchMPI = launchMPI or any(
        [x == 'AllReduce' for x in node_strategy.values()])
    return launchMPI, launchPS, node_strategy, devices, min_worker_num


def assign_context_by_traverse_nodes(
    node_list: List[Op],
    ctx: DLContext,
) -> None:
    from .dataloader import DataloaderOp
    from .optimizer import OptimizerOp
    from .gpu_ops.Variable import PlaceholderOp

    def get_index(raw_ctx: DeviceGroup, ctx: DLContext) -> Tuple[int, bool]:
        dp_index = -1
        for i, c in enumerate(raw_ctx.workers):
            if ctx == c:
                dp_index = i
        return dp_index, dp_index >= 0

    def assign_ctx(node: Op) -> None:
        cur_ctx = node.raw_ctx
        if node in visited:
            return
        visited[node] = True
        if isinstance(node, DataloaderOp):
            return
        elif isinstance(node, PlaceholderOp):
            dp_index, local_dp = get_index(cur_ctx, ctx)
            if local_dp:
                node.ctx = ctx
        elif isinstance(node, OptimizerOp):
            for ori_grad in node.inputs:
                assign_ctx(ori_grad)
        else:
            dp_index, local_dp = get_index(cur_ctx, ctx)
            for i, n in enumerate(node.inputs):
                if isinstance(n, DataloaderOp):
                    if local_dp:
                        n.set_dp_rank(dp_index, cur_ctx.worker_num)
                    continue
                assign_ctx(n)

            if local_dp:
                node.ctx = ctx

    visited = {}

    for node in node_list:
        assign_ctx(node)


class DistConfig(object):
    def __init__(
        self,
        file: Optional[str] = None,
        num_local_servers: int = 0,
        num_local_workers: int = 1
    ) -> None:
        if file is None:
            assert num_local_workers > 0, \
                'Please specify the configuration file or set the number of local workers.'
            self.settings = {'nodes': [{
                'host': socket.gethostname(),
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

    def __str__(self) -> str:
        return '\n'.join([
            'Cluster: {',
            '  Chief: %s,' % self.chief,
            '  Servers(%d): %s,' % (self.num_servers, self.servers),
            '  Workers(%d): %s,' % (self.num_workers, self.workers),
            '}',
        ])

    def __iter__(self) -> Iterator[List[Dict[Any]]]:
        return iter(self.settings['nodes'])

    def save(self, path: str) -> None:
        with open(path, 'w') as fw:
            yaml.dump(self.settings, fw)

    def make_ps_config(self) -> Dict[str, Union[str, int]]:
        port = self.get_available_port(self.chief_address)
        return {
            'DMLC_PS_ROOT_URI': self.chief_address,
            'DMLC_PS_ROOT_PORT': port,
            'DMLC_NUM_WORKER': self.num_workers,
            'DMLC_NUM_SERVER': self.num_servers,
            'DMLC_PS_VAN_TYPE': 'p3'
        }

    def get_available_port(self, localhost: str) -> int:
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
