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
        pass

    def set_raw_ctxs(self):
        raise NotImplementedError


class DataParallel(Strategy):
    def __init__(self, aggregate=None):
        super().__init__()
        self.settings = DistConfig('/tmp/hetu_config.yml')
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
        visited = set()
        appending = list(eval_node_list)
        while appending:
            node = appending.pop(0)
            if node in visited:
                continue
            appending.extend(list(node.inputs))
            if isinstance(node, PlaceholderOp) and node.trainable and not node.is_embed:
                node.raw_ctx = self.raw_ctx
            else:
                node.raw_ctx = self.embedding_raw_ctx
            visited.add(node)

        return self.raw_ctx
