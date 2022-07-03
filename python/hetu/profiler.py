from time import time
import numpy as np
import hetu as ht
import pickle
from copy import copy
from collections import namedtuple, defaultdict
from ctypes import cast, byref, c_void_p, c_int
from .stream import create_event_handle, create_stream_handle
from .gpu_ops.Variable import PlaceholderOp
from .gpu_ops.EmbeddingLookUp import EmbeddingLookUp, EmbeddingLookUp_Gradient
from .gpu_ops.Sum import SumOp
from .gpu_ops.SoftmaxCrossEntropySparse import SoftmaxCrossEntropySparseOp, SoftmaxCrossEntropySparseGradientOp
from ._base import _LIB
import pynvml
from enum import Enum


class NCCLOP(Enum):
    AllReduce = 0
    AllGather = 1
    ReduceScatter = 2
    Reduce = 3
    Broadcast = 4


class BaseProfiler(object):
    def __init__(self):
        self.idx = 0  # index of feed arrays

    def renew_instances(self):
        raise NotImplementedError

    def set_ctx(self, ctx):
        self.ctx = ctx
        self.use_gpu = ht.is_gpu_ctx(ctx)
        self.stream = create_stream_handle(ctx) if self.use_gpu else None

    def init_memory(self, shape, seed):
        arr = ht.empty(shape, self.ctx)
        if self.use_gpu:
            from .gpu_links import normal_init
            normal_init(arr, 0., 0.01, seed, self.stream)
            self.stream.sync()
        else:
            from .cpu_links import normal_init
            from ._base import DNNL_LIB
            if DNNL_LIB['cpu_NormalInit']:
                normal_init(arr, 0., 0.01, seed)
            else:
                arr[:] = np.random.normal(
                    loc=0., scale=0.01, size=shape).astype(np.float32)
        return arr


class HetuProfiler(BaseProfiler):
    def __init__(self, computing_nodes, feed_shapes, node_to_arr_map, ctx=ht.gpu(0)):
        # we don't profile following nodes:
        # PS op: ParameterServerCommunicateOp, ParameterServerSparsePullOp
        # AllReduce op: AllReduceCommunicateOp
        # CPU-GPU transfer op: DataH2DOp, DataD2HOp, DataD2HSparseOp
        self.computing_nodes = computing_nodes
        # here the feed shapes should include DataloaderOp and PlaceholderOp
        self.feed_shapes = feed_shapes
        self.node_to_arr_map = node_to_arr_map
        self.set_ctx(ctx)
        self.timer = {node: 0.0 for node in self.computing_nodes}
        self.timer['all'] = 0.0
        self.idx = 0  # index of feed arrays
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.nodes_type = None

    def free_mem(self):
        cur_meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return cur_meminfo.free

    def renew_nodes(self, computing_nodes, feed_shapes, node_to_arr_map):
        self.computing_nodes = computing_nodes
        self.feed_shapes = feed_shapes
        self.has_indexedslices = any([isinstance(sh[0], tuple)
                                      for sh in self.feed_shapes.values()])
        self.node_to_arr_map = node_to_arr_map
        self.timer = {node: 0.0 for node in self.computing_nodes}
        self.timer['all'] = 0.0
        self.idx = 0
        self.nodes_type = type(computing_nodes[0]) if len(
            computing_nodes) == 1 else None
        if self.nodes_type == EmbeddingLookUp:
            assert len(
                self.feed_shapes) == 2, 'Embedding Lookup has only 2 inputs, got {} / {}; node type'.format(self.feed_shapes, self.computing_nodes, self.nodes_type)
            embed_node = computing_nodes[0]
            # here we have to determine the compute function,
            # since this is determined in forward_hook
            # TODO: whether and how to profile PS embedding lookup?
            if self.use_gpu:
                comp_func = embed_node._compute_gpu
            else:
                from ._base import DNNL_LIB
                if DNNL_LIB['cpu_EmbeddingLookup']:
                    comp_func = embed_node._compute_cpu_dnnl
                else:
                    comp_func = embed_node._compute_cpu_numpy
            embed_node.compute = comp_func

    def renew_sparse_update_optimizer(self, node, shape, indices_shape, values_shape):
        self.computing_nodes = [node]
        self.feed_shapes = (shape, indices_shape, values_shape)
        self.node_to_arr_map = {node: None}
        self.timer = {node: 0.0 for node in self.computing_nodes}
        self.timer['all'] = 0.0
        self.idx = 0
        self.nodes_type = EmbeddingLookUp_Gradient

    def renew_instances(self, num_instances=5):
        if self.nodes_type == EmbeddingLookUp:
            # this is for embedding lookup
            self.renew_embedding_instances(num_instances+5)
        elif self.nodes_type == EmbeddingLookUp_Gradient:
            # this is for sparse updates
            self.renew_sparse_update_instances(num_instances+5)
        elif self.nodes_type == SumOp and self.has_indexedslices:
            # consider indexedslices in inputs
            self.renew_sumop_instances(num_instances+5)
        elif self.nodes_type in (SoftmaxCrossEntropySparseOp, SoftmaxCrossEntropySparseGradientOp):
            self.renew_softmaxcrossentropysparse_instances(num_instances)
        else:
            self.feed_values = []
            all_size = sum([np.prod(shape, dtype=int)
                            for shape in self.feed_shapes.values()])
            cnt = 0
            seed = np.int64(time())
            # theoretically we need the free memory to be larger than all_size * 4
            # but some operators need workspace during computation
            # TODO: if workspace is controlled by operator in the future,
            # re-consider the judgement here
            while self.free_mem() > all_size * 8 and cnt < num_instances:
                self.feed_values.append({node: self.init_memory(
                    shape, seed+cnt) for node, shape in self.feed_shapes.items()})
                cnt += 1
            self.num_instances = cnt
            assert cnt > 0, 'Space not enough for profile!'

    def get_lookup_sampler(self, vocab_size, ignore_rate=0.):
        if vocab_size > 100000:
            # here we assume using recommendation system model
            assert False, 'Not support large-scale vocabulary size in embedding table now.'
        elif vocab_size > 10000:
            # here we assume using language model
            def zipf_sampler(shape):
                size = np.prod(shape, dtype=int)
                mask = np.random.uniform(0, 1, size=size) < ignore_rate
                arr = np.random.zipf(1.5, size=size).astype(np.float32)
                arr[mask] = -1
                for i in range(len(arr)):
                    if arr[i] > vocab_size:
                        arr[i] = np.random.randint(0, vocab_size)
                return arr.reshape(shape)
            sampler = zipf_sampler
        else:
            # here we assume using normal embedding table, with uniform distribution
            def normal_sampler(shape):
                return np.random.randint(0, vocab_size, size=shape)
            sampler = normal_sampler
        return sampler

    def renew_embedding_instances(self, num_instances=5):
        self.feed_values = {}
        shapes = list(self.feed_shapes.values())
        nodes = list(self.feed_shapes.keys())
        vocab_size, embed_dim = shapes[0]
        sampler = self.get_lookup_sampler(vocab_size)
        embed_size = vocab_size * embed_dim
        assert self.free_mem() > embed_size * \
            4, 'Space not enough for profiling embedding lookup!'
        self.feed_values[nodes[0]] = ht.empty(shapes[0], ctx=self.ctx)
        self.feed_values[nodes[1]] = []

        rest_size = np.prod(shapes[1], dtype=int)
        cnt = 0
        while self.free_mem() > rest_size * 4 and cnt < num_instances:
            self.feed_values[nodes[1]].append(
                ht.array(sampler(shapes[1]), ctx=self.ctx))
            cnt += 1
        self.num_instances = cnt
        assert cnt > 0, 'Space not enough for profile!'

    def renew_sparse_update_instances(self, num_instances=5):
        from .ndarray import IndexedSlices
        self.feed_values = []
        shape, indices_shape, values_shape = self.feed_shapes
        sampler = self.get_lookup_sampler(shape[0])
        size = np.prod(indices_shape, dtype=int) + \
            np.prod(values_shape, dtype=int)
        cnt = 0
        while self.free_mem() > size * 4 and cnt < num_instances:
            self.feed_values.append(IndexedSlices(indices=ht.array(sampler(
                indices_shape), ctx=self.ctx), values=ht.empty(values_shape, ctx=self.ctx), dense_shape=shape))
            cnt += 1
        self.num_instances = cnt
        assert cnt > 0, 'Space not enough for profile!'

    def renew_sumop_instances(self, num_instances=5):
        from .ndarray import IndexedSlices
        self.feed_values = {}
        indexedslices = [isinstance(sh[0], tuple)
                         for sh in self.feed_shapes.values()]
        shape = self.node_to_arr_map[self.computing_nodes[0]].shape
        sampler = self.get_lookup_sampler(shape[0])
        for (n, sh), indslc in zip(self.feed_shapes.items(), indexedslices):
            if indslc:
                self.feed_values[n] = []
                for _ in range(num_instances):
                    self.feed_values[n].append(IndexedSlices(indices=ht.array(sampler(
                        sh[0]), ctx=self.ctx), values=ht.empty(sh[1], ctx=self.ctx), dense_shape=shape))
            else:
                self.feed_values[n] = ht.empty(shape, ctx=self.ctx)
        self.num_instances = num_instances

    def renew_softmaxcrossentropysparse_instances(self, num_instances=5):
        self.feed_values = []
        node = self.computing_nodes[0]
        shape = self.feed_shapes[node.inputs[0]]
        # the ignore rate is for ignore indices, here use for bert
        # TODO: set ignore rate according to specific datasets
        sampler = self.get_lookup_sampler(shape[1], ignore_rate=0.9)
        all_size = sum([np.prod(shape, dtype=int)
                        for shape in self.feed_shapes.values()])
        cnt = 0
        seed = np.int64(time())
        # theoretically we need the free memory to be larger than all_size * 4
        # but some operators need workspace during computation
        # TODO: if workspace is controlled by operator in the future,
        # re-consider the judgement here
        while self.free_mem() > all_size * 8 and cnt < num_instances:
            cur_arr = {}
            for i, n in enumerate(node.inputs):
                if i == 1:
                    cur_arr[n] = ht.array(
                        sampler(self.feed_shapes[n]), ctx=self.ctx)
                else:
                    cur_arr[n] = self.init_memory(
                        self.feed_shapes[n], seed+cnt)
            self.feed_values.append(cur_arr)
            cnt += 1
        self.num_instances = cnt
        assert cnt > 0, 'Space not enough for profile!'

    def step_idx(self):
        self.idx = (self.idx + 1) % self.num_instances

    def feed_in_arr_map(self):
        if self.nodes_type == EmbeddingLookUp:
            nodes = list(self.feed_shapes.keys())
            self.node_to_arr_map[nodes[0]] = self.feed_values[nodes[0]]
            self.node_to_arr_map[nodes[1]
                                 ] = self.feed_values[nodes[1]][self.idx]
        elif self.nodes_type == EmbeddingLookUp_Gradient:
            node = self.computing_nodes[0].inputs[0]
            self.node_to_arr_map[node] = self.feed_values[self.idx]
        elif self.nodes_type == SumOp and self.has_indexedslices:
            for n in self.computing_nodes[0].inputs:
                values = self.feed_values[n]
                if isinstance(values, list):
                    self.node_to_arr_map[n] = values[self.idx]
                else:
                    self.node_to_arr_map[n] = values
        else:
            feed_value = self.feed_values[self.idx]
            for node, value in feed_value.items():
                self.node_to_arr_map[node] = value
        self.step_idx()

    def pure_compute(self):
        for node in self.computing_nodes:
            input_vals = [self.node_to_arr_map[n] for n in node.inputs]
            node_val = self.node_to_arr_map[node]
            node.compute(input_vals, node_val, self.stream)

    def profile(self, num_iterations=100, profiler='gpu'):
        assert profiler in ('cpu', 'gpu')
        self.renew_instances()
        # we don't record the first 5 iterations
        for _ in range(5):
            self.feed_in_arr_map()
            self.pure_compute()
        if self.use_gpu:
            self.stream.sync()

        if profiler == 'cpu':
            for _ in range(5, 5 + num_iterations):
                self.feed_in_arr_map()
                for node in self.computing_nodes:
                    input_vals = [self.node_to_arr_map[n] for n in node.inputs]
                    node_val = self.node_to_arr_map[node]
                    start = time()
                    node.compute(input_vals, node_val, self.stream)
                    if self.use_gpu:
                        self.stream.sync()
                    ending = time()
                    self.timer[node] += (ending - start) * 1000

        else:
            assert self.use_gpu
            start_event = create_event_handle(self.ctx)
            ending_event = create_event_handle(self.ctx)
            for _ in range(5, 5 + num_iterations):
                self.feed_in_arr_map()
                for node in self.computing_nodes:
                    input_vals = [self.node_to_arr_map[n] for n in node.inputs]
                    node_val = self.node_to_arr_map[node]
                    start_event.record(self.stream)
                    node.compute(input_vals, node_val, self.stream)
                    ending_event.record(self.stream)
                    ending_event.sync()
                    duration = ending_event.time_since(start_event)
                    self.timer[node] += duration

        for node in self.timer:
            self.timer[node] /= num_iterations
        self.clean()
        return self.timer

    def profile_all(self, num_iterations=100, profiler='gpu'):
        def get_result():
            # we don't record the first 5 iterations
            if profiler == 'cpu':
                for _ in range(5):
                    self.feed_in_arr_map()
                    self.pure_compute()
                if self.use_gpu:
                    self.stream.sync()
                start = time()
                for _ in range(5, 5 + num_iterations):
                    self.feed_in_arr_map()
                    self.pure_compute()
                if self.use_gpu:
                    self.stream.sync()
                ending = time()
                result = (ending - start) * 1000
            else:
                assert self.use_gpu
                start_event = create_event_handle(self.ctx)
                ending_event = create_event_handle(self.ctx)
                for _ in range(5):
                    self.feed_in_arr_map()
                    self.pure_compute()
                start_event.record(self.stream)
                for _ in range(5, 5 + num_iterations):
                    self.feed_in_arr_map()
                    self.pure_compute()
                ending_event.record(self.stream)
                ending_event.sync()
                duration = ending_event.time_since(start_event)
                result = duration
            return result / num_iterations

        assert profiler in ('cpu', 'gpu')
        if self.nodes_type == EmbeddingLookUp_Gradient:
            self.renew_instances(num_iterations)
        else:
            self.renew_instances()
        if self.num_instances == 1:
            # we repeat several times in this case
            result = get_result()
            for _ in range(2):
                self.renew_instances(1)
                result += get_result()
            result /= 3
        else:
            result = get_result()

        self.timer['all'] = result
        self.clean()
        return self.timer

    def profile_n_log(self, log_file, profiler='cpu'):
        timer = self.profile(profiler=profiler)
        with open(log_file, 'w') as fw:
            for k, v in timer.items():
                print(k, v, 'ms', file=fw, flush=True)

    def clean(self):
        del self.feed_values
        self.computing_nodes = []
        self.feed_shapes = {}
        self.node_to_arr_map = {}
        _LIB.clear_chunk()


class NCCLProfiler(BaseProfiler):
    def __init__(self):
        from .gpu_ops.executor import wrapped_mpi_nccl_init, get_mpi_communicate, new_group_comm
        from .communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t
        self.nccl_comm = wrapped_mpi_nccl_init()
        self.mpi_comm = get_mpi_communicate()
        nrank = self.nccl_comm.nrank
        assert nrank > 1
        self.set_ctx(ht.gpu(self.nccl_comm.dev_id))
        self.ar_dtype = ncclDataType_t.ncclFloat32
        self.ar_op = ncclRedOp_t.ncclSum
        self.idx = 0
        self.is_root = (self.mpi_comm.rank) == 0
        if self.is_root:
            self.start_event = create_event_handle(self.ctx)
            self.ending_event = create_event_handle(self.ctx)

        from .context import DeviceGroup, DistConfig
        self.settings = DistConfig('/tmp/hetu_config.yml')
        self.workers = {host['host']: host['workers']
                        for host in self.settings}
        all_num_workers = sum(self.workers.values())
        assert max(self.workers.values(
        )) == self.workers[self.settings.chief], 'Chief should have the most workers to enable profiling!'
        # initialize group communicators
        local_topo_comb = {
            2: [(0, 1)],
            3: [(0, 1), (0, 2), (0, 1, 2)],
            4: [(0, 1), (0, 2), (0, 1, 2), (0, 1, 2, 3)],
            5: [(0, 1), (0, 2), (0, 4), (0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 3, 4)],
            6: [(0, 1), (0, 2), (0, 4), (0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 4, 5), (0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5)],
            7: [(0, 1), (0, 2), (0, 4), (0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 4, 5), (0, 1, 4, 6), (0, 2, 4, 6), (0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5, 6)],
            8: [(0, 1), (0, 2), (0, 4), (0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 4, 5), (0, 1, 4, 6), (0, 2, 4, 6), (0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4, 5, 6, 7)],
        }[self.workers[self.settings.chief]]
        self.group_comms = {}
        for comb in local_topo_comb:
            if len(comb) == all_num_workers:
                self.group_comms[comb] = self.nccl_comm
            else:
                self.group_comms[comb] = new_group_comm(
                    DeviceGroup([ht.gpu(dev) for dev in comb]))
        if len(self.workers) > 1:
            alloc_dict = self.get_alloc_among_hosts()
            for key, value in alloc_dict.items():
                if sum(value.values()) == all_num_workers:
                    self.group_comms[(-1,) + key] = self.nccl_comm
                else:
                    self.group_comms[(-1,) + key] = new_group_comm(DeviceGroup([ht.rgpu(host, dev)
                                                                                for host, num_worker in value.items() for dev in range(num_worker)]))
        else:
            self.host_index = 0

    def get_alloc_among_hosts(self):
        def get_alloc(depth, rest, max_num, cur_comb):
            if rest == 0:
                results.append(tuple(x for x in cur_comb if x != 0))
            elif depth == num_hosts - 1:
                assert rest <= max_num
                cur_comb[depth] = rest
                results.append(tuple(x for x in cur_comb if x != 0))
                cur_comb[depth] = 0
            else:
                rest_dev_num = (num_hosts - depth)
                min_num = (rest + rest_dev_num - 1) // rest_dev_num
                for cur_num in range(min_num, max_num + 1):
                    cur_comb[depth] = cur_num
                    get_alloc(depth+1, rest - cur_num, cur_num, cur_comb)
                    cur_comb[depth] = 0
        workers = self.workers
        num_hosts = len(workers)
        num_workers = sum(workers.values())
        cur_num_workers = 2
        results = []
        sorted_workers = sorted(
            list(workers.items()), key=lambda x: (-x[1], x[0] != self.settings.chief))
        if len(sorted_workers) > 1:
            self.index1_host = sorted_workers[1][0]
        self.host_index = {x[0]: i for i, x in enumerate(sorted_workers)}[
            self.mpi_comm.hostname]
        while cur_num_workers <= num_workers:
            cur_num_hosts = 2
            while cur_num_hosts <= num_hosts:
                get_alloc(0, cur_num_workers, cur_num_workers -
                          1, [0 for _ in range(cur_num_hosts)])
                cur_num_hosts += 1
            cur_num_workers += 1
        all_results = {}
        for res in results:
            flag = True
            new_res = {}
            for required, provided in zip(res, sorted_workers):
                if required > provided[1]:
                    flag = False
                    break
                else:
                    new_res[provided[0]] = required
            if flag:
                assert self.settings.chief in new_res
                all_results[res] = new_res
        return all_results

    def renew_instances(self, shape, nrank=None, primitive=None):
        feed_value = self.init_memory(shape, np.int64(time()))
        if primitive is not None:
            assert nrank is not None
            if primitive == NCCLOP.AllGather:
                shape = (nrank * shape[0],) + shape[1:]
            elif primitive == NCCLOP.ReduceScatter:
                shape = (shape[0] // nrank,) + shape[1:]
        output_value = ht.empty(shape, ctx=self.ctx)
        return feed_value, output_value

    def profile_allreduce(self, size, devices, num_iterations=10, primitive=NCCLOP.AllReduce):
        def get_void_pointer(var):
            return cast(byref(var), c_void_p)

        if self.mpi_comm.rank == 0:
            info = (c_int * 2)(size, len(devices))
        else:
            info = (c_int * 2)()
        self.mpi_comm.MPI_Broadcast(get_void_pointer(info), 8)
        size, ndev = info
        devices = (c_int * ndev)(*devices)
        self.mpi_comm.MPI_Broadcast(get_void_pointer(devices), 4 * ndev)
        devices = tuple(devices)
        assert devices[0] == - \
            1 or 0 in devices, 'Root device should be in profiling devices.'
        duration = 0
        if (devices[0] != -1 and self.host_index == 0 and self.mpi_comm.local_rank in devices) or \
                (devices[0] == -1 and self.host_index + 1 < len(devices) and self.mpi_comm.local_rank < devices[self.host_index + 1]):
            comm = self.group_comms[devices]
            feed_value, output_value = self.renew_instances(
                (size,), nrank=comm.nrank, primitive=primitive)
            operation = {
                NCCLOP.AllReduce: comm.dlarrayNcclAllReduce,
                NCCLOP.AllGather: comm.dlarrayAllGather,
                NCCLOP.ReduceScatter: comm.dlarrayReduceScatter,
                NCCLOP.Reduce: comm.dlarrayNcclReduce,
                NCCLOP.Broadcast: comm.dlarrayBroadcast,
            }[primitive]
            args = {
                NCCLOP.AllReduce: [feed_value, output_value, self.ar_dtype, self.ar_op, self.stream],
                NCCLOP.AllGather: [feed_value, output_value, self.ar_dtype, self.stream],
                NCCLOP.ReduceScatter: [feed_value, output_value, self.ar_dtype, self.ar_op, self.stream],
                NCCLOP.Reduce: [feed_value, output_value, 0, self.ar_dtype, self.ar_op, self.stream],
                NCCLOP.Broadcast: [feed_value, output_value, self.ar_dtype, 0, self.stream],
            }[primitive]
            # warming up
            for _ in range(5):
                operation(*args)
            if self.is_root:
                self.start_event.record(self.stream)
            for _ in range(5, 5 + num_iterations):
                operation(*args)
            if self.is_root:
                self.ending_event.record(self.stream)
                self.ending_event.sync()
                duration = self.ending_event.time_since(
                    self.start_event) / num_iterations
            del feed_value
            del output_value
        return duration

    def profile_sendrecv(self, size, devices, num_iterations=10):

        def get_void_pointer(var):
            return cast(byref(var), c_void_p)

        if self.mpi_comm.rank == 0:
            assert devices[0] == 0 and devices[1] in (1, 2, 4, 8)
            info = (c_int * 2)(size, devices[1])
        else:
            info = (c_int * 2)()
        self.mpi_comm.MPI_Broadcast(get_void_pointer(info), 8)
        size, target = info
        devices = (0, target)
        duration = 0
        if self.mpi_comm.rank == 0:
            comm = self.nccl_comm
            if target < 8:
                real_target = self.mpi_comm.getRankFromDevice(
                    self.settings.chief, target)
            else:
                real_target = self.mpi_comm.getRankFromDevice(
                    self.index1_host, 0)
            feed_value, output_value = self.renew_instances((size,))
            for _ in range(5):
                comm.dlarraySend(
                    feed_value, self.ar_dtype, real_target, self.stream)
                comm.dlarrayRecv(
                    output_value, self.ar_dtype, real_target, self.stream)
            self.start_event.record(self.stream)
            for _ in range(5, 5 + num_iterations):
                comm.dlarraySend(
                    feed_value, self.ar_dtype, real_target, self.stream)
                comm.dlarrayRecv(
                    output_value, self.ar_dtype, real_target, self.stream)
            self.ending_event.record(self.stream)
            self.ending_event.sync()
            duration = self.ending_event.time_since(
                self.start_event) / num_iterations / 2
            del feed_value
            del output_value
        elif (devices[1] in (1, 2, 4) and self.host_index == 0 and self.mpi_comm.local_rank == target) or \
                (devices[1] == 8 and self.host_index == 1 and self.mpi_comm.local_rank == 0):
            comm = self.nccl_comm
            feed_value, output_value = self.renew_instances((size,))
            output_value = ht.empty((size,), ctx=self.ctx)
            for _ in range(5 + num_iterations):
                comm.dlarrayRecv(
                    output_value, self.ar_dtype, 0, self.stream)
                comm.dlarraySend(
                    feed_value, self.ar_dtype, 0, self.stream)
            self.stream.sync()
            del feed_value
            del output_value
        return duration


class HetuSimulator(object):
    def __init__(self, feed_shapes, ctx, mpi_comm, num_ctxs, pix=True, cache_path='/tmp/hetu_cached_exetime.bin'):
        # use sythetic data for profiling is not good enough for some nodes:
        # such as softmax_cross_entropy_sparse, need to consider ignore_index
        # such as sparse updates, the position embedding has a fixed pattern
        # TODO: consider whether and how to use real data to profile
        self.pix = pix  # indicate whether two GPU share a PCIe bridge
        self.mpi_comm = mpi_comm
        if mpi_comm.nrank > 1:
            self.nccl_profiler = NCCLProfiler()
        if mpi_comm.rank == 0:
            self.feed_shapes = feed_shapes
            self.num_ctxs = num_ctxs
            self.ctx = ctx

            # initialize cache
            self.cache_path = cache_path
            try:
                with open(self.cache_path, 'rb') as fr:
                    self.cached_exetime = pickle.load(fr)
            except:
                self.cached_exetime = {}
            self.cached_placeholders = [PlaceholderOp(
                'test_node', ctx=self.ctx)]
            self.cached_optimizer = None
            self.cached_config = namedtuple(
                'Config', 'placeholder_to_arr_map')(placeholder_to_arr_map={})

            # initialize profiler
            self.profiler = HetuProfiler([], {}, {})

            # these nodes we don't profile
            from .gpu_ops.BatchNorm import Batch_Normalization_Gradient_of_DataOp, Batch_Normalization_Gradient_of_ScaleOp, Batch_Normalization_Gradient_of_BiasOp
            from .gpu_ops.LayerNorm import Layer_Normalization_Gradient_of_DataOp, Layer_Normalization_Gradient_of_ScaleOp, Layer_Normalization_Gradient_of_BiasOp
            from .gpu_ops.Reshape import Array_ReshapeOp, Array_Reshape_GradientOp
            self.no_computing_nodes = (
                PlaceholderOp,
                Batch_Normalization_Gradient_of_DataOp, Batch_Normalization_Gradient_of_ScaleOp, Batch_Normalization_Gradient_of_BiasOp,
                Layer_Normalization_Gradient_of_DataOp, Layer_Normalization_Gradient_of_ScaleOp, Layer_Normalization_Gradient_of_BiasOp,
                EmbeddingLookUp_Gradient,
                Array_ReshapeOp, Array_Reshape_GradientOp,
            )

    def init_empty_optimizer(self, optimizer, cached=False):
        new_optimizer = copy(optimizer)
        del new_optimizer.loss
        del new_optimizer.params
        del new_optimizer.backward2forward
        del new_optimizer.forward2backward
        if cached:
            self.cached_optimizer = new_optimizer
            self.cached_placeholders[0].on_cpu = False
            self.cached_placeholders[0].on_gpu = True
        return new_optimizer

    def profile_new_case(self, new_node, input_shapes, output_shape=None):
        new_node.ctx = self.ctx
        new_node.on_cpu = False
        new_node.on_gpu = True
        num_cur_ph = len(self.cached_placeholders)
        num_inputs = len(input_shapes)
        if num_cur_ph < num_inputs:
            self.cached_placeholders.extend([PlaceholderOp(
                'test_node', ctx=self.ctx) for _ in range(num_inputs - num_cur_ph)])
        # return 0.
        new_node.inputs = self.cached_placeholders[:len(input_shapes)]
        new_shape = new_node.infer_shape([sh for sh in input_shapes])
        if output_shape is not None:
            assert output_shape == new_shape, \
                'Inferred shape not correct: {}, {} vs {}.'.format(
                    new_node, output_shape, new_shape)
        node_to_arr_map = {new_node: ht.empty(
            new_shape, ctx=self.ctx) if new_shape is not None else None}
        self.profiler.renew_nodes(
            [new_node], {n: sh for n, sh in zip(new_node.inputs, input_shapes)}, node_to_arr_map)
        try:
            result = self.profiler.profile_all(
                num_iterations=10, profiler='gpu')['all']
        except:
            # if memory not enough
            result = 1e20
        del node_to_arr_map
        return result

    def profile_new_sum_case(self, new_node, input_shapes, output_shape):
        # input shapes should contains indices shape and values shape if indexedslices
        new_node.ctx = self.ctx
        new_node.on_cpu = False
        new_node.on_gpu = True
        num_cur_ph = len(self.cached_placeholders)
        num_inputs = len(input_shapes)
        indexedslices = [isinstance(sh[0], tuple) for sh in input_shapes]
        if num_cur_ph < num_inputs:
            self.cached_placeholders.extend([PlaceholderOp(
                'test_node', ctx=self.ctx) for _ in range(num_inputs - num_cur_ph)])
        # return 0.
        new_node.inputs = self.cached_placeholders[:len(input_shapes)]
        new_node.callbacks = [
            new_node._indexed_gpu_callback if indslcs else new_node._simple_gpu_callback for indslcs in indexedslices]
        node_to_arr_map = {new_node: ht.empty(
            output_shape, ctx=self.ctx)}
        self.profiler.renew_nodes(
            [new_node], {n: sh for n, sh in zip(new_node.inputs, input_shapes)}, node_to_arr_map)
        result = self.profiler.profile_all(
            num_iterations=10, profiler='gpu')
        del node_to_arr_map
        return result['all']

    def profile_new_sparse_case(self, new_node, shape, indices_shape, values_shape):
        new_node.ctx = self.ctx
        new_node.on_cpu = False
        new_node.on_gpu = True
        num_cur_ph = len(self.cached_placeholders)
        num_inputs = 1
        if num_cur_ph < num_inputs:
            self.cached_placeholders.extend([PlaceholderOp(
                'test_node', ctx=self.ctx) for _ in range(num_inputs - num_cur_ph)])
        # return 0.
        new_node.inputs = self.cached_placeholders[:1]
        self.profiler.renew_sparse_update_optimizer(
            new_node, shape, indices_shape, values_shape)
        result = self.profiler.profile_all(
            num_iterations=10, profiler='gpu')
        return result['all']

    def get_node_time(self, node, input_shapes, output_shape):
        node_type = type(node)
        if node_type in self.no_computing_nodes:
            return 0.
        key = (node_type, output_shape) + tuple(input_shapes)
        if key not in self.cached_exetime:
            new_node = copy(node)
            new_node.inputs = []
            if hasattr(new_node, 'compute_to_be_config'):
                new_node.compute_to_be_config = True
            # if hasattr(new_node, 'grad_node'):
            #     new_node.grad_node = None
            # if hasattr(new_node, 'grad_nodes'):
            #     del new_node.grad_nodes
            if hasattr(new_node, 'forward_node'):
                from .gpu_ops.BatchNorm import Batch_Normalization_GradientOp
                from .gpu_ops.LayerNorm import Layer_Normalization_GradientOp
                from .gpu_ops.InstanceNorm2d import Instance_Normalization2d_GradientOp
                from .gpu_ops.Dropout import Dropout_Gradient_recomputeOp
                del new_node.forward_node
                seed = np.int64(time())
                if node_type == Batch_Normalization_GradientOp:
                    temp_shape = (input_shapes[0][1],)
                    forward_node = namedtuple('forward_node', ['save_mean', 'save_var'])(
                        save_mean=self.profiler.init_memory(
                            temp_shape, seed),
                        save_var=self.profiler.init_memory(
                            temp_shape, seed+1),
                    )
                    new_node.forward_node = forward_node
                    new_node.tmp_gradient_bn_scale = ht.empty(
                        input_shapes[2], self.ctx)
                    new_node.tmp_gradient_bn_bias = ht.empty(
                        input_shapes[2], self.ctx)
                    new_node.tmp_gradient_in_arr = ht.empty(
                        input_shapes[1], self.ctx)
                elif node_type == Layer_Normalization_GradientOp:
                    temp_shape = tuple(input_shapes[0][:-1]) + (1,)
                    forward_node = namedtuple('forward_node', ['save_mean', 'save_var'])(
                        save_mean=self.profiler.init_memory(temp_shape, seed),
                        save_var=self.profiler.init_memory(temp_shape, seed+1),
                    )
                    new_node.forward_node = forward_node
                    new_node.tmp_gradient_ln_scale = ht.empty(
                        input_shapes[2], self.ctx)
                    new_node.tmp_gradient_ln_bias = ht.empty(
                        input_shapes[2], self.ctx)
                    new_node.tmp_gradient_in_arr = ht.empty(
                        input_shapes[0], self.ctx)
                elif node_type == Instance_Normalization2d_GradientOp:
                    temp_shape = tuple(input_shapes[0][:-2]) + (1, 1)
                    forward_node = namedtuple('forward_node', ['save_mean', 'save_var', 'eps'])(
                        save_mean=self.profiler.init_memory(temp_shape, seed),
                        save_var=self.profiler.init_memory(temp_shape, seed+1),
                        eps=0.0000001,
                    )
                    new_node.forward_node = forward_node
                elif node_type == Dropout_Gradient_recomputeOp:
                    import ctypes
                    forward_node = namedtuple('forward_node', ['seed'])(
                        seed=ctypes.c_ulonglong(int(time()))
                    )
                    new_node.forward_node = forward_node
            if node_type == SumOp:
                result = self.profile_new_sum_case(
                    new_node, input_shapes, output_shape)
            else:
                result = self.profile_new_case(
                    new_node, input_shapes, output_shape)
            self.cached_exetime[key] = result
        return self.cached_exetime[key]

    def get_split_time(self, input_shape, axes, inds, splits, return_shape=False):
        from .gpu_ops.Split import split_op, SplitOp
        output_shape = self.get_split_shape(
            dict(zip(axes, splits)), input_shape)
        # TODO: consider whether add indices in key
        key = (SplitOp, output_shape, input_shape)
        if key not in self.cached_exetime:
            new_node = split_op(
                self.cached_placeholders[0], axes, inds, splits, ctx=self.ctx)
            self.cached_exetime[key] = self.profile_new_case(
                new_node, [input_shape])
        if return_shape:
            return self.cached_exetime[key], output_shape
        else:
            return self.cached_exetime[key]

    def get_concatenate_time(self, input_shapes, axis, return_shape=False):
        from .gpu_ops.Concatenate import concatenate_op, ConcatenateOp
        output_shape = self.get_concatenate_shape(input_shapes, axis)
        key = (ConcatenateOp, output_shape) + tuple(input_shapes)
        if key not in self.cached_exetime:
            new_node = concatenate_op(
                self.cached_placeholders, axis=axis, ctx=self.ctx)
            self.cached_exetime[key] = self.profile_new_case(
                new_node, input_shapes)
        if return_shape:
            return self.cached_exetime[key], output_shape
        else:
            return self.cached_exetime[key]

    def get_sum_time(self, input_shapes, return_shape=False):
        # here we assume no indexedslices
        from .gpu_ops.Sum import sum_op, SumOp
        key = (SumOp, input_shapes[0]) + tuple(input_shapes)
        if key not in self.cached_exetime:
            new_node = sum_op(self.cached_placeholders, ctx=self.ctx)
            self.cached_exetime[key] = self.profile_new_case(
                new_node, input_shapes)
        if return_shape:
            return self.cached_exetime[key], input_shapes[0]
        else:
            return self.cached_exetime[key]

    def get_update_time(self, shape, sparse_shape=None):
        from .optimizer import OptimizerOp
        key = (OptimizerOp, type(self.cached_optimizer), shape)
        is_sparse_update = sparse_shape is not None
        if is_sparse_update:
            indices_shape, values_shape = sparse_shape
            key += (indices_shape, values_shape)
        if key not in self.cached_exetime:
            param = self.cached_placeholders[0]
            self.cached_optimizer.params = [param]
            if is_sparse_update:
                assert shape[0] <= 100000, 'Not support too large table.'
            self.cached_config.placeholder_to_arr_map[param] = ht.empty(
                shape, ctx=self.ctx)
            self.cached_optimizer.initiated = False
            self.cached_optimizer.initiate_states(self.cached_config)
            new_node = OptimizerOp(
                self.cached_placeholders, self.cached_optimizer)
            new_node.comm_mode = None
            if is_sparse_update:
                self.cached_exetime[key] = self.profile_new_sparse_case(
                    new_node, shape, indices_shape, values_shape)
            else:
                self.cached_exetime[key] = self.profile_new_case(
                    new_node, [shape])
            self.cached_optimizer.uninitiate_states()
            del self.cached_config.placeholder_to_arr_map[param]
        return self.cached_exetime[key]

    def get_comm_time(self, from_device, to_device, shape):
        from .gpu_ops.PipelineSend import PipelineSendOp
        cur_size = np.prod(shape, dtype=int)
        distance = self.get_dev_distance(from_device, to_device)
        key = (PipelineSendOp, cur_size, distance)
        if key not in self.cached_exetime:
            # send signal
            signal = c_int(3)
            self.mpi_comm.MPI_Broadcast(
                cast(byref(signal), c_void_p), 4, root=0)
            # profile send/recv
            self.cached_exetime[key] = self.nccl_profiler.profile_sendrecv(
                cur_size, [0, distance])
        return self.cached_exetime[key]

    def get_group_comm_time(self, dev_n_shape):
        # now only consider adjacent devices are in single PCIe switch
        # dev_n_shape is a list of tuples (from_device, to_device, shape)
        send_time = self.HostDictionary(
            self.nccl_profiler.workers, pix=self.pix)
        recv_time = self.HostDictionary(
            self.nccl_profiler.workers, pix=self.pix)
        for from_dev, to_dev, shape in dev_n_shape:
            comm_time = self.get_comm_time(from_dev, to_dev, shape)
            send_time.add_to_channel(from_dev, comm_time)
            recv_time.add_to_channel(to_dev, comm_time)
        return max(send_time.all_values() + recv_time.all_values())

    def get_allreduce_time(self, shape, device_group, primitive=NCCLOP.AllReduce):
        from .gpu_ops.AllReduceCommunicate import AllReduceCommunicateOp

        def add_to_cache(key, dev_list):
            if key not in self.cached_exetime:
                # send signal
                self.mpi_comm.MPI_Broadcast(
                    cast(byref(signal), c_void_p), 4, root=0)
                # profile allreduce
                self.cached_exetime[key] = self.nccl_profiler.profile_allreduce(
                    cur_size, dev_list, primitive=primitive)

        hostnames = set([dev.hostname for dev in device_group])
        signal = {
            NCCLOP.AllReduce: c_int(2),
            NCCLOP.AllGather: c_int(4),
            NCCLOP.ReduceScatter: c_int(5),
            NCCLOP.Reduce: c_int(6),
            NCCLOP.Broadcast: c_int(7),
        }[primitive]
        cur_size = np.prod(shape, dtype=int)
        if len(hostnames) == 1:
            device_group = sorted([dev.device_id for dev in device_group])
            partial = len(device_group)
            assert 1 < partial <= 8
            if partial not in (2, 4):
                key = (AllReduceCommunicateOp, cur_size,
                       partial) + (primitive.value,)
                add_to_cache(key, list(range(partial)))
            elif partial == 2:
                d0, d1 = device_group
                distance = 0
                while d0 != d1:
                    distance += 1
                    d0 //= 2
                    d1 //= 2
                key = (AllReduceCommunicateOp, cur_size, 2,
                       distance) + (primitive.value,)
                add_to_cache(key, [0, 2 ** (distance - 1)])
            elif partial == 4:
                if device_group in ([0, 1, 2, 3], [4, 5, 6, 7]):
                    device_group = [0, 1, 2, 3]
                else:
                    devs = [d // 2 for d in device_group]
                    same_bucket = len(np.unique(devs))
                    if same_bucket == 4:
                        device_group = [0, 2, 4, 6]
                    elif same_bucket == 2:
                        device_group = [0, 1, 4, 5]
                    else:
                        devs = [d // 2 for d in devs]
                        if devs[0] == devs[1] and devs[2] == devs[3]:
                            device_group = [0, 1, 4, 6]
                        else:
                            device_group = [0, 1, 2, 4]
                key = (AllReduceCommunicateOp, cur_size,
                       4, tuple(device_group)) + (primitive.value,)
                add_to_cache(key, device_group)
        else:
            # multiple hosts
            cnts = defaultdict(int)
            for dev in device_group:
                cnts[dev.hostname] += 1
            multi_host_topo = (-1,) + \
                tuple(sorted(cnts.values(), key=lambda x: -x))
            key = (AllReduceCommunicateOp, cur_size,
                   multi_host_topo) + (primitive.value,)
            add_to_cache(key, multi_host_topo)
        return self.cached_exetime[key]

    def wrapped_get_allreduce_time(self, shape, device_group, status, primitive=NCCLOP.AllReduce, dim=None):
        if dim is None:
            dim = {
                NCCLOP.AllReduce: -2,
                NCCLOP.AllGather: 0,
                NCCLOP.ReduceScatter: -2,
                NCCLOP.Reduce: -2,
                NCCLOP.Broadcast: -1,  # here need target status and device group
            }[primitive]
        allreduce_devices = status.get_devices_by_dim(
            device_group, dim, dp_index=0, mp_index=0, return_device_group=False)
        assert allreduce_devices is not None, 'No need to allreduce, node status {}, device group {}.'.format(
            status, device_group)
        return self.get_allreduce_time(shape, allreduce_devices, primitive=primitive)

    def get_allgather_time(self, indices_shape, value_shape, device_group):
        return self.get_allreduce_time(indices_shape, device_group, primitive=NCCLOP.AllGather) + \
            self.get_allreduce_time(
                value_shape, device_group, primitive=NCCLOP.AllGather)

    def wrapped_get_allgather_time(self, indices_shape, value_shape, device_group, status):
        return self.wrapped_get_allreduce_time(indices_shape, device_group, status, primitive=NCCLOP.AllGather, dim=-2) + \
            self.wrapped_get_allreduce_time(
                value_shape, device_group, status, primitive=NCCLOP.AllGather, dim=-2)

    def get_general_comm_time(self, pre_status, tar_status, pre_rawctx, tar_rawctx, shape, use_nccl_collectives=True):
        deduce_mp = (pre_status != tar_status or pre_rawctx != tar_rawctx) \
            and (pre_status.is_dist() or tar_status.is_dist())
        if deduce_mp:
            prev_state, prev_duplicate, prev_order = pre_status.get_all()
            target_state, target_duplicate, target_order = tar_status.get_all()
            prev_partial = pre_status.partial
            prev_devices = pre_rawctx.workers[0] if pre_rawctx.is_mp else [
                pre_rawctx.workers[0]]
            target_devices = tar_rawctx.workers[0] if tar_rawctx.is_mp else [
                tar_rawctx.workers[0]]
            same_ctxs = (prev_devices == target_devices and isinstance(
                prev_devices, tuple))
            bcast_ctxs = False
            rdc_ctxs = False
            if isinstance(target_devices, tuple):
                num_prev = len(prev_devices)
                num_cur = len(target_devices)
                if num_prev < num_cur and num_cur % num_prev == 0:
                    bcast_ctxs = True
                    last_ind = None
                    gap = num_cur // num_prev
                    for dev in prev_devices:
                        if dev not in target_devices:
                            bcast_ctxs = False
                            break
                        else:
                            ind = target_devices.index(dev)
                            if last_ind is None:
                                last_ind = ind
                            elif ind - last_ind != gap:
                                bcast_ctxs = False
                                break
            if isinstance(prev_devices, tuple):
                num_prev = len(prev_devices)
                num_cur = len(target_devices)
                if num_cur < num_prev and num_prev % num_cur == 0:
                    rdc_ctxs = True
                    last_ind = None
                    gap = num_prev // num_cur
                    for dev in target_devices:
                        if dev not in prev_devices:
                            rdc_ctxs = False
                            break
                        else:
                            ind = prev_devices.index(dev)
                            if last_ind is None:
                                last_ind = ind
                            elif ind - last_ind != gap:
                                rdc_ctxs = False
                                break
            if use_nccl_collectives:
                input_shape = self.get_split_shape(pre_status.state, shape)
                if same_ctxs and pre_status.check_allreduce(tar_status):
                    # here we use allreduce instead of all2all
                    return self.wrapped_get_allreduce_time(input_shape, pre_rawctx, pre_status)
                elif same_ctxs and pre_status.check_allgather(tar_status):
                    # here we use allgather instead of all2all
                    return self.wrapped_get_allreduce_time(input_shape, pre_rawctx, pre_status, primitive=NCCLOP.AllGather)
                elif same_ctxs and pre_status.check_reducescatter(tar_status):
                    # here we use reducescatter instead of all2all
                    return self.wrapped_get_allreduce_time(input_shape, pre_rawctx, pre_status, primitive=NCCLOP.ReduceScatter)
                elif bcast_ctxs and pre_status.check_broadcast(tar_status):
                    # here we use broadcast instead of all2all
                    return self.wrapped_get_allreduce_time(input_shape, tar_rawctx, tar_status, primitive=NCCLOP.Broadcast)
                elif rdc_ctxs and pre_status.check_reduce(tar_status):
                    # here we use reduce instead of all2all
                    return self.wrapped_get_allreduce_time(input_shape, pre_rawctx, pre_status, primitive=NCCLOP.Reduce)

            send_time = self.HostDictionary(
                self.nccl_profiler.workers, pix=self.pix)
            recv_time = self.HostDictionary(
                self.nccl_profiler.workers, pix=self.pix)
            prev_shape = self.get_split_shape(pre_status.state, shape)
            all_times = []
            shape_buffer = defaultdict(dict)
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
                        split_time, output_shape = self.get_split_time(
                            prev_shape, keys, indices, splits, return_shape=True)
                        all_times.append(split_time)
                    else:
                        output_shape = prev_shape
                    if prev_devices[mp_index] != target_devices[device_index]:
                        from_dev, to_dev = prev_devices[mp_index], target_devices[device_index]
                        comm_time = self.get_comm_time(
                            from_dev, to_dev, output_shape)
                        send_time.add_to_channel(from_dev, comm_time)
                        recv_time.add_to_channel(to_dev, comm_time)
                    shape_buffer[mp_index][device_index] = output_shape
                    device_index += 1
                else:
                    cur_dim = target_order[depth]
                    if cur_dim < 0:
                        assert cur_dim == -1, 'Target node status must not enable partial.'
                        cur_index = cur_state_index.get(cur_dim, 0)
                        if prev_duplicate % target_duplicate == 0:
                            # at `cur_dim` dimension we need to send one output
                            multiple = prev_duplicate // target_duplicate
                            assert cur_index % multiple == 0
                            device_index += cur_index // multiple * \
                                loop_sizes[depth]
                            cross_send(split_cur_state,
                                       split_target_state, depth+1, need_split)
                            device_index += (prev_duplicate - 1 -
                                             cur_index) // multiple * loop_sizes[depth]
                        elif target_duplicate % prev_duplicate == 0:
                            # at `cur_dim` dimension we need to split and send some outputs
                            multiple = target_duplicate // prev_duplicate
                            device_index += cur_index * \
                                multiple * loop_sizes[depth]
                            for index in range(multiple):
                                cross_send(split_cur_state,
                                           split_target_state, depth+1, True)
                            device_index += (prev_duplicate - 1 -
                                             cur_index) * multiple * loop_sizes[depth]
                        else:
                            assert False
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

            loop_sizes = tar_status.get_loop_sizes()
            for mp_index in range(pre_rawctx.mp_dev_num):
                cur_state_index = pre_status.map_dev_to_index(
                    mp_index, containing_duplicate=True)
                if prev_partial == 1 and prev_duplicate > target_duplicate and cur_state_index.get(-1, 0) % (prev_duplicate // target_duplicate) != 0:
                    pass
                else:
                    device_index = 0
                    cross_send({}, {}, 0, False)
                    assert device_index == len(target_devices)

            # receive next
            def cross_receive(depth):
                nonlocal device_index
                if depth == len(prev_order):
                    output_shape = shape_buffer[device_index][mp_index]
                    device_index += 1
                else:
                    cur_dim = prev_order[depth]
                    if cur_dim == -2:
                        # sum op task
                        input_shapes = [cross_receive(
                            depth+1) for _ in range(prev_partial)]
                        sum_time, output_shape = self.get_sum_time(
                            input_shapes, return_shape=True)
                        all_times.append(sum_time)
                    elif cur_dim == -1:
                        prev_index = cur_state_index.get(cur_dim, 0)
                        if prev_duplicate % target_duplicate == 0:
                            multiple = prev_duplicate // target_duplicate
                            device_index += prev_index * \
                                multiple * loop_sizes[depth]
                            output_shape = cross_receive(depth+1)
                            device_index += ((target_duplicate - prev_index)
                                             * multiple - 1) * loop_sizes[depth]
                        elif target_duplicate % prev_duplicate == 0:
                            multiple = target_duplicate // prev_duplicate
                            device_index += prev_index // multiple * \
                                loop_sizes[depth]
                            output_shape = cross_receive(depth+1)
                            device_index += (target_duplicate - 1 -
                                             prev_index) // multiple * loop_sizes[depth]
                        else:
                            assert False
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
                                output_shape = cross_receive(depth+1)
                            else:
                                # concatenate op task
                                input_shapes = [cross_receive(
                                    depth+1) for _ in range(multiple)]
                                concatenate_time, output_shape = self.get_concatenate_time(
                                    input_shapes, cur_dim, return_shape=True)
                                all_times.append(concatenate_time)
                            device_index += (tar_st - 1 - cur_st) * \
                                multiple * loop_sizes[depth]
                        elif tar_st % prev_state[cur_dim] == 0:
                            # at `cur_dim` dimension we need to specify one input
                            multiple = tar_st // prev_state[cur_dim]
                            device_index += cur_st // multiple * \
                                loop_sizes[depth]
                            output_shape = cross_receive(depth+1)
                            device_index += (tar_st - 1 -
                                             cur_st) // multiple * loop_sizes[depth]
                        else:
                            assert False, 'The dispatch state (%d, %d) at dimension %d is invalid.' % (
                                prev_state[cur_dim], tar_st, cur_dim)
                return output_shape

            loop_sizes = pre_status.get_loop_sizes()
            for mp_index in range(tar_rawctx.mp_dev_num):
                cur_state_index = tar_status.map_dev_to_index(
                    mp_index, containing_duplicate=True)
                device_index = 0
                cross_receive(0)
                assert device_index == len(prev_devices)

            group_comm_time = max(
                send_time.all_values() + recv_time.all_values())
            return sum(all_times) + group_comm_time
        else:
            # check parallel + data parallel
            assert pre_rawctx.worker_num == tar_rawctx.worker_num == 1, \
                'In flexflow, the worker number should be 1!'
            pre_rawctx.check_mp_num(tar_rawctx.mp_dev_num)
            if pre_rawctx.mp_dev_num == 1:
                if pre_rawctx.workers[0] != tar_rawctx.workers[0]:
                    from_dev, to_dev = pre_rawctx.workers[0], tar_rawctx.workers[0]
                    comm_time = self.get_comm_time(from_dev, to_dev, shape)
                    return comm_time
                else:
                    return 0.
            else:
                # here in the same model parallel
                assert pre_rawctx == tar_rawctx
                return 0.

    def get_split_shape(self, parts, shape):
        shape = list(shape)
        if isinstance(parts, list):
            parts = {k: v for k, v in enumerate(parts) if v != 1}
        for i, pts in parts.items():
            assert shape[i] % pts == 0
            shape[i] //= pts
        return tuple(shape)

    def get_concatenate_shape(self, input_shapes, dim):
        shape = list(input_shapes[0])
        for sh in input_shapes[1:]:
            shape[dim] += sh[dim]
        return tuple(shape)

    def get_dev_distance(self, from_device, to_device):
        if from_device.hostname == to_device.hostname:
            from_dev = from_device.device_id
            to_dev = to_device.device_id
            distance = 1
            while from_dev != to_dev:
                distance *= 2
                from_dev //= 2
                to_dev //= 2
            distance //= 2
        else:
            distance = 8
        return distance

    def profile_allreduce(self, *args, **kargs):
        self.nccl_profiler.profile_allreduce(*args, **kargs)

    def profile_sendrecv(self, *args, **kargs):
        self.nccl_profiler.profile_sendrecv(*args, **kargs)

    def write_cache(self):
        if self.cache_path is not None:
            with open(self.cache_path, 'wb') as fw:
                pickle.dump(self.cached_exetime, fw)

    class HostDictionary(object):
        def __init__(self, workers, func_init_items=None, pix=True):
            self.pix = pix
            if func_init_items is None:
                if self.pix:
                    def func_init_items(key, num_workers): return [
                        0 for _ in range((num_workers + 1) // 2)]
                else:
                    def func_init_items(key, num_workers): return [
                        0 for _ in range(num_workers)]
            self.contents = {key: func_init_items(
                key, num_workers) for key, num_workers in workers.items()}

        def __getitem__(self, key):
            return self.contents[key.hostname][key.device_id]

        def __setitem__(self, key, value):
            self.contents[key.hostname][key.device_id] = value

        def get_comm_send(self, key):
            # for FlexFlow strategy
            if self.pix:
                return self.contents[key.hostname][key.device_id // 2 * 2]
            else:
                return self.contents[key.hostname][key.device_id * 2]

        def set_comm_send(self, key, value):
            # for FlexFlow strategy
            if self.pix:
                self.contents[key.hostname][key.device_id // 2 * 2] = value
            else:
                self.contents[key.hostname][key.device_id * 2] = value

        def get_comm_recv(self, key):
            # for FlexFlow strategy
            if self.pix:
                return self.contents[key.hostname][key.device_id // 2 * 2 + 1]
            else:
                return self.contents[key.hostname][key.device_id * 2 + 1]

        def set_comm_recv(self, key, value):
            # for FlexFlow strategy
            if self.pix:
                self.contents[key.hostname][key.device_id // 2 * 2 + 1] = value
            else:
                self.contents[key.hostname][key.device_id * 2 + 1] = value

        def add_to_channel(self, key, value):
            if self.pix:
                self.contents[key.hostname][key.device_id // 2] += value
            else:
                self.contents[key.hostname][key.device_id] += value

        def all_values(self):
            return sum(self.contents.values(), [])

        def __repr__(self):
            return str(self.contents)
