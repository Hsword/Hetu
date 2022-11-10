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
