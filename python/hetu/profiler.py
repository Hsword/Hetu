from time import time
import numpy as np
import hetu as ht
from .stream import create_event_handle, create_stream_handle


class BaseProfiler(object):
    def __init__(self):
        self.idx = 0  # index of feed arrays

    def renew_instances(self):
        raise NotImplementedError

    def inc_idx(self):
        self.idx = (self.idx + 1) % self.num_instances


class HetuProfiler(BaseProfiler):
    def __init__(self, computing_nodes, feed_shapes, node_to_arr_map, ctx=ht.gpu(0)):
        # we don't profile following nodes:
        # PS op: ParameterServerCommunicateOp, ParameterServerSparsePullOp
        # AllReduce op: AllReduceCommunicateOp
        # CPU-GPU transfer op: DataH2DOp, DataD2HOp, DataD2HSparseOp
        self.computing_nodes = computing_nodes
        # here the feed shapes should include DataloaderOp and PlaceholderOp
        self.feed_shapes = feed_shapes
        self.ctx = ctx
        self.node_to_arr_map = node_to_arr_map
        self.use_gpu = ht.is_gpu_ctx(ctx)
        self.stream = create_stream_handle(ctx) if self.use_gpu else None
        self.timer = {node: 0.0 for node in self.computing_nodes}
        self.timer['all'] = 0.0
        self.idx = 0  # index of feed arrays

    def renew_nodes(self, computing_nodes, feed_shapes, node_to_arr_map):
        self.computing_nodes = computing_nodes
        self.feed_shapes = feed_shapes
        self.node_to_arr_map = node_to_arr_map
        self.timer = {node: 0.0 for node in self.computing_nodes}
        self.timer['all'] = 0.0
        self.idx = 0

    def renew_instances(self, num_instances=10):
        self.feed_values = {node: [ht.array(np.random.normal(scale=0.01, size=shape).astype(
            np.float32), ctx=self.ctx) for _ in range(num_instances)] for node, shape in self.feed_shapes.items()}
        self.num_instances = num_instances

    def profile(self, num_iterations=100, profiler='gpu'):
        assert profiler in ('cpu', 'gpu')
        self.renew_instances()
        # we don't record the first 5 iterations
        for _ in range(5):
            for node, values in self.feed_values.items():
                self.node_to_arr_map[node] = values[self.idx]
            for node in self.computing_nodes:
                input_vals = [self.node_to_arr_map[n] for n in node.inputs]
                node_val = self.node_to_arr_map[node]
                node.compute(input_vals, node_val, self.stream)
            self.inc_idx()
        if self.use_gpu:
            self.stream.sync()

        if profiler == 'cpu':
            for _ in range(5, 5 + num_iterations):
                for node, values in self.feed_values.items():
                    self.node_to_arr_map[node] = values[self.idx]
                for node in self.computing_nodes:
                    input_vals = [self.node_to_arr_map[n] for n in node.inputs]
                    node_val = self.node_to_arr_map[node]
                    start = time()
                    node.compute(input_vals, node_val, self.stream)
                    if self.use_gpu:
                        self.stream.sync()
                    ending = time()
                    self.timer[node] += (ending - start) * 1000
                self.inc_idx()

        else:
            assert self.use_gpu
            start_event = create_event_handle(self.ctx)
            ending_event = create_event_handle(self.ctx)
            for _ in range(5, 5 + num_iterations):
                for node, values in self.feed_values.items():
                    self.node_to_arr_map[node] = values[self.idx]
                for node in self.computing_nodes:
                    input_vals = [self.node_to_arr_map[n] for n in node.inputs]
                    node_val = self.node_to_arr_map[node]
                    start_event.record(self.stream)
                    node.compute(input_vals, node_val, self.stream)
                    ending_event.record(self.stream)
                    ending_event.sync()
                    duration = ending_event.time_since(start_event)
                    self.timer[node] += duration
                self.inc_idx()

        for node in self.timer:
            self.timer[node] /= num_iterations
        return self.timer

    def profile_all(self, num_iterations=100, profiler='gpu'):
        assert profiler in ('cpu', 'gpu')
        self.renew_instances()
        # we don't record the first 5 iterations
        if profiler == 'cpu':
            for _ in range(5):
                for node, values in self.feed_values.items():
                    self.node_to_arr_map[node] = values[self.idx]
                for node in self.computing_nodes:
                    input_vals = [self.node_to_arr_map[n] for n in node.inputs]
                    node_val = self.node_to_arr_map[node]
                    node.compute(input_vals, node_val, self.stream)
                self.inc_idx()
            if self.use_gpu:
                self.stream.sync()
            start = time()
            for _ in range(5, 5 + num_iterations):
                for node, values in self.feed_values.items():
                    self.node_to_arr_map[node] = values[self.idx]
                for node in self.computing_nodes:
                    input_vals = [self.node_to_arr_map[n] for n in node.inputs]
                    node_val = self.node_to_arr_map[node]
                    node.compute(input_vals, node_val, self.stream)
                self.inc_idx()
            if self.use_gpu:
                self.stream.sync()
            ending = time()
            self.timer['all'] = (ending - start) * 1000

        else:
            assert self.use_gpu
            start_event = create_event_handle(self.ctx)
            ending_event = create_event_handle(self.ctx)
            for _ in range(5):
                for node, values in self.feed_values.items():
                    self.node_to_arr_map[node] = values[self.idx]
                for node in self.computing_nodes:
                    input_vals = [self.node_to_arr_map[n] for n in node.inputs]
                    node_val = self.node_to_arr_map[node]
                    node.compute(input_vals, node_val, self.stream)
                self.inc_idx()
            start_event.record(self.stream)
            for _ in range(5, 5 + num_iterations):
                for node, values in self.feed_values.items():
                    self.node_to_arr_map[node] = values[self.idx]
                for node in self.computing_nodes:
                    input_vals = [self.node_to_arr_map[n] for n in node.inputs]
                    node_val = self.node_to_arr_map[node]
                    node.compute(input_vals, node_val, self.stream)
                self.inc_idx()
            ending_event.record(self.stream)
            ending_event.sync()
            duration = ending_event.time_since(start_event)
            self.timer['all'] = duration

        self.timer['all'] /= num_iterations
        return self.timer

    def profile_n_log(self, log_file, profiler='cpu'):
        timer = self.profile(profiler=profiler)
        with open(log_file, 'w') as fw:
            for k, v in timer.items():
                print(k, v, 'ms', file=fw, flush=True)


class NCCLProfiler(BaseProfiler):
    def __init__(self):
        from .gpu_ops.executor import wrapped_mpi_nccl_init, get_mpi_communicate, new_group_comm
        from .stream import create_stream_handle
        from .communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t
        self.nccl_comm = wrapped_mpi_nccl_init()
        self.mpi_comm = get_mpi_communicate()
        nrank = self.nccl_comm.nrank
        assert nrank in (2, 4, 8)
        self.ctx = ht.gpu(self.nccl_comm.dev_id)
        self.stream = create_stream_handle(self.ctx)
        self.ar_dtype = ncclDataType_t.ncclFloat32
        self.ar_op = ncclRedOp_t.ncclSum
        self.idx = 0
        self.is_root = (self.mpi_comm.rank) == 0
        if self.is_root:
            self.start_event = create_event_handle(self.ctx)
            self.ending_event = create_event_handle(self.ctx)

        # initialize group communicators
        topo_comb = {
            2: [],
            4: [(0, 1), (0, 2)],
            8: [(0, 1), (0, 2), (0, 4), (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 4, 5), (0, 1, 4, 6), (0, 2, 4, 6)],
        }[nrank]
        self.group_comms = {tuple(range(nrank)): self.nccl_comm}
        from .context import DeviceGroup
        for comb in topo_comb:
            self.group_comms[comb] = new_group_comm(
                DeviceGroup([ht.gpu(dev) for dev in comb]))

    def renew_instances(self, shape, num_instances=10):
        self.feed_values = [ht.array(np.random.normal(scale=0.01, size=shape).astype(
            np.float32), ctx=self.ctx) for _ in range(num_instances)]
        self.num_instances = num_instances

    def profile_allreduce(self, size, devices, num_iterations=10):
        from ctypes import c_void_p, c_int, cast, byref

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
        assert 0 in devices, 'Root device should be in profiling devices.'
        duration = 0
        if self.mpi_comm.rank in devices:
            self.renew_instances((size,))
            output_value = ht.empty((size,), ctx=self.ctx)
            comm = self.group_comms[devices]
            # warming up
            for _ in range(5):
                comm.dlarrayNcclAllReduce(
                    self.feed_values[self.idx], output_value, self.ar_dtype, self.ar_op, self.stream)
                self.inc_idx()
            if self.is_root:
                self.start_event.record(self.stream)
            for _ in range(5, 5 + num_iterations):
                comm.dlarrayNcclAllReduce(
                    self.feed_values[self.idx], output_value, self.ar_dtype, self.ar_op, self.stream)
                self.inc_idx()
            if self.is_root:
                self.ending_event.record(self.stream)
                self.ending_event.sync()
                duration = self.ending_event.time_since(
                    self.start_event) / num_iterations
        return duration

    def profile_sendrecv(self, size, devices, num_iterations=10):
        from ctypes import c_void_p, c_int, cast, byref

        def get_void_pointer(var):
            return cast(byref(var), c_void_p)

        if self.mpi_comm.rank == 0:
            assert devices[0] == 0 and devices[1] in (1, 2, 4)
            info = (c_int * 2)(size, devices[1])
        else:
            info = (c_int * 2)()
        self.mpi_comm.MPI_Broadcast(get_void_pointer(info), 8)
        size, target = info
        devices = (0, target)
        duration = 0
        if self.mpi_comm.rank == 0:
            comm = self.nccl_comm
            self.renew_instances((size,))
            output_value = ht.empty((size,), ctx=self.ctx)
            for _ in range(5):
                comm.dlarraySend(
                    self.feed_values[self.idx], self.ar_dtype, target, self.stream)
                comm.dlarrayRecv(
                    output_value, self.ar_dtype, target, self.stream)
                self.inc_idx()
            self.start_event.record(self.stream)
            for _ in range(5, 5 + num_iterations):
                comm.dlarraySend(
                    self.feed_values[self.idx], self.ar_dtype, target, self.stream)
                comm.dlarrayRecv(
                    output_value, self.ar_dtype, target, self.stream)
                self.inc_idx()
            self.ending_event.record(self.stream)
            self.ending_event.sync()
            duration = self.ending_event.time_since(
                self.start_event) / num_iterations / 2
        elif self.mpi_comm.rank == target:
            comm = self.nccl_comm
            self.renew_instances((size,))
            output_value = ht.empty((size,), ctx=self.ctx)
            for _ in range(5 + num_iterations):
                comm.dlarrayRecv(
                    output_value, self.ar_dtype, 0, self.stream)
                comm.dlarraySend(
                    self.feed_values[self.idx], self.ar_dtype, 0, self.stream)
                self.inc_idx()
            self.stream.sync()
        return duration
