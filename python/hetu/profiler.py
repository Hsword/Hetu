from time import time
import numpy as np
import hetu as ht
from .stream import create_event_handle, create_stream_handle


class HetuProfiler(object):
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

    def renew_nodes(self, computing_nodes, feed_shapes, node_to_arr_map, single=False):
        self.computing_nodes = computing_nodes
        self.feed_shapes = feed_shapes
        self.node_to_arr_map = node_to_arr_map
        self.timer = {node: 0.0 for node in self.computing_nodes}
        self.timer['all'] = 0.0
        self.idx = 0

    def renew_instances(self, num_instances=10):
        self.feed_values = {node: [ht.array(np.random.normal(scale=0.01, size=shape).astype(np.float32), ctx=self.ctx) for _ in range(
            num_instances)] for node, shape in self.feed_shapes.items()}
        self.num_instances = num_instances

    def inc_idx(self):
        self.idx = (self.idx + 1) % self.num_instances

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
