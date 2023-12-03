from __future__ import absolute_import
from .BatchNorm import Batch_NormalizationOp
from .. import ndarray
from .._base import DNNL_LIB
from ..cpu_links import array_set as cpu_array_set
from .AllReduceCommunicate import AllReduceCommunicateOp
from .ParameterServerCommunicate import ParameterServerCommunicateOp, ParameterServerSparsePullOp
from .DataTransfer import DataH2DOp, DataD2HOp, DataD2HSparseOp
from ..communicator.mpi_nccl_comm import GroupStart, GroupEnd
from ..stream import Event
from .PipelineSend import PipelineSendOp
from .PipelineReceive import PipelineReceiveOp
from .Dropout import DropoutOp
from .executor import SubExecutor
from ..stream import create_event_handle
from time import time
from collections import defaultdict
import contextlib


class HetuTimer(object):
    def __init__(self) -> None:
        self.timer = defaultdict(float)
        self.cnt = 0

    @contextlib.contextmanager
    def __call__(self, key, stream=None):
        yield

    def clearTimer(self):
        self.timer.clear()
        self.cnt = 0

    def step(self):
        self.cnt += 1

    def logOut(self, path, rank, log_level='node', clear=True, multiplier=1):
        if self.cnt == 0:
            print('No records.')
            return
        assert log_level in ('node', 'type')
        path = path.format(rank)
        multiplier /= self.cnt
        with open(path, 'w') as fw:
            all_time = sum(self.timer.values(), 0) * multiplier
            if log_level == 'node':
                for node, comp_time in self.timer.items():
                    comp_time = comp_time * multiplier
                    print(node, node.inputs, comp_time, file=fw, flush=True)
            else:
                type_timer = defaultdict(float)
                for node, comp_time in self.timer.items():
                    if isinstance(node, (PipelineReceiveOp, PipelineSendOp)):
                        type_timer[PipelineSendOp] += comp_time
                    else:
                        type_timer[type(node)] += comp_time
                for node_type, comp_time in type_timer.items():
                    comp_time = comp_time * multiplier
                    print('{}: {}'.format(node_type.__name__,
                                          comp_time), file=fw, flush=True)
            print('All_time: {}'.format(all_time), file=fw, flush=True)
        if clear:
            self.clearTimer()


class HetuCPUTimer(HetuTimer):
    def __init__(self) -> None:
        super().__init__()

    @contextlib.contextmanager
    def __call__(self, key, stream=None):
        start = time()
        yield
        if stream is not None:
            stream.sync()
        ending = time()
        cur_time = (ending - start) * 1000
        self.timer[key] += cur_time


class HetuGPUTimer(HetuTimer):
    def __init__(self, ctx) -> None:
        super().__init__()
        self.start_event = create_event_handle(ctx)
        self.ending_event = create_event_handle(ctx)

    @contextlib.contextmanager
    def __call__(self, key, stream):
        self.start_event.record(stream)
        yield
        self.ending_event.record(stream)
        stream.sync()
        cur_time = self.ending_event.time_since(
            self.start_event)
        self.timer[key] += cur_time


def make_timer(timer=None, ctx=None):
    if timer is None:
        return HetuTimer()
    elif timer == 'cpu':
        return HetuCPUTimer()
    elif timer == 'gpu':
        return HetuGPUTimer(ctx)
    else:
        assert False, 'Timer must be in (None, cpu, gpu).'


class TimerSubExecutor(SubExecutor):
    def __init__(self, name, eval_node_list, config, timer='gpu'):
        super().__init__(name, eval_node_list, config)
        self.timer = make_timer(timer, self.config.context)

    def compute(self, computing_nodes, arr_map):
        # computing
        grouping_nodes = []
        cur_ind = -1

        def make_group():
            p2p_stream = self.config.p2p_stream
            with self.timer(grouping_nodes[0], p2p_stream):
                GroupStart()
                for node in grouping_nodes:
                    node.compute([arr_map[n] for n in node.inputs],
                                 arr_map[node], p2p_stream)
                GroupEnd()
            for node in grouping_nodes:
                node.event.record(p2p_stream)
            grouping_nodes.clear()
        for node in computing_nodes:
            if node.on_cpu and isinstance(arr_map[node], ndarray.NDArray):
                if DNNL_LIB['cpu_ArraySet'] and not isinstance(node, DataD2HOp):
                    cpu_array_set(arr_map[node], 0.0)
                else:
                    # here we suppose not using DNNL_LIB
                    # arr_map[node][:] = np.zeros(self.node_to_shape_map[node]).astype(np.float32)
                    pass

            if isinstance(node, (PipelineSendOp, PipelineReceiveOp)):
                for n in node.inputs:
                    if n.event:
                        n.event.sync()
                if len(grouping_nodes) > 0 and self.config.layer_indices[node] != cur_ind:
                    make_group()
                if len(grouping_nodes) == 0:
                    cur_ind = self.config.layer_indices[node]
                grouping_nodes.append(node)
                continue
            else:
                if len(grouping_nodes) > 0:
                    make_group()

                input_vals = [arr_map[n] for n in node.inputs]
                node_val = arr_map[node]

                node_type = type(node)
                cur_stream = self.node_type_to_stream_map.get(
                    node_type, self.comp_stream)

                with self.timer(node, cur_stream):
                    if node_type in (DropoutOp, Batch_NormalizationOp):
                        node.compute(input_vals, node_val, cur_stream,
                                     inference=self.inference)
                    else:
                        node.compute(input_vals, node_val, cur_stream)

        self.timer.step()

        if len(grouping_nodes) > 0:
            make_group()

    def clearTimer(self):
        self.timer.clearTimer()

    def logOut(self, path, log_level='node', clear=True, multiplier=1):
        self.timer.logOut(path, self.config.rank, log_level, clear, multiplier)


def make_texecutor(timer):
    def get_sub_executor(name, eval_node_list, config):
        return TimerSubExecutor(name, eval_node_list, config, timer)
    return get_sub_executor
