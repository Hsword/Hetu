from .gpu_ops.AllReduceCommunicate import AllReduceCommunicateOp
from .gpu_ops.EmbeddingLookUp import EmbeddingLookUp, EmbeddingLookUp_Gradient
from .gpu_ops.DataTransfer import DataD2HSparseOp, DataH2DSparseOp, DataH2DOp
from .gpu_ops.LayerNorm import Layer_Normalization_Gradient_of_DataOp, Layer_Normalization_Gradient_of_ScaleOp, Layer_Normalization_Gradient_of_BiasOp
from .gpu_ops.BatchNorm import Batch_Normalization_Gradient_of_DataOp, Batch_Normalization_Gradient_of_ScaleOp, Batch_Normalization_Gradient_of_BiasOp
from .gpu_ops.Variable import PlaceholderOp
from .gpu_ops.EmbeddingLookUp import EmbeddingLookUp
from .gpu_ops.Dropout import DropoutOp
from .gpu_ops.Sum import SparseSumOp
from .gpu_ops.PipelineReceive import PipelineReceiveOp
from .gpu_ops.PipelineSend import PipelineSendOp
from .gpu_ops.StopGradient import StopGradientOp
from .gpu_ops.Unique import UniqueIndicesOffsetsOp
from .dataloader import DataloaderOp, GNNDataLoaderOp
from .optimizer import OptimizerOp
from . import ndarray

from collections import defaultdict
import pynvml
import numpy as np


class HetuMemoryPool(object):
    def __init__(self):
        # here the indexed_nodes only used for flexflow
        self.indexed_nodes = (EmbeddingLookUp_Gradient, DataD2HSparseOp, DataH2DSparseOp, SparseSumOp)
        self.ln_bn_grad_nodes = (Batch_Normalization_Gradient_of_DataOp, Batch_Normalization_Gradient_of_ScaleOp, Batch_Normalization_Gradient_of_BiasOp,
                                 Layer_Normalization_Gradient_of_DataOp, Layer_Normalization_Gradient_of_ScaleOp, Layer_Normalization_Gradient_of_BiasOp,
                                 UniqueIndicesOffsetsOp)
        self.no_compute_nodes = (StopGradientOp, DataloaderOp, GNNDataLoaderOp)

    def compute_memory_reuse_plan(self, computing_nodes, node_to_shape, eval_node_list):
        persistent_nodes = self.form_persistent_nodes(
            eval_node_list, node_to_shape)
        # compute output deg
        outdeg = {}
        memory_pool = defaultdict(list)
        reuse_map = {}
        for node in computing_nodes:
            outdeg[node] = 0
            for n in node.inputs:
                if n in outdeg:
                    outdeg[n] += 1

        def release_node(node):
            if node not in computing_nodes:
                return
            outdeg[node] -= 1
            if outdeg[node] > 0 or node in persistent_nodes or isinstance(node, self.indexed_nodes):
                return
            assert outdeg[node] == 0
            if node.inplace:
                for n in node.inputs:
                    release_node(n)
            else:
                assert not node.use_indexed_slices, node.name
                memory_pool[(node_to_shape[node], node.ctx,
                             node.dtype)].append(node)

        for node in computing_nodes:
            if node.inplace:
                continue
            shape = node_to_shape[node]
            key = (shape, node.ctx, node.dtype)
            if shape is None or node in persistent_nodes or isinstance(node, self.indexed_nodes):
                pass
            elif len(memory_pool[key]) > 0:
                reuse_map[node] = memory_pool[key].pop()
            for n in node.inputs:
                release_node(n)
        return reuse_map

    def form_persistent_nodes(self, eval_node_list, _node_to_shape):
        persistent_nodes = set(eval_node_list)
        for node in _node_to_shape:
            if isinstance(node, DataH2DOp):
                persistent_nodes.add(node)
            elif isinstance(node, AllReduceCommunicateOp):
                persistent_nodes.add(node.inputs[0])
                persistent_nodes.add(node)
            elif isinstance(node, PipelineReceiveOp):
                persistent_nodes.add(node)
            elif isinstance(node, PipelineSendOp):
                persistent_nodes.add(node.inputs[0])
        return persistent_nodes

    def memory_plan(self, computing_nodes, node_to_shape_map, node_to_arr_map, config, eval_node_list, indexed_slices_shape):
        placeholder_to_arr_map = config.placeholder_to_arr_map
        inference = not any([isinstance(node, OptimizerOp)
                             for node in eval_node_list])
        param_psval_map = config.infer_ps_map if inference else config.ps_map
        reuse_map = self.compute_memory_reuse_plan(
            computing_nodes, node_to_shape_map, eval_node_list)
        for node, shape in node_to_shape_map.items():
            if isinstance(node, PlaceholderOp):
                if placeholder_to_arr_map[node] is not None:
                    node_to_arr_map[node] = placeholder_to_arr_map[node]
                elif node not in node_to_arr_map:
                    node_to_arr_map[node] = None
            elif not isinstance(node, self.no_compute_nodes):
                # add for OptimizerOp and ParameterServerOp
                if shape is None:
                    node_to_arr_map[node] = None
                elif node in indexed_slices_shape:
                    ind_shape, val_shape = indexed_slices_shape[node]
                    indices = ndarray.empty(
                        ind_shape, node.ctx, dtype=np.int32)
                    values = ndarray.empty(val_shape, node.ctx)
                    node_to_arr_map[node] = ndarray.IndexedSlices(
                        indices=indices, values=values, dense_shape=shape)
                elif isinstance(node, EmbeddingLookUp) and (config.use_sparse_pull or config.cstable_policy) and config.prefetch:
                    node_to_arr_map[node] = param_psval_map[node.inputs[0]]
                else:
                    if node.on_gpu:
                        if node.inplace or isinstance(node, self.indexed_nodes):
                            node_to_arr_map[node] = ndarray.NDArray(None)
                        elif inference and isinstance(node, DropoutOp):
                            node_to_arr_map[node] = node_to_arr_map[node.inputs[0]]
                        else:
                            result = reuse_map.get(node, node)
                            if result is node:
                                node_to_arr_map[node] = ndarray.empty(
                                    shape, ctx=node.ctx, dtype=node.dtype)
                            else:
                                node_to_arr_map[node] = node_to_arr_map[result]
                    else:
                        node_to_arr_map[node] = ndarray.empty(
                            shape, ctx=node.ctx, dtype=node.dtype)
                    if isinstance(node, self.ln_bn_grad_nodes):
                        # for batch normailzation, pass array to the real gradient node
                        node.pass_grad_array(node_to_arr_map[node])
            elif isinstance(node, StopGradientOp):
                node_to_arr_map[node] = node_to_arr_map[node.inputs[0]]

    def start_simulate(self, devices):
        # we simply assume the environment is homogeneous
        # TODO: support heterogeneous environment
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # since there're memory fragments, we assume only use 90%
        # TODO: need a more accurate fragments prediction
        free_memory = int(meminfo.free * 0.9 / 4)
        memory_free = {dev: free_memory for dev in devices}
        return memory_free

    def test_memory(self, devices, task_graph):
        # this function is only for flexflow now
        # here we don't consider eval nodes as persistent nodes, since they occupy little memory
        # TODO: re-design to support other strategies
        def add_usage(device, size):
            if isinstance(size, tuple):
                size = int(np.prod(size, dtype=int))
            memory_free[device] -= size

        # compute output deg
        memory_free = self.start_simulate(devices)
        dev_outdeg = {dev: {} for dev in devices}
        memory_pool = {dev: defaultdict(list) for dev in devices}
        for task in task_graph:
            if task.name.startswith('update'):
                continue
            dev = task.device
            if isinstance(dev, tuple):
                is_group = (task.name == 'group_comm')
                is_allreduce = (task.name == 'allreduce')
                if is_group:
                    for t in task.contents:
                        # communication in another stream, not reuse
                        add_usage(t.device[1], t.shape)
                elif is_allreduce:
                    for d in dev:
                        # allreduce in another stream, not reuse
                        add_usage(d, task.shape)
                else:
                    # communication in another stream, not reuse
                    add_usage(dev[1], task.shape)
            else:
                shape = task.shape
                if shape is not None and not isinstance(task.original_node, self.indexed_nodes):
                    if task.memory_persistent:
                        # variables need the memory
                        add_usage(dev, shape)
                    else:
                        dev_outdeg[dev][task] = len(task.outputs)
                        if len(memory_pool[dev][shape]) > 0:
                            memory_pool[dev][shape].pop()
                        else:
                            add_usage(dev, task.shape)
                for t in task.inputs:
                    if t.shape is not None and t.device == dev and not t.memory_persistent and not isinstance(t.original_node, self.indexed_nodes):
                        dev_outdeg[dev][t] -= 1
                        if dev_outdeg[dev][t] == 0:
                            memory_pool[dev][t.shape].append(t)

        # get result
        result = 0
        for value in memory_free.values():
            if value < 0:
                result -= value
        return result
