""" library to take autodiff and execute a computation graph """
from __future__ import absolute_import
from .BatchNorm import Batch_NormalizationOp
import numpy as np
from scipy.sparse import spmatrix, coo_matrix
from .. import ndarray
from .._base import DNNL_LIB
from ..cpu_links import array_set as cpu_array_set
from .Variable import PlaceholderOp  # add for optimizer
from ..dataloader import DataloaderOp, GNNDataLoaderOp
from .AllReduceCommunicate import AllReduceCommunicateOp
from .ParameterServerCommunicate import ParameterServerCommunicateOp, ParameterServerSparsePullOp, parameterServerSparsePull_op
from .AddElewise import add_op
from .DataTransfer import DataH2DOp, DataD2HOp, DataD2HSparseOp
from ..communicator.mpi_nccl_comm import GroupStart, GroupEnd
from .EmbeddingLookUp import EmbeddingLookUp, EmbeddingLookUp_Gradient
from ..optimizer import OptimizerOp
from . import OnesLike
from ..stream import create_stream_handle, Event
from ..context import get_current_context, get_launch_config_by_traverse_nodes, assign_context_by_traverse_nodes, DeviceGroup
from .PipelineSend import PipelineSendOp
from .PipelineReceive import PipelineReceiveOp
from .AddElewise import AddOp
from .Split import SplitOp
from .Concat import ConcatOp
from .Dropout import DropoutOp
from .LayerNorm import Layer_NormalizationOp
from .OnesLike import OnesLikeOp
from operator import add
from functools import reduce
import ctypes
import os
from time import time
from ..communicator.mpi_nccl_comm import ncclDataType_t, GroupStart, GroupEnd


def path_to_lib(name):
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(curr_path, '../../../build/lib/')
    return os.path.join(lib_path, name)


def wrapped_mpi_nccl_init(init_nccl=True, devices=None):
    from ..communicator.mpi_nccl_comm import mpi_communicator
    global mpi_comm
    global nccl_comm
    if 'mpi_comm' not in globals():
        mpi_comm = mpi_communicator(devices=devices)
        if 'nccl_comm' not in globals():
            nccl_comm = mpi_comm.ncclInit() if init_nccl else None
    return nccl_comm


def new_group_comm(devices_context=None):
    assert 'mpi_comm' in globals()
    global mpi_comm
    if devices_context is None:
        comm = mpi_comm.ncclInit()
    else:
        comm = mpi_comm.ncclGroupInit(devices_context)
    return comm


def get_nccl_communicate():
    global nccl_comm
    return nccl_comm


def get_worker_communicate():
    global ps_comm
    return ps_comm


def worker_init():
    global ps_comm
    ll = ctypes.cdll.LoadLibrary
    ps_comm = ll(path_to_lib("libps.so"))
    ps_comm.Init()


def worker_finish():
    ps_comm.Finalize()


def server_init():
    global ps_comm
    ll = ctypes.cdll.LoadLibrary
    ps_comm = ll(path_to_lib("libps.so"))
    ps_comm.Init()
    ps_comm.StartServer()


def server_finish():
    ps_comm.Finalize()


def scheduler_init():
    global ps_comm
    ll = ctypes.cdll.LoadLibrary
    ps_comm = ll(path_to_lib("libps.so"))
    ps_comm.Init()


def scheduler_finish():
    ps_comm.Finalize()


class HetuConfig(object):
    __slots__ = [
        'eval_node_list',
        'train_name',
        'val_name',
        'context',
        'seed',
        'np_rand',
        'comm_mode',
        'node_strategy',
        'context_launch',
        'ps_comm',
        'nccl_comm',
        'local_rank',
        'rank',
        'nrank',
        'p2p_stream',
        'comp_stream',
        'nccl_stream',
        'h2d_stream',
        'd2h_stream',
        'h2d_ops',
        'd2h_ops',
        'ps_map',
        'infer_ps_map',
        'dataloader_ops',
        'use_sparse_pull',
        'cstable_policy',
        'inference',
        'enable_lazy',
        'bsp',
        'prefetch',
        'cache_bound',
        'log_path',
        'my_eval_nodes',
        'param_allreduce_group',
        'placeholder_to_arr_map',
        'gpipe',
        'pipedream',
        'dynamic_memory',
        'layer_indices',
    ]

    def __init__(
        self,
        eval_node_list,
        train_name,
        val_name,
        ctx=None,
        seed=None,
        comm_mode=None,
        use_sparse_pull=True,
        cstable_policy=None,
        bsp=False,
        prefetch=True,
        enable_lazy=True,
        cache_bound=100,
        log_path=None,
        gpipe=False,
        pipedream=False,
        dynamic_memory=False,
    ):
        '''
        context: default device context
        comm_mode: communication mode, should be one of the following
            None       -> Single GPU
            PS         -> Parameter Server
            AllRedeuce -> MPI AllReduce
            Hybrid     -> Parameter Server for Sparse Parameter and MPI AllReduce for Dense Parameter
        '''
        self.gpipe = gpipe
        self.pipedream = pipedream

        self.eval_node_list = eval_node_list
        self.train_name = train_name
        self.val_name = val_name

        self.dynamic_memory = dynamic_memory

        # check context
        if ctx is None:
            ctx = get_current_context()
        assert ctx, 'Default context should be determined.'

        self.comm_mode = comm_mode
        self.node_strategy = {}
        local_gpu_devices = None
        context_launch = isinstance(ctx, DeviceGroup)
        self.context_launch = context_launch
        if context_launch:
            # with context usage
            launchMPI, launchPS, self.node_strategy, devices = get_launch_config_by_traverse_nodes(
                eval_node_list, ctx)
            local_gpu_devices = sorted(
                [dev.device_id for dev in devices if dev.local and ndarray.is_gpu_ctx(dev)])
            if not launchMPI and not launchPS:
                self.comm_mode = None
            elif launchMPI and not launchPS:
                self.comm_mode = 'AllReduce'
            elif not launchMPI and launchPS:
                self.comm_mode = 'PS'
            else:
                self.comm_mode = 'Hybrid'
            # in pipeline or model parallel we have to initialize another p2p stream
            init_p2p_stream = len(devices) != len(ctx)

        # variables initialization
        self.seed = seed if seed is not None else np.int64(time())
        self.np_rand = np.random.RandomState(self.seed)

        # get attribute of communication mode
        self.ps_comm = None
        self.nccl_comm = None
        self.local_rank = None
        self.rank = None
        self.nrank = None
        ps_nrank = None
        if self.comm_mode == 'PS' or self.comm_mode == 'Hybrid':
            worker_init()
            self.ps_comm = get_worker_communicate()
            ps_rank = int(self.ps_comm.rank())
            ps_nrank = int(
                os.environ['DMLC_NUM_WORKER']) if 'DMLC_NUM_WORKER' in os.environ else 1
        if self.comm_mode == "Hybrid" or self.comm_mode == "AllReduce":
            self.nccl_comm = wrapped_mpi_nccl_init(devices=local_gpu_devices)
        elif context_launch:
            self.nccl_comm = wrapped_mpi_nccl_init(
                init_nccl=init_p2p_stream, devices=local_gpu_devices)
        if self.nccl_comm is not None:
            self.local_rank = self.nccl_comm.local_rank
            device_id = self.nccl_comm.dev_id
            self.rank = self.nccl_comm.rank
            self.nrank = self.nccl_comm.nrank
            if ps_nrank:
                assert ps_nrank == self.nrank
        elif self.comm_mode == 'PS':
            self.rank = ps_rank
            self.nrank = ps_nrank
            if context_launch:
                global mpi_comm
                self.local_rank = mpi_comm.local_rank
                device_id = mpi_comm.dev_id
        elif context_launch:
            assert len(devices) == 1
            device_id = devices.pop().device_id

        self.my_eval_nodes = eval_node_list
        self.p2p_stream = None
        self.param_allreduce_group = {}
        self.layer_indices = None
        if context_launch:
            # comm_mode is None <=> only 1 model parallel instance
            self.context = ndarray.gpu(device_id)
            self.p2p_stream = create_stream_handle(
                self.context) if init_p2p_stream else None
            self.my_eval_nodes, self.param_allreduce_group, self.layer_indices = assign_context_by_traverse_nodes(
                eval_node_list, self.context, self.nccl_comm, self.p2p_stream)
            for param in self.param_allreduce_group:
                self.node_strategy[param] = 'AllReduce'
            if self.param_allreduce_group != {}:
                if self.comm_mode is None:
                    self.comm_mode = 'AllReduce'
                if self.comm_mode == 'PS':
                    self.comm_mode = 'Hybrid'
        else:
            self.context = ctx

        on_gpu = ndarray.is_gpu_ctx(self.context)

        self.nccl_stream = None
        if self.comm_mode == "Hybrid" or self.comm_mode == "AllReduce":
            if on_gpu:
                self.nccl_stream = create_stream_handle(self.context)
            self.nccl_comm = get_nccl_communicate()

        # define streams
        self.comp_stream = create_stream_handle(
            self.context) if on_gpu else None
        self.h2d_stream = create_stream_handle(
            self.context) if on_gpu else None
        self.d2h_stream = create_stream_handle(
            self.context) if on_gpu else None

        self.use_sparse_pull = use_sparse_pull if self.comm_mode == 'PS' or self.comm_mode == "Hybrid" else False
        self.cstable_policy = cstable_policy if self.comm_mode == 'PS' or self.comm_mode == "Hybrid" else None
        self.prefetch = prefetch if self.comm_mode == 'PS' or self.comm_mode == 'Hybrid' else False
        if self.cstable_policy is not None:
            self.cstable_policy = self.cstable_policy.upper()
            self.use_sparse_pull = False

        self.h2d_ops = {}
        self.d2h_ops = {}
        self.ps_map = {}
        self.infer_ps_map = {}
        self.enable_lazy = False and enable_lazy  # now we don't use lazy
        self.bsp = bsp
        self.cache_bound = int(cache_bound)
        if self.dynamic_memory:
            self.enable_lazy = False

        self.log_path = log_path
        if log_path is not None and (self.comm_mode == 'PS' or self.comm_mode == "Hybrid"):
            assert os.path.isdir(
                log_path), 'Need to specify a work directory to save logs.'
            self.ps_comm.startRecord(ctypes.c_char_p(bytes(log_path, 'utf-8')))

        self.placeholder_to_arr_map = dict()
        topo_sort_with_hook(self.my_eval_nodes, self)


class Executor(object):
    """Executor computes values for given set of nodes in computation graph."""

    def __init__(self, eval_node_dict, config=None, **kargs):
        """
        Parameters
        ----------
        eval_node_dict: dict of list of nodes whose values need to be computed.
        """
        if not isinstance(eval_node_dict, dict):
            eval_node_dict = {'default': eval_node_dict}
        train_name, val_name = None, None
        for k, v in eval_node_dict.items():
            if any([isinstance(node, OptimizerOp) for node in v]):
                # get the last subexecutor containing optimizer as train for ps op
                train_name = k
            else:
                # get the last subexecutor not containing optimizer as val for ps op
                val_name = k
        all_eval_nodes = list(set(reduce(add, eval_node_dict.values())))
        if config is None:
            config = HetuConfig(eval_node_list=all_eval_nodes,
                                train_name=train_name, val_name=val_name, **kargs)
        assert isinstance(
            config, HetuConfig), 'Config type %s invalid.' % str(type(config))

        self.eval_node_dict = eval_node_dict
        self.config = config

        def get_sub_executor(k):
            if config.gpipe and k == "train":
                return SubExecutor4Gpipe
            elif config.pipedream and k == "train":
                return SubExecutor4Pipedream
            return SubExecutor

        self.subexecutor = {k: get_sub_executor(k)(k, v, config) for k, v in eval_node_dict.items()}

        self.topo_order = find_topo_sort(config.my_eval_nodes)
        self.param_nodes = [node for node in self.topo_order if isinstance(
            node, PlaceholderOp) and node.trainable]
        self.comm_mode = self.config.comm_mode
        self.ps_comm = self.config.ps_comm
        self.local_rank = self.config.local_rank
        self.rank = self.config.rank

    def run(self, name='default', eval_node_list={}, feed_dict={}, convert_to_numpy_ret_vals=False, **kwargs):
        return self.subexecutor[name].run(eval_node_list, feed_dict, convert_to_numpy_ret_vals, **kwargs)

    @property
    def batch_num(self):
        assert len(
            self.subexecutor) == 1, 'Batch num should be used with only 1 subexecutor.'
        return list(self.subexecutor.values())[0].batch_num

    def get_batch_num(self, name='default'):
        return self.subexecutor[name].batch_num

    def save(self, file_path):
        assert os.path.isdir(
            file_path), 'Need to specify a work directory to save parameters.'
        if self.comm_mode in (None, 'AllReduce'):
            # when using allreduce, users need to specify the worker whose rank equals 0 to save
            for node in self.param_nodes:
                np.save(os.path.join(file_path, node.name + '.npy'),
                        self.config.placeholder_to_arr_map[node].asnumpy())
        else:
            self.ps_comm.BarrierWorker()
            if self.config.rank == 0:
                for node in self.param_nodes:
                    if node.is_embed or self.comm_mode == 'PS':
                        node.event.sync()
                        nodeid = ctypes.c_int(node.id)
                        self.ps_comm.SaveParam(
                            nodeid, ctypes.c_char_p(bytes(file_path, 'utf-8')))
                        self.ps_comm.Wait(nodeid)
                    else:
                        np.save(os.path.join(file_path, node.name + '.npy'),
                                self.config.placeholder_to_arr_map[node].asnumpy())
            self.ps_comm.BarrierWorker()

    def load(self, file_path):
        assert os.path.isdir(
            file_path), 'Need to specify a work directory to load parameters.'
        if self.comm_mode in (None, 'AllReduce'):
            for node in self.param_nodes:
                self.config.placeholder_to_arr_map[node][:] = np.load(
                    os.path.join(file_path, node.name + '.npy'))
        else:
            self.ps_comm.BarrierWorker()
            if self.config.rank == 0:
                for node in self.param_nodes:
                    if node.is_embed or self.comm_mode == 'PS':
                        node.event.sync()
                        nodeid = ctypes.c_int(node.id)
                        self.ps_comm.LoadParam(
                            nodeid, ctypes.c_char_p(bytes(file_path, 'utf-8')))
                        node.event.update()
            self.ps_comm.BarrierWorker()
            for node in self.topo_order:
                if isinstance(node, PlaceholderOp) and node.trainable and not node.is_embed:
                    if self.comm_mode == 'PS':
                        node.event.sync()
                        nodeid = ctypes.c_int(node.id)
                        self.ps_comm.Pull(
                            nodeid, self.config.ps_map[node].handle)
                        node.event.update()
                    else:
                        self.config.placeholder_to_arr_map[node][:] = np.load(
                            os.path.join(file_path, node.name + '.npy'))
                elif isinstance(node, EmbeddingLookUp) and self.config.prefetch:
                    node.event.sync()
                    nodeid = ctypes.c_int(node.inputs[0].id)
                    self.ps_comm.SparsePull(nodeid, node.inputs[1].get_next_arr(
                        self.name).handle, self.config.ps_map[node.inputs[0]].handle)
                    node.event.update()
            self.ps_comm.BarrierWorker()

    def recordLoads(self):
        for node in self.config.ps_map:
            node.event.sync()
        self.ps_comm.getLoads()

    def __del__(self):
        if self.config.comp_stream is not None:
            self.config.comp_stream.sync()
        if self.config.h2d_stream is not None:
            self.config.h2d_stream.sync()
        if self.config.d2h_stream is not None:
            self.config.d2h_stream.sync()
        if self.config.nccl_stream is not None:
            self.config.nccl_stream.sync()
        for node in self.param_nodes:
            if node.event:
                node.event.sync()
        if self.comm_mode in ('PS', 'Hybrid'):
            worker_finish()


class SubExecutor4Gpipe(object):
    def __init__(self, name, eval_node_list, config):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        topo_order: list of nodes in topological order
        node_to_shape_map: dict from node to shape of the node
        node_to_arr_maps: a list of [dict from node to ndarray.NDArray allocated for node]
        feed_shapes: shapes of feed_dict from last run(...)
        """
        self.name = name
        self.eval_node_list = config.my_eval_nodes
        self.config = config
        self.inference = not any([isinstance(node, OptimizerOp) for node in eval_node_list])
        self.global_eval_nodes = eval_node_list

        self.topo_order = find_topo_sort(self.eval_node_list)

        # split the topo into two parts: forward and backward
        for i in range(len(self.topo_order)):
            if isinstance(self.topo_order[i], PipelineSendOp):
                pivot_idx = i
                break
        if config.local_rank == config.nrank - 1:
            self.forward_topo_order = self.topo_order[:pivot_idx]
            self.backward_topo_order = self.topo_order[pivot_idx:]
        else:
            self.forward_topo_order = self.topo_order[:pivot_idx+1]
            self.backward_topo_order = self.topo_order[pivot_idx+1:]

        """
        print("gpu {}'s topo: ".format(config.local_rank),
              [(x.name,x.desc) for x in self.topo_order])
        print("gpu {}'s forward topo: ".format(config.local_rank),
              [(x.name,x.desc) for x in self.forward_topo_order])
        print("gpu {}'s backward topo: ".format(config.local_rank),
              [(x.name,x.desc) for x in self.backward_topo_order])
        """

        # main structures, nodes' shapes and arrays
        self.node_to_shape_map = {}
        self.node_to_arr_maps = []

        # inherit from configurations
        self.comm_mode = self.config.comm_mode
        self.ps_comm = self.config.ps_comm
        self.nccl_comm = self.config.nccl_comm
        self.comp_stream = self.config.comp_stream
        self.h2d_stream = self.config.h2d_stream
        self.d2h_stream = self.config.d2h_stream
        self.nccl_stream = self.config.nccl_stream
        self.param_psval_map = self.config.ps_map
        self.use_sparse_pull = self.config.use_sparse_pull
        self.cstable_policy = self.config.cstable_policy
        self.use_p2p = self.config.p2p_stream is not None

        # assisting structures, improve performance
        self.need_feed_nodes = []
        self.param_nodes = []
        self.dataloader_nodes = []
        self.computing_nodes = []
        for node in self.topo_order:
            if isinstance(node, DataloaderOp) or isinstance(node, GNNDataLoaderOp):
                self.dataloader_nodes.append(node)
            elif isinstance(node, PlaceholderOp):
                if node.shape is None:
                    self.need_feed_nodes.append(node)
                elif node.trainable:
                    self.param_nodes.append(node)
            elif not ((self.use_sparse_pull or self.cstable_policy) and isinstance(node, EmbeddingLookUp) and self.config.prefetch):
                self.computing_nodes.append(node)

    def infer_shape(self, feed_shapes):
        """Given shapes of feed_dict nodes, infer shape for all nodes in graph.

        Implementation note:
        Iteratively calls node.infer_shape to infer shapes.
        Node shapes stored in self.node_to_shape_map.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        self.node_to_shape_map = {}
        for node in self.topo_order:
            if node in feed_shapes:
                self.node_to_shape_map[node] = tuple(feed_shapes[node])
            else:
                input_shapes = [self.node_to_shape_map[n] for n in node.inputs]
                cur_shape = node.infer_shape(input_shapes)
                self.node_to_shape_map[node] = cur_shape if cur_shape is None else tuple(
                    cur_shape)

    def memory_plan(self):
        """Allocates ndarray.NDArray for every node except feed_dict nodes.
        Parameters
        ----------
        """

        def assign_one_to_all_maps(node, tensor_value):
            # assign the same obj across all maps, careful
            for mp in self.node_to_arr_maps:
                mp[node] = tensor_value

        for node, shape in self.node_to_shape_map.items():
            if isinstance(node, PlaceholderOp):
                if self.config.placeholder_to_arr_map[node] is not None:
                    assign_one_to_all_maps(
                        node, self.config.placeholder_to_arr_map[node])
                elif node not in self.node_to_arr_maps[0]:
                    assign_one_to_all_maps(node, None)
            elif not isinstance(node, DataloaderOp) and not isinstance(node, GNNDataLoaderOp):
                # add for OptimizerOp and ParameterServerOp
                if shape is None:
                    assign_one_to_all_maps(node, None)
                    continue
                if isinstance(node, (EmbeddingLookUp_Gradient, DataD2HSparseOp)):
                    raise NotImplementedError
                    continue
                if isinstance(node, EmbeddingLookUp) and (self.use_sparse_pull or self.cstable_policy) and self.config.prefetch:
                    raise NotImplementedError
                    continue
                if node.on_gpu:
                    if node.inplace:
                        for mp in self.node_to_arr_maps:
                            mp[node] = ndarray.NDArray(None)
                    elif self.inference and isinstance(node, DropoutOp):
                        for mp in self.node_to_arr_maps:
                            mp[node] = mp[node.inputs[0]]
                    else:
                        for mp in self.node_to_arr_maps:
                            mp[node] = ndarray.empty(shape, ctx=node.ctx)
                else:
                    for mp in self.node_to_arr_maps:
                        mp[node] = ndarray.empty(shape, ctx=node.ctx)

    def run(self, eval_node_list, feed_dicts_list, convert_to_numpy_ret_vals, batch_num):
        if feed_dicts_list:
            assert batch_num == len(feed_dicts_list), "Feed dicts list invalid"

        if not self.node_to_arr_maps:
            self.node_to_arr_maps = [dict() for _ in range(batch_num)]
            need_reallocation = True
        else:
            need_reallocation = False

        feed_shapes = {}

        # get feed in values
        for idx in range(batch_num):
            cur_node_to_arr_map = self.node_to_arr_maps[idx]
            feed_dict = feed_dicts_list[idx] if feed_dicts_list else {}
            for node, value in feed_dict.items():
                if node not in self.need_feed_nodes:
                    continue
                local_shape = tuple(value.shape)
                local_realloc = node not in cur_node_to_arr_map
                assert self.node_to_shape_map.get(node, local_shape) == local_shape
                if node.on_cpu:
                    assert isinstance(value, (np.ndarray, spmatrix, ndarray.NDArray)), \
                        "feed_dict value type not supported"
                    if isinstance(value, np.ndarray):
                        if local_realloc:
                            self.node_to_arr_map[node] = ndarray.empty(
                                local_shape, ctx=node.ctx)
                        self.node_to_arr_map[node][:] = value
                    else:
                        self.node_to_arr_map[node] = value
                else:
                    if isinstance(value, np.ndarray):
                        if local_realloc:
                            cur_node_to_arr_map[node] = ndarray.array(value, ctx=node.ctx)
                        else:
                            cur_node_to_arr_map[node][:] = value
                    elif isinstance(value, spmatrix):
                        value = coo_matrix(value)
                        value = ndarray.sparse_array(value.data,
                                                     (value.row, value.col), shape=local_shape, ctx=node.ctx)
                        cur_node_to_arr_map[node] = value
                    elif isinstance(value, ndarray.NDArray):
                        if value.ctx == node.ctx:
                            cur_node_to_arr_map[node] = value
                        else:
                            if local_realloc:
                                cur_node_to_arr_map[node] = ndarray.empty(local_shape, ctx=node.ctx)
                            else:
                                cur_node_to_arr_map[node][:] = value
                    elif isinstance(value, ndarray.ND_Sparse_Array):
                        cur_node_to_arr_map[node] = value
                    else:
                        assert False, "feed_dict value type not supported"

                if node not in feed_shapes:
                    feed_shapes[node] = local_shape
                else:
                    assert feed_shapes[node] == local_shape

            for node in self.dataloader_nodes:
                self.node_to_arr_maps[idx][node] = node.get_arr(self.name)
                feed_shapes[node] = node.get_cur_shape(self.name)

        # reallocation, infer shapes and allocate memory
        if need_reallocation:
            self.infer_shape(feed_shapes)
            self.memory_plan()

        saved_opt = None
        # computing
        for cur_topo in [self.forward_topo_order, self.backward_topo_order]:
            node_maps = self.node_to_arr_maps if self.forward_topo_order else self.node_to_arr_maps[::-1]
            for cur_node_to_arr_map in node_maps:
                for node in self.computing_nodes:
                    if node not in cur_topo:
                        continue

                    if isinstance(node, OptimizerOp):
                        saved_opt = node
                        continue

                    input_vals = [cur_node_to_arr_map[n] for n in node.inputs]
                    node_val = cur_node_to_arr_map[node]

                    if isinstance(node, (DropoutOp, Batch_NormalizationOp, Layer_NormalizationOp)):
                        node.compute(input_vals, node_val, self.comp_stream, inference=self.inference)
                    else:
                        node.compute(input_vals, node_val, self.comp_stream)

        # apply gradient update after all calculations for microbatches are finished
        for cur_node_to_arr_map in self.node_to_arr_maps:
            input_vals = [cur_node_to_arr_map[n] for n in saved_opt.inputs]
            node_val = cur_node_to_arr_map[saved_opt]
            saved_opt.compute(input_vals, node_val, self.comp_stream)

        self.comp_stream.sync()

        # get results
        results = []
        for cur_node_to_arr_map in self.node_to_arr_maps:
            cur_res = []
            for n in self.global_eval_nodes:
                if n in cur_node_to_arr_map and cur_node_to_arr_map[n] is not None:
                    cur_res.append(cur_node_to_arr_map[n].asnumpy())
            results.append(cur_res)

        return results


class SubExecutor4Pipedream(object):
    def __init__(self, name, eval_node_list, config):
        self.name = name
        self.config = config
        self.inference = not any([isinstance(node, OptimizerOp) for node in eval_node_list])
        self.eval_node_list = config.my_eval_nodes
        self.global_eval_nodes = eval_node_list
        self.node_to_shape_map = {}
        self.topo_order = find_topo_sort(self.eval_node_list)

        # split the topo into two parts: forward and backward
        for i in range(len(self.topo_order)):
            if isinstance(self.topo_order[i], PipelineSendOp):
                pivot_idx = i
                break

        if config.local_rank == config.nrank - 1:
            # node before oneslike belong to forward
            self.forward_topo_order = self.topo_order[:pivot_idx]
            self.backward_topo_order = self.topo_order[pivot_idx:]
        else:
            self.forward_topo_order = self.topo_order[:pivot_idx+1]
            self.backward_topo_order = self.topo_order[pivot_idx+1:]

        def move_send_op(tp):
            """
            move send op to the tail so that we can wrap send/recv pairs
            with nccl groupcall to avoid deadlock
            """
            saved_send = None
            for n in tp:
                if isinstance(n, PipelineSendOp):
                    saved_send = n
                    break
            tp.remove(saved_send)
            tp.append(saved_send)

        if config.local_rank != 0:
            move_send_op(self.backward_topo_order)
        """
        print("gpu {}'s topo: ".format(config.local_rank),
              [x.desc for x in self.topo_order])
        print("gpu {}'s forward topo: ".format(config.local_rank),
              [x.desc for x in self.forward_topo_order])
        print("gpu {}'s backward topo: ".format(config.local_rank),
              [x.desc for x in self.backward_topo_order])
        print("")
        """

        """
        for each micro batch, we need:
        * a version of tensors to store intermediate values for gradients computation
        * at most one version of weights
        """
        self.batch_to_tensor_maps = dict()  # store intermediate tensors(all nodes except weights)

        # inherit from configurations
        self.comm_mode = self.config.comm_mode
        self.ps_comm = self.config.ps_comm
        self.nccl_comm = self.config.nccl_comm
        self.comp_stream = self.config.comp_stream
        self.h2d_stream = self.config.h2d_stream
        self.d2h_stream = self.config.d2h_stream
        self.nccl_stream = self.config.nccl_stream
        self.param_psval_map = self.config.ps_map
        self.use_sparse_pull = self.config.use_sparse_pull
        self.cstable_policy = self.config.cstable_policy
        self.use_p2p = self.config.p2p_stream is not None

        # assisting structures, improve performance
        self.need_feed_nodes = []
        self.param_nodes = []
        self.dataloader_nodes = []
        self.computing_nodes = []
        for node in self.topo_order:
            if isinstance(node, DataloaderOp) or isinstance(node, GNNDataLoaderOp):
                self.dataloader_nodes.append(node)
            elif isinstance(node, PlaceholderOp):
                if node.shape is None:
                    self.need_feed_nodes.append(node)
                elif node.trainable:
                    self.param_nodes.append(node)
            elif not ((self.use_sparse_pull or self.cstable_policy) and isinstance(node, EmbeddingLookUp) and self.config.prefetch):
                self.computing_nodes.append(node)
        self.batch_num = 0
        self.init_need_allocation = False

    def infer_shape(self, feed_shapes):
        """Given shapes of feed_dict nodes, infer shape for all nodes in graph.

        Implementation note:
        Iteratively calls node.infer_shape to infer shapes.
        Node shapes stored in self.node_to_shape_map.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        self.node_to_shape_map = {}
        for node in self.topo_order:
            if node in feed_shapes:
                self.node_to_shape_map[node] = tuple(feed_shapes[node])
            else:
                input_shapes = [self.node_to_shape_map[n] for n in node.inputs]
                cur_shape = node.infer_shape(input_shapes)
                self.node_to_shape_map[node] = cur_shape if cur_shape is None else tuple(
                    cur_shape)

    def memory_plan(self, batch_id):
        """Allocates ndarray.NDArray for every node except feed_dict nodes.
        Parameters
        ----------
        batch_id: current batch
        """

        for node, shape in self.node_to_shape_map.items():
            mp = self.batch_to_tensor_maps[batch_id]

            if isinstance(node, PlaceholderOp):
                if self.config.placeholder_to_arr_map[node] is not None:
                    orig = self.config.placeholder_to_arr_map[node]
                    copied = None
                    if isinstance(orig, np.ndarray):
                        copied = orig.copy()
                    elif isinstance(orig, ndarray.NDArray):
                        # enable async copy
                        copied = ndarray.empty(orig.shape, orig.ctx)
                        copied._async_copyfrom(orig, self.comp_stream)
                    else:
                        raise ValueError
                    mp[node] = copied
                elif node not in mp:
                    mp[node] = None
            elif not isinstance(node, DataloaderOp) and not isinstance(node, GNNDataLoaderOp):
                # add for OptimizerOp and ParameterServerOp
                if shape is None:
                    mp[node] = None
                elif isinstance(node, (EmbeddingLookUp_Gradient, DataD2HSparseOp)):
                    mp[node] = ndarray.IndexedSlices(dense_shape=shape)
                elif self.inference and isinstance(node, DropoutOp):
                    mp[node] = mp[node.inputs[0]]
                else:
                    mp[node] = ndarray.empty(shape, ctx=node.ctx)

    def copy_latest_weight(self):
        """
            In pipedream, the weight used in forward pass are used again in the backward pass.
            However, gradients should be applied to the latest model.
            Thus, we need to copy latest model weight.
        """
        # the last worker has only one copy of weight, no need to copy
        if self.config.rank == self.config.nrank - 1:
            return
        oldest = min(self.batch_to_tensor_maps.keys())
        for node in self.config.placeholder_to_arr_map:
            if isinstance(node, PlaceholderOp) and node.trainable:
                dst_tensor = self.batch_to_tensor_maps[oldest][node]
                src_tensor = self.config.placeholder_to_arr_map[node]
                dst_tensor._async_copyfrom(src_tensor, self.comp_stream)
                # after optimizer update, this dst tensor will become the latest model weight
                # so we set let placeholder_to_arr_map point to it
                self.config.placeholder_to_arr_map[node] = dst_tensor


    def run(self, eval_node_list, feed_dict_list, convert_to_numpy_ret_vals, batch_num):
        rank = self.config.rank
        nrank = self.config.nrank

        last_vacant_batch = -1
        in_flight_batches = []
        start_group_call_idx = nrank - rank
        scheduler = pipedream_scheduler(rank, nrank)

        results_list = []

        while True:
            batch_id, cur_schedule = next(scheduler)

            cur_topo = self.backward_topo_order if cur_schedule == 1 else self.forward_topo_order

            if cur_schedule == 0:
                if batch_id > batch_num:
                    """
                    add necessary group call to finish the pipeline
                    """
                    if len(in_flight_batches) == 0:
                        if rank != 0:  # self.config.nrank - 1:
                            GroupEnd()
                        break
                    else:
                        # still have unfinished micro-batches to do backward
                        if rank == 0:
                            GroupStart()
                        else:
                            GroupEnd()
                            GroupStart()
                        continue

                in_flight_batches.append(batch_id)

                if last_vacant_batch == -1:
                    # no old NDArray to reuse, allocate new
                    if batch_id not in self.batch_to_tensor_maps:
                        self.batch_to_tensor_maps[batch_id] = dict()
                else:
                    # change ownership of old array and reuse
                    self.batch_to_tensor_maps[batch_id] = self.batch_to_tensor_maps.pop(last_vacant_batch)
                    last_vacant_batch = -1

                feed_shapes = {}
                need_reallocation = self.init_need_allocation
                if self.batch_to_tensor_maps[batch_id] == dict():
                    need_reallocation = True
                # get dataloader values
                for node in self.dataloader_nodes:
                    local_shape = node.get_cur_shape(self.name)
                    local_realloc = local_shape != self.node_to_shape_map.get(node, None)
                    need_reallocation = need_reallocation or local_realloc
                    self.batch_to_tensor_maps[batch_id][node] = node.get_arr(self.name)
                    feed_shapes[node] = local_shape

                # reallocation, infer shapes and allocate memory
                if need_reallocation:
                    self.init_need_allocation = False
                    if self.node_to_shape_map == {}:
                        self.infer_shape(feed_shapes)
                    self.memory_plan(batch_id)

            else:
                in_flight_batches.pop(0)

            # compute, same logic for backward and forward
            for node in self.computing_nodes:
                if node not in cur_topo:
                    continue

                node_val = self.batch_to_tensor_maps[batch_id][node]

                input_vals = []
                for n in node.inputs:
                    input_vals.append(self.batch_to_tensor_maps[batch_id][n])

                if isinstance(node, PipelineSendOp):
                    """
                    to avoid deadlock of PipelineSend/PipelineRecv pairs,
                    we need wrap them in group call.

                    Forward compute topo: [... node1 node2 SendOp]
                    Backward compute topo: [RecvOp node3 node4 ...]

                    for each rank, we need to insert GroupStart before the ending
                    SendOp of each forward phase, and insert GroupEnd after the first
                    RecvOp of the next backward phase
                    """
                    group_call = False
                    if rank == 0 and batch_id >= start_group_call_idx:
                        group_call = True
                    if rank == nrank - 1:
                        group_call = True
                    if rank not in (0, nrank - 1):
                        if cur_schedule == 1 or batch_id >= start_group_call_idx:
                            group_call = True
                    node.compute(input_vals, node_val, self.comp_stream, group_call=group_call)

                elif isinstance(node, PipelineReceiveOp):
                    group_call = False
                    if rank == 0:
                        group_call = True
                    if rank == nrank - 1 and batch_id > start_group_call_idx:
                        group_call = True
                    if rank not in (0, nrank - 1):
                        if cur_schedule == 1 or batch_id > start_group_call_idx:
                            group_call = True
                    node.compute(input_vals, node_val, self.comp_stream, group_call=group_call)

                elif isinstance(node, (DropoutOp, Batch_NormalizationOp, Layer_NormalizationOp)):
                    node.compute(input_vals, node_val,
                                 self.comp_stream, inference=self.inference)

                elif isinstance(node, OptimizerOp):
                    self.copy_latest_weight()
                    node.compute(input_vals, node_val, self.comp_stream, self.batch_to_tensor_maps[batch_id])

                else:
                    node.compute(input_vals, node_val, self.comp_stream)

            self.comp_stream.sync()

            # save result as numpy(must), because the tensor will be reused later
            if cur_schedule == 1:
                tmp_results = []
                for n in self.global_eval_nodes:
                    if n in self.batch_to_tensor_maps[batch_id]:
                        r = self.batch_to_tensor_maps[batch_id][n]
                        tmp_results.append(r.asnumpy() if r is not None else None)
                results_list.append(tmp_results)

            # after update, mark the vacant maps
            if cur_schedule == 1:
                #assert last_vacant_batch == -1, "last_vacant_batch error, check the logic of code"
                last_vacant_batch = batch_id

            # end of scheduling loop
        # release for the next run
        self.batch_to_tensor_maps = {}

        return results_list


class SubExecutor(object):
    def __init__(self, name, eval_node_list, config):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        topo_order: list of nodes in topological order
        node_to_shape_map: dict from node to shape of the node
        node_to_arr_map: dict from node to ndarray.NDArray allocated for node
        feed_shapes: shapes of feed_dict from last run(...)
        """
        self.name = name
        self.eval_node_list = eval_node_list
        self.config = config
        inference = not any([isinstance(node, OptimizerOp)
                             for node in eval_node_list])
        self.inference = inference

        if config.gpipe or config.pipedream:
            assert self.inference
            self.eval_node_list = []
            # Remove the last pipeline send on worker 1...n-1 ( not needed in inference stage) and the optimizer
            remove_send = 1 if config.rank > 0 else 0
            for node in config.my_eval_nodes[::-1]:
                if remove_send and isinstance(node, PipelineSendOp):
                    remove_send = 0
                elif not isinstance(node, OptimizerOp):
                    self.eval_node_list.append(node)
            self.global_eval_nodes = eval_node_list

        if inference == False:
            self.topo_order = find_topo_sort(self.eval_node_list)
        else:  # in inference phase
            if self.config.use_sparse_pull == True or self.config.cstable_policy is not None:
                # insert ps_sparse_pull_op
                self.topo_order = find_topo_sort_inference(self.eval_node_list)
                # fetch sparse parameter
                fetch_sparse_parameter_value(self.topo_order, self.config)
            else:
                self.topo_order = find_topo_sort(self.eval_node_list)

        self.topo_order = reorder_for_group(
            self.topo_order, self.config.layer_indices)

        # main structures, nodes' shapes and arrays
        self.node_to_shape_map = {}
        self.node_to_arr_map = {}

        # inherit from configurations
        self.comm_mode = self.config.comm_mode
        self.ps_comm = self.config.ps_comm
        self.nccl_comm = self.config.nccl_comm
        self.comp_stream = self.config.comp_stream
        self.h2d_stream = self.config.h2d_stream
        self.d2h_stream = self.config.d2h_stream
        self.nccl_stream = self.config.nccl_stream
        self.param_psval_map = self.config.infer_ps_map if self.inference else self.config.ps_map
        self.use_sparse_pull = self.config.use_sparse_pull
        self.cstable_policy = self.config.cstable_policy
        self.use_p2p = self.config.p2p_stream is not None
        self.dynamic_memory = self.config.dynamic_memory

        # assisting structures, improve performance
        self.need_feed_nodes = []
        self.param_nodes = []
        self.dataloader_nodes = []
        self.computing_nodes = []

        # structures related to memory pool, used when dynamic_memory == True
        self.node_outdeg_map = {}
        self.node_ref_cnt = {}
        self.memory_pool = {}
        for node in self.topo_order:
            self.node_outdeg_map[node] = 0
            self.node_ref_cnt[node] = None

        for node in self.topo_order:
            if isinstance(node, DataloaderOp) or isinstance(node, GNNDataLoaderOp):
                self.dataloader_nodes.append(node)
            elif isinstance(node, PlaceholderOp):
                if node.shape is None:
                    self.need_feed_nodes.append(node)
                elif node.trainable:
                    self.param_nodes.append(node)
            elif not ((self.use_sparse_pull or self.cstable_policy) and isinstance(node, EmbeddingLookUp) and self.config.prefetch):
                self.computing_nodes.append(node)
            for n in node.inputs:
                self.node_outdeg_map[n] += 1
        self.batch_num = set([node.get_batch_num(self.name)
                              for node in self.dataloader_nodes])
        assert len(self.batch_num) <= 1, 'Batch num not conform.'
        self.batch_num = None if len(
            self.batch_num) == 0 else self.batch_num.pop()
        self.init_need_allocation = (self.need_feed_nodes == []) and (
            self.dataloader_nodes == [])

    def update_executor(self, eval_node_list):
        self.eval_node_list = eval_node_list
        inference = not any([isinstance(node, OptimizerOp)
                             for node in eval_node_list])
        self.inference = inference

        if self.config.p2p_stream and self.inference == True:
            raise NotImplementedError

        if inference == False:
            self.topo_order = find_topo_sort(self.eval_node_list)
        else:  # in inference phase
            if self.config.use_sparse_pull == True or self.config.cstable_policy is not None:
                # insert ps_sparse_pull_op
                self.topo_order = find_topo_sort_inference(self.eval_node_list)
                # fetch sparse parameter
                fetch_sparse_parameter_value(self.topo_order, self.config)
            else:
                self.topo_order = find_topo_sort(self.eval_node_list)

        self.topo_order = reorder_for_group(
            self.topo_order, self.config.layer_indices)

        # main structures, nodes' shapes and arrays
        self.node_to_shape_map = {}
        self.node_to_arr_map = {}

        # assisting structures, improve performance
        self.need_feed_nodes = []
        self.param_nodes = []
        self.dataloader_nodes = []
        self.computing_nodes = []

        # structures related to memory pool, used when dynamic_memory == True
        self.node_outdeg_map = {}
        self.node_ref_cnt = {}
        self.memory_pool = {}
        for node in self.topo_order:
            self.node_outdeg_map[node] = 0
            self.node_ref_cnt[node] = None

        for node in self.topo_order:
            if isinstance(node, DataloaderOp) or isinstance(node, GNNDataLoaderOp):
                self.dataloader_nodes.append(node)
            elif isinstance(node, PlaceholderOp):
                if node.shape is None:
                    self.need_feed_nodes.append(node)
                elif node.trainable:
                    self.param_nodes.append(node)
            elif not ((self.use_sparse_pull or self.cstable_policy) and isinstance(node, EmbeddingLookUp) and self.config.prefetch):
                self.computing_nodes.append(node)
            for n in node.inputs:
                self.node_outdeg_map[n] += 1
        self.batch_num = set([node.get_batch_num(self.name)
                              for node in self.dataloader_nodes])
        assert len(self.batch_num) <= 1, 'Batch num not conform.'
        self.batch_num = None if len(
            self.batch_num) == 0 else self.batch_num.pop()
        self.init_need_allocation = (self.need_feed_nodes == []) and (
            self.dataloader_nodes == [])

    def infer_shape(self, feed_shapes):
        """Given shapes of feed_dict nodes, infer shape for all nodes in graph.

        Implementation note:
        Iteratively calls node.infer_shape to infer shapes.
        Node shapes stored in self.node_to_shape_map.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """

        def make_group():
            if len(grouping_nodes) > 1:
                GroupStart()
            temp_res = {node: ndarray.empty(
                (4,), ctx=node.ctx) for node in grouping_nodes}
            nccl_comm = self.config.nccl_comm
            p2p_stream = self.config.p2p_stream
            for node in grouping_nodes:
                if isinstance(node, PipelineSendOp):
                    shape = self.node_to_shape_map[node.inputs[0]]
                    if len(shape) < 4:
                        shape = [0] * (4 - len(shape)) + list(shape)
                    # construct and send
                    temp_res[node][:] = np.array(shape)
                    nccl_comm.dlarraySend(temp_res[node],
                                          ncclDataType_t.ncclFloat32,
                                          node.const_attr,
                                          p2p_stream)
                else:
                    nccl_comm.dlarrayRecv(temp_res[node],
                                          ncclDataType_t.ncclFloat32,
                                          node.const_attr,
                                          p2p_stream)
            if len(grouping_nodes) > 1:
                GroupEnd()
            p2p_stream.sync()
            for node in grouping_nodes:
                if isinstance(node, PipelineSendOp):
                    self.node_to_shape_map[node] = None
                else:
                    shape_arr = [int(x) for x in list(
                        temp_res[node].asnumpy()) if x != 0]
                    self.node_to_shape_map[node] = tuple(shape_arr)
            grouping_nodes.clear()

        self.node_to_shape_map = {}
        grouping_nodes = []
        cur_ind = -1
        for node in self.topo_order:
            if node in feed_shapes:
                self.node_to_shape_map[node] = tuple(feed_shapes[node])
            else:
                if isinstance(node, (PipelineSendOp, PipelineReceiveOp)):
                    if len(grouping_nodes) > 0 and self.config.layer_indices[node] != cur_ind:
                        make_group()
                    if len(grouping_nodes) == 0:
                        cur_ind = self.config.layer_indices[node]
                    grouping_nodes.append(node)
                    continue
                elif len(grouping_nodes) > 0:
                    make_group()
                input_shapes = [self.node_to_shape_map[n] for n in node.inputs]
                cur_shape = node.infer_shape(input_shapes)
                self.node_to_shape_map[node] = cur_shape if cur_shape is None else tuple(
                    cur_shape)
        if len(grouping_nodes) > 0:
            make_group()

    def from_memory_pool(self, key, node):
        if key in self.memory_pool:
            self.node_to_arr_map[node] = self.memory_pool[key].pop()
            if not len(self.memory_pool[key]):
                del self.memory_pool[key]
            return True
        else:
            return False

    def to_memory_pool(self, key, node):
        if key not in self.memory_pool:
            self.memory_pool[key] = []
        self.memory_pool[key].append(self.node_to_arr_map[node])
        self.node_to_arr_map[node] = None

    def node_memory_plan(self, node):
        """Allocates ndarray.NDArray for the specified node, used when dynamic_memory == True
        Parameters
        ----------
        """
        shape = self.node_to_shape_map[node]
        if isinstance(node, PlaceholderOp):
            if self.config.placeholder_to_arr_map[node] is not None:
                self.node_to_arr_map[node] = self.config.placeholder_to_arr_map[node]
            elif node not in self.node_to_arr_map:
                self.node_to_arr_map[node] = None
        elif not isinstance(node, DataloaderOp) and not isinstance(node, GNNDataLoaderOp):
            # add for OptimizerOp and ParameterServerOp
            if shape is None:
                self.node_to_arr_map[node] = None
                return
            if isinstance(node, (EmbeddingLookUp_Gradient, DataD2HSparseOp)):
                if not self.from_memory_pool((shape, 'IndexedSlices'), node):
                    self.node_to_arr_map[node] = ndarray.IndexedSlices(
                        dense_shape=shape)
                return
            if isinstance(node, EmbeddingLookUp) and (self.use_sparse_pull or self.cstable_policy) and self.config.prefetch:
                self.node_to_arr_map[node] = self.param_psval_map[node.inputs[0]]
                return
            if node.on_gpu:
                if node.inplace:
                    self.node_to_arr_map[node] = ndarray.NDArray(None)
                elif self.inference and isinstance(node, DropoutOp):
                    self.node_to_arr_map[node] = self.node_to_arr_map[node.inputs[0]]
                else:
                    if not self.from_memory_pool(shape, node):
                        self.node_to_arr_map[node] = ndarray.empty(
                            shape, ctx=node.ctx)
            else:
                if not self.from_memory_pool(shape, node):
                    self.node_to_arr_map[node] = ndarray.empty(
                        shape, ctx=node.ctx)

    def memory_plan(self):
        """Allocates ndarray.NDArray for every node except feed_dict nodes.
        Parameters
        ----------
        """
        for node, shape in self.node_to_shape_map.items():
            if isinstance(node, PlaceholderOp):
                if self.config.placeholder_to_arr_map[node] is not None:
                    self.node_to_arr_map[node] = self.config.placeholder_to_arr_map[node]
                elif node not in self.node_to_arr_map:
                    self.node_to_arr_map[node] = None
            elif not isinstance(node, DataloaderOp) and not isinstance(node, GNNDataLoaderOp):
                # add for OptimizerOp and ParameterServerOp
                if shape is None:
                    self.node_to_arr_map[node] = None
                    continue
                if isinstance(node, (EmbeddingLookUp_Gradient, DataD2HSparseOp)):
                    self.node_to_arr_map[node] = ndarray.IndexedSlices(
                        dense_shape=shape)
                    continue
                if isinstance(node, EmbeddingLookUp) and (self.use_sparse_pull or self.cstable_policy) and self.config.prefetch:
                    self.node_to_arr_map[node] = self.param_psval_map[node.inputs[0]]
                    continue
                if isinstance(node, AllReduceCommunicateOp) and isinstance(node.inputs[0], EmbeddingLookUp_Gradient):
                    self.node_to_arr_map[node] = ndarray.IndexedSlices(
                        dense_shape=shape)
                    continue
                if node.on_gpu:
                    if node.inplace:
                        self.node_to_arr_map[node] = ndarray.NDArray(None)
                    elif self.inference and isinstance(node, DropoutOp):
                        self.node_to_arr_map[node] = self.node_to_arr_map[node.inputs[0]]
                    else:
                        self.node_to_arr_map[node] = ndarray.empty(
                            shape, ctx=node.ctx)
                else:
                    self.node_to_arr_map[node] = ndarray.empty(
                        shape, ctx=node.ctx)

    def run(self, eval_node_list={}, feed_dict={}, convert_to_numpy_ret_vals=False):
        """
        Parameters
        ----------
        feed_dict: a dictionary of node->np.ndarray supplied by user.
        convert_to_numpy_ret_vals: whether to convert ret vals to np.array

        Returns
        -------
        A list of values for nodes in eval_node_list. NDArray or np.ndarray.
        """
        assert len(feed_dict) == len(
            self.need_feed_nodes) or self.use_p2p, 'Feed dict invalid.'
        if eval_node_list != {} and eval_node_list != self.eval_node_list:
            self.update_executor(eval_node_list)

        feed_shapes = {}
        need_reallocation = self.init_need_allocation

        # get feed in values
        for node, value in feed_dict.items():
            if self.use_p2p and node not in self.need_feed_nodes:
                continue
            assert node in self.need_feed_nodes, 'Only allow feed in PlaceholderOp with no values, here got %s:%s.' % (
                str(type(node)), node.name)
            local_shape = tuple(value.shape)
            local_realloc = local_shape != self.node_to_shape_map.get(
                node, None)
            need_reallocation = need_reallocation or local_realloc
            if node.on_cpu:
                assert isinstance(value, (np.ndarray, spmatrix, ndarray.NDArray)), \
                    "feed_dict value type not supported"
                if isinstance(value, np.ndarray):
                    if local_realloc:
                        self.node_to_arr_map[node] = ndarray.empty(
                            local_shape, ctx=node.ctx)
                    self.node_to_arr_map[node][:] = value
                else:
                    self.node_to_arr_map[node] = value
            else:
                if isinstance(value, np.ndarray):
                    if local_realloc:
                        self.node_to_arr_map[node] = ndarray.array(
                            value, ctx=node.ctx)
                    else:
                        self.node_to_arr_map[node][:] = value
                elif isinstance(value, spmatrix):
                    value = coo_matrix(value)
                    value = ndarray.sparse_array(value.data,
                                                 (value.row, value.col), shape=local_shape, ctx=node.ctx)
                    self.node_to_arr_map[node] = value
                elif isinstance(value, ndarray.NDArray):
                    if value.ctx == node.ctx:
                        self.node_to_arr_map[node] = value
                    else:
                        if local_realloc:
                            self.node_to_arr_map[node] = ndarray.empty(
                                local_shape, ctx=node.ctx)
                        else:
                            self.node_to_arr_map[node][:] = value
                elif isinstance(value, ndarray.ND_Sparse_Array):
                    self.node_to_arr_map[node] = value
                else:
                    assert False, "feed_dict value type not supported"
            feed_shapes[node] = local_shape

        # get dataloader values
        for node in self.dataloader_nodes:
            local_shape = node.get_cur_shape(self.name)
            local_realloc = local_shape != self.node_to_shape_map.get(
                node, None)
            need_reallocation = need_reallocation or local_realloc
            self.node_to_arr_map[node] = node.get_arr(self.name)
            feed_shapes[node] = local_shape

        # in pipedream, we should retrieve the latest model parameter.
        if self.config.pipedream:
            self.node_to_arr_map.update(self.config.placeholder_to_arr_map)

        # reallocation, infer shapes and allocate memory
        if need_reallocation:
            self.init_need_allocation = False
            self.infer_shape(feed_shapes)
            if not self.dynamic_memory:
                self.memory_plan()

        # computing
        grouping_nodes = []
        cur_ind = -1

        def make_group():
            if len(grouping_nodes) > 1:
                GroupStart()
            nccl_comm = self.config.nccl_comm
            p2p_stream = self.config.p2p_stream
            for node in grouping_nodes:
                if isinstance(node, PipelineSendOp):
                    nccl_comm.dlarraySend(self.node_to_arr_map[node.inputs[0]],
                                          ncclDataType_t.ncclFloat32,
                                          node.const_attr,
                                          p2p_stream)
                else:
                    nccl_comm.dlarrayRecv(self.node_to_arr_map[node],
                                          ncclDataType_t.ncclFloat32,
                                          node.const_attr,
                                          p2p_stream)
            if len(grouping_nodes) > 1:
                GroupEnd()
            for node in grouping_nodes:
                if isinstance(node, PipelineReceiveOp):
                    node.event.record(p2p_stream)
            grouping_nodes.clear()
        for node in self.computing_nodes:
            if self.dynamic_memory:
                # allocate memory for the node when dynamic_memory == True
                if self.node_ref_cnt[node] is None or need_reallocation:
                    self.node_memory_plan(node)
                    self.node_ref_cnt[node] = self.node_outdeg_map[node]
                for n in node.inputs:
                    if n not in self.node_to_arr_map:
                        self.node_memory_plan(n)
                        self.node_ref_cnt[n] = self.node_outdeg_map[n]

            if node.on_cpu and isinstance(self.node_to_arr_map[node], ndarray.NDArray):
                if DNNL_LIB['cpu_ArraySet'] and not isinstance(node, DataD2HOp):
                    cpu_array_set(self.node_to_arr_map[node], 0.0)
                else:
                    # here we suppose not using DNNL_LIB
                    # self.node_to_arr_map[node][:] = np.zeros(self.node_to_shape_map[node]).astype(np.float32)
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

                for n in node.inputs:
                    if n.event:
                        n.event.sync()
                input_vals = [self.node_to_arr_map[n] for n in node.inputs]
                node_val = self.node_to_arr_map[node]

                if isinstance(node, (ParameterServerCommunicateOp, ParameterServerSparsePullOp)):
                    # Here we use d2h stream in ps op, since the stream is used for d2h data transfer.
                    # Please take care at this part.
                    node.compute(input_vals, node_val, self.d2h_stream)

                elif isinstance(node, AllReduceCommunicateOp):
                    node.compute(input_vals, node_val, self.nccl_stream)

                elif isinstance(node, DataH2DOp):
                    node.compute(input_vals, node_val, self.h2d_stream)

                elif isinstance(node, (DataD2HOp, DataD2HSparseOp)):
                    node.compute(input_vals, node_val, self.d2h_stream)

                elif isinstance(node, (DropoutOp, Batch_NormalizationOp, Layer_NormalizationOp)):
                    node.compute(input_vals, node_val,
                                 self.comp_stream, inference=self.inference)
                    if isinstance(node.event, Event):
                        # for d2h op / eval nodes / nodes before [allreduce or ps nodes or pipelinesend nodes]
                        node.event.record(self.comp_stream)

                else:
                    node.compute(input_vals, node_val, self.comp_stream)
                    if isinstance(node.event, Event):
                        # for d2h op / eval nodes / nodes before [allreduce or ps nodes or pipelinesend nodes]
                        node.event.record(self.comp_stream)

            if self.dynamic_memory:
                # free nodes whose reference count is 0 when dynamic_memory == True
                for n in node.inputs:
                    if n in self.computing_nodes:
                        self.node_ref_cnt[n] -= 1
                        if self.node_ref_cnt[n] <= 0 and n not in self.eval_node_list:
                            self.node_ref_cnt[n] = None
                            key = self.node_to_shape_map[n]
                            if key is not None:
                                if n.op_type in ['EmbeddingLookUp_Gradient', 'DataD2HSparseOp']:
                                    key = (key, 'IndexedSlices')
                                self.to_memory_pool(key, n)
                            else:
                                del self.node_to_arr_map[n]

        if len(grouping_nodes) > 0:
            make_group()

        for n in self.eval_node_list:
            # every node in eval_node_list should have an event (except dataloader/optimizer...)
            if n.event:
                n.event.sync()

        # get results
        results = [self.node_to_arr_map[n] for n in self.eval_node_list]
        if convert_to_numpy_ret_vals:
            for i in range(len(results)):
                if results[i] is not None:
                    results[i] = results[i].asnumpy()

        # remap to original order in model parallel
        if self.use_p2p:
            results = filter(lambda x : x[0] in self.global_eval_nodes,
                zip(self.eval_node_list, results))
            results = [x[1] for x in results]

        return results


def gradients(output_node, node_list, insert_grad=None):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.
    insert_grad: used to assign gradient to output_node in model parallel.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """
    if isinstance(output_node, list):
        node_to_output_grads_list = {
            output_node[i]: [OnesLike.oneslike_op(output_node[i])] if insert_grad is None
            else [insert_grad[i]] for i in range(len(output_node))
        }
    else:
        node_to_output_grads_list = {
            output_node: [OnesLike.oneslike_op(output_node)] if insert_grad is None else [
                insert_grad]
        }
        output_node = [output_node]
    node_to_output_grad = {}
    # Traverse forward graph in reverse topological order
    reverse_topo_order = reversed(find_topo_sort(output_node))
    for node in reverse_topo_order:
        # here the ctx for embedding lookup is a workaround
        # TODO: when implement PS strategy for context semantics, modify here
        if isinstance(node, EmbeddingLookUp):
            output_grad = sum_node_list(
                node_to_output_grads_list[node], node_to_output_grads_list[node][0].raw_ctx)
        else:
            output_grad = sum_node_list(
                node_to_output_grads_list[node], node.raw_ctx)
        if output_grad is None:
            for n in node.inputs:
                if n not in node_to_output_grads_list:
                    node_to_output_grads_list[n] = []
            continue
        node_to_output_grad[node] = output_grad
        input_grads_list = node.gradient(output_grad)
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            # Calculate partial adjoint for input nodes.
            node_to_output_grads_list[node.inputs[i]].append(
                input_grads_list[i])

    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list

##################
# Helper Methods #
##################


def topo_sort_with_hook(node_list, config):
    visited = set()
    for node in node_list:
        topo_sort_dfs_with_hook(node, visited, config)


def topo_sort_dfs_with_hook(node, visited, config):
    if node in visited:
        return
    visited.add(node)
    node.backward_hook(config)
    # move param from node to config
    if isinstance(node, PlaceholderOp):
        config.placeholder_to_arr_map[node] = node.tensor_value
        node.tensor_value = None
    for n in node.inputs:
        topo_sort_dfs_with_hook(n, visited, config)
    node.forward_hook(config)


def find_topo_sort(node_list):
    """Given a list of nodes, return a topo ordering of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a
    topological sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def find_topo_sort_inference(node_list):
    topo_order = find_topo_sort(node_list)
    embedding_list = list()
    embedding_outputs = dict()
    embedding_cnt = dict()
    for node in topo_order:
        if isinstance(node, EmbeddingLookUp):
            embedding_outputs[node] = list()
            embedding_cnt[node] = 0
            embedding_list.append(node)
        else:
            for input_node in node.inputs:
                if isinstance(input_node, EmbeddingLookUp):
                    embedding_outputs[input_node].append(node)
                    embedding_cnt[input_node] += 1
    topo_order_inference = list()
    for node in topo_order:
        topo_order_inference.append(node)
        for embedding in embedding_list:
            if node in embedding_outputs[embedding]:
                embedding_cnt[embedding] -= 1
            if embedding_cnt[embedding] == 0:
                topo_order_inference.append(parameterServerSparsePull_op(
                    embedding, embedding_outputs[embedding]))
                embedding_list.remove(embedding)

    return topo_order_inference


def fetch_sparse_parameter_value(node_list, config):
    for node in node_list:
        if isinstance(node, ParameterServerSparsePullOp):
            node.forward_hook(config)


def fetch_dense_parameter_value(node_list, config):
    assert config.comm_mode in ('PS', 'Hybrid')
    topo_order = find_topo_sort(node_list)
    val_list = []
    # get var list
    for node in topo_order:
        if isinstance(node, PlaceholderOp) and node.trainable:
            val_list.append(node)
    for node in val_list:
        if config.use_sparse_pull and node.is_embed:
            continue
        else:
            pull_val = ndarray.empty(node.shape, ctx=ndarray.cpu(0))
            config.ps_comm.Pull(node.id, pull_val.handle)
            config.infer_ps_map[node] = pull_val
            config.placeholder_to_arr_map[node] = pull_val
        node.event.update()


def sum_node_list(node_list, ctx):
    """Custom sum func to avoid creating redundant nodes in Python sum func."""
    node_list = [n for n in node_list if n is not None]
    if node_list == []:
        return None
    sum_node = node_list[0]
    for n in node_list[1:]:
        sum_node = add_op(sum_node, n, ctx=ctx)
    return sum_node


def pipedream_scheduler(rank, nrank):
    """
    used in pipedream; 0: forward, 1: backward

    the pipeline schedule is 1F1B in steady phase like following:
        * -- means bubble
        * 1F means forward of micro-batch1
        * 2B means backward of micro-batch2

    gpu0: 1F -- 2F -- 3F -- 4F 1B 5F 2B 6F 3B 7F 4B -- 5B -- 6B -- 7B
    gpu1:    1F -- 2F -- 3F 1B 4F 2B 5F 3B 6F 4B 7F 5B -- 6B -- 7B
    gpu2:       1F -- 2F 1B 3F 2B 4F 3B 5F 4B 6F 5B 7F 6B -- 7B
    gpu3:          1F 1B 2F 2B 3F 3B 4F 4B 5F 5B 6F 6B 7F 7B

    """
    batch_id_fwd, batch_id_bwd = 0, 0
    for _ in range(nrank - rank):
        batch_id_fwd += 1
        yield (batch_id_fwd, 0)
    while True:
        batch_id_bwd += 1
        yield (batch_id_bwd, 1)
        batch_id_fwd += 1
        yield (batch_id_fwd, 0)


def reorder_for_group(topo_order, layer_indices):
    if layer_indices is None:
        return topo_order
    # here we reorder for 2 reasons:
    # 1. group consecutive pipeline send/recv ops
    # 2. reorder pipeline send/recv ops according to grouping indices
    has_pipeline_ops = set([layer_indices[x] for x in layer_indices if isinstance(
        x, (PipelineSendOp, PipelineReceiveOp))])
    labels = {}
    for node in topo_order:
        if isinstance(node, (DataH2DOp, DataD2HOp, DataD2HSparseOp)):
            layer_indices[node] = layer_indices[node.inputs[0]] + 0.5
        cur_with_pipeline = layer_indices[node] in has_pipeline_ops
        if cur_with_pipeline and isinstance(node, SplitOp):
            labels[node] = 1
        elif isinstance(node, (PipelineSendOp, PipelineReceiveOp)):
            labels[node] = 2
        elif cur_with_pipeline and isinstance(node, (AddOp, ConcatOp)):
            labels[node] = 3
        else:
            labels[node] = 0

    topo_order = sorted(topo_order, key=lambda x: 10 *
                        layer_indices[x] + labels[x])
    return topo_order
