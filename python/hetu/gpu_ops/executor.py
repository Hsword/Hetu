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
from .EmbeddingLookUp import EmbeddingLookUp, EmbeddingLookUp_Gradient
from ..optimizer import OptimizerOp
from . import OnesLike
from ..stream import create_stream_handle, Event
from ..context import get_current_context, get_launch_config_by_traverse_nodes, assign_context_by_traverse_nodes, DeviceGroup
from .PipelineSend import PipelineSendOp
from .PipelineReceive import PipelineReceiveOp
from .Dropout import DropoutOp
from .LayerNorm import Layer_NormalizationOp
from .OnesLike import OnesLikeOp
from operator import add
from functools import reduce
import ctypes
import os
from time import time


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
        pipedream=False
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
        self.seed = seed if seed else np.int64(time())
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

        self.my_eval_nodes = eval_node_list
        self.p2p_stream = None
        self.param_allreduce_group = {}
        if context_launch:
            # comm_mode is None <=> only 1 model parallel instance
            self.context = ndarray.gpu(device_id)
            self.p2p_stream = create_stream_handle(
                self.context) if init_p2p_stream else None
            self.my_eval_nodes, trainable_params, has_send_recv = assign_context_by_traverse_nodes(
                eval_node_list, self.context, self.nccl_comm, self.p2p_stream)
            if (self.comm_mode == "Hybrid" or self.comm_mode == "AllReduce") and has_send_recv:
                # here we need to use group communicator to implement allreduce,
                # since not all processes use the same group
                groups = set([n.raw_ctx for n in trainable_params])
                temp_group_comms = {}
                for group in groups:
                    temp_group_comms[group] = new_group_comm(group)
                self.param_allreduce_group = {
                    n: temp_group_comms[n.raw_ctx] for n in trainable_params}
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
        
        if config.gpipe:
            self.subexecutor = {k: SubExecutor4Gpipe(k, v, config) for k, v in eval_node_dict.items()}
        elif config.pipedream:
            self.subexecutor = {k: SubExecutor4Pipedream(k, v, config) for k, v in eval_node_dict.items()}
        else:
            self.subexecutor = {k: SubExecutor(k, v, config) for k, v in eval_node_dict.items()}

        self.topo_order = find_topo_sort(config.my_eval_nodes)
        self.param_nodes = [node for node in self.topo_order if isinstance(
            node, PlaceholderOp) and node.trainable]
        self.comm_mode = self.config.comm_mode
        self.ps_comm = self.config.ps_comm
        self.local_rank = self.config.local_rank
        self.rank = self.config.rank

    def run(self, name='default', eval_node_list={}, feed_dict={}, convert_to_numpy_ret_vals=False):
        return self.subexecutor[name].run(eval_node_list, feed_dict, convert_to_numpy_ret_vals)

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
        self.eval_node_list = eval_node_list
        self.config = config
        inference = not any([isinstance(node, OptimizerOp) for node in eval_node_list])
        self.inference = inference

        if config.p2p_stream:
            self.run_results_indices = [eval_node_list.index(node) if node in eval_node_list else -1 for node in config.my_eval_nodes]
            self.eval_node_list = config.my_eval_nodes
            self.global_eval_nodes = eval_node_list

        if inference == False:
            self.topo_order = find_topo_sort(self.eval_node_list)
        else: # in inference phase
            if self.config.use_sparse_pull == True or self.config.cstable_policy is not None:
                # topo_sort_with_hook(self.eval_node_list, self.config)
                # insert ps_sparse_pull_op
                self.topo_order = find_topo_sort_inference(self.eval_node_list)
                # fetch sparse parameter
                fetch_sparse_parameter_value(self.topo_order, self.config)
            else:
                self.topo_order = find_topo_sort(self.eval_node_list)

        # split the topo into two parts: forward and backward
        # split critia for gpu0~gpu n-2: nodes before the first send op belong to forward
        # split critia for gpu n-1: nodes after the first oneslike op belong to forward
        pivot_idx = 0
        for i in range(len(self.topo_order)):
            if isinstance(self.topo_order[i], (PipelineSendOp, OnesLikeOp)):
                pivot_idx = i
                break
        if config.local_rank == config.nrank - 1:
            self.forward_topo_order = self.topo_order[:pivot_idx]
            self.backward_topo_order = self.topo_order[pivot_idx:]
        else:
            self.forward_topo_order = self.topo_order[:pivot_idx+1]
            self.backward_topo_order = self.topo_order[pivot_idx+1:]

        """
        print("gpu {}'s topo: ".format(config.local_rank), [(x.name,x.desc) for x in self.topo_order])
        print("gpu {}'s forward topo: ".format(config.local_rank), [(x.name,x.desc) for x in self.forward_topo_order])
        print("gpu {}'s backward topo: ".format(config.local_rank), [(x.name,x.desc) for x in self.backward_topo_order])
        """

        self.micro_batches_num = -1

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
        self.batch_num = set([node.get_batch_num(self.name) for node in self.dataloader_nodes])
        assert len(self.batch_num) <= 1, 'Batch num not conform.'
        self.batch_num = None if len(self.batch_num) == 0 else self.batch_num.pop()
        self.init_need_allocation = (self.need_feed_nodes == []) and (self.dataloader_nodes == [])

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
                self.node_to_shape_map[node] = cur_shape if cur_shape is None else tuple(cur_shape)
            # print(node.name, self.node_to_shape_map[node])

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
                    assign_one_to_all_maps(node, self.config.placeholder_to_arr_map[node])
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

    def run(self, eval_node_list={}, feed_dicts_list=[], convert_to_numpy_ret_vals=False):
        """
        Parameters
        ----------
        feed_dict: a dictionary of node->np.ndarray supplied by user.
        convert_to_numpy_ret_vals: whether to convert ret vals to np.array

        Returns
        -------
        A list of values for nodes in eval_node_list. NDArray or np.ndarray.
        """
        assert feed_dicts_list and all([isinstance(x, dict) for x in feed_dicts_list]) and \
               len(set([len(x) for x in feed_dicts_list])) == 1, 'Feed dicts list invalid'
        assert len(feed_dicts_list[0]) == len(self.need_feed_nodes) or self.use_p2p, 'Feed dict invalid.'

        if eval_node_list != {} and eval_node_list != self.eval_node_list:
            print("wrong eval_node_list")
            raise ValueError

        if self.micro_batches_num == -1:
            self.micro_batches_num = len(feed_dicts_list)
        else:
            assert self.micro_batches_num == len(feed_dicts_list), "Feed dicts list invalid"

        if not self.node_to_arr_maps:
            self.node_to_arr_maps = [dict() for _ in range(self.micro_batches_num)]

        feed_shapes = {}
        need_reallocation = self.init_need_allocation

        # get feed in values
        for idx, feed_dict in enumerate(feed_dicts_list):
            cur_node_to_arr_map = self.node_to_arr_maps[idx]
            for node, value in feed_dict.items():
                if self.use_p2p and node not in self.need_feed_nodes:
                    continue
                assert node in self.need_feed_nodes, 'Only allow feed in PlaceholderOp with no values, here got %s:%s.' % (str(type(node)), node.name)
                local_shape = tuple(value.shape)
                local_realloc = local_shape != self.node_to_shape_map.get(node, None)
                local_realloc = local_realloc or (node not in cur_node_to_arr_map)
                need_reallocation = need_reallocation or local_realloc
                if node.on_cpu:
                    print("node must be on gpu in gpipe mode")
                    raise AttributeError
                else:
                    if isinstance(value, np.ndarray):
                        if local_realloc:
                            cur_node_to_arr_map[node] = ndarray.array(value, ctx=node.ctx)
                        else:
                            cur_node_to_arr_map[node][:] = value
                    elif isinstance(value, spmatrix):
                        value = coo_matrix(value)
                        value = ndarray.sparse_array(value.data,
                                (value.row, value.col), shape = local_shape, ctx=node.ctx)
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

        assert len(self.dataloader_nodes) == 0, "gpipe mode currently does not support dataloader"

        # reallocation, infer shapes and allocate memory
        if need_reallocation:
            self.init_need_allocation = False
            self.infer_shape(feed_shapes)
            self.memory_plan()

        saved_opt = None
        # computing
        for cur_topo in [self.forward_topo_order, self.backward_topo_order]:
            if cur_topo == self.backward_topo_order:
                self.node_to_arr_maps = self.node_to_arr_maps[::-1]
            for cur_node_to_arr_map in self.node_to_arr_maps:
                for node in self.computing_nodes:
                    if node not in cur_topo:
                        continue

                    if isinstance(node, OptimizerOp):
                        saved_opt = node
                        continue

                    assert node.on_gpu, "node must be on gpu in gpipe mode"

                    input_vals = [cur_node_to_arr_map[n] for n in node.inputs]
                    node_val = cur_node_to_arr_map[node]

                    for n in node.inputs:
                        if n.event:
                            n.event.sync()

                    # print("gpu{} start compute node: ".format(self.config.local_rank), node.name, node.desc)
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

                    elif isinstance(node, (PipelineSendOp, PipelineReceiveOp)):
                        node.compute(input_vals, node_val)

                    elif isinstance(node, (DropoutOp, Batch_NormalizationOp, Layer_NormalizationOp)):
                        node.compute(input_vals, node_val, self.comp_stream, inference=self.inference)
                        if isinstance(node.event, Event):
                            # for d2h op / eval nodes / nodes before [allreduce or ps nodes or pipelinesend nodes]
                            node.event.record(self.comp_stream)

                    else:
                        node.compute(input_vals, node_val, self.comp_stream)
                        if isinstance(node.event, Event):
                            # for d2h op / eval nodes / nodes before [allreduce or ps nodes or pipelinesend nodes]
                            node.event.record(self.comp_stream)

                # every forward or backward pass should sync
                for n in self.eval_node_list:
                    if n in cur_topo and n.event:
                        n.event.sync()
            # recover the order of maps
            if cur_topo == self.backward_topo_order:
                self.node_to_arr_maps = self.node_to_arr_maps[::-1]

        # apply gradient update after all calculations for microbatches are finished
        for n in saved_opt.inputs:
            if n.event:
                n.event.sync()

        for cur_node_to_arr_map in self.node_to_arr_maps:
            input_vals = [cur_node_to_arr_map[n] for n in saved_opt.inputs]
            node_val = cur_node_to_arr_map[saved_opt]
            saved_opt.compute(input_vals, node_val, self.comp_stream)

        for n in self.eval_node_list:
            if n.event:
                n.event.sync()

        # get results
        results = []
        for cur_node_to_arr_map in self.node_to_arr_maps:
            cur_res = [cur_node_to_arr_map[n] for n in self.eval_node_list]
            if convert_to_numpy_ret_vals:
                for i in range(len(cur_res)):
                    if cur_res[i] is not None:
                        cur_res[i] = cur_res[i].asnumpy()
            results.append(cur_res)

        # remap to original order in model parallel
        if self.use_p2p:
            for idx in range(len(results)):
                cur_res = results[idx]
                new_res = [None for _ in self.global_eval_nodes]
                for i, j in enumerate(self.run_results_indices):
                    new_res[j] = cur_res[i]
                results[idx] = new_res

        return results

class SubExecutor4Pipedream(object):
    def __init__(self, name, eval_node_list, config):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        topo_order: list of nodes in topological order
        node_to_shape_map: dict from node to shape of the node
        batch_to_tensor_maps: a dict of maps storing intermediate tensors for each micro-batch 
        batch_to_weight_maps: a dict of maps storing params tensors for each micro-batch 
        feed_shapes: shapes of feed_dict from last run(...)

        """
        self.name = name
        self.eval_node_list = eval_node_list
        self.config = config
        inference = not any([isinstance(node, OptimizerOp) for node in eval_node_list])
        self.inference = inference
      
        if config.p2p_stream:
            self.run_results_indices = [eval_node_list.index(node) if node in eval_node_list else -1 for node in config.my_eval_nodes]
            self.eval_node_list = config.my_eval_nodes
            self.global_eval_nodes = eval_node_list

        if inference == False:
            self.topo_order = find_topo_sort(self.eval_node_list)
        else: # in inference phase
            if self.config.use_sparse_pull == True or self.config.cstable_policy is not None:
                # topo_sort_with_hook(self.eval_node_list, self.config)
                # insert ps_sparse_pull_op
                self.topo_order = find_topo_sort_inference(self.eval_node_list)
                # fetch sparse parameter
                fetch_sparse_parameter_value(self.topo_order, self.config)
            else:
                self.topo_order = find_topo_sort(self.eval_node_list)

        # split the topo into two parts: forward and backward
        # split critia for gpu0~gpu n-2: nodes before(including) the first send op belong to forward
        # split critia for gpu n-1: nodes before(not including) the first oneslike op belong to forward
        pivot_idx = 0
        for i in range(len(self.topo_order)):
            if isinstance(self.topo_order[i], (OnesLikeOp, PipelineSendOp)):
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
        print("gpu {}'s topo: ".format(config.local_rank), [x.desc for x in self.topo_order])
        print("gpu {}'s forward topo: ".format(config.local_rank), [x.desc for x in self.forward_topo_order])
        print("gpu {}'s backward topo: ".format(config.local_rank), [x.desc for x in self.backward_topo_order])
        print("")
        """

        # TODO: should support delicate workload asjustments in the future
        self.micro_batches_num = config.nrank

        # main structures, nodes' shapes and arrays
        self.node_to_shape_map = {}
        """
        for each micro batch, we need:
        * a version of tensors to store intermediate values for gradients computation
        * at most one version of weights
        """
        self.batch_to_tensor_maps = dict() # store intermediate tensors(all nodes except weights)
        self.batch_to_weight_maps = dict() # store weights of different version

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
        self.batch_num = set([node.get_batch_num(self.name) for node in self.dataloader_nodes])
        assert len(self.batch_num) <= 1, 'Batch num not conform.'
        self.batch_num = None if len(self.batch_num) == 0 else self.batch_num.pop()
        self.init_need_allocation = (self.need_feed_nodes == []) and (self.dataloader_nodes == [])
        if self.use_p2p:
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
                self.node_to_shape_map[node] = cur_shape if cur_shape is None else tuple(cur_shape)
            # print(node.name, self.node_to_shape_map[node])

    def memory_plan(self, batch_id):
        """Allocates ndarray.NDArray for every node except feed_dict nodes.
        Parameters
        ----------
        batch_id: current batch
        """

        for node, shape in self.node_to_shape_map.items():
            # for node to be feed, we ingore them
            if isinstance(node, PlaceholderOp) and node.trainable:
                cur_node_to_arr_map = self.batch_to_weight_maps[batch_id]
            else:
                cur_node_to_arr_map = self.batch_to_tensor_maps[batch_id]

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
                    cur_node_to_arr_map[node] = copied
                elif node not in cur_node_to_arr_map:
                    cur_node_to_arr_map[node] = None
            elif not isinstance(node, DataloaderOp) and not isinstance(node, GNNDataLoaderOp):
                # add for OptimizerOp and ParameterServerOp
                if shape is None:
                    cur_node_to_arr_map[node] = None
                    continue
                if isinstance(node, (EmbeddingLookUp_Gradient, DataD2HSparseOp)):
                    cur_node_to_arr_map[node] = ndarray.IndexedSlices(dense_shape=shape)
                    continue
                if isinstance(node, EmbeddingLookUp) and (self.use_sparse_pull or self.cstable_policy) and self.config.prefetch:
                    cur_node_to_arr_map[node] = self.param_psval_map[node.inputs[0]]
                    continue
                if node.on_gpu:
                    if node.inplace:
                        cur_node_to_arr_map[node] = ndarray.NDArray(None)
                    elif self.inference and isinstance(node, DropoutOp):
                        if node.inputs[0] in self.batch_to_weight_maps[batch_id]:
                            cur_node_to_arr_map[node] = self.batch_to_weight_maps[node.inputs[0]]
                        else:
                            cur_node_to_arr_map[node] = self.batch_to_tensor_maps[node.inputs[0]]
                    else:
                        cur_node_to_arr_map[node] = ndarray.empty(shape, ctx=node.ctx)
                else:
                    cur_node_to_arr_map[node] = ndarray.empty(shape, ctx=node.ctx)

    

    def run(self, eval_node_list={}, feed_generator={}, convert_to_numpy_ret_vals=False):
        #from .my_log import get_logger
        #log = get_logger(self.config.local_rank)

        """
        Parameters
        ----------
        feed_generator: a dictionary of node->np.ndarray supplied by user.
        convert_to_numpy_ret_vals: whether to convert ret vals to np.array

        Returns
        -------
        A list of values for nodes in eval_node_list. NDArray or np.ndarray.
        """

        from inspect import isgenerator
        assert isinstance(feed_generator, dict) and all([isgenerator(x) for x in feed_generator.values()]), 'Feed generator invalid'
        assert len(feed_generator) == len(self.need_feed_nodes) or self.use_p2p, 'Feed generator invalid.'

        if eval_node_list != {} and eval_node_list != self.eval_node_list:
            raise ValueError("wrong eval_node_list")

        local_rank = self.config.local_rank
        nrank = self.config.nrank

        batch_id = 0
        last_vacant_batch = -1
        in_flight_batches = []
        start_group_call_idx = nrank - local_rank
        scheduler = get_scheduler(local_rank, nrank)

        results_list = []

        while True:
            cur_schedule = next(scheduler)
            # if cur_schedule == 0: forward
            # if cur_schedule == 1: backward
            
            cur_topo = self.backward_topo_order
            if cur_schedule == 0:
                cur_topo = self.forward_topo_order

            """
            we do following things only at forward stage
            0. initialization
            1. get feed values
            2. maybe memory plan
            """
            if cur_schedule == 0:

                feed_dict = {}
                try:
                    for n, g in feed_generator.items():
                        feed_dict[n] = next(g)
                except StopIteration:
                    #print("no enough data after {} batches, finish all training!".format(batch_id))
                    """
                    add necessary group call to finish the pipeline
                    """
                    from ..communicator.mpi_nccl_comm import GroupStart, GroupEnd
                    if len(in_flight_batches) == 0:
                        if self.config.local_rank != 0:# self.config.nrank - 1:
                            GroupEnd()
                        break
                    else:
                        # still have unfinished micro-batches to do backward
                        if self.config.local_rank == 0:
                            GroupStart()
                        elif self.config.local_rank == self.config.nrank - 1:
                            raise ValueError
                        else:
                            GroupEnd()
                            GroupStart()
                        continue
                except Exception as e:
                    sys.exit()

                batch_id += 1
                in_flight_batches.append(batch_id)
                #print("rank: {}, schedule: {}, batch: {}".format(local_rank, cur_schedule, batch_id))

                if last_vacant_batch == -1:
                    # no old NDArray to reuse, allocate new
                    if batch_id not in self.batch_to_tensor_maps:
                        self.batch_to_tensor_maps[batch_id] = dict()
                    if batch_id not in self.batch_to_weight_maps:
                        self.batch_to_weight_maps[batch_id] = dict()
                else:
                    # change ownership of old array and reuse
                    self.batch_to_weight_maps[batch_id] = self.batch_to_weight_maps.pop(last_vacant_batch)
                    self.batch_to_tensor_maps[batch_id] = self.batch_to_tensor_maps.pop(last_vacant_batch)
                    last_vacant_batch = -1

                feed_shapes = {}
                need_reallocation = self.init_need_allocation
                if self.batch_to_tensor_maps[batch_id] == dict():
                    need_reallocation = True
                # get feed in values
                for node, value in feed_dict.items():
                    if self.use_p2p and node not in self.need_feed_nodes:
                        continue
                    assert node in self.need_feed_nodes, 'Only allow feed in PlaceholderOp with no values, here got %s:%s.' % (str(type(node)), node.name)
                    
                    if node.trainable:
                        cur_node_to_arr_map = self.batch_to_weight_maps[batch_id]
                    else:
                        cur_node_to_arr_map = self.batch_to_tensor_maps[batch_id]

                    local_shape = tuple(value.shape)
                    local_realloc = local_shape != self.node_to_shape_map.get(node, None)
                    local_realloc = local_realloc or (node not in cur_node_to_arr_map)
                    need_reallocation = need_reallocation or local_realloc

                    if node.on_cpu:
                        raise AttributeError("node must be on gpu in pipedream mode")
                    else:
                        if isinstance(value, np.ndarray):

                            if local_realloc:
                                #cur_node_to_arr_map[node] = ndarray.array(value, ctx=node.ctx)
                                cur_node_to_arr_map[node] = ndarray.empty(local_shape, ctx=node.ctx)
                                cur_node_to_arr_map[node].async_h2d(value, self.comp_stream)
                            else:
                                # reinit, use async_copy
                                #cur_node_to_arr_map[node][:] = ndarray.array(value, ctx=node.ctx)
                                cur_node_to_arr_map[node].async_h2d(value, self.comp_stream)

                        elif isinstance(value, spmatrix):
                            value = coo_matrix(value)
                            value = ndarray.sparse_array(value.data,
                                    (value.row, value.col), shape = local_shape, ctx=node.ctx)
                            cur_node_to_arr_map[node] = value
                        elif isinstance(value, ndarray.NDArray):
                            raise ValueError
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


                # reallocation, infer shapes and allocate memory
                if need_reallocation:
                    self.init_need_allocation = False
                    if self.node_to_shape_map == {}:
                        self.infer_shape(feed_shapes)
                    self.memory_plan(batch_id)

                tmp_batch_id = batch_id

            else:
                """
                doing backward, we choose the oldest in-flight batch,
                we use tmp_batch_id to access the according maps
                """
                tmp_batch_id = in_flight_batches.pop(0)
                #print("rank: {}, schedule: {}, batch: {}".format(local_rank, cur_schedule, tmp_batch_id))
                
            assert len(self.dataloader_nodes) == 0, "pipedream currently does not support dataloader"

            #compute, same logic for backward and forward
            for node in self.computing_nodes:
                if node not in cur_topo:
                    continue

                if node.on_cpu:
                    raise ValueError("node must be on gpu in pipedream mode")

                for n in node.inputs:
                    if n.event:
                        n.event.sync()

                node_val = None
                if isinstance(node, PlaceholderOp) and node.trainable:
                    node_val = self.batch_to_weight_maps[tmp_batch_id][node]
                else:
                    node_val = self.batch_to_tensor_maps[tmp_batch_id][node]

                input_vals = []
                for n in node.inputs:
                    if isinstance(n, PlaceholderOp) and n.trainable:
                        input_vals.append(self.batch_to_weight_maps[tmp_batch_id][n])
                    else:
                        input_vals.append(self.batch_to_tensor_maps[tmp_batch_id][n])


                #print("start compute node: {}".format(node.desc))
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

                elif isinstance(node, PipelineSendOp):
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
                    if local_rank == 0 and tmp_batch_id >= start_group_call_idx:
                        group_call = True
                    if local_rank == nrank - 1:
                        group_call = True
                    if local_rank not in (0, nrank - 1):
                        if cur_schedule == 1 or tmp_batch_id >= start_group_call_idx:
                            group_call = True
                    node.compute(input_vals, node_val, group_call=group_call)
                
                elif isinstance(node, PipelineReceiveOp):
                    group_call = False
                    if local_rank == 0:
                        group_call = True
                    if local_rank == nrank - 1 and tmp_batch_id > start_group_call_idx:
                        group_call = True
                    if local_rank not in (0, nrank - 1):
                        if cur_schedule == 1 or tmp_batch_id > start_group_call_idx:
                            group_call = True
                    node.compute(input_vals, node_val, group_call=group_call)

                elif isinstance(node, (DropoutOp, Batch_NormalizationOp, Layer_NormalizationOp)):
                    node.compute(input_vals, node_val, self.comp_stream, inference=self.inference)
                    if isinstance(node.event, Event):
                        # for d2h op / eval nodes / nodes before [allreduce or ps nodes or pipelinesend nodes]
                        node.event.record(self.comp_stream)

                elif isinstance(node, OptimizerOp):
                    node.compute(input_vals, node_val, self.comp_stream, self.batch_to_weight_maps[tmp_batch_id])

                else:
                    node.compute(input_vals, node_val, self.comp_stream)
                    if isinstance(node.event, Event):
                        # for d2h op / eval nodes / nodes before [allreduce or ps nodes or pipelinesend nodes]
                        node.event.record(self.comp_stream)


            # we should do necessary syncing to avoid deadlock
            for n in self.eval_node_list:
                if n.event:
                    n.event.sync()

            #print("finish sync for batch{} at rank{}".format(tmp_batch_id, self.config.local_rank))

            # save result as numpy(must), because the tensor will be reused later
            if cur_schedule == 1:
                tmp_results = []
                for n in self.eval_node_list:
                    if isinstance(n, PlaceholderOp) and n.trainable:
                        r = self.batch_to_weight_maps[tmp_batch_id][n]
                    else:
                        r = self.batch_to_tensor_maps[tmp_batch_id][n]
                    if r is not None:
                        tmp_results.append(r.asnumpy())
                    else:
                        tmp_results.append(None)
                results_list.append(tmp_results)

            # after update, mark the vacant maps
            if cur_schedule == 1:
                #assert last_vacant_batch == -1, "last_vacant_batch error, check the logic of code"
                last_vacant_batch = tmp_batch_id

            # end of scheduling loop

        # update the lastest param to global self.placeholder_to_arr_map
        for n, v in self.batch_to_weight_maps[batch_id].items():
            self.config.placeholder_to_arr_map[n] = v

        # remap to original order in model parallel
        if self.use_p2p:
            for idx in range(len(results_list)):
                tmp_results = results_list[idx]
                new_results = [None for _ in self.global_eval_nodes]
                for i, j in enumerate(self.run_results_indices):
                    new_results[j] = tmp_results[i]
                results_list[idx] = new_results

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

        if config.p2p_stream:
            self.run_results_indices = [eval_node_list.index(
                node) if node in eval_node_list else -1 for node in config.my_eval_nodes]
            self.eval_node_list = config.my_eval_nodes
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

        # main structures, nodes' shapes and arrays
        self.node_to_shape_map = {}
        self.node_to_arr_map = {}

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

        # reallocation, infer shapes and allocate memory
        if need_reallocation:
            self.init_need_allocation = False
            self.infer_shape(feed_shapes)
            self.memory_plan()

        # computing
        for node in self.computing_nodes:
            if node.on_cpu and isinstance(self.node_to_arr_map[node], ndarray.NDArray):
                if DNNL_LIB['cpu_ArraySet'] and not isinstance(node, DataD2HOp):
                    cpu_array_set(self.node_to_arr_map[node], 0.0)
                else:
                    # here we suppose not using DNNL_LIB
                    # self.node_to_arr_map[node][:] = np.zeros(self.node_to_shape_map[node]).astype(np.float32)
                    pass

            input_vals = [self.node_to_arr_map[n] for n in node.inputs]
            node_val = self.node_to_arr_map[node]

            for n in node.inputs:
                if n.event:
                    n.event.sync()

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

            elif isinstance(node, (PipelineSendOp, PipelineReceiveOp)):
                node.compute(input_vals, node_val)

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
            new_results = [None for _ in self.global_eval_nodes]
            for i, j in enumerate(self.run_results_indices):
                new_results[j] = results[i]
            results = new_results

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

def get_scheduler(rank, nrank):
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
    assert 0 <= rank < nrank, "invalid args for get_scheduler"
    prefix_forward = [0 for _ in range(nrank - rank)]
    for p in prefix_forward:
        yield p
    while True:
        yield 1
        yield 0
