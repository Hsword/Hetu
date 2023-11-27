from __future__ import absolute_import, annotations
from .BatchNorm import Batch_NormalizationOp, Batch_Normalization_Gradient_of_DataOp, Batch_Normalization_Gradient_of_ScaleOp, Batch_Normalization_Gradient_of_BiasOp
from .LayerNorm import Layer_NormalizationOp, Layer_Normalization_Gradient_of_DataOp, Layer_Normalization_Gradient_of_ScaleOp, Layer_Normalization_Gradient_of_BiasOp
from .MultiplyConst import MulByConstOp
import numpy as np
from scipy.sparse import spmatrix, coo_matrix
from .. import ndarray
from .._base import DNNL_LIB
from ..gpu_links import array_set
from ..cpu_links import array_set as cpu_array_set
from .Variable import PlaceholderOp  # add for optimizer
from ..dataloader import DataloaderOp, GNNDataLoaderOp
from .AllReduceCommunicate import AllReduceCommunicateOp, AllReduceCommunicateP2POp
from .AllGatherCommunicate import AllGatherCommunicateOp
from .ReduceScatterCommunicate import ReduceScatterCommunicateOp
from .ReduceCommunicate import ReduceCommunicateOp
from .BroadcastCommunicate import BroadcastCommunicateOp
from .AllToAll import AllToAllOp
from .ParameterServerCommunicate import ParameterServerCommunicateOp, ParameterServerSparsePullOp, parameterServerSparsePull_op
from .Sum import sum_op, SumOp, SparseSumOp
from .SumSparseGradient import sum_sparse_gradient_op
from .StopGradient import StopGradientOp
from .DataTransfer import DataH2DOp, DataD2HOp, DataD2HSparseOp, DataH2DSparseOp
from ..communicator.mpi_nccl_comm import ncclDataType_t, GroupStart, GroupEnd
from .EmbeddingLookUp import EmbeddingLookUp, EmbeddingLookUp_Gradient
from .Unique import UniqueIndicesOffsetsOp
from ..optimizer import OptimizerOp
from . import OnesLike
from ..stream import create_stream_handle, Event
from ..context import get_current_context, get_launch_config_by_traverse_nodes, DeviceGroup, GraphStatus
from ..memory_pool import HetuMemoryPool
from .PipelineSend import PipelineSendOp
from .PipelineReceive import PipelineReceiveOp
from .Split import SplitOp
from .Concatenate import ConcatenateOp
from .Dropout import DropoutOp
from operator import add
from functools import reduce
import ctypes
import os
from time import time
import pickle
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..communicator.mpi_nccl_comm import NCCL_Communicator, MPI_Communicator
    from ..ndarray import DLContext, NDArray, ND_Sparse_Array, IndexedSlices
    from ..distributed_strategies.base import Strategy
    from .Node import Op
    from ctypes import CDLL
    from typing import Type, Optional, Union, Tuple, List, Dict, Set
    FEEDINS = Union[np.ndarray, spmatrix, NDArray, ND_Sparse_Array]
    OP_LIST = List[Op]
    ARR_MAP = Dict[Op, Tuple[NDArray, ND_Sparse_Array, IndexedSlices]]
    SHAPE_MAP = Dict[Op, Tuple[int, ...]]


def path_to_lib(name: str) -> str:
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(curr_path, '../../../build/lib/')
    return os.path.join(lib_path, name)


def wrapped_mpi_nccl_init(
    init_nccl: bool = True,
    devices: Optional[List[int]] = None
) -> NCCL_Communicator:
    from ..communicator.mpi_nccl_comm import mpi_communicator
    global mpi_comm
    global nccl_comm
    if 'mpi_comm' not in globals():
        mpi_comm = mpi_communicator(devices=devices)
        if 'nccl_comm' not in globals():
            nccl_comm = mpi_comm.ncclInit() if init_nccl else None
    return nccl_comm


def new_group_comm(devices_context: DeviceGroup = None) -> NCCL_Communicator:
    assert 'mpi_comm' in globals()
    global mpi_comm
    if devices_context is None:
        comm = mpi_comm.ncclInit()
    else:
        devices_context = devices_context.get_sorted()
        comm = mpi_comm.ncclGroupInit(devices_context)
    return comm


def get_mpi_communicate() -> MPI_Communicator:
    global mpi_comm
    return mpi_comm


def get_nccl_communicate() -> NCCL_Communicator:
    global nccl_comm
    return nccl_comm


def get_worker_communicate() -> CDLL:
    global ps_comm
    return ps_comm


def worker_init() -> None:
    global ps_comm
    ll = ctypes.cdll.LoadLibrary
    ps_comm = ll(path_to_lib("libps.so"))
    ps_comm.Init()


def worker_finish() -> None:
    ps_comm.Finalize()


def server_init() -> None:
    global ps_comm
    ll = ctypes.cdll.LoadLibrary
    ps_comm = ll(path_to_lib("libps.so"))
    ps_comm.Init()
    ps_comm.StartServer()


def server_finish() -> None:
    ps_comm.Finalize()


def scheduler_init() -> None:
    global ps_comm
    ll = ctypes.cdll.LoadLibrary
    ps_comm = ll(path_to_lib("libps.so"))
    ps_comm.Init()


def scheduler_finish() -> None:
    ps_comm.Finalize()


class HetuConfig(object):
    __slots__ = [
        'eval_node_list',
        'train_name',
        'val_name',
        'context',
        'seed',
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
        'dataloader_states',
        'use_sparse_pull',
        'cstable_policy',
        'inference',
        'enable_lazy',
        'bsp',
        'prefetch',
        'cache_bound',
        'log_path',
        'logger',
        'my_eval_nodes',
        'placeholder_to_arr_map',
        'pipeline',
        'all_forward_nodes',
        'pp_rank',
        'pp_nrank',
        'dp_rank',
        'dp_nrank',
        'min_dp_nrank',
        'lcm_dp_nrank',
        'use_preduce',
        'layer_indices',
        'dist_strategy',
        'graph_status',
        'memory_pool',
        'overlap',
        'use_nccl_collectives',
    ]

    def __init__(
        self,
        eval_node_list: OP_LIST,
        train_name: str,
        val_name: str,
        ctx: Union[None, DLContext, DeviceGroup] = None,
        seed: Optional[int] = None,
        comm_mode: Optional[str] = None,
        use_sparse_pull: bool = True,
        cstable_policy: Optional[str] = None,
        bsp: int = -1,
        prefetch: bool = True,
        enable_lazy: bool = False,
        cache_bound: int = 100,
        log_path: Optional[str] = None,
        logger: Optional[str] = None,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        pipeline: Optional[str] = None,
        dist_strategy: Optional[Strategy] = None,
        use_preduce: bool = False,
        overlap: bool = True,
        use_nccl_collectives: bool = True,
    ):
        '''
        context: default device context
        comm_mode: communication mode, should be one of the following
            None       -> Single GPU
            PS         -> Parameter Server
            AllRedeuce -> MPI AllReduce
            Hybrid     -> Parameter Server for Sparse Parameter and MPI AllReduce for Dense Parameter
        '''
        assert pipeline in (None, "gpipe", "pipedream", "hetpipe")
        self.pipeline = pipeline
        self.use_preduce = use_preduce
        self.overlap = overlap
        self.use_nccl_collectives = use_nccl_collectives

        self.eval_node_list = eval_node_list
        self.train_name = train_name
        self.val_name = val_name

        # check context
        self.dist_strategy = dist_strategy
        self.graph_status = GraphStatus(eval_node_list)
        if self.pipeline is not None:
            self.graph_status.extend_oplayers()
            self.all_forward_nodes = set(find_topo_sort(
                self.graph_status.forward_node_list))
            self.graph_status.shrink_oplayers()
        self.memory_pool = HetuMemoryPool()

        if self.dist_strategy is None:
            if ctx is None:
                ctx = get_current_context()
            context_launch = isinstance(ctx, DeviceGroup)
            if context_launch:
                self.graph_status.parse_graph_with_dispatch()
        else:
            self.dist_strategy.use_nccl_collectives = self.use_nccl_collectives
            self.dist_strategy.set_overlap(self.overlap)
            if self.dist_strategy.use_dispatch:
                ctx = self.dist_strategy.set_raw_ctxs(self.graph_status)
                self.graph_status.parse_graph_with_dispatch()
            else:
                ctx = self.dist_strategy.set_raw_ctxs_n_states(
                    self.graph_status, self.memory_pool)
            context_launch = isinstance(ctx, DeviceGroup)
        assert ctx, 'Default context should be determined.'
        self.graph_status.extend_oplayers()

        self.comm_mode = comm_mode
        self.node_strategy = {}
        local_gpu_devices = None
        self.context_launch = context_launch

        if context_launch:
            # with context usage
            launchMPI, launchPS, self.node_strategy, devices, min_worker_num, lcm_worker_num = get_launch_config_by_traverse_nodes(
                eval_node_list, ctx, pipeline)
            local_gpu_devices = sorted(
                [dev.device_id for dev in devices if dev.local and ndarray.is_gpu_ctx(dev)])
            if not launchMPI and not launchPS:
                self.comm_mode = None
            elif launchMPI and (not launchPS or self.use_preduce):
                self.comm_mode = 'AllReduce'
            elif not launchMPI and launchPS:
                self.comm_mode = 'PS'
            else:
                self.comm_mode = 'Hybrid'
            # in pipeline or model parallel we have to initialize another p2p stream
            init_p2p_stream = len(devices) != ctx.worker_num

        # variables initialization
        seed = seed if seed is not None else np.int64(time())
        from ..random import set_random_seed, get_seed
        set_random_seed(seed)

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
            topo_sort_register_ps(
                eval_node_list, self.ps_comm, self.comm_mode, get_seed(), cstable_policy)
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
        # with open('topo{}.txt'.format(self.rank), 'w') as fw:
        #     for node in find_topo_sort(self.eval_node_list):
        #         print(node, node.inputs, file=fw, flush=True)

        self.my_eval_nodes = eval_node_list
        self.p2p_stream = None
        self.layer_indices = None
        if context_launch:
            # comm_mode is None <=> only 1 model parallel instance
            self.context = ndarray.gpu(device_id)
            if self.pipeline is not None:
                from ..distributed_strategies import PipeOptSearching
                if isinstance(dist_strategy, PipeOptSearching):
                    self.pp_rank = self.rank * dist_strategy.num_parts // self.nrank
                    self.pp_nrank = dist_strategy.num_parts
                    self.dp_rank = 0
                    self.dp_nrank = 0
                else:
                    self.pp_rank, self.pp_nrank, self.dp_rank, self.dp_nrank = get_pipeline_stage_info(
                        eval_node_list, self.context)
                self.min_dp_nrank = min_worker_num
                self.lcm_dp_nrank = lcm_worker_num
            self.p2p_stream = create_stream_handle(
                self.context) if init_p2p_stream else None
            self.my_eval_nodes, self.param_allreduce_group, self.layer_indices = self.graph_status.assign_context_by_traverse_nodes(
                self.context, self.nccl_comm, self.use_nccl_collectives)
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
        # use another stream for data parallel
        if self.comm_mode == "Hybrid" or self.comm_mode == "AllReduce":
            if self.overlap and on_gpu:
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

        self.h2d_ops: Dict[Op, Union[DataH2DOp, DataH2DSparseOp]] = {}
        self.d2h_ops: Dict[Op, Union[DataD2HOp, DataD2HSparseOp]] = {}
        self.ps_map: Dict[Op, NDArray] = {}
        self.infer_ps_map: Dict[Op, NDArray] = {}
        self.enable_lazy = enable_lazy
        self.bsp = bsp
        self.cache_bound = int(cache_bound)

        self.log_path = log_path
        if log_path is not None and (self.comm_mode == 'PS' or self.comm_mode == "Hybrid"):
            assert os.path.isdir(
                log_path), 'Need to specify a work directory to save logs.'
            self.ps_comm.startRecord(ctypes.c_char_p(bytes(log_path, 'utf-8')))
        if logger is not None:
            assert project is not None and run_name is not None
            if logger == 'wandb':
                from ..logger import WandbLogger
                self.logger = WandbLogger(
                    project, run_name, run_id, self.rank, self.nrank, self.context, self.nccl_comm, self.nccl_stream)
            elif logger == 'hetu':
                from ..logger import HetuLogger
                self.logger = HetuLogger(
                    self.rank, self.nrank, self.context, self.nccl_comm, self.nccl_stream)
            else:
                raise ValueError
        else:
            self.logger = None

        self.placeholder_to_arr_map = dict()
        topo_sort_with_hook(self.my_eval_nodes, self)


def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


class Executor(object):
    """Executor computes values for given set of nodes in computation graph."""

    def __init__(
        self,
        eval_node_dict: Union[Dict[str, OP_LIST], OP_LIST],
        config: HetuConfig = None,
        timing: Optional[str] = None,
        **kargs
    ) -> None:
        """
        Parameters
        ----------
        eval_node_dict: dict of list of nodes whose values need to be computed.
        """
        if not isinstance(eval_node_dict, dict):
            eval_node_dict = {'default': eval_node_dict}
        eval_node_dict = {k: list(flatten(v))
                          for k, v in eval_node_dict.items()}
        train_name, val_name = None, None
        for k, v in eval_node_dict.items():
            if any([isinstance(node, OptimizerOp) for node in v]):
                # get the last subexecutor containing optimizer as train for ps op
                train_name = k
            else:
                # get the last subexecutor not containing optimizer as val for ps op
                val_name = k
        all_eval_nodes = []
        for n in reduce(add, eval_node_dict.values()):
            if n not in all_eval_nodes:
                all_eval_nodes.append(n)
        if config is None:
            config = HetuConfig(eval_node_list=all_eval_nodes,
                                train_name=train_name, val_name=val_name, **kargs)
        assert isinstance(
            config, HetuConfig), 'Config type %s invalid.' % str(type(config))

        self.eval_node_dict: Dict[str: OP_LIST] = eval_node_dict
        self.config = config
        self.logger = self.config.logger

        def get_sub_executor(k: str) -> Type[SubExecutor]:
            if timing:
                from .timer_subexecutor import make_texecutor
                return make_texecutor(timing)
            if config.pipeline == "gpipe" and k in ("train", "default"):
                from .gpipe_subexecutor import SubExecutor4Gpipe
                return SubExecutor4Gpipe
            elif config.pipeline in ("pipedream", "hetpipe") and k in ("train", "default"):
                from .pipedream_subexecutor import SubExecutor4Pipedream
                return SubExecutor4Pipedream
            return SubExecutor

        self.subexecutor = {k: get_sub_executor(k)(
            k, v, config) for k, v in eval_node_dict.items()}

        self.topo_order = find_topo_sort(config.my_eval_nodes)
        # with open('topo{}.txt'.format(self.config.rank), 'w') as fw:
        #     for node in self.topo_order:
        #         if isinstance(node, (PipelineSendOp, PipelineReceiveOp)):
        #             print(node, node.inputs, node.const_attr,
        #                   file=fw, flush=True)
        #         else:
        #             print(node, node.inputs, file=fw, flush=True)
        self.param_nodes = [node for node in self.topo_order if isinstance(
            node, PlaceholderOp) and node.trainable]
        self.comm_mode = self.config.comm_mode
        self.ps_comm = self.config.ps_comm
        self.local_rank = self.config.local_rank
        self.rank = self.config.rank

    def profile(
        self,
        feed_shapes: SHAPE_MAP,
        log_file: str,
        profiler: str = 'cpu',
        name: str = 'default'
    ) -> None:
        self.subexecutor[name].profile(
            feed_shapes, log_file, profiler=profiler)

    def set_config(self, attrs):
        self.logger.set_config(attrs)

    def log(self, name, value):
        self.logger.wrapped_log(name, value)

    def multi_log(self, results):
        for k, v in results.items():
            self.log(k, v)

    def step_logger(self):
        self.logger.step()

    def run(
        self,
        name: str = 'default',
        eval_node_list: OP_LIST = [],
        feed_dict: Dict[Op, FEEDINS] = {},
        convert_to_numpy_ret_vals: bool = False,
        inference: Optional[bool] = None,
        **kwargs
    ) -> List[Union[None, np.ndarray, NDArray]]:
        return self.subexecutor[name].run(eval_node_list, feed_dict, convert_to_numpy_ret_vals, inference=inference, **kwargs)

    @property
    def ctx(self) -> DLContext:
        return self.config.context

    @property
    def batch_num(self) -> int:
        assert len(
            self.subexecutor) == 1, 'Batch num should be used with only 1 subexecutor.'
        return list(self.subexecutor.values())[0].batch_num

    def get_batch_num(self, name: str = 'default') -> int:
        return self.subexecutor[name].batch_num

    def sync_all_streams(self):
        if self.config.comp_stream is not None:
            self.config.comp_stream.sync()
        if self.config.h2d_stream is not None:
            self.config.h2d_stream.sync()
        if self.config.d2h_stream is not None:
            self.config.d2h_stream.sync()
        if self.config.nccl_stream is not None:
            self.config.nccl_stream.sync()

    def state_dict(self):
        state_dict = {}
        for node, value in self.config.placeholder_to_arr_map.items():
            if value is not None:
                state_dict[node.name] = value.asnumpy()
            else:
                # feed node
                assert node.shape is None
        return state_dict

    def save(self, file_path: str, file_name: str, others: Optional[dict] = None) -> None:
        if others is None:
            others = {}
        else:
            assert 'state_dict' not in others

        self.sync_all_streams()
        assert os.path.isdir(
            file_path), 'Need to specify a work directory to save parameters.'
        assert others is None or 'state_dict' not in others
        if self.comm_mode in (None, 'AllReduce'):
            # when using allreduce, users need to specify the worker whose rank equals 0 to save
            state_dict = self.state_dict()
        else:
            state_dict = {}
            self.ps_comm.BarrierWorker()
            if self.config.rank == 0:
                for node, value in self.config.placeholder_to_arr_map.items():
                    if node.is_embed or self.comm_mode == 'PS':
                        node.event.sync()
                        nodeid = ctypes.c_int(node.id)
                        self.ps_comm.SaveParam(
                            nodeid, ctypes.c_char_p(bytes(file_path, 'utf-8')))
                        self.ps_comm.Wait(nodeid)
                    else:
                        state_dict[node.name] = value.asnumpy()
            self.ps_comm.BarrierWorker()

        others['state_dict'] = state_dict
        from ..random import get_seed_status
        others['seed'] = get_seed_status()
        with open(os.path.join(file_path, file_name), "wb") as writer:
            pickle.dump(others, writer, protocol=4)

    def load(self, file_path: str, file_name: str, consider_splits: bool = False) -> None:
        assert os.path.isdir(
            file_path), 'Need to specify a work directory to load parameters.'

        with open(os.path.join(file_path, file_name), 'rb') as reader:
            state_dict = pickle.load(reader)
        variables = state_dict['state_dict']
        seeds = state_dict['seed']
        self.load_seeds(seeds)
        self.load_dict(variables, file_path, consider_splits)

    def load_seeds(self, seeds):
        from ..random import set_random_seed, reset_seed_seqnum, step_seqnum
        set_random_seed(seeds[0])
        reset_seed_seqnum()
        step_seqnum(seeds[1])

    def load_dict(
        self,
        state_dict: Dict[Op, np.ndarray],
        file_path: str = None,
        consider_splits: bool = False
    ) -> None:
        self.sync_all_streams()
        if self.comm_mode in (None, 'AllReduce'):
            for node in self.config.placeholder_to_arr_map:
                if node.name in state_dict:
                    value = state_dict[node.name]
                    if consider_splits and node.reshaped:
                        value = node.reshape_tensor(value)
                    pre_shape = self.config.placeholder_to_arr_map[node].shape
                    cur_shape = value.shape
                    assert pre_shape == cur_shape, 'Shape not conform! Got {} and {} for {}.'.format(
                        pre_shape, cur_shape, node.name)
                    self.config.placeholder_to_arr_map[node][:] = value
                # else:
                #     print(list(state_dict.keys()))
                #     print(node.name)
                #     assert False
        else:
            assert file_path is not None
            self.ps_comm.BarrierWorker()
            if self.config.rank == 0:
                for node in self.config.placeholder_to_arr_map:
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
                        if node.name in state_dict:
                            self.config.placeholder_to_arr_map[node][:] = state_dict[node.name]
                elif isinstance(node, EmbeddingLookUp) and self.config.prefetch:
                    node.event.sync()
                    nodeid = ctypes.c_int(node.inputs[0].id)
                    self.ps_comm.SparsePull(nodeid, node.inputs[1].get_next_arr(
                        self.name).handle, self.config.ps_map[node.inputs[0]].handle)
                    node.event.update()
            self.ps_comm.BarrierWorker()

    def set_dataloader_batch_index(self, name, index):
        self.subexecutor[name].set_dataloader_batch_index(index)

    def recordLoads(self) -> None:
        for node in self.config.ps_map:
            node.event.sync()
        self.ps_comm.getLoads()

    def reduceMean(self, arr: np.ndarray, root: int = 0) -> np.ndarray:
        # only used for loss and accuracy, etc.
        # the communicator is formed by the context of loss
        if 'loss' in self.config.param_allreduce_group:
            comm = self.config.param_allreduce_group['loss']
            if comm.local_rank == -1:
                # in case local context is not in communicator
                return arr
            local_array = ndarray.array(arr, ndarray.cpu())
            comm.dlarrayNcclReduce(local_array, local_array, root)
            comm.stream.sync()
            return local_array.asnumpy() / comm.nrank
        else:
            return arr

    def gatherPredict(self, arr: np.ndarray) -> np.ndarray:
        # only used for predicted y.
        # the communicator is formed by the context of loss
        if 'loss' in self.config.param_allreduce_group:
            comm = self.config.param_allreduce_group['loss']
            if comm.local_rank == -1:
                # in case local context is not in communicator
                return arr
            new_shape = list(arr.shape)
            new_shape[0] *= comm.nrank
            local_array = ndarray.array(arr, ndarray.cpu())
            global_array = ndarray.empty(tuple(new_shape), ndarray.cpu())
            comm.dlarrayAllGather(
                local_array, global_array, ncclDataType_t.ncclFloat32)
            comm.stream.sync()
            return global_array.asnumpy()
        else:
            return arr

    def logOut(self, *args, name:  str = 'default', **kargs) -> None:
        self.subexecutor[name].logOut(*args, **kargs)

    def clearTimer(self, name: str = 'default') -> None:
        self.subexecutor[name].clearTimer()

    def return_tensor_values(self) -> None:
        for k, v in self.config.placeholder_to_arr_map.items():
            k.tensor_value = v

    def __del__(self) -> None:
        self.sync_all_streams()
        for node in self.param_nodes:
            if node.event:
                node.event.sync()
        if self.comm_mode in ('PS', 'Hybrid'):
            worker_finish()


class SubExecutor(object):
    def __init__(self, name: str, eval_node_list: OP_LIST, config: HetuConfig) -> None:
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

        if config.pipeline and self.inference:
            self.eval_node_list = []
            # Remove the last pipeline send on worker 1...n-1 ( not needed in inference stage) and the optimizer
            remove_send = 1 if config.pp_rank > 0 else 0
            for node in config.my_eval_nodes[::-1]:
                if remove_send and isinstance(node, PipelineSendOp):
                    remove_send = 0
                elif not isinstance(node, OptimizerOp):
                    self.eval_node_list.append(node)
            self.global_eval_nodes = eval_node_list
        elif config.p2p_stream:
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

        self.topo_order = reorder_for_group(
            self.topo_order, self.config.layer_indices)
        # self.logNodes(self.topo_order, 'real_graph{}.txt')

        # main structures, nodes' shapes and arrays
        self.node_to_shape_map: SHAPE_MAP = {}
        self.node_to_arr_map: ARR_MAP = {}

        # inherit from configurations
        self.comm_mode = self.config.comm_mode
        self.ps_comm = self.config.ps_comm
        self.nccl_comm = self.config.nccl_comm
        self.comp_stream = self.config.comp_stream
        self.h2d_stream = self.config.h2d_stream
        self.d2h_stream = self.config.d2h_stream
        self.p2p_stream = self.config.p2p_stream
        self.nccl_stream = self.config.nccl_stream
        self.param_psval_map = self.config.infer_ps_map if self.inference else self.config.ps_map
        self.use_sparse_pull = self.config.use_sparse_pull
        self.cstable_policy = self.config.cstable_policy
        self.use_p2p = self.config.p2p_stream is not None

        # assisting structures, improve performance
        self.need_feed_nodes: OP_LIST = []
        self.param_nodes: OP_LIST = []
        self.dataloader_nodes: OP_LIST = []
        self.computing_nodes: OP_LIST = []

        ln_bn_grad_nodes = (Batch_Normalization_Gradient_of_DataOp, Batch_Normalization_Gradient_of_ScaleOp, Batch_Normalization_Gradient_of_BiasOp,
                            Layer_Normalization_Gradient_of_DataOp, Layer_Normalization_Gradient_of_ScaleOp, Layer_Normalization_Gradient_of_BiasOp,
                            UniqueIndicesOffsetsOp,)
        no_compute_nodes = ln_bn_grad_nodes + (StopGradientOp,)

        for node in self.topo_order:
            if isinstance(node, (DataloaderOp, GNNDataLoaderOp)):
                self.dataloader_nodes.append(node)
            elif isinstance(node, PlaceholderOp):
                if node.shape is None:
                    self.need_feed_nodes.append(node)
                elif node.trainable:
                    self.param_nodes.append(node)
            elif not ((self.use_sparse_pull or self.cstable_policy) and isinstance(node, EmbeddingLookUp) and self.config.prefetch) and not isinstance(node, no_compute_nodes):
                self.computing_nodes.append(node)
        self._batch_num = None
        self.init_need_allocation = (self.need_feed_nodes == []) and (
            self.dataloader_nodes == [])

        self.node_type_to_stream_map = {
            ParameterServerCommunicateOp: self.d2h_stream,
            ParameterServerSparsePullOp: self.d2h_stream,
            DataD2HOp: self.d2h_stream,
            DataD2HSparseOp: self.d2h_stream,
            DataH2DSparseOp: self.h2d_stream,
            DataH2DOp: self.h2d_stream,
            AllGatherCommunicateOp: self.p2p_stream,
            ReduceScatterCommunicateOp: self.p2p_stream,
            ReduceCommunicateOp: self.p2p_stream,
            BroadcastCommunicateOp: self.p2p_stream,
            AllReduceCommunicateP2POp: self.p2p_stream,
            AllToAllOp: self.nccl_stream,
        }
        if self.config.overlap:
            self.node_type_to_stream_map[AllReduceCommunicateOp] = self.nccl_stream

    @property
    def batch_num(self) -> int:
        if self._batch_num is None:
            batch_num = set([node.get_batch_num(self.name)
                             for node in self.dataloader_nodes])
            assert len(batch_num) <= 1, 'Batch num not conform.'
            batch_num = None if len(
                batch_num) == 0 else batch_num.pop()
            self._batch_num = batch_num
        return self._batch_num

    def set_dataloader_batch_index(self, index):
        for op in self.dataloader_nodes:
            op.set_batch_index(self.name, index)

    def profile(
        self,
        feed_shapes: SHAPE_MAP,
        log_file: str,
        profiler: str = 'cpu'
    ) -> None:
        # !!! we should profile before using distributed settings
        # !!! so here we don't consider multiple devices
        # !!! no self.use_p2p, no node reshape, no pipeline configuration
        # !!! not support sparse input now
        # !!! not support dynamic memory now
        assert profiler in ('cpu', 'gpu')
        assert len(feed_shapes) == len(self.need_feed_nodes) + \
            len(self.dataloader_nodes)

        need_reallocation = self.init_need_allocation

        for node, shape in feed_shapes.items():
            assert node in self.need_feed_nodes or node in self.dataloader_nodes
            local_realloc = shape != self.node_to_shape_map.get(node, None)
            need_reallocation = need_reallocation or local_realloc
            if local_realloc:
                self.node_to_arr_map[node] = ndarray.empty(
                    shape, ctx=node.ctx)

        if need_reallocation:
            self.init_need_allocation = False
            self.infer_shape(feed_shapes)
            self.memory_plan()

        from ..profiler import HetuProfiler
        if not hasattr(self, 'profiler'):
            self.profiler = HetuProfiler(
                self.computing_nodes, feed_shapes, self.node_to_arr_map, ctx=self.config.context)
        self.profiler.profile_n_log(log_file, profiler=profiler)

    def update_executor(self, eval_node_list: OP_LIST) -> None:
        # !!!! Not in use
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
        self._batch_num = None
        self.init_need_allocation = (self.need_feed_nodes == []) and (
            self.dataloader_nodes == [])

    def infer_shape(self, feed_shapes: SHAPE_MAP) -> None:
        """Given shapes of feed_dict nodes, infer shape for all nodes in graph.

        Implementation note:
        Iteratively calls node.infer_shape to infer shapes.
        Node shapes stored in self.node_to_shape_map.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """

        def make_group() -> None:
            def encode_shape(shape):
                shape = list(shape)
                if len(shape) < 4:
                    shape = [0] * (4 - len(shape)) + shape
                return shape

            def decode_shape(shape):
                assert len(shape) == 4
                return tuple(int(x) for x in list(shape) if x != 0)

            temp_res = {}
            for node in grouping_nodes:
                if isinstance(node, PipelineReceiveOp):
                    size = 12 if node.use_indexed_slices else 4
                    temp_res[node] = [ndarray.empty(
                        (size,), ctx=node.ctx) for _ in node.const_attr]
            nccl_comm = self.config.nccl_comm
            p2p_stream = self.config.p2p_stream
            dtype = ncclDataType_t.ncclFloat32
            GroupStart()
            for node in grouping_nodes:
                if isinstance(node, PipelineSendOp):
                    shape = encode_shape(
                        self.node_to_shape_map[node.inputs[0]])
                    if node.use_indexed_slices:
                        ind_shape, val_shape = self.indexed_slices_shape[node]
                        ind_shape = encode_shape(ind_shape)
                        val_shape = encode_shape(val_shape)
                        shape = shape + ind_shape + val_shape
                    # construct and send
                    new_res = ndarray.array(np.array(shape), ctx=node.ctx)
                    for dest in node.const_attr:
                        nccl_comm.dlarraySend(new_res,
                                              dtype,
                                              dest,
                                              p2p_stream)
                else:
                    for res, src in zip(temp_res[node], node.const_attr):
                        nccl_comm.dlarrayRecv(res,
                                              dtype,
                                              src,
                                              p2p_stream)
            GroupEnd()
            p2p_stream.sync()
            for node in grouping_nodes:
                if isinstance(node, PipelineSendOp):
                    self.node_to_shape_map[node] = None
                else:
                    shape_arr = None
                    for arr in temp_res[node]:
                        cur_res = arr.asnumpy()
                        assert shape_arr is None or (
                            shape_arr == cur_res).all()
                        shape_arr = cur_res
                    self.node_to_shape_map[node] = decode_shape(shape_arr[:4])
                    if node.use_indexed_slices:
                        self.indexed_slices_shape[node] = (decode_shape(
                            shape_arr[4:8]), decode_shape(shape_arr[8:]))
            grouping_nodes.clear()

        self.node_to_shape_map = {}
        self.indexed_slices_shape = {}
        grouping_nodes = []
        cur_ind = -1
        for node in self.topo_order:
            if isinstance(node, EmbeddingLookUp_Gradient):
                if len(node.inputs) == 2:
                    self.indexed_slices_shape[node] = (
                        self.node_to_shape_map[node.inputs[1]], self.node_to_shape_map[node.inputs[0]])
                else:
                    self.indexed_slices_shape[node] = (
                        node.index.shape, self.node_to_shape_map[node.inputs[0]])
            elif isinstance(node, (DataD2HSparseOp, PipelineSendOp)) and node.use_indexed_slices:
                self.indexed_slices_shape[node] = self.indexed_slices_shape[node.inputs[0]]
            elif isinstance(node, AllReduceCommunicateOp) and node.use_indexed_slices:
                ind_shape, val_shape = self.indexed_slices_shape[node.inputs[0]]
                ind_shape = list(ind_shape)
                val_shape = list(val_shape)
                ind_shape[0] *= node.comm.nrank
                val_shape[0] *= node.comm.nrank
                self.indexed_slices_shape[node] = (
                    tuple(ind_shape), tuple(val_shape))
            if node in feed_shapes:
                self.node_to_shape_map[node] = tuple(feed_shapes[node])
            else:
                if isinstance(node, (PipelineSendOp, PipelineReceiveOp)):
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
        # with open('shape{}.txt'.format(self.config.rank), 'w') as fw:
        #     for node, shape in self.node_to_shape_map.items():
        #         print(node, node.inputs, shape, file=fw, flush=True)

    def memory_plan(self) -> None:
        self.config.memory_pool.memory_plan(
            self.computing_nodes, self.node_to_shape_map, self.node_to_arr_map, self.config, self.eval_node_list, self.indexed_slices_shape)

    def get_feed_value(self, arr_map: ARR_MAP, node: Op, value: FEEDINS) -> Tuple[Tuple[int, ...], bool]:
        if node.reshaped:
            value = node.reshape_tensor(value)
        local_shape = tuple(value.shape)
        local_realloc = local_shape != self.node_to_shape_map.get(
            node, None)
        value_dtype = ndarray.convert_dtype(value.dtype)
        if node.dtype != value_dtype:
            message = 'Node dtype (original {}) will be set by value dtype {}.'.format(
                node.dtype.__name__.upper(), value_dtype.__name__.upper())
            warnings.warn(message)
            node.dtype = value_dtype
        if node.on_cpu:
            assert isinstance(value, (np.ndarray, spmatrix, ndarray.NDArray)), \
                "feed_dict value type not supported"
            if isinstance(value, np.ndarray):
                if local_realloc:
                    arr_map[node] = ndarray.empty(
                        local_shape, ctx=node.ctx, dtype=node.dtype)
                arr_map[node][:] = value
            else:
                arr_map[node] = value
        else:
            if isinstance(value, np.ndarray):
                if local_realloc:
                    arr_map[node] = ndarray.array(
                        value, ctx=node.ctx, dtype=node.dtype)
                else:
                    arr_map[node][:] = value
            elif isinstance(value, spmatrix):
                value = coo_matrix(value)
                value = ndarray.sparse_array(value.data,
                                             (value.row, value.col), shape=local_shape, ctx=node.ctx)
                arr_map[node] = value
            elif isinstance(value, ndarray.NDArray):
                if value.ctx == node.ctx:
                    arr_map[node] = value
                else:
                    if local_realloc:
                        arr_map[node] = ndarray.empty(
                            local_shape, ctx=node.ctx)
                    arr_map[node][:] = value
            elif isinstance(value, ndarray.ND_Sparse_Array):
                arr_map[node] = value
            else:
                assert False, "feed_dict value type not supported"
        return local_shape, local_realloc

    def run(
        self,
        eval_node_list: OP_LIST = [],
        feed_dict: Dict[Op, FEEDINS] = {},
        convert_to_numpy_ret_vals: bool = False,
        dataloader_step=True,
        inference=None,
    ) -> List[Union[None, np.ndarray, NDArray]]:
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
        if eval_node_list != [] and eval_node_list != self.eval_node_list:
            self.update_executor(eval_node_list)

        feed_shapes = {}
        need_reallocation = self.init_need_allocation

        # get feed in values
        for node, value in feed_dict.items():
            if self.use_p2p and node not in self.need_feed_nodes:
                continue
            assert node in self.need_feed_nodes, 'Only allow feed in PlaceholderOp with no values, here got %s:%s.' % (
                str(type(node)), node.name)
            local_shape, local_realloc = self.get_feed_value(
                self.node_to_arr_map, node, value)
            need_reallocation = need_reallocation or local_realloc
            feed_shapes[node] = local_shape

        # get dataloader values
        for node in self.dataloader_nodes:
            if dataloader_step:
                cur_value = node.get_arr(self.name)
            else:
                cur_value = node.get_next_arr(self.name)
            self.node_to_arr_map[node] = cur_value
            local_shape = cur_value.shape
            feed_shapes[node] = local_shape
            local_realloc = local_shape != self.node_to_shape_map.get(
                node, None)
            need_reallocation = need_reallocation or local_realloc

        # in pipedream, we should retrieve the latest model parameter.
        if self.config.pipeline == "pipedream":
            self.node_to_arr_map.update(self.config.placeholder_to_arr_map)

        # reallocation, infer shapes and allocate memory
        if need_reallocation:
            self.init_need_allocation = False
            self.infer_shape(feed_shapes)
            self.memory_plan()

        self.compute(self.computing_nodes,
                     self.node_to_arr_map, inference=inference)

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
        if self.config.pipeline:
            results = filter(lambda x: x[0] in self.global_eval_nodes,
                             zip(self.eval_node_list, results))
            results = [x[1] for x in results]
        elif self.use_p2p:
            new_results = [None for _ in self.global_eval_nodes]
            for i, j in enumerate(self.run_results_indices):
                new_results[j] = results[i]
            results = new_results

        return results

    def compute(self, computing_nodes: OP_LIST, arr_map: ARR_MAP, inference: Optional[bool] = None) -> None:
        # computing
        grouping_nodes = []
        cur_ind = -1

        def make_group() -> None:
            p2p_stream = self.config.p2p_stream
            GroupStart()
            for node in grouping_nodes:
                node.compute([arr_map[n] for n in node.inputs],
                             arr_map[node], p2p_stream)
            GroupEnd()
            for node in grouping_nodes:
                node.event.record(p2p_stream)
            grouping_nodes.clear()
        if inference is None:
            inference = self.inference
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
                grouping_nodes.append(node)
                continue
            else:
                if len(grouping_nodes) > 0:
                    make_group()

                for n in node.inputs:
                    if n.event:
                        n.event.sync()
                input_vals = [arr_map[n] for n in node.inputs]
                node_val = arr_map[node]

                node_type = type(node)
                cur_stream = self.node_type_to_stream_map.get(
                    node_type, self.comp_stream)

                if node_type in (DropoutOp, Batch_NormalizationOp, MulByConstOp):
                    node.compute(input_vals, node_val, cur_stream,
                                 inference=inference)
                else:
                    node.compute(input_vals, node_val, cur_stream)

                if isinstance(node.event, Event):
                    # for d2h op / eval nodes / nodes before [allreduce or ps nodes or pipelinesend nodes]
                    # for allreduce op
                    node.event.record(cur_stream)
                # if cur_stream is not None:
                #     cur_stream.sync()
                #     if isinstance(node_val, ndarray.NDArray) and np.any(np.isnan(node_val.asnumpy())):
                #         print('appear nan!!!', node, node.inputs)
                #         np.save('problem.npy', input_vals[0].asnumpy())
                #         exit()

        if len(grouping_nodes) > 0:
            make_group()

    def logNodes(self, node_list: OP_LIST, log_path: str) -> None:
        if self.config.rank is not None:
            log_path = log_path.format(self.config.rank)
        with open(log_path, 'w') as fw:
            for node in node_list:
                print(node, node.inputs, file=fw, flush=True)


def gradients(
    output_node: Op,
    node_list: OP_LIST,
    insert_grad: Optional[Op] = None,
    return_all: bool = False
):
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
    from .ReduceMean import ReduceMeanOp
    from .Division import DivConstOp, DivOp
    from .AddElewise import AddOp
    from .AddConst import AddByConstOp
    from .MinusElewise import MinusOp
    from .Sum import SumOp
    add_ops = (AddOp, AddByConstOp, SumOp)
    # TODO: add support for Csrmm, MatrixDot, Sigmoid, Sqrt, Where
    # key: backward node; value: corresponding forward node (that generates it) and index of its inputs
    backward2forward: Dict[Op, Tuple[Op, int]] = {}
    # key: forward node; value: list of generated backward nodes
    forward2backward: Dict[Optional[Op], OP_LIST] = {}
    if not isinstance(output_node, list):
        output_node = [output_node]
    if insert_grad is None:
        insert_grad = [OnesLike.oneslike_op(
            outnode) for outnode in output_node]
    elif not isinstance(insert_grad, list):
        insert_grad = [insert_grad]
    forward2backward[None] = insert_grad
    for output, grad in zip(output_node, insert_grad):
        backward2forward[grad] = (output, -1)
    node_to_output_grads_list = {node: [grad] if grad is not None else []
                                 for node, grad in zip(output_node, insert_grad)}
    node_to_output_grad = {}
    # Traverse forward graph in reverse topological order
    reverse_topo_order: OP_LIST = reversed(find_topo_sort(output_node))
    for node in reverse_topo_order:
        # here the ctx for embedding lookup is a workaround
        # TODO: when implement PS strategy for context semantics, modify here
        if isinstance(node, EmbeddingLookUp):
            if len(node_to_output_grads_list[node]) == 0:
                output_grad, is_new = None, False
            else:
                output_grad, is_new = sum_node_list(
                    node_to_output_grads_list[node], node_to_output_grads_list[node][0].raw_ctx)
        else:
            if isinstance(node, PlaceholderOp):
                shape = node.shape
            else:
                shape = None
            output_grad, is_new = sum_node_list(
                node_to_output_grads_list[node], node.raw_ctx, shape)
        if output_grad is None:
            for n in node.inputs:
                if n not in node_to_output_grads_list:
                    node_to_output_grads_list[n] = []
            continue
        node_to_output_grad[node] = output_grad
        input_grads_list = node.gradient(output_grad)
        if input_grads_list is not None:
            # TODO: not consider following nodes in forward2backward, can be improved
            # DistGCN_15d, MatrixDot, Sigmoid, Sqrt, Where
            value = (node, -1)
            if isinstance(node, add_ops):
                # these nodes don't generate new nodes
                forward2backward[node] = []
            elif isinstance(node, MinusOp):
                forward2backward[node] = [input_grads_list[1]]
            else:
                forward2backward[node] = [
                    n for n in input_grads_list if n is not None]
            if isinstance(node, (ReduceMeanOp, Batch_NormalizationOp, Layer_NormalizationOp)):
                forward2backward[node].append(input_grads_list[0].inputs[0])
                if isinstance(node, ReduceMeanOp):
                    backward2forward[input_grads_list[0].inputs[0]] = (
                        node, 0)
                else:
                    backward2forward[input_grads_list[0].inputs[0]] = value
            elif isinstance(node, DivConstOp):
                temp = input_grads_list[0].inputs[0]
                forward2backward[node].extend([temp, temp.inputs[0]])
                backward2forward[temp] = value
                backward2forward[temp.inputs[0]] = value
            elif isinstance(node, DivOp):
                temp = input_grads_list[1].inputs[0]
                forward2backward[node].extend(
                    [input_grads_list[0].inputs[0], temp, temp.inputs[0]])
                backward2forward[input_grads_list[0].inputs[0]] = value
                backward2forward[temp] = value
                backward2forward[temp.inputs[0]] = value
        if is_new:
            # sum op
            assert output_grad not in backward2forward
            backward2forward[output_grad] = (node, -1)
            if node not in forward2backward:
                assert isinstance(
                    node, PlaceholderOp), 'In this case the input grads is None.'
                forward2backward[node] = [output_grad]
            else:
                forward2backward[node].append(output_grad)
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            # Calculate partial adjoint for input nodes.
            if input_grads_list[i] is not None:
                node_to_output_grads_list[node.inputs[i]].append(
                    input_grads_list[i])
            cur_key = input_grads_list[i]
            if cur_key is not None and not isinstance(node, add_ops) and not (isinstance(node, (MinusOp)) and i == 0):
                assert cur_key not in backward2forward
                backward2forward[cur_key] = (node, i)

    grad_node_list = [node_to_output_grad[node] for node in node_list]
    if return_all:
        return grad_node_list, backward2forward, forward2backward
    else:
        return grad_node_list

##################
# Helper Methods #
##################


# this function synchronously initialize meta information and do the initialization on ps,
# Will modify PS worker state, PS server parmeter initialization
# Won't modify config, computation graph and executor state
def topo_sort_register_ps(
    node_list: OP_LIST,
    ps_comm: CDLL,
    comm_mode: Optional[str],
    seed: int,
    cstable_policy: Optional[str]
) -> None:
    visited = set()
    for node in node_list:
        if isinstance(node, OptimizerOp):
            opt = node.optimizer.get_config()

    def _topo_sort_register_ps(node: Op) -> None:
        if node in visited:
            return
        visited.add(node)
        if isinstance(node, PlaceholderOp) and node.trainable and (comm_mode == "PS" or node.is_embed):
            node_type = int(node.is_embed)
            if node_type and cstable_policy is not None:
                node_type = 2
            node.initializer.init_on_ps(
                ps_comm, node.id, node_type, seed=seed + node.id, opt=opt)
        for n in node.inputs:
            _topo_sort_register_ps(n)

    for node in node_list:
        _topo_sort_register_ps(node)


def get_pipeline_stage_info(node_list: OP_LIST, ctx: DLContext) -> Tuple[int, int, int, int]:
    # TODO: use a device group to judge pipeline articulation
    # TODO: the pipline rank/nrank cannot express noncontiguous stages
    stage_index = {}

    def _is_pipeline_articulation(n0: Op, n1: Op) -> bool:
        if isinstance(n0, OptimizerOp):
            return True
        w0 = n0.raw_ctx.workers
        w1 = n1.raw_ctx.workers
        if len(w0) == 0 or len(w1) == 0 or w0 == w1:
            return False
        if len(w0) != len(w1):
            # different dp nrank => different pipeline stage
            return True
        for ww0, ww1 in zip(w0, w1):
            if not isinstance(ww0, tuple):
                ww0 = (ww0,)
            if not isinstance(ww1, tuple):
                ww1 = (ww1,)
            if len(set(ww0).intersection(set(ww1))) > 0:
                # has intersection, in the same stage
                return False
        return True

    def _traverse(node: Op) -> None:
        if node in stage_index:
            return
        stage_index[node] = 0
        for n in node.inputs:
            _traverse(n)
            if _is_pipeline_articulation(node, n):
                stage_index[node] = max(stage_index[n] + 1, stage_index[node])
            else:
                stage_index[node] = max(stage_index[n], stage_index[node])

    for node in node_list:
        _traverse(node)

    # for a n stage pipeline, we have 2 * ( n - 1) pipeline articulation and 1 optimizer,
    # thus max(stage_index.values()) = 2n -1
    total_stage = (max(stage_index.values()) + 1) // 2
    total_stage = max(1, total_stage)  # handle corner case
    # find out my stage index, which is the biggest stage number in the forward pass
    my_stage = set()
    # dp rank (data parallel rank) is used to let dataloader know which part of data it should load
    my_dp_rank = None
    my_dp_nrank = None
    for node, stage in stage_index.items():
        if isinstance(node, OptimizerOp):
            continue
        node.raw_ctx.set_index(ctx)
        if node.raw_ctx.local_dp and stage < total_stage:
            dp_index = node.raw_ctx.dp_index
            dp_nrank = node.raw_ctx.worker_num
            assert my_dp_rank in (dp_index, None)
            assert my_dp_nrank in (dp_nrank, None)
            my_dp_rank = dp_index
            my_dp_nrank = dp_nrank
            my_stage.add(stage)
    my_stage = max(my_stage)
    assert my_stage >= 0
    return my_stage, total_stage, my_dp_rank, my_dp_nrank


def topo_sort_with_hook(node_list: OP_LIST, config: HetuConfig) -> None:
    visited = set()
    for node in node_list:
        topo_sort_dfs_with_hook(node, visited, config)


def topo_sort_dfs_with_hook(node: Op, visited: Set[Op], config: HetuConfig) -> None:
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


def find_topo_sort(node_list: OP_LIST) -> OP_LIST:
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


def topo_sort_dfs(node: Op, visited: Set[Op], topo_order: OP_LIST) -> None:
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def find_topo_sort_inference(node_list: OP_LIST) -> OP_LIST:
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


def fetch_sparse_parameter_value(node_list: OP_LIST, config: HetuConfig) -> None:
    for node in node_list:
        if isinstance(node, ParameterServerSparsePullOp):
            node.forward_hook(config)


def fetch_dense_parameter_value(node_list: OP_LIST, config: HetuConfig) -> None:
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


def sum_node_list(node_list: OP_LIST, ctx: Optional[DeviceGroup], shape: Optional[Tuple[int]] = None) -> Tuple[Optional[Op], bool, bool]:
    """Custom sum func to avoid creating redundant nodes in Python sum func."""
    node_list = [n for n in node_list if n is not None]
    if node_list == []:
        return None, False
    elif len(node_list) == 1:
        return node_list[0], False
    else:
        if any([isinstance(n, tuple) for n in node_list]):
            inputs = []
            dtype = None
            for n in node_list:
                if isinstance(n, tuple):
                    inputs.append((n[0], n[2]))
                    assert dtype in (None, n[2].dtype)
                    dtype = n[2].dtype
                else:
                    inputs.append(n)
                    assert dtype in (None, n.dtype)
                    dtype = n.dtype
            return sum_sparse_gradient_op(shape, *inputs, dtype=dtype, ctx=ctx), True
        else:
            return sum_op(node_list, ctx=ctx), True


def reorder_for_group(topo_order: OP_LIST, layer_indices: Optional[Dict[Op, Union[int, float]]]) -> OP_LIST:
    if layer_indices is None:
        return topo_order
    # here we reorder for 2 reasons:
    # 1. group consecutive pipeline send/recv ops
    # 2. reorder pipeline send/recv ops according to grouping indices
    has_pipeline_ops = set([layer_indices[x] for x in layer_indices if isinstance(
        x, (PipelineSendOp, PipelineReceiveOp))])
    if len(has_pipeline_ops) == 0:
        # if no pipeline send/recv, no reorder; a workaround for PS
        # TODO: better plan?
        return topo_order
    labels = {}
    for node in topo_order:
        if isinstance(node, (DataH2DOp, DataD2HOp, DataD2HSparseOp)):
            layer_indices[node] = layer_indices[node.inputs[0]] + 0.5
        cur_with_pipeline = layer_indices[node] in has_pipeline_ops
        if cur_with_pipeline and isinstance(node, SplitOp):
            labels[node] = 1
        elif isinstance(node, (PipelineSendOp, PipelineReceiveOp)):
            labels[node] = 2
        elif cur_with_pipeline and isinstance(node, (SumOp, ConcatenateOp)):
            labels[node] = 3
        else:
            labels[node] = 0

    topo_order = sorted(topo_order, key=lambda x: 10 *
                        layer_indices[x] + labels[x])
    return topo_order
