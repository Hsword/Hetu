""" library to take autodiff and execute a computation graph """
from __future__ import absolute_import
from .BatchNorm import Batch_NormalizationOp
import numpy as np
from scipy.sparse import spmatrix, coo_matrix
from .. import ndarray
from ..stream import create_stream_handle, Event

from .PipelineSend import PipelineSendOp
from .PipelineReceive import PipelineReceiveOp
from .Sum import SumOp
from .Split import SplitOp
from .Concatenate import ConcatenateOp
from .Dropout import DropoutOp
from .LayerNorm import Layer_NormalizationOp
from .OnesLike import OnesLikeOp
from ..communicator.mpi_nccl_comm import ncclDataType_t, GroupStart, GroupEnd
from .ParameterServerCommunicate import ParameterServerCommunicateOp, ParameterServerSparsePullOp, parameterServerSparsePull_op
from .Variable import PlaceholderOp  # add for optimizer
from ..dataloader import DataloaderOp, GNNDataLoaderOp
from ..optimizer import OptimizerOp
from .AllReduceCommunicate import AllReduceCommunicateOp
from .EmbeddingLookUp import EmbeddingLookUp, EmbeddingLookUp_Gradient
from .DataTransfer import DataH2DOp, DataD2HOp, DataD2HSparseOp

from .executor import find_topo_sort

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
