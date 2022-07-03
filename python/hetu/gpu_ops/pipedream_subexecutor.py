""" library to take autodiff and execute a computation graph """
from __future__ import absolute_import
import numpy as np
from .. import ndarray
from ..stream import Event

from .PipelineSend import PipelineSendOp
from .PipelineReceive import PipelineReceiveOp
from .Dropout import DropoutOp
from .BatchNorm import Batch_NormalizationOp
from .LayerNorm import Layer_NormalizationOp
from ..communicator.mpi_nccl_comm import GroupStart, GroupEnd
from .ParameterServerCommunicate import ParameterServerCommunicateOp, ParameterServerSparsePullOp
from .Variable import PlaceholderOp  # add for optimizer
from ..dataloader import DataloaderOp, GNNDataLoaderOp
from ..optimizer import OptimizerOp
from .AllReduceCommunicate import AllReduceCommunicateOp
from .EmbeddingLookUp import EmbeddingLookUp_Gradient
from .DataTransfer import DataH2DOp, DataD2HOp, DataD2HSparseOp
from ..gpu_links import matrix_elementwise_add, matrix_elementwise_multiply_by_const
from ..preduce import PartialReduce
from .pipeline_subexecutor import SubExecutor4Pipe


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


class SubExecutor4Pipedream(SubExecutor4Pipe):
    def __init__(self, name, eval_node_list, config):
        super().__init__(name, eval_node_list, config)
        self.use_ps = any([isinstance(node, ParameterServerCommunicateOp)
                           for node in self.topo_order])
        self.ctx = self.config.context

        self.forward_topo_order, self.backward_topo_order = self.partitions

        def move_send_op(tp):
            """
            move send op to the tail so that we can wrap send/recv pairs
            with nccl groupcall to avoid deadlock
            """
            for n in tp:
                if isinstance(n, PipelineSendOp):
                    tp.remove(n)
                    tp.append(n)
                    break

        move_send_op(self.backward_topo_order)

        # store intermediate tensors(all nodes except weights)
        self.batch_to_tensor_maps = dict()

        self.init_need_allocation = False
        if self.config.pipeline == "hetpipe":
            self.grad_accum_map = {}  # map weight node to gradients
            self.h2d_map = {}  # map weight to d2h node
            self.skip_h2d = set()
            for node in self.topo_order:
                if isinstance(node, DataH2DOp) and isinstance(node.inputs[0], PlaceholderOp) and node.inputs[0].trainable:
                    self.h2d_map[node.inputs[0]] = node
        elif config.use_preduce:
            self.preduce = PartialReduce(config.pp_rank)
            self.preduce_partner = None
            self.all_reduce_param_map = dict(
                zip(self.opt.inputs, self.opt.optimizer.params))

    def memory_plan(self, batch_id):
        """Allocates ndarray.NDArray for every node except feed_dict nodes.
        Parameters
        ----------
        batch_id: current batch
        """

        for node, shape in self.node_to_shape_map.items():
            mp = self.batch_to_tensor_maps[batch_id]

            if isinstance(node, PlaceholderOp):
                orig = self.config.placeholder_to_arr_map[node]
                if self.config.pipeline == "hetpipe":
                    if node not in self.grad_accum_map:
                        self.grad_accum_map[node] = ndarray.empty(
                            orig.shape, self.ctx)
                if node.raw_ctx.servers:
                    # use parameter servers, in this case weights are treated like activations (use H2D)
                    mp[node] = orig
                else:
                    if isinstance(orig, np.ndarray):
                        copied = orig.copy()
                    elif isinstance(orig, ndarray.NDArray):
                        # enable async copy
                        copied = ndarray.empty(orig.shape, orig.ctx)
                        copied._async_copyfrom(orig, self.comp_stream)
                    else:
                        raise ValueError
                    mp[node] = copied
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

    def copy_latest_weight(self, node):
        """
            In pipedream, the weight used in forward pass are used again in the backward pass.
            However, gradients should be applied to the latest model.
            Thus, we need to copy latest model weight.
        """
        if self.use_ps:
            return
        if node.trainable:
            oldest = min(self.batch_to_tensor_maps.keys())
            dst_tensor = self.batch_to_tensor_maps[oldest][node]
            src_tensor = self.config.placeholder_to_arr_map[node]
            # the last worker has only one copy of weight, no need to copy
            if src_tensor is not dst_tensor:
                dst_tensor._async_copyfrom(src_tensor, self.comp_stream)
            # after optimizer update, this dst tensor will become the latest model weight
            # so we set let placeholder_to_arr_map point to it
            self.config.placeholder_to_arr_map[node] = dst_tensor

    def update_gradient_local(self, ps_node, init_value, need_sync):
        node_id = ps_node.ps_id
        cur_batch_id = min(self.batch_to_tensor_maps.keys())
        grad_tensor = self.batch_to_tensor_maps[cur_batch_id][ps_node.inputs[0]]
        dst_tensor = self.grad_accum_map[ps_node.parameter]
        if init_value:
            dst_tensor._async_copyfrom(grad_tensor, self.comp_stream)
        else:
            matrix_elementwise_add(
                grad_tensor, dst_tensor, dst_tensor, stream=self.comp_stream)

        if not need_sync:
            last_fwd_batch_id = max(self.batch_to_tensor_maps.keys())
            h2d_node = self.h2d_map[ps_node.parameter]
            latest_weight = self.batch_to_tensor_maps[last_fwd_batch_id][h2d_node]
            current_weight = self.batch_to_tensor_maps[cur_batch_id][h2d_node]
            if cur_batch_id != last_fwd_batch_id:
                current_weight._async_copyfrom(latest_weight, self.comp_stream)

            self.run_optimizer(current_weight, grad_tensor)
            self.skip_h2d.add(h2d_node)

    def run_optimizer(self, weight_tensor, grad_tensor):
        matrix_elementwise_multiply_by_const(
            grad_tensor, -self.opt.optimizer.learning_rate, grad_tensor, self.comp_stream)
        matrix_elementwise_add(weight_tensor, grad_tensor,
                               weight_tensor, stream=self.comp_stream)

    def run(self, eval_node_list, feed_dict_list, convert_to_numpy_ret_vals, batch_num):
        rank = self.config.pp_rank
        nrank = self.config.pp_nrank

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
                    self.batch_to_tensor_maps[batch_id] = self.batch_to_tensor_maps.pop(
                        last_vacant_batch)
                    last_vacant_batch = -1

                feed_shapes = {}
                need_reallocation = self.init_need_allocation
                if self.batch_to_tensor_maps[batch_id] == dict():
                    need_reallocation = True
                # get dataloader values
                for node in self.dataloader_nodes:
                    local_shape = node.get_cur_shape(self.name)
                    local_realloc = local_shape != self.node_to_shape_map.get(
                        node, None)
                    need_reallocation = need_reallocation or local_realloc
                    self.batch_to_tensor_maps[batch_id][node] = node.get_arr(
                        self.name)
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
            for node in cur_topo:

                node_val = self.batch_to_tensor_maps[batch_id][node]
                input_vals = []
                for n in node.inputs:
                    input_vals.append(self.batch_to_tensor_maps[batch_id][n])
                    if n.event:
                        n.event.sync()

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
                    node.compute(input_vals, node_val,
                                 self.comp_stream, group_call=group_call)

                elif isinstance(node, PipelineReceiveOp):
                    group_call = False
                    if rank == 0:
                        group_call = True
                    if rank == nrank - 1 and batch_id > start_group_call_idx:
                        group_call = True
                    if rank not in (0, nrank - 1):
                        if cur_schedule == 1 or batch_id > start_group_call_idx:
                            group_call = True
                    node.compute(input_vals, node_val,
                                 self.comp_stream, group_call=group_call)

                elif isinstance(node, DataH2DOp):
                    if self.config.pipeline == "hetpipe" and node in self.skip_h2d:
                        self.skip_h2d.remove(node)
                    else:
                        node.compute(input_vals, node_val, self.h2d_stream)

                elif isinstance(node, (DataD2HOp, DataD2HSparseOp)):
                    node.compute(input_vals, node_val, self.d2h_stream)

                elif isinstance(node, AllReduceCommunicateOp):
                    if self.config.use_preduce:
                        if not self.preduce_partner:
                            self.preduce_partner = self.preduce.get_partner(
                                max_worker=self.inputs[0].raw_ctx.worker_num)
                        weight_node = self.all_reduce_param_map[node]
                        self.copy_latest_weight(weight_node)
                        weight_tensor = self.config.placeholder_to_arr_map[weight_node]
                        self.run_optimizer(weight_tensor, input_vals[0])
                        self.comp_stream.sync()
                        self.preduce.preduce(
                            weight_tensor, self.preduce_partner, stream=self.nccl_stream)
                        node.event.record(self.nccl_stream)
                    else:
                        node.compute(input_vals, node_val, self.nccl_stream)

                elif isinstance(node, (ParameterServerCommunicateOp, ParameterServerSparsePullOp)):
                    if self.config.pipeline == "hetpipe":
                        need_sync = (batch_id % self.config.pp_nrank == 0) or \
                            (batch_id == batch_num)
                        self.update_gradient_local(node, init_value=(
                            batch_id % self.config.pp_nrank) == 1, need_sync=need_sync)
                        if need_sync:
                            input_vals = [self.grad_accum_map[node.parameter]]
                            self.comp_stream.sync()
                            node.compute(input_vals, node_val, self.d2h_stream)
                    else:
                        node.compute(input_vals, node_val, self.d2h_stream)

                elif isinstance(node, (DropoutOp, Batch_NormalizationOp, Layer_NormalizationOp)):
                    node.compute(input_vals, node_val,
                                 self.comp_stream, inference=self.inference)

                elif isinstance(node, OptimizerOp):
                    if self.config.use_preduce:
                        self.preduce_partner = None  # renew partner for the next iteration
                    else:
                        for weight_node in self.config.placeholder_to_arr_map:
                            self.copy_latest_weight(weight_node)
                        node.compute(input_vals, node_val, self.comp_stream,
                                     self.batch_to_tensor_maps[batch_id])

                else:
                    node.compute(input_vals, node_val, self.comp_stream)
                    if isinstance(node.event, Event):
                        # for d2h op / eval nodes / nodes before [allreduce or ps nodes or pipelinesend nodes]
                        node.event.record(self.comp_stream)

            self.comp_stream.sync()

            # save result as numpy(must), because the tensor will be reused later
            if cur_schedule == 1:
                tmp_results = []
                for n in self.global_eval_nodes:
                    if n in self.batch_to_tensor_maps[batch_id]:
                        r = self.batch_to_tensor_maps[batch_id][n]
                        tmp_results.append(
                            r.asnumpy() if r is not None else None)
                results_list.append(tmp_results)

            # after update, mark the vacant maps
            if cur_schedule == 1:
                #assert last_vacant_batch == -1, "last_vacant_batch error, check the logic of code"
                last_vacant_batch = batch_id

            # end of scheduling loop
        # release for the next run
        self.batch_to_tensor_maps = {}
        if self.config.pipeline == 'hetpipe':
            self.skip_h2d.clear()

        return results_list
