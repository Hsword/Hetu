
from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from ..gpu_links import matrix_elementwise_multiply_by_const
from ..ndarray import cpu
from .. import stream
import os
import numpy as np
import ctypes


class ParameterServerCommunicateOp(Op):

    def __init__(self, nodeA, parameter, optimizer):
        super().__init__(ParameterServerCommunicateOp, [nodeA], nodeA.ctx)
        self.on_gpu = ndarray.is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu
        self.parameter = parameter
        self.optimizer = optimizer
        # the optimizer not implemented yet! only SGD is supported, calculate on worker
        # the optimizer only support fixed learning rate, no scheduler supported.
        # TODO: implement optimizer on Servers(already implemented, not in use) and Caches(not implemented yet)
        # TODO: implement learning rate scheduler
        self.learning_rate = -optimizer[1][0]
        self.ps_id = ctypes.c_int(self.parameter.id)
        self.psevent = None

    def _get_event(self, input_val, stream_handle):
        if stream_handle:
            self.push_val.async_d2h(input_val, stream_handle, self.psevent)
            evt = self.psevent.handle
        else:
            input_val.copyto(self.push_val)
            evt = None
        return evt

    def _compute_asp_prefetch(self, input_vals, output_val, stream_handle=None):
        self._mult_lr(input_vals[0], stream_handle)
        self._update_event(self._push_pull(input_vals[0], stream_handle))

    def _compute_ssp_prefetch(self, input_vals, output_val, stream_handle=None):
        self._mult_lr(input_vals[0], stream_handle)
        self._wait(self._push(input_vals[0], stream_handle))
        self.comm.ssp_sync(self.ps_id, self.ssp_version)
        self._update_event(self._pull())
        self.ssp_version += 1

    def _compute_bsp_prefetch(self, input_vals, output_val, stream_handle=None):
        self._mult_lr(input_vals[0], stream_handle)
        self._wait(self._push(input_vals[0], stream_handle))
        self.comm.BarrierWorker()
        self._update_event(self._pull())

    def _compute_no_prefetch(self, input_vals, output_val, stream_handle=None):
        self._mult_lr(input_vals[0], stream_handle)
        self._update_event(self._push(input_vals[0], stream_handle))

    def _mult_lr_sparse_cpu(self, input_val, stream_handle):
        input_val.values[:] = input_val.values.asnumpy() * self.learning_rate

    def _mult_lr_dense_cpu(self, input_val, stream_handle):
        input_val[:] = input_val.asnumpy() * self.learning_rate

    def _mult_lr_dense_gpu(self, input_val, stream_handle):
        matrix_elementwise_multiply_by_const(
            input_val, self.learning_rate, input_val, stream_handle)

    def _push_pull_cache(self, input_val, stream_handle):
        return self.cache.embedding_push_pull(
            pullkeys=self.dl_node.get_next_arr(self.dl_name), dest=self.sparse_pull_val,
            pushkeys=input_val.indices, grads=input_val.values
        )

    def _push_pull_sparse_cpu(self, input_val, stream_handle):
        return self.comm.SSPushPull(self.ps_id, input_val.indices.handle, input_val.values.handle,
                                    self.dl_node.get_next_arr(self.dl_name).handle, self.sparse_pull_val.handle, None)

    def _push_pull_halfsparse_cpu(self, input_val, stream_handle):
        return self.comm.SDPushPull(self.ps_id, input_val.indices.handle, input_val.values.handle, self.pull_val.handle, None)

    def _push_pull_dense_cpu(self, input_val, stream_handle):
        return self.comm.DDPushPull(self.ps_id, input_val.handle, self.pull_val.handle, None)

    def _push_pull_dense_gpu(self, input_val, stream_handle):
        evt = self._get_event(input_val, stream_handle)
        return self.comm.DDPushPull(self.ps_id, self.push_val.handle, self.pull_val.handle, evt)

    def _push_cache(self, input_val, stream_handle):
        return self.cache.embedding_update(input_val.indices, input_val.values)

    def _push_sparse_cpu(self, input_val, stream_handle):
        return self.comm.SparsePush(self.ps_id, input_val.indices.handle, input_val.values.handle, None)

    def _push_dense_cpu(self, input_val, stream_handle):
        return self.comm.Push(self.ps_id, input_val.handle, None)

    def _push_dense_gpu(self, input_val, stream_handle):
        evt = self._get_event(input_val, stream_handle)
        return self.comm.Push(self.ps_id, self.push_val.handle, evt)

    def _pull_cache(self):
        return self.cache.embedding_lookup(self.dl_node.get_next_arr(self.dl_name), self.sparse_pull_val)

    def _pull_sparse(self):
        return self.comm.SparsePull(self.ps_id, self.dl_node.get_next_arr(self.dl_name).handle, self.sparse_pull_val.handle)

    def _pull_dense(self):
        return self.comm.Pull(self.ps_id, self.pull_val.handle)

    def _wait_cache(self, ts):
        ts.wait()

    def _wait_ps(self, ts):
        self.comm.Wait(self.ps_id)

    def _update_event_cache(self, ts):
        self.parameter.event.update_ts(ts)

    def _update_event_ps(self, ts):
        self.parameter.event.update()

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return None

    def forward_hook(self, config):
        # disable inplace if not lazy execution
        # previously we use array reshape lazy callback to do this, which is deprecated (not efficient)
        self.inputs[0].inplace = False

        self.ctx = self.inputs[0].ctx
        self.on_gpu = ndarray.is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu

        if self.parameter.is_embed and self.on_gpu:
            self.on_cpu = True
            self.on_gpu = False
            self.ctx = cpu()
            self.inputs[0] = self.add_transfer_op(
                self.inputs[0], self.ctx, config.h2d_ops, config.d2h_ops)

        if self.on_gpu and self.inputs[0].event is None:
            self.inputs[0].event = stream.create_event_handle(self.ctx)

        self.comm = config.ps_comm
        node_shape = self.parameter.shape

        # using cache
        if config.cstable_policy is not None and self.parameter.is_embed:
            assert len(node_shape) == 2
            from hetu.cstable import CacheSparseTable
            self._wait = self._wait_cache
            self._update_event = self._update_event_cache
            self._mult_lr = self._mult_lr_sparse_cpu
            if config.bsp == 0 and config.prefetch:
                self._push = self._push_cache
                self._pull = self._pull_cache
                self.compute = self._compute_bsp_prefetch
            elif config.prefetch:
                self._push_pull = self._push_pull_cache
                self.compute = self._compute_asp_prefetch
            else:
                self._push = self._push_cache
                self.compute = self._compute_no_prefetch
            limit = node_shape[0] // 10  # TODO: need tuning
            # only worker 0 will do the initialization on server,
            # this function synchronously initialize meta information and do the initialization,
            # ALREADY has barrier!
            self.cache = CacheSparseTable(
                limit, node_shape[0], node_shape[1], self.parameter.id, config.cstable_policy, config.cache_bound)
            self.parameter.cache = self.cache
            if config.prefetch:
                self.dl_name = config.train_name
                self.dl_node = self.inputs[0].inputs[1]
                local_shape = list(self.dl_node.get_cur_shape(self.dl_name))
                local_shape.append(node_shape[-1])
                self.sparse_pull_val = ndarray.empty(
                    tuple(local_shape), ctx=ndarray.cpu(0))
                self.parameter.event.update_ts(self.cache.embedding_lookup(
                    self.dl_node.get_next_arr(self.dl_name), self.sparse_pull_val))
                config.ps_map[self.parameter] = self.sparse_pull_val
            return

        # initialize
        self_sparse = self.parameter.is_embed and config.use_sparse_pull
        if self.on_gpu:
            self.push_val = ndarray.empty(node_shape, ctx=ndarray.cpu(0))
            if config.d2h_stream:
                self.psevent = stream.create_event_handle(self.ctx)
        if self_sparse:
            if config.prefetch:
                self.dl_name = config.train_name
                self.dl_node = self.inputs[0].inputs[1]
                local_shape = list(self.dl_node.get_cur_shape(self.dl_name))
                local_shape.append(node_shape[-1])
                self.sparse_pull_val = ndarray.empty(
                    tuple(local_shape), ctx=ndarray.cpu(0))
                self.comm.SparsePull(self.ps_id, self.dl_node.get_next_arr(
                    self.dl_name).handle, self.sparse_pull_val.handle)
                config.ps_map[self.parameter] = self.sparse_pull_val
                self.parameter.event.update()
        else:
            self.pull_val = ndarray.empty(node_shape, ctx=ndarray.cpu(0))
            self.comm.Pull(self.ps_id, self.pull_val.handle)
            config.ps_map[self.parameter] = self.pull_val
            config.placeholder_to_arr_map[self.parameter] = self.pull_val
            self.parameter.event.update()

        # config compute function
        self._wait = self._wait_ps
        self._update_event = self._update_event_ps
        if self_sparse:
            self._mult_lr = self._mult_lr_sparse_cpu
            self._push = self._push_sparse_cpu
            self._pull = self._pull_sparse
            self._push_pull = self._push_pull_sparse_cpu
        elif self.parameter.is_embed:
            self._mult_lr = self._mult_lr_sparse_cpu
            self._push = self._push_sparse_cpu
            self._pull = self._pull_dense
            self._push_pull = self._push_pull_halfsparse_cpu
        elif self.on_cpu:
            self._mult_lr = self._mult_lr_dense_cpu
            self._push = self._push_dense_cpu
            self._pull = self._pull_dense
            self._push_pull = self._push_pull_dense_cpu
        else:
            self._mult_lr = self._mult_lr_dense_gpu
            self._push = self._push_dense_gpu
            self._pull = self._pull_dense
            self._push_pull = self._push_pull_dense_gpu
        if config.bsp >= 0 and (config.prefetch or not self_sparse):
            self.compute = self._compute_ssp_prefetch
            self.ssp_version = 0
            self.comm.ssp_init(
                self.ps_id, self.inputs[0].raw_ctx.worker_num, config.bsp)
        elif config.prefetch or not self_sparse:
            self.compute = self._compute_asp_prefetch
        else:
            self.compute = self._compute_no_prefetch

# 只在正向图插入sparse pull的op dense pull的op在init时完成


class ParameterServerSparsePullOp(Op):
    def __init__(self, node, deps_node):
        super().__init__(ParameterServerSparsePullOp,
                         [node] + deps_node, node.ctx)
        self.on_gpu = ndarray.is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu
        self.parameter = node.inputs[0]
        self.ps_id = ctypes.c_int(self.parameter.id)
        self.psevent = None

    def compute(self, input_vals, output_val, stream_handle=None):
        comm = self.comm
        if self.use_cache_table:
            ts = self.cache.embedding_lookup(
                self.dl_node.get_next_arr(self.dl_name), self.sparse_pull_val)
            self.parameter.event.update_ts(ts)
            return
        assert self.on_cpu == True
        assert isinstance(input_vals[0], ndarray.NDArray)
        comm.SparsePull(self.ps_id, self.dl_node.get_next_arr(
            self.dl_name).handle, self.sparse_pull_val.handle)
        self.parameter.event.update()

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return None

    def forward_hook(self, config):
        self.comm = config.ps_comm
        self.use_cache_table = config.cstable_policy is not None
        node_shape = self.parameter.shape
        assert (
            config.use_sparse_pull or self.use_cache_table) and self.parameter.is_embed
        self.dl_name = config.val_name
        self.dl_node = self.inputs[0].inputs[1]
        local_shape = list(self.dl_node.get_cur_shape(self.dl_name))
        local_shape.append(node_shape[-1])
        self.sparse_pull_val = ndarray.empty(
            tuple(local_shape), ctx=ndarray.cpu(0))
        config.infer_ps_map[self.parameter] = self.sparse_pull_val
        if self.use_cache_table:
            self.cache = self.parameter.cache
            self.parameter.event.sync()
            ts = self.cache.embedding_lookup(
                self.dl_node.get_next_arr(self.dl_name), self.sparse_pull_val)
            self.parameter.event.update_ts(ts)
        else:
            self.parameter.event.sync()
            self.comm.SparsePull(self.ps_id, self.dl_node.get_next_arr(
                self.dl_name).handle, self.sparse_pull_val.handle)
            self.parameter.event.update()


def parameterServerCommunicate_op(node, parameter, optimizer):
    """Make a new instance of ParameterServerCommunicateOp and call the instance.

    Parameters:
    ----
    node : Node
        The Node to do allreduce
    parameter: Node
        The parameter Node that corresponding to the gradient
    learning_rate: float
        Adjusted learning rate

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ParameterServerCommunicateOp(node, parameter, optimizer)


def parameterServerSparsePull_op(parameter, deps_node):
    """Make a new instance of ParameterServerCommunicateOp and call the instance.

    Parameters:
    ----
    node : Node
        The Node to do Pull data
    parameter: Node
        The parameter Node that corresponding to the gradient

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ParameterServerSparsePullOp(parameter, deps_node)
