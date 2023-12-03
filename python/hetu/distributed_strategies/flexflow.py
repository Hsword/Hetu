import numpy as np
from random import choice, random
from collections import defaultdict

from ..gpu_ops.Variable import PlaceholderOp
from ..gpu_ops.DataTransfer import DataD2HSparseOp
from ..optimizer import OptimizerOp
from .base import BaseSearchingStrategy
from ..ndarray import rgpu


class FlexFlowSearching(BaseSearchingStrategy):
    def __init__(self, feed_shapes, time_budget=-1, round_budget=-1, unit_round_budget=-1, alpha=0.05, **kargs):
        # DEPRECATED! Not maintained; not support prune_status in context or nccl primitives.
        # in FlexFlow paper, no duplicate considered
        assert 'include_duplicate' not in kargs
        super().__init__(feed_shapes, include_duplicate=False, **kargs)
        self.time_budget = time_budget
        self.round_budget = round_budget
        self.unit_round_budget = unit_round_budget
        self.alpha = alpha

    def budget_judge(self, start, ending, best_emerge_time, iter_num):
        # we consider both time budget and round budget
        if self.round_budget >= 0 and self.time_budget < 0:
            return iter_num < self.round_budget
        else:
            return iter_num < self.round_budget or \
                ending - start < self.time_budget or \
                (self.time_budget < 0 and 2 *
                 (ending - best_emerge_time) < ending - start)

    def searching(self, graph_status, memory_pool):
        from time import time
        if self.unit_round_budget >= 0:
            cnt = 0
            for node, value in self.search_space.items():
                cnt += len(value[1])
            self.round_budget = self.unit_round_budget * cnt
        all_possible_nodes = [
            k for k, v in self.search_space.items() if v[0]]
        print('No configuration loaded. Start autotuning...')
        # infer states using partial information
        start = time()
        meta_cur_state_map = graph_status.copy_cur_state_to()
        for node, value in self.merging.items():
            graph_status.node_cur_state_map[node] = meta_cur_state_map[value]
        graph_status.complete_state_map_with_partial_information(prune=False)
        simulation_result, memory_excess = self.make_graph_n_simulate(
            graph_status, memory_pool)
        graph_status.reset_status()
        assert memory_excess == 0, 'Data parallel is out of memory by {} MB! Not a good start point.'.format(
            memory_excess / 1024 / 1024 * 4)

        print('Initial data parallel configuration generated; simulation result: {:.3f}ms.'.format(
            simulation_result))
        best_cur_status = {node.name: meta_cur_state_map[node]
                           for node in self.search_space}
        best_raw_ctx = {node.name: node.raw_ctx for node in self.search_space}
        best_simulation = simulation_result
        best_emerge_time = time()
        prev_simulation = simulation_result
        cnt = 0

        ending = time()
        cnt += 1
        while self.budget_judge(start, ending, best_emerge_time, cnt - 1):
            # sample new configuration
            changing_node = choice(all_possible_nodes)
            new_status = choice(self.search_space[changing_node][1])
            new_raw_ctx = choice(self.device_candidates[new_status.dev_num])
            ori_status = meta_cur_state_map[changing_node]
            while (new_status.state == ori_status.state and new_status.dev_num == ori_status.dev_num and new_raw_ctx == changing_node.raw_ctx):
                new_status = choice(self.search_space[changing_node][1])
                new_raw_ctx = choice(
                    self.device_candidates[new_status.dev_num])
            print('Search round {}'.format(cnt))
            print('    Change node {} from {} in {} to {} in {}.'.format(
                changing_node, ori_status, changing_node.raw_ctx, new_status, new_raw_ctx))
            ori_raw_ctx = changing_node.raw_ctx
            self.set_group_raw_ctx(changing_node, new_raw_ctx)
            meta_cur_state_map[changing_node] = new_status
            graph_status.copy_cur_state_from(meta_cur_state_map)
            for node, value in self.merging.items():
                graph_status.node_cur_state_map[node] = meta_cur_state_map[value]
            graph_status.complete_state_map_with_partial_information(
                prune=False)
            new_simulation, memory_excess = self.make_graph_n_simulate(
                graph_status, memory_pool)
            graph_status.reset_status()
            memory_excess_in_MB = memory_excess / 1024 / 1024 * 4
            print('    Simulation result {:.3f}ms. Memory exceeds {}MB.'.format(
                new_simulation, memory_excess_in_MB))
            # penalize 1ms for 1MB exceeded, as in FlexFlow
            new_simulation += memory_excess_in_MB
            if new_simulation < prev_simulation:
                # move to new strategy
                print(
                    '    Better than last simulation; move to new configuration.')
                if new_simulation < best_simulation and memory_excess == 0:
                    # ignore out-of-memory cases
                    best_cur_status = {node.name: meta_cur_state_map[node]
                                       for node in self.search_space}
                    best_raw_ctx = {
                        node.name: node.raw_ctx for node in self.search_space}
                    best_simulation = new_simulation
                    best_emerge_time = time()
                    print('    Reach the best simulation so far!')
                prev_simulation = new_simulation
            else:
                # probably move to new strategy
                threshold = np.exp(
                    self.alpha * (prev_simulation - new_simulation))
                print('    Worse than last simulation; the probability of moving is {:.3f}.'.format(
                    threshold))
                if random() < threshold:
                    # move to new strategy
                    print('    Move to new configuration.')
                    prev_simulation = new_simulation
                else:
                    # not move
                    print('    Keep the last configuration.')
                    meta_cur_state_map[changing_node] = ori_status
                    self.set_group_raw_ctx(changing_node, ori_raw_ctx)

                # TODO: delta change current states, target states, task graph, simulation result(maybe need to profile)
            ending = time()
            cnt += 1
        graph_status.copy_cur_state_from(meta_cur_state_map)
        self.simulator.write_cache()

        print('The simulation result of the best strategy discovered is: {:.3f}ms.'.format(
            best_simulation))
        return best_cur_status, best_raw_ctx

    class TaskNode(object):
        def __init__(self, name, device, inputs=[], shape=-1, memory_persistent=False, original_node=None):
            self.name = name
            self.device = device
            self.inputs = inputs
            if shape == -1:
                shape = self.inputs[0].shape
            self.shape = shape
            self.outputs = []
            for task in inputs:
                task.add_output(self)
            self.exeTime = None
            self.readyTime = 0
            self.startTime = 0
            self.endTime = 0
            self.contents = None
            self.state = 0  # 0 for incomplete, 1 for complete
            # TODO: consider DataloaderOp, GNNDataloaderOp, EmbeddingLookup(in PS)
            # check whether is parameter, useful in memory pool
            self.memory_persistent = memory_persistent
            self.original_node = original_node  # only for memory pool check
            if original_node is not None and original_node.use_indexed_slices:
                self.indexedslices = True
                self.real_shape = (self.inputs[1].shape, self.inputs[0].shape)
            elif isinstance(original_node, DataD2HSparseOp):
                # actually this part not implemented
                raise NotImplementedError
                self.indexedslices = True
                self.real_shape = self.inputs[0].shape
            else:
                self.indexedslices = False

        def add_output(self, out_node):
            self.outputs.append(out_node)

        def set_exetime(self, exeTime):
            self.exeTime = exeTime

        def __repr__(self):
            return self.name

        def __lt__(self, other):
            return self.readyTime < other.readyTime

        def make_memory_persistent(self):
            self.memory_persistent = True

    def make_task_node(self, node, index, shape, inputs=[], memory_persistent=False):
        from ..gpu_ops.Sum import SumOp
        name = '{}_{}'.format(node.name, index)
        device_group = node.raw_ctx.workers[0]
        if isinstance(device_group, tuple):
            device = device_group[index]
        else:
            assert index == 0
            device = device_group
        task = self.TaskNode(name, device, inputs=inputs, shape=shape,
                             memory_persistent=memory_persistent, original_node=node)
        if isinstance(node, SumOp):
            input_shapes = [
                n.real_shape if n.indexedslices else n.shape for n in task.inputs]
        else:
            input_shapes = [n.shape for n in task.inputs]
        task.set_exetime(self.simulator.get_node_time(
            node, input_shapes, task.shape))
        return task

    def make_split_task_node(self, op_index, input_task, axes, inds, splits):
        name = 'split_{}'.format(op_index)
        device = input_task.device
        shape = self.simulator.get_split_shape(
            dict(zip(axes, splits)), input_task.shape)
        task = self.TaskNode(name, device, inputs=[input_task, ], shape=shape)
        task.set_exetime(self.simulator.get_split_time(
            input_task.shape, axes, inds, splits))
        return task

    def make_concatenate_task_node(self, device, inputs, axis):
        name = 'concatenate_dim{}'.format(axis)
        input_shapes = [n.shape for n in inputs]
        task = self.TaskNode(name, device, inputs=inputs,
                             shape=self.simulator.get_concatenate_shape(input_shapes, axis))
        task.set_exetime(
            self.simulator.get_concatenate_time(input_shapes, axis))
        return task

    def make_sum_task_node(self, device, inputs):
        task = self.TaskNode('sum_partial', device, inputs=inputs)
        task.set_exetime(self.simulator.get_sum_time(
            [n.shape for n in inputs]))
        return task

    def make_update_task_node(self, device_group, index, input_task):
        name = 'update_{}'.format(index)
        if isinstance(device_group, tuple):
            device = device_group[index]
        else:
            assert index == 0
            device = device_group
        task = self.TaskNode(name, device, inputs=[input_task, ])
        task.make_memory_persistent()
        if input_task.indexedslices:
            sparse_shape = input_task.real_shape
        else:
            sparse_shape = None
        update_time = self.simulator.get_update_time(
            task.shape, sparse_shape=sparse_shape)
        task.set_exetime(update_time)
        return task

    def make_comm_task_node(self, from_device, to_device, prev_task):
        assert not prev_task.indexedslices
        name = 'comm_{}_to_{}'.format(from_device, to_device)
        device = (from_device, to_device)
        task = self.TaskNode(name, device, inputs=[prev_task, ])
        prev_task.make_memory_persistent()
        task.set_exetime(self.simulator.get_comm_time(
            from_device, to_device, task.shape))
        return task

    def make_group_comm_task_node(self, tasks):
        if len(tasks) == 0:
            return None
        name = 'group_comm'
        device = tuple(t.device for t in tasks)
        task = self.TaskNode(name, device, shape=None)
        task.inputs = list(set([ti for t in tasks for ti in t.inputs]))
        task.set_exetime(self.simulator.get_group_comm_time(
            [(t.device[0], t.device[1], t.shape) for t in tasks]))
        task.contents = tasks
        return task

    def make_allreduce_task_nodes(self, device_group, prev_tasks, status):
        name = 'allreduce'
        state, duplicate, order = status.get_all()
        partial = status.partial
        shape = prev_tasks[0].shape
        is_allgather = prev_tasks[0].indexedslices
        num_splits = status.dev_num // partial
        allreduce_tasks = []
        update_tasks = []
        interval = 1
        par_dim = order.index(-2)
        for cur_order in order[par_dim+1:]:
            interval *= state[cur_order]
        macro_interval = interval * partial
        for index in range(num_splits):
            start = (index // interval) * \
                macro_interval + (index % interval)
            indices = range(start, start + macro_interval, interval)
            allreduce_devices = [
                device_group.workers[0][ind] for ind in indices]
            devices = tuple(allreduce_devices)
            task = self.TaskNode(name, devices, inputs=[
                prev_tasks[ind] for ind in indices])
            if is_allgather:
                indices_shape, value_shape = prev_tasks[0].real_shape
                task.set_exetime(self.simulator.wrapped_get_allgather_time(
                    indices_shape, value_shape, device_group, status))
                task.indexedslices = True
                new_shape0 = list(indices_shape)
                new_shape1 = list(value_shape)
                new_shape0[0] *= partial
                new_shape1[0] *= partial
                task.real_shape = (tuple(new_shape0), tuple(new_shape1))
            else:
                task.set_exetime(self.simulator.wrapped_get_allreduce_time(
                    shape, device_group, status))
            allreduce_tasks.append(task)
            update_tasks.extend([self.make_update_task_node(
                device_group.workers[0], ind, task) for ind in indices])
        for task in prev_tasks:
            task.make_memory_persistent()
        for task in allreduce_tasks:
            task.make_memory_persistent()
        return allreduce_tasks, update_tasks

    def init_task_graph(self, graph_status):

        def init_task(node, eval_node=False):
            if node not in node_to_task_map:
                if isinstance(node, PlaceholderOp):
                    status = node_cur_state_map[node]
                    new_shape = self.simulator.get_split_shape(
                        status.state, self.feed_shapes.get(node, node.shape))
                    cur_tasks = [self.make_task_node(
                        node, i, new_shape, memory_persistent=True) for i in range(status.dev_num)]

                elif isinstance(node, OptimizerOp):
                    self.simulator.init_empty_optimizer(
                        node.optimizer, cached=True)
                    cur_tasks = []
                    for grad, param in zip(node.inputs, node.optimizer.params):
                        temp_tasks = init_comm_task(grad, param)
                        partial = node_cur_state_map[grad].partial
                        if partial is not None and partial > 1:
                            # allreduce tasks + update tasks
                            allreduce_tasks, update_tasks = self.make_allreduce_task_nodes(
                                param.raw_ctx, temp_tasks, node_cur_state_map[grad])
                            task_topo_order.extend(allreduce_tasks)
                            cur_tasks.extend(update_tasks)
                        else:
                            # update task
                            cur_tasks.extend([self.make_update_task_node(
                                param.raw_ctx.workers[0], i, t) for i, t in enumerate(temp_tasks)])
                else:
                    inputs = {}
                    for n in node.inputs:
                        inputs[n] = init_comm_task(n, node)
                    cur_tasks = [self.make_task_node(node, i, node.naive_infer_shape(
                        [n.shape for n in ns]), inputs=list(ns)) for i, ns in enumerate(zip(*inputs.values()))]

                node_to_task_map[node] = cur_tasks
                if eval_node:
                    for task in cur_tasks:
                        task.make_memory_persistent()
                task_topo_order.extend(cur_tasks)
            return node_to_task_map[node]

        def init_comm_task(prev, node):
            prev_tasks = init_task(prev)
            prev_ctx, cur_ctx = prev.raw_ctx, node.raw_ctx
            cur_stat, tar_stat = node_cur_state_map[prev], node_tar_state_map[prev]
            deduce_mp = node in tar_stat \
                and (cur_stat != tar_stat[node] or prev_ctx != cur_ctx) \
                and (cur_stat.is_dist() or tar_stat[node].is_dist())
            if deduce_mp:
                generated_splits = []
                generated_comb = []
                comm_tasks = []
                key = (tar_stat[node].content_hash(), cur_ctx)
                if not key in recv_src[prev]:
                    cur_tasks = []
                    task_buffer = defaultdict(dict)
                    prev_devices = prev_ctx.workers[0] if prev_ctx.is_mp else [
                        prev_ctx.workers[0]]
                    cur_devices = cur_ctx.workers[0] if cur_ctx.is_mp else [
                        cur_ctx.workers[0]]
                    prev_state, prev_duplicate, prev_order = cur_stat.get_all()
                    prev_partial = cur_stat.partial
                    target_state, target_duplicate, target_order = tar_stat[node].get_all(
                    )

                    # send first
                    def cross_send(split_cur_state, split_target_state, depth, need_split):
                        nonlocal device_index
                        if depth == len(target_order):
                            if need_split:
                                keys = list(
                                    split_target_state.keys())
                                indices = [split_cur_state[k]
                                           for k in keys]
                                splits = [split_target_state[k]
                                          for k in keys]
                                # split op
                                res_task = self.make_split_task_node(
                                    device_index, prev_tasks[mp_index], keys, indices, splits)
                                generated_splits.append(res_task)
                            else:
                                res_task = prev_tasks[mp_index]
                            if prev_devices[mp_index] != cur_devices[device_index]:
                                res_task = self.make_comm_task_node(
                                    prev_devices[mp_index], cur_devices[device_index], res_task)
                                comm_tasks.append(res_task)
                            task_buffer[mp_index][device_index] = res_task
                            device_index += 1
                        else:
                            cur_dim = target_order[depth]
                            if cur_dim < 0:
                                assert cur_dim == -1, 'Target node status must not enable partial.'
                                cur_index = cur_state_index.get(cur_dim, 0)
                                if prev_duplicate % target_duplicate == 0:
                                    # at `cur_dim` dimension we need to send one output
                                    multiple = prev_duplicate // target_duplicate
                                    assert cur_index % multiple == 0
                                    device_index += cur_index // multiple * \
                                        loop_sizes[depth]
                                    cross_send(split_cur_state,
                                               split_target_state, depth+1, need_split)
                                    device_index += (prev_duplicate - 1 -
                                                     cur_index) // multiple * loop_sizes[depth]
                                elif target_duplicate % prev_duplicate == 0:
                                    # at `cur_dim` dimension we need to split and send some outputs
                                    multiple = target_duplicate // prev_duplicate
                                    device_index += cur_index * \
                                        multiple * loop_sizes[depth]
                                    for index in range(multiple):
                                        cross_send(split_cur_state,
                                                   split_target_state, depth+1, True)
                                    device_index += (prev_duplicate - 1 -
                                                     cur_index) * multiple * loop_sizes[depth]
                                else:
                                    assert False
                            else:
                                pre_st = prev_state.get(cur_dim, 1)
                                cur_st = cur_state_index.get(
                                    cur_dim, 0)
                                if pre_st % target_state[cur_dim] == 0:
                                    # at `cur_dim` dimension we need to send one output
                                    multiple = pre_st // target_state[cur_dim]
                                    device_index += cur_st // multiple * \
                                        loop_sizes[depth]
                                    split_cur_state[cur_dim] = 0
                                    split_target_state[cur_dim] = 1
                                    cross_send(split_cur_state,
                                               split_target_state, depth+1, need_split)
                                    device_index += (pre_st - 1 -
                                                     cur_st) // multiple * loop_sizes[depth]
                                elif target_state[cur_dim] % pre_st == 0:
                                    # at `cur_dim` dimension we need to split and send some outputs
                                    multiple = target_state[cur_dim] // pre_st
                                    device_index += cur_st * \
                                        multiple * \
                                        loop_sizes[depth]
                                    for index in range(multiple):
                                        split_cur_state[cur_dim] = index
                                        split_target_state[cur_dim] = multiple
                                        cross_send(split_cur_state,
                                                   split_target_state, depth+1, True)
                                    device_index += (pre_st - 1 -
                                                     cur_st) * multiple * loop_sizes[depth]
                                else:
                                    assert False, 'The dispatch state (%d, %d) at dimension %d is invalid.' % (
                                        pre_st, target_state[cur_dim], cur_dim)

                    loop_sizes = tar_stat[node].get_loop_sizes()
                    for mp_index in range(prev_ctx.mp_dev_num):
                        cur_state_index = cur_stat.map_dev_to_index(
                            mp_index, containing_duplicate=True)
                        if cur_stat.partial == 1 and prev_duplicate > target_duplicate and cur_state_index.get(-1, 0) % (prev_duplicate // target_duplicate) != 0:
                            pass
                        else:
                            device_index = 0
                            cross_send({}, {}, 0, False)
                            assert device_index == len(cur_devices)

                    # receive next
                    def cross_receive(depth):
                        nonlocal device_index
                        if depth == len(prev_order):
                            res_task = task_buffer[device_index][mp_index]
                            device_index += 1
                        else:
                            cur_dim = prev_order[depth]
                            if cur_dim == -2:
                                res_task = self.make_sum_task_node(cur_devices[mp_index], [
                                    cross_receive(depth+1) for _ in range(prev_partial)])
                                generated_comb.append(res_task)
                            elif cur_dim == -1:
                                # TODO: consider how to choose the copy with minimal communication
                                # now we use following rules:
                                # if prev_duplicate < target_duplicate, then each prev send to some targets
                                # else, each target receive from the first duplicate in the group
                                prev_index = cur_state_index.get(cur_dim, 0)
                                if prev_duplicate % target_duplicate == 0:
                                    multiple = prev_duplicate // target_duplicate
                                    device_index += prev_index * \
                                        multiple * loop_sizes[depth]
                                    res_task = cross_receive(depth+1)
                                    device_index += ((target_duplicate - prev_index)
                                                     * multiple - 1) * loop_sizes[depth]
                                elif target_duplicate % prev_duplicate == 0:
                                    multiple = target_duplicate // prev_duplicate
                                    device_index += prev_index // multiple * \
                                        loop_sizes[depth]
                                    res_task = cross_receive(depth+1)
                                    device_index += (target_duplicate - 1 -
                                                     prev_index) // multiple * loop_sizes[depth]
                                else:
                                    assert False
                            else:
                                tar_st = target_state.get(cur_dim, 1)
                                cur_st = cur_state_index.get(
                                    cur_dim, 0)
                                if prev_state[cur_dim] % tar_st == 0:
                                    # at `cur_dim` dimension we need to concat some inputs
                                    multiple = prev_state[cur_dim] // tar_st
                                    device_index += cur_st * \
                                        multiple * loop_sizes[depth]
                                    if multiple == 1:
                                        res_task = cross_receive(depth+1)
                                    else:
                                        # concatenate op task
                                        inputs = [cross_receive(
                                            depth+1) for _ in range(multiple)]
                                        res_task = self.make_concatenate_task_node(
                                            cur_devices[mp_index], inputs, cur_dim)
                                        generated_comb.append(res_task)
                                    device_index += (tar_st - 1 - cur_st) * \
                                        multiple * loop_sizes[depth]
                                elif tar_st % prev_state[cur_dim] == 0:
                                    # at `cur_dim` dimension we need to specify one input
                                    multiple = tar_st // prev_state[cur_dim]
                                    device_index += cur_st // multiple * \
                                        loop_sizes[depth]
                                    res_task = cross_receive(depth+1)
                                    device_index += (tar_st - 1 -
                                                     cur_st) // multiple * loop_sizes[depth]
                                else:
                                    assert False, 'The dispatch state (%d, %d) at dimension %d is invalid.' % (
                                        prev_state[cur_dim], tar_st, cur_dim)
                        return res_task

                    loop_sizes = cur_stat.get_loop_sizes()
                    for mp_index in range(cur_ctx.mp_dev_num):
                        cur_state_index = tar_stat[node].map_dev_to_index(
                            mp_index, containing_duplicate=True)
                        device_index = 0
                        cur_tasks.append(cross_receive(0))
                        assert device_index == len(prev_devices)

                    recv_src[prev][key] = cur_tasks
                task_topo_order.extend(generated_splits)
                group_comm_task = self.make_group_comm_task_node(comm_tasks)
                for t in comm_tasks:
                    group_comm_map[t] = group_comm_task
                if group_comm_task is not None:
                    task_topo_order.append(group_comm_task)
                task_topo_order.extend(generated_comb)
                return recv_src[prev][key]
            else:
                # check parallel + data parallel
                generated_tasks = []
                assert prev_ctx.worker_num == cur_ctx.worker_num == 1, \
                    'In flexflow, the worker number should be 1!'
                prev_ctx.check_mp_num(cur_ctx.mp_dev_num)
                if prev_ctx.mp_dev_num == 1:
                    if prev_ctx.workers[0] != cur_ctx.workers[0]:
                        if cur_ctx not in recv_src[prev]:
                            res_task = self.make_comm_task_node(
                                prev_ctx.workers[0], cur_ctx.workers[0], prev_tasks[0])
                            generated_tasks.append(res_task)
                            recv_src[prev][cur_ctx] = [res_task]
                        res_task = recv_src[prev][cur_ctx][0]
                    else:
                        res_task = prev_tasks[0]
                    task_topo_order.extend(generated_tasks)
                    return [res_task]
                else:
                    # here in the same model parallel
                    assert prev_ctx == cur_ctx
                    return prev_tasks
        node_to_task_map = {}
        group_comm_map = {}
        task_topo_order = []
        recv_src = defaultdict(dict)
        graph_status.extend_oplayers()
        node_cur_state_map, node_tar_state_map = graph_status.get_state_maps()
        for node in graph_status.node_list:
            init_task(node, eval_node=True)
        for task in task_topo_order:
            changed = False
            for i, t in enumerate(task.inputs):
                if t in group_comm_map:
                    task.inputs[i] = group_comm_map[t]
                    changed = True
            if changed:
                task.inputs = list(set(task.inputs))
            changed = False
            for i, t in enumerate(task.outputs):
                if t in group_comm_map:
                    task.outputs[i] = group_comm_map[t]
                    changed = True
            if changed:
                task.outputs = list(set(task.outputs))
        for task in set(group_comm_map.values()):
            outputs = sum([t.outputs for t in task.contents], [])
            task.outputs = list(set(outputs))
        graph_status.shrink_oplayers()
        return task_topo_order

    def full_simulate(self, task_graph):
        def func_init_items(key, num_workers): return [self.TaskNode(
            'NULL', rgpu(key, i), shape=()) for i in range(num_workers)]
        last_tasks = self.simulator.HostDictionary(
            self.simulator.nccl_profiler.workers, func_init_items)

        if self.simulator.pix:
            def func_init_items_comm(key, num_workers): return [self.TaskNode(
                'NULL', rgpu(key, i), shape=()) for i in range(num_workers + (num_workers % 2))]
        else:
            def func_init_items_comm(key, num_workers): return [self.TaskNode(
                'NULL', rgpu(key, i), shape=()) for i in range(2 * num_workers)]
        last_comm_tasks = self.simulator.HostDictionary(
            self.simulator.nccl_profiler.workers, func_init_items_comm, pix=self.simulator.pix)
        for task in task_graph:
            assert all([t.state == 1 for t in task.inputs])
            dev = task.device
            task.state = 1
            if isinstance(dev, tuple):
                # communication
                is_group = (task.name == 'group_comm')
                is_allreduce = (task.name == 'allreduce')
                startTime = task.readyTime
                if is_group:
                    for d in dev:
                        startTime = max(
                            startTime,
                            # last_tasks[d[0]].endTime,
                            last_comm_tasks.get_comm_send(d[0]).endTime,
                            last_comm_tasks.get_comm_recv(d[1]).endTime,
                        )
                elif is_allreduce:
                    for d in dev:
                        startTime = max(
                            startTime,
                            # last_tasks[d].endTime,
                            last_comm_tasks.get_comm_send(d).endTime,
                            last_comm_tasks.get_comm_recv(d).endTime,
                        )
                else:
                    startTime = max(
                        startTime,
                        # last_tasks[dev[0]].endTime,
                        last_comm_tasks.get_comm_send(dev[0]).endTime,
                        last_comm_tasks.get_comm_recv(dev[1]).endTime,
                    )
                task.startTime = startTime
                if is_group:
                    for d in dev:
                        last_comm_tasks.set_comm_send(d[0], task)
                        last_comm_tasks.set_comm_recv(d[1], task)
                elif is_allreduce:
                    for d in dev:
                        last_comm_tasks.set_comm_send(d, task)
                        last_comm_tasks.set_comm_recv(d, task)
                else:
                    last_comm_tasks.set_comm_send(dev[0], task)
                    last_comm_tasks.set_comm_recv(dev[1], task)
            else:
                task.startTime = max(task.readyTime, last_tasks[dev].endTime)
                last_tasks[dev] = task
            task.endTime = task.startTime + task.exeTime
            for t in task.outputs:
                t.readyTime = max(t.readyTime, task.endTime)
        result = 0
        for task in last_tasks.all_values():
            result = max(result, task.endTime)
        return result

    def make_graph_n_simulate(self, graph_status, memory_pool=None):
        task_graph = self.init_task_graph(graph_status)
        if memory_pool is not None:
            memory_excess = memory_pool.test_memory(
                self.all_devices, task_graph)
        else:
            memory_excess = None
        # self.log_task_graph(task_graph, log_level='node')
        return self.full_simulate(task_graph), memory_excess

    def log_task_graph(self, task_topo, log_path='task_graph_{}.txt', log_level='node'):
        def if_local(devs, idx):
            if isinstance(devs, tuple):
                return any([d.device_id == idx for d in devs])
            else:
                return devs.device_id == idx
        assert log_level in ('node', 'type')
        # only for debug
        if log_level == 'node':
            diff_files = (log_path.format(0) != log_path)
            if diff_files:
                for i in range(self.num_ctxs):
                    local_sum = 0.
                    cur_log_path = log_path.format(i)
                    with open(cur_log_path, 'w') as fw:
                        for task in task_topo:
                            if task.contents is None:
                                if not if_local(task.device, i):
                                    continue
                                print(task, task.exeTime, file=fw, flush=True)
                            else:
                                is_local = False
                                for t in task.contents:
                                    if if_local(t.device, i):
                                        is_local = True
                                        print(t, t.exeTime,
                                              file=fw, flush=True)
                                if not is_local:
                                    continue
                            local_sum += task.exeTime
                        print('All_time:', local_sum, file=fw, flush=True)
            else:
                with open(log_path, 'w') as fw:
                    for task in task_topo:
                        if task.exeTime == 0.0:
                            continue
                        print(task, task.inputs, task.exeTime, task.device,
                              task.readyTime, task.startTime, task.endTime, file=fw, flush=True)
        else:
            # TODO: use multiple files to express different device
            log_path = log_path.format('')
            from ..gpu_ops.Split import SplitOp
            from ..gpu_ops.Concatenate import ConcatenateOp
            from ..gpu_ops.Sum import SumOp
            from ..gpu_ops.PipelineSend import PipelineSendOp
            from ..gpu_ops.AllReduceCommunicate import AllReduceCommunicateOp
            new_dict = defaultdict(float)
            all_time = 0.
            for task in task_topo:
                name = task.name
                cur_time = task.exeTime
                if task.original_node is not None:
                    key = type(task.original_node)
                elif name.startswith('split'):
                    key = SplitOp
                elif name.startswith('sum'):
                    key = SumOp
                elif name.startswith('concatenate'):
                    key = ConcatenateOp
                elif name.startswith('update'):
                    key = OptimizerOp
                elif name.startswith('comm') or name.startswith('group_comm'):
                    key = PipelineSendOp
                elif name.startswith('allreduce'):
                    key = AllReduceCommunicateOp
                else:
                    assert False, 'Invalid task: {}.'.format(str(task))
                if key != AllReduceCommunicateOp:
                    cur_time /= self.num_ctxs
                new_dict[key] += cur_time
                all_time += cur_time
            with open(log_path, 'w') as fw:
                for node_type, comp_time in new_dict.items():
                    if comp_time == 0.:
                        continue
                    print('{}: {}'.format(node_type.__name__,
                                          comp_time), file=fw, flush=True)
                print('All_time: {}'.format(all_time), file=fw, flush=True)

    def simulate_time(self, graph_status, best_cur_status, best_raw_ctx):
        for node in self.node_group:
            graph_status.node_cur_state_map[node] = best_cur_status[node.name]
            self.set_group_raw_ctx(node, best_raw_ctx[node.name])
        for node, value in self.merging.items():
            # for merging backbone nodes
            graph_status.node_cur_state_map[node] = best_cur_status[value.name]
        graph_status.complete_state_map_with_partial_information(prune=False)
        assert self.mpi_comm.rank == 0
        simulation_result, _ = self.make_graph_n_simulate(graph_status)
        graph_status.copy_cur_state_from(best_cur_status)
        return simulation_result
