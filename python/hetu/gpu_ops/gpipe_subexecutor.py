from __future__ import absolute_import

from .pipeline_subexecutor import SubExecutor4Pipe
from .timer_subexecutor import TimerSubExecutor, make_timer


class SubExecutor4Gpipe(SubExecutor4Pipe):
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
        super().__init__(name, eval_node_list, config, reserve_opt=False)
        self.node_to_arr_maps = []
        timer = None
        self.timer = make_timer(timer, self.config.context)
        if timer is not None:
            self.compute = lambda *args: TimerSubExecutor.compute(self, *args)

    def memory_plan(self, batch_num):
        num_loop = batch_num // len(self.schedule)
        need_memory_indices = [i for i, sched in enumerate(
            self.schedule * num_loop) if sched]
        for mp in self.iter_valid_arr_map(need_memory_indices):
            self.config.memory_pool.memory_plan(
                self.computing_nodes, self.node_to_shape_map, mp, self.config, self.eval_node_list, self.indexed_slices_shape)

    def run(self, eval_node_list, feed_dicts_list, convert_to_numpy_ret_vals, batch_num=None):
        if batch_num is None:
            batch_num = self.config.lcm_dp_nrank
        if feed_dicts_list:
            assert batch_num == len(feed_dicts_list), "Feed dicts list invalid"

        if not self.node_to_arr_maps:
            self.node_to_arr_maps = [dict() for _ in range(batch_num)]
            need_reallocation = True
        else:
            need_reallocation = False

        cur_schedule = self.step_index(batch_num)
        self.valid_indices = [
            i for i, cur_exec in enumerate(cur_schedule) if cur_exec]
        feed_shapes = {}

        # get feed in values
        for idx in self.valid_indices:
            cur_node_to_arr_map = self.node_to_arr_maps[idx]
            feed_dict = feed_dicts_list[idx] if feed_dicts_list else {}
            for node, value in feed_dict.items():
                if node not in self.need_feed_nodes:
                    continue
                local_shape, local_realloc = self.get_feed_value(
                    cur_node_to_arr_map, node, value)
                need_reallocation = need_reallocation or local_realloc
                if node not in feed_shapes:
                    feed_shapes[node] = local_shape
                else:
                    assert feed_shapes[node] == local_shape

            for node in self.dataloader_nodes:
                cur_node_to_arr_map[node] = node.get_arr(self.name)
                local_shape = node.get_cur_shape(self.name)
                if node not in feed_shapes:
                    feed_shapes[node] = local_shape
                else:
                    assert feed_shapes[node] == local_shape

        # reallocation, infer shapes and allocate memory
        if need_reallocation:
            self.infer_shape(feed_shapes)
            self.memory_plan(batch_num)

        # computing
        for cur_subgraph in self.partitions:
            for cur_node_to_arr_map in self.iter_valid_arr_map():
                self.compute(cur_subgraph, cur_node_to_arr_map)

        # apply gradient update after all calculations for microbatches are finished
        if self.opt is not None:
            for cur_node_to_arr_map in self.iter_valid_arr_map():
                input_vals = [cur_node_to_arr_map[n] for n in self.opt.inputs]
                node_val = cur_node_to_arr_map[self.opt]
                with self.timer(self.opt, self.comp_stream):
                    self.opt.compute(input_vals, node_val, self.comp_stream)

        self.comp_stream.sync()

        # get results
        results = [[cur_node_to_arr_map[n]
                    for cur_node_to_arr_map in self.iter_valid_arr_map()] for n in self.eval_node_list]
        if convert_to_numpy_ret_vals:
            for i in range(len(results)):
                for j in range(len(results[i])):
                    if results[i][j] is not None:
                        results[i][j] = results[i][j].asnumpy()
        for i in range(len(results)):
            if all([x is None for x in results[i]]):
                results[i] = None

        # remap to original order in model parallel
        new_results = [None for _ in self.global_eval_nodes]
        for i, j in enumerate(self.run_results_indices):
            new_results[j] = results[i]
        results = new_results

        return results

    def iter_valid_arr_map(self, indices=None):
        if indices is None:
            indices = self.valid_indices
        for idx in indices:
            yield self.node_to_arr_maps[idx]

    def clearTimer(self):
        return TimerSubExecutor.clearTimer(self)

    def logOut(self, path, log_level='node', clear=True):
        return TimerSubExecutor.logOut(self, path, log_level, clear, multiplier=len(self.valid_indices))
