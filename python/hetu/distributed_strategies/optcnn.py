from copy import copy, deepcopy
from collections import defaultdict
from ..optimizer import OptimizerOp
from ..gpu_ops.Sum import SumOp
from ..gpu_ops.OnesLike import OnesLikeOp
from .base import BaseSearchingStrategy


class OptCNNSearching(BaseSearchingStrategy):
    # drawbacks of OptCNN's cost model:
    # 1. not considering the influence from neighbor nodes
    #    if the neighbor nodes has the same status, then the communication from father has already calculated
    # TODO: whether or not considering such topology and how?
    # 2. not consider memory constraints
    # TODO: whether or not add memory planner?
    # 3. not support partial split
    # TODO: add partial?
    def __init__(self, feed_shapes, **kargs):
        # in OptCNN paper, no duplicate considered
        super().__init__(feed_shapes, include_duplicate=False, **kargs)
        self.status_n_ctxs = None
        self.debugging = False
        self.use_nccl_collectives = True

    def searching(self, graph_status, memory_pool):
        # make coarse graph and initialize dp table, according to node group
        self.reduce_device_candidates()
        graph_status.extend_oplayers()
        self.node_status_map = dict()

        self.add_group_optimizer_node(graph_status.opt)
        node_cur_state_map, node_raw_ctx_map, min_time = self.search_part(
            graph_status)
        print('The simulated optimized execution time is: {}ms.'.format(min_time))
        graph_status.shrink_oplayers()
        if self.status_n_ctxs is None:
            return {node.name: node_cur_state_map[node] for node in self.search_space}, {node.name: node_raw_ctx_map[node] for node in self.search_space}
        else:
            # for load with simulate
            return min_time

    def add_group_optimizer_node(self, opt_node):
        # add partial optimizer node into node group
        params = opt_node.optimizer.params
        opt_map_grad = defaultdict(list)
        opt_map_param = defaultdict(list)
        self.simulator.init_empty_optimizer(
            opt_node.optimizer, cached=True)
        for i, n in enumerate(opt_node.inputs):
            cur_backbone_node = self.inverse_node_group[n]
            assert cur_backbone_node is self.inverse_node_group[params[i]]
            opt_map_grad[cur_backbone_node].append(n)
            opt_map_param[cur_backbone_node].append(params[i])
        for key in opt_map_param.keys():
            new_opt = self.simulator.init_empty_optimizer(
                opt_node.optimizer)
            new_opt.params = opt_map_param[key]
            new_opt_op = OptimizerOp(opt_map_grad[key], new_opt)
            self.node_group[key].append(new_opt_op)

    def search_part(self, graph_status, forward_node_list=None, stop_nodes=None, return_graph=False):
        # make coarse graph and initialize dp table, according to node group
        self.graph_status = graph_status
        inverse_graph_nodes = dict()
        if return_graph:
            self.cross_node = {}

        def need_stop(node):
            return stop_nodes is not None and node in stop_nodes

        def forward_dfs(node):
            if node in visited or need_stop(node):
                return
            visited.add(node)
            backbone_node = self.inverse_node_group[node]
            if backbone_node not in inverse_graph_nodes:
                new_node = self.init_graph_node(backbone_node)
                graph_nodes[new_node] = backbone_node
                inverse_graph_nodes[backbone_node] = new_node
            for n in node.inputs:
                forward_dfs(n)
                if not need_stop(n):
                    input_backbone_node = self.inverse_node_group[n]
                    if input_backbone_node is not backbone_node:
                        new_edge = self.init_graph_edge(
                            n, input_backbone_node, backbone_node, return_graph)
                        input_node, output_node = inverse_graph_nodes[
                            input_backbone_node], inverse_graph_nodes[backbone_node]
                        key = (input_node, output_node)
                        new_edge.set_io(input_node, output_node)
                        if key in graph_edges:
                            new_edge = self.combine_edges(
                                graph_edges[key], new_edge)
                        else:
                            input_node.add_output(new_edge)
                            output_node.add_input(new_edge)
                        graph_edges[key] = new_edge

        def wrapped_get_best_key(graph_node, key=None):
            backbone_node = graph_nodes[graph_node]
            best_key = graph_node.get_best_key(key=key)
            node_cur_state_map[backbone_node] = self.node_status_map[best_key[0]]
            node_raw_ctx_map[backbone_node] = best_key[1]

        visited = set()
        graph_nodes = dict()
        graph_edges = dict()
        if forward_node_list is None:
            forward_node_list = graph_status.forward_node_list
        for node in forward_node_list:
            forward_dfs(node)
        for graph_node in graph_nodes:
            # now we simply add allreduce time into computing time
            # TODO: re-think the combination
            graph_node.combine_ar_time(self.overlap)
        if return_graph:
            comp_time = {backbone_node: copy(
                gnode.costs) for backbone_node, gnode in inverse_graph_nodes.items()}
            comm_time = {(graph_nodes[k[0]], graph_nodes[k[1]]): copy(
                gedge.costs) for k, gedge in graph_edges.items()}
        remaining_nodes = set(graph_nodes.keys())
        deleted_nodes = list()
        min_time, best_key, cur_node = self.shrink_graph(
            remaining_nodes, deleted_nodes, graph_edges, 0)
        node_cur_state_map = dict()
        node_raw_ctx_map = dict()
        wrapped_get_best_key(cur_node, best_key)
        for graph_node in deleted_nodes[::-1]:
            wrapped_get_best_key(graph_node)
        self.simulator.write_cache()
        if return_graph:
            for key in comp_time:
                comp_time[key] = comp_time[key][(
                    str(node_cur_state_map[key]), node_raw_ctx_map[key])]
            for key in comm_time:
                pstatus = str(node_cur_state_map[key[0]])
                prawctx = node_raw_ctx_map[key[0]]
                tstatus = str(node_cur_state_map[key[1]])
                trawctx = node_raw_ctx_map[key[1]]
                comm_time[key] = comm_time[key][(
                    (pstatus, prawctx), (tstatus, trawctx))]
            return node_cur_state_map, node_raw_ctx_map, min_time, comp_time, comm_time
        else:
            return node_cur_state_map, node_raw_ctx_map, min_time

    def shrink_graph(self, remain_nodes, del_nodes, cur_graph_edges, depth):
        num_rem = len(remain_nodes)
        while num_rem > 1:
            for graph_node in tuple(remain_nodes):
                if graph_node.has_single_io():
                    self.eliminate_node(
                        graph_node, remain_nodes, del_nodes, cur_graph_edges)
            if num_rem == len(remain_nodes):
                # in this case we are handling complex network such as Bert
                # some work solve this case by using heuristics (TensorOpt[Arxiv2004.10856])
                # we simply use brute-force search
                flag = False
                for graph_node in tuple(remain_nodes):
                    if len(graph_node.inputs) == 0:
                        remain_nodes.remove(graph_node)
                        for out_edge in graph_node.outputs:
                            out_node = out_edge.output
                            out_node.remove_input(out_edge)
                            cur_graph_edges.pop((graph_node, out_node))
                        flag = True
                        break
                assert flag
                best_final_key = None
                best_cur_key = None
                best_last_node = None
                best_del_nodes = None
                best_memo = None
                min_time = None
                for key in graph_node.keys():
                    cur_node_time = graph_node[key]
                    memo = {}
                    new_remain_nodes = deepcopy(remain_nodes, memo)
                    new_graph_edges = deepcopy(cur_graph_edges, memo)
                    for out_edge in graph_node.outputs:
                        out_node = out_edge.output
                        new_out_node = memo[id(out_node)]
                        for out_key in out_node.keys():
                            new_out_node.costs[out_key] += (
                                cur_node_time + out_edge[(key, out_key)])
                    new_del_nodes = list()
                    new_min_time, new_best_key, new_cur_node = self.shrink_graph(
                        new_remain_nodes, new_del_nodes, new_graph_edges, depth+1)
                    if min_time is None or new_min_time < min_time:
                        best_final_key = new_best_key
                        best_cur_key = key
                        best_last_node = new_cur_node
                        best_del_nodes = new_del_nodes
                        best_memo = memo
                        min_time = new_min_time
                del_nodes.append(graph_node)
                graph_node.set_best_keys(
                    {(None, None): best_cur_key}, None, None)
                best_memo[None] = None
                for node in best_del_nodes:
                    ori_node = best_memo[node]
                    del_nodes.append(ori_node)
                    ori_node.copy_best_keys(node, best_memo)
                return min_time, best_final_key, best_memo[best_last_node]
            num_rem = len(remain_nodes)
        cur_node = remain_nodes.pop()
        assert cur_node.inputs == [] and cur_node.outputs == []
        min_time = None
        best_key = None
        for key in cur_node.keys():
            new_time = cur_node[key]
            if min_time is None or new_time < min_time:
                min_time = new_time
                best_key = key
        return min_time, best_key, cur_node

    class GraphNode(object):
        def __init__(self, costs, ar_time, name):
            self.costs = costs
            self.ar_time = ar_time
            self.inputs = []
            self.outputs = []
            self.best_keys = None
            self.name = name

        def add_input(self, edge):
            self.inputs.append(edge)

        def add_output(self, edge):
            self.outputs.append(edge)

        def has_single_io(self):
            return len(self.inputs) <= 1 and len(self.outputs) <= 1

        def keys(self):
            return self.costs.keys()

        def __getitem__(self, index):
            return self.costs[index]

        def redirect_input(self, ori_edge, new_edge):
            index = self.inputs.index(ori_edge)
            self.inputs[index] = new_edge

        def redirect_output(self, ori_edge, new_edge):
            index = self.outputs.index(ori_edge)
            self.outputs[index] = new_edge

        def remove_input(self, edge):
            self.inputs.remove(edge)

        def remove_output(self, edge):
            self.outputs.remove(edge)

        def set_best_keys(self, best_keys, in_node, out_node):
            del self.inputs
            del self.outputs
            self.in_node = in_node
            self.out_node = out_node
            self.best_keys = best_keys

        def copy_best_keys(self, other, memo):
            self.set_best_keys(
                other.best_keys, memo[other.in_node], memo[other.out_node])

        def get_best_key(self, key=None):
            if key is None:
                ind1, ind2 = self.in_node, self.out_node
                if ind1 is not None:
                    ind1 = ind1.best_key
                if ind2 is not None:
                    ind2 = ind2.best_key
                self.best_key = self.best_keys[(ind1, ind2)]
            else:
                self.best_key = key
            del self.best_keys
            return self.best_key

        def combine_ar_time(self, overlap):
            for key in self.ar_time:
                if overlap:
                    self.costs[key] = max(self.costs[key], self.ar_time[key])
                else:
                    self.costs[key] = self.costs[key] + self.ar_time[key]
            del self.ar_time

        def __deepcopy__(self, memo):
            new_node = OptCNNSearching.GraphNode(
                copy(self.costs), None, self.name)
            # explicitly add into memo, or falling into endless loop
            memo[id(self)] = new_node
            new_node.inputs = [deepcopy(inp, memo) for inp in self.inputs]
            new_node.outputs = [deepcopy(outp, memo) for outp in self.outputs]
            memo[new_node] = self
            return new_node

        def __repr__(self):
            return self.name

    class GraphEdge(object):
        def __init__(self, costs):
            self.costs = costs
            # need to initiate the communication time, including split/concat/sum

        def set_io(self, input_node, output_node):
            self.input = input_node
            self.output = output_node

        def __getitem__(self, index):
            return self.costs[index]

        def __deepcopy__(self, memo):
            new_edge = OptCNNSearching.GraphEdge(copy(self.costs))
            memo[id(self)] = new_edge
            new_edge.set_io(deepcopy(self.input, memo),
                            deepcopy(self.output, memo))
            return new_edge

        def __repr__(self):
            return '({}, {})'.format(self.input.name, self.output.name)

    def init_graph_node(self, backbone_node):
        costs = {}
        ar_time = {}
        cur_group_topo = self.node_group[backbone_node]
        for backbone_node_status in self.get_status_cands(backbone_node):
            with self.wrapped_complete_partial_graph(backbone_node, backbone_node_status, self.graph_status) as local_node_cur_state_map:
                repr_node_status = str(backbone_node_status)
                if repr_node_status not in self.node_status_map:
                    self.node_status_map[repr_node_status] = backbone_node_status
                raw_ctxs = self.get_rawctx_cands(
                    backbone_node, backbone_node_status)
                result = self.get_group_compute_time(
                    cur_group_topo, local_node_cur_state_map, raw_ctxs)
                for raw_ctx in raw_ctxs:
                    costs[(repr_node_status, raw_ctx)] = result[raw_ctx]
                    ar_time[(repr_node_status, raw_ctx)] = self.get_group_allreduce_time(
                        cur_group_topo[-1], local_node_cur_state_map, raw_ctx)
        return self.GraphNode(costs, ar_time, backbone_node.name)

    def init_graph_edge(self, cur_node, pre_backbone_node, tar_backbone_node, return_graph=False):
        shape = self.node_to_shape_map[cur_node]
        if return_graph:
            self.cross_node[(pre_backbone_node, tar_backbone_node)] = cur_node
        costs = {}
        for pre_backbone_node_status in self.get_status_cands(pre_backbone_node):
            with self.wrapped_complete_partial_graph(
                    pre_backbone_node, pre_backbone_node_status, self.graph_status) as pre_cur_state_map:
                pre_status = pre_cur_state_map[cur_node]
                for tar_backbone_node_status in self.get_status_cands(tar_backbone_node):
                    with self.wrapped_complete_partial_graph(
                            tar_backbone_node, tar_backbone_node_status, self.graph_status) as tar_cur_state_map:
                        if tar_backbone_node not in tar_cur_state_map:
                            tar_cur_state_map[tar_backbone_node] = tar_backbone_node_status
                        tar_status = tar_cur_state_map[cur_node]
                        pre_ctx_cands = self.get_rawctx_cands(
                            pre_backbone_node, pre_backbone_node_status)
                        tar_ctx_cands = self.get_rawctx_cands(
                            tar_backbone_node, tar_backbone_node_status)
                        for pre_rawctx in pre_ctx_cands:
                            for tar_rawctx in tar_ctx_cands:
                                # here we only consider the forward edges
                                # so we multiply the results with 2
                                # TODO: consider whether and how to express backward edges?
                                comm_time = self.simulator.get_general_comm_time(
                                    pre_status, tar_status, pre_rawctx, tar_rawctx, shape, use_nccl_collectives=self.use_nccl_collectives)
                                if self.debugging:
                                    with open('optcnn.txt', 'a') as fw:
                                        print('comm', comm_time,
                                              cur_node, file=fw, flush=True)
                                costs[((str(pre_backbone_node_status), pre_rawctx),
                                       (str(tar_backbone_node_status), tar_rawctx))] = 2 * comm_time
        return self.GraphEdge(costs)

    def get_group_compute_time(self, group_topo, node_cur_state_map, raw_ctxs):
        # time without allreduce
        local_node_to_shape_map = dict()
        partial_buffer = set()
        for node, value in node_cur_state_map.items():
            if self.node_to_shape_map[node] is None:
                local_node_to_shape_map[node] = self.node_to_shape_map[node]
            else:
                local_node_to_shape_map[node] = self.simulator.get_split_shape(
                    value.state, self.node_to_shape_map[node])
        all_compute_time = 0.
        for node in group_topo:
            if not isinstance(node, OptimizerOp):
                for n in node.inputs:
                    # handle partial -> duplicate communication
                    prev, curr = node_cur_state_map[n], node_cur_state_map[node]
                    if prev.enable_partial and prev.partial > 1 and n in group_topo and (not curr.enable_partial or curr.partial == 1) and not isinstance(node, OnesLikeOp):
                        partial_buffer.add(n)
            if isinstance(node, OptimizerOp):
                for i, n in enumerate(node.inputs):
                    if n.use_indexed_slices:
                        # considering after allgather
                        shape_ind = list(local_node_to_shape_map[n.inputs[1]])
                        shape_value = list(
                            local_node_to_shape_map[n.inputs[0]])
                        partial = node_cur_state_map[n].partial
                        shape_ind[0] *= partial
                        shape_value[0] *= partial
                        sparse_shape = (tuple(shape_ind), tuple(shape_value))
                    else:
                        sparse_shape = None
                    update_time = self.simulator.get_update_time(
                        local_node_to_shape_map[n], sparse_shape=sparse_shape)
                    if self.debugging:
                        with open('optcnn.txt', 'a') as fw:
                            print('update', update_time,
                                  node.optimizer.params[i], file=fw, flush=True)
                    all_compute_time += update_time
            else:
                if isinstance(node, SumOp):
                    input_shapes = []
                    for n in node.inputs:
                        if n.use_indexed_slices:
                            input_shapes.append(
                                (local_node_to_shape_map[n.inputs[1]], local_node_to_shape_map[n.inputs[0]]))
                        else:
                            input_shapes.append(local_node_to_shape_map[n])
                else:
                    input_shapes = [local_node_to_shape_map[n]
                                    for n in node.inputs]
                exe_time = self.simulator.get_node_time(
                    node, input_shapes, local_node_to_shape_map[node])
                if self.debugging:
                    with open('optcnn.txt', 'a') as fw:
                        print('execute', exe_time, node, file=fw, flush=True)
                all_compute_time += exe_time
        results = {}
        for rctx in raw_ctxs:
            comm_time = 0.
            for node in partial_buffer:
                # considering partial for the backward generated sumop
                ori_status = node_cur_state_map[node]
                tar_status = ori_status.remove_partial()
                cur_comm_time = self.simulator.get_general_comm_time(
                    ori_status, tar_status, rctx, rctx, self.node_to_shape_map[node], use_nccl_collectives=self.use_nccl_collectives)
                comm_time += cur_comm_time
                if self.debugging:
                    with open('optcnn.txt', 'a') as fw:
                        print('comm', comm_time,
                              node, file=fw, flush=True)
            results[rctx] = all_compute_time + comm_time * 2
        return results

    def get_group_allreduce_time(self, node, node_cur_state_map, raw_ctx):
        allreduce_time = 0.
        if isinstance(node, OptimizerOp):
            # the node is not optimizer => the group has no parameters
            for i, n in enumerate(node.inputs):
                cur_status = node_cur_state_map[n]
                if cur_status.enable_partial and cur_status.partial > 1:
                    # TODO: now we simply add allreduce time; consider how to overlap?
                    if n.use_indexed_slices:
                        ind_node, val_node = n.inputs[1], n.inputs[0]
                        indices_shape = self.simulator.get_split_shape(
                            node_cur_state_map[ind_node].state, self.node_to_shape_map[ind_node])
                        values_shape = self.simulator.get_split_shape(
                            node_cur_state_map[val_node].state, self.node_to_shape_map[val_node])
                        cur_time = self.simulator.wrapped_get_allgather_time(
                            indices_shape, values_shape, raw_ctx, cur_status)
                    else:
                        shape = self.simulator.get_split_shape(
                            cur_status.state, self.node_to_shape_map[n])
                        cur_time = self.simulator.wrapped_get_allreduce_time(
                            shape, raw_ctx, cur_status)
                    if self.debugging:
                        with open('optcnn.txt', 'a') as fw:
                            print('allreduce', cur_time,
                                  n, file=fw, flush=True)
                    allreduce_time += cur_time
        return allreduce_time

    def eliminate_node(self, graph_node, remaining_nodes, deleted_nodes, graph_edges):
        num_inputs = len(graph_node.inputs)
        num_outputs = len(graph_node.outputs)
        assert num_inputs in (0, 1)
        assert num_outputs in (0, 1)
        if num_inputs == 0 and num_outputs == 0:
            # last single node
            return
        if num_inputs == 1:
            in_edge = graph_node.inputs[0]
            in_node = in_edge.input
            in_keys = in_node.keys()
            graph_edges.pop((in_node, graph_node))
        else:
            in_node = None
            in_keys = [None]
        if num_outputs == 1:
            out_edge = graph_node.outputs[0]
            out_node = out_edge.output
            out_keys = out_node.keys()
            graph_edges.pop((graph_node, out_node))
        else:
            out_node = None
            out_keys = [None]
        best_keys = dict()
        best_costs = dict()
        for in_key in in_keys:
            for out_key in out_keys:
                best_key = None
                min_time = None
                for mid_key in graph_node.keys():
                    new_time = graph_node[mid_key]
                    if in_key is not None:
                        new_time += in_edge[(in_key, mid_key)]
                    if out_key is not None:
                        new_time += out_edge[(mid_key, out_key)]
                    if min_time is None or new_time < min_time:
                        min_time = new_time
                        best_key = mid_key
                best_keys[(in_key, out_key)] = best_key
                best_costs[(in_key, out_key)] = min_time
        if num_inputs == 0:
            for out_key in out_keys:
                out_node.costs[out_key] += best_costs[(None, out_key)]
            out_node.remove_input(out_edge)
        elif num_outputs == 0:
            for in_key in in_keys:
                in_node.costs[in_key] += best_costs[(in_key, None)]
            in_node.remove_output(in_edge)
        else:
            new_edge = self.GraphEdge(best_costs)
            new_edge.set_io(in_node, out_node)
            in_node.redirect_output(in_edge, new_edge)
            out_node.redirect_input(out_edge, new_edge)
            key = (in_node, out_node)
            if key in graph_edges:
                new_edge = self.combine_edges(
                    graph_edges[key], new_edge, existing=True)
            graph_edges[key] = new_edge
        graph_node.set_best_keys(best_keys, in_node, out_node)
        remaining_nodes.remove(graph_node)
        deleted_nodes.append(graph_node)

    def combine_edges(self, ori_edge, new_edge, existing=False):
        in_node, out_node = ori_edge.input, ori_edge.output
        assert in_node == new_edge.input and out_node == new_edge.output
        result_edge = self.GraphEdge(
            {k: v + new_edge[k] for k, v in ori_edge.costs.items()})
        result_edge.set_io(in_node, out_node)
        if existing:
            # new edge already exists in nodes
            in_node.remove_output(new_edge)
            out_node.remove_input(new_edge)
        in_node.redirect_output(ori_edge, result_edge)
        out_node.redirect_input(ori_edge, result_edge)
        return result_edge

    def simulate_time(self, graph_status, best_cur_status, best_raw_ctx):
        self.status_n_ctxs = dict()
        for name in best_cur_status:
            self.status_n_ctxs[name] = (
                best_cur_status[name], best_raw_ctx[name])
        return self.searching(graph_status, None)

    def get_status_cands(self, backbone_node):
        if self.status_n_ctxs is None:
            cur_space = self.search_space[backbone_node]
            cands = cur_space[1]
        else:
            cands = [self.status_n_ctxs[backbone_node.name][0]]
        return cands

    def get_rawctx_cands(self, backbone_node, status):
        if self.status_n_ctxs is None:
            unfixed = self.search_space[backbone_node][0]
            cands = self.device_candidates[status.dev_num] if unfixed else [
                backbone_node.raw_ctx]
        else:
            cands = [self.status_n_ctxs[backbone_node.name][1]]
        return cands
