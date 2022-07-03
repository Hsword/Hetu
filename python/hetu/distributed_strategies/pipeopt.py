from .optcnn import OptCNNSearching
from .base import BaseSearchingStrategy, OpLayer
from ..context import DeviceGroup
from ..ndarray import gpu, rgpu
import os
import os.path as osp


class PipeOptSearching(OptCNNSearching):
    def __init__(self, feed_shapes, save_dir=None, load_dir=None, batch_num_factor=1, ignore_batch_size=[], **kargs):
        # simple strategy combining pipeline and optcnn
        BaseSearchingStrategy.__init__(self, feed_shapes, **kargs)
        self.status_n_ctxs = None
        self.debugging = False
        self.use_nccl_collectives = True
        self.save_dir = save_dir
        self.load_dir = load_dir
        self.ignore_batch_size = ignore_batch_size
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        self.batch_num = batch_num_factor * len(self.all_devices)

    def searching(self, graph_status, memory_pool):
        # make coarse graph and initialize dp table, according to node group
        # self.init_pipeline_states(graph_status)
        num_group = len(self.coarse_topo_order)
        self.level_worker_num = self.check_valid_topo()

        mp_candidates = [1]
        while mp_candidates[-1] < self.num_ctxs:
            mp_candidates.append(mp_candidates[-1] * 2)
        self.all_node_to_shape_map = {}
        all_feed_shapes = {}
        graph_status.extend_oplayers()
        for cand in mp_candidates:
            key = self.num_ctxs // cand
            new_feed_shapes = {}
            valid = True
            for node, shape in self.feed_shapes.items():
                if node in self.ignore_batch_size:
                    new_feed_shapes[node] = shape
                else:
                    assert shape[0] % self.batch_size == 0
                    if shape[0] % cand != 0:
                        valid = False
                        break
                    cur_shape = list(shape)
                    cur_shape[0] //= cand
                    new_feed_shapes[node] = tuple(cur_shape)
            if valid:
                self.all_node_to_shape_map[key] = self.infer_global_shapes(
                    new_feed_shapes, graph_status)
                all_feed_shapes[key] = new_feed_shapes

        self.init_hybrid_pipeline_states(graph_status, all_feed_shapes)
        graph_status.shrink_oplayers()

        # we assume there's only 2 levels: devices in the same machine, and multiple machines
        # same as simple pipe
        level_worker_num = self.level_worker_num
        keys = []
        m1 = 1
        m2 = 1
        while m1 <= level_worker_num[1]:
            keys.append((m1, m2))
            m1 *= 2
        m1 //= 2
        m2 *= 2
        while m2 <= level_worker_num[2]:
            keys.append((m1, m2))
            m2 *= 2
        num_all_workers = level_worker_num[1] * level_worker_num[2]

        best_result = None
        best_split = None
        best_key = None
        for key in keys:
            cur_worker_num = key[0] * key[1]
            per_machine_part = level_worker_num[1] // key[0]
            parts = num_all_workers // cur_worker_num
            if parts > num_group:
                # number of workers larger than number of layers
                continue
            accum_time = self.all_accum_times[cur_worker_num]
            cross_time = self.all_cross_times[cur_worker_num]
            time_level = {(i, j): accum_time[j] - accum_time[i-1]
                          for i in range(1, num_group+1) for j in range(i, num_group+1)}
            comm_times = {}
            for i in range(num_group):
                for diff in range(1, num_group - i):
                    j = i + diff
                    pnode = self.coarse_topo_order[i]
                    tnode = self.coarse_topo_order[j]
                    tkey = (pnode, tnode)
                    if diff == 1:
                        comm_times[(i + 1, j + 1)] = 0.
                        if tkey in cross_time:
                            comm_times[(i + 1, j + 1)] = cross_time[tkey]
                    else:
                        comm_times[(i + 1, j + 1)] = comm_times[(i + 1, j)]
                        for m in range(i, j):
                            cnode = self.coarse_topo_order[m]
                            ckey = (cnode, tnode)
                            if ckey in cross_time:
                                comm_times[(i + 1, j + 1)] += cross_time[ckey]
                    time_level[(i + 1, j + 1)] += comm_times[(i + 1, j + 1)]
            cur_coeff = self.batch_num // cur_worker_num - 1
            maxcomp_dp = {}
            comm_dp = {}
            cur_split = {}
            for p in range(parts):
                for j in range(1 + p, num_group + 1):
                    if p == 0:
                        maxcomp_dp[(p, j)] = time_level[(1, j)]
                        comm_dp[(p, j)] = 0.
                    else:
                        cur_ans = -1
                        cur_res = None
                        res_max_comp = None
                        res_comm = None
                        for s in range(p, j):
                            cur_comm = self.get_group_comm_time(
                                1, s, j, cur_worker_num, p % per_machine_part == 0) + comm_dp[(p-1, s)]
                            cur_max_comp = max(
                                maxcomp_dp[(p-1, s)], time_level[(s+1, j)], cur_comm)
                            temp_result = cur_max_comp
                            if cur_res is None or temp_result < cur_res:
                                cur_ans = s
                                cur_res = temp_result
                                res_max_comp = cur_max_comp
                                res_comm = cur_comm
                        assert cur_ans > 0
                        cur_split[(p, j)] = cur_ans
                        maxcomp_dp[(p, j)] = res_max_comp
                        comm_dp[(p, j)] = res_comm
            cur_best_result = cur_coeff * maxcomp_dp[(parts-1, num_group)] + comm_dp[(
                parts-1, num_group)] + time_level[(1, num_group)]
            # consider bubbles
            if best_result is None or cur_best_result < best_result:
                best_result = cur_best_result
                best_split = cur_split
                best_key = key

        cur_worker_num = best_key[0] * best_key[1]
        self.num_parts = self.num_ctxs // cur_worker_num
        best_status_map = self.all_status_maps[cur_worker_num]
        best_rawctx_map = self.all_rawctx_maps[cur_worker_num]
        parts = self.num_parts
        cur_end = num_group
        layers = []
        for p in range(1, parts)[::-1]:
            cur_start = best_split[(p, cur_end)]
            layers.append((cur_start+1, cur_end))
            cur_end = cur_start
        layers.append((1, cur_end))
        layers = layers[::-1]

        node_raw_ctx_map = {}
        gpu_lind = 0
        gpu_rind = best_key[0]
        host_lind = 0
        host_rind = best_key[1]
        hosts = list(self.workers.keys())
        final_splits = {}

        original_ctx_group = [rgpu(hostname, i) for hostname in hosts[host_lind:host_rind]
                              for i in range(gpu_lind, gpu_rind)]

        for p, (lind, rind) in enumerate(layers):
            cur_ctxs = [rgpu(hostname, i) for hostname in hosts[host_lind:host_rind]
                        for i in range(gpu_lind, gpu_rind)]
            gpu_lind = gpu_rind
            if gpu_lind == self.level_worker_num[1]:
                gpu_lind = 0
                host_lind = host_rind
                host_rind = host_lind + best_key[1]
            gpu_rind = gpu_lind + best_key[0]
            for i in range(lind - 1, rind):
                cur_node = self.coarse_topo_order[i]
                node_raw_ctx_map[cur_node.name] = self.remap_ctxs(
                    best_rawctx_map[cur_node], original_ctx_group, cur_ctxs)
            final_splits[(lind, rind)] = DeviceGroup(tuple(cur_ctxs))
        print('Pipeline partition:', final_splits)
        self.simulator.write_cache()
        print('The simulated optimized execution time is: {}ms.'.format(best_result))
        return {node.name: value for node, value in best_status_map.items()}, node_raw_ctx_map

    def init_hybrid_pipeline_states(self, graph_status, new_feed_shapes):
        self.global_search_space = self.search_space
        self.global_device_candidates = self.device_candidates
        self.global_node_to_shape_map = self.node_to_shape_map
        self.global_feed_shapes = self.feed_shapes
        self.all_accum_times = {}
        self.all_cross_times = {}
        self.all_cross_nodes = {}
        self.all_status_maps = {}
        self.all_rawctx_maps = {}
        self.node_status_map = dict()
        self.add_group_optimizer_node(graph_status.opt)
        for dev_num, mapping in self.all_node_to_shape_map.items():
            self.search_space, self.device_candidates = self.form_candidates(
                dev_num)
            self.node_to_shape_map = mapping
            self.feed_shapes = new_feed_shapes[dev_num]
            from_load = False
            try:
                best_cur_status, best_raw_ctx, _ = BaseSearchingStrategy.load_json(
                    self, osp.join(self.load_dir, 'config_d{}.json'.format(dev_num)))
                self.status_n_ctxs = dict()
                for name in best_cur_status:
                    self.status_n_ctxs[name] = (
                        best_cur_status[name], best_raw_ctx[name])
                status_map, rawctx_map, min_time, comp_time, cross_time = self.search_part(
                    graph_status, return_graph=True)
                from_load = True
            except:
                status_map, rawctx_map, min_time, comp_time, cross_time = self.search_part(
                    graph_status, return_graph=True)
            if self.save_dir is not None and not from_load:
                BaseSearchingStrategy.save_json(self, {node.name: st for node, st in status_map.items()}, {node.name: rc for node, rc in rawctx_map.items()}, osp.join(
                    self.save_dir, 'config_d{}.json'.format(dev_num)))
            accum_time = [0.]
            for node in self.coarse_topo_order:
                cur_group_time = comp_time[node]
                accum_time.append(accum_time[-1] + cur_group_time)
            self.all_accum_times[dev_num] = accum_time
            self.all_cross_times[dev_num] = cross_time
            self.all_cross_nodes[dev_num] = self.cross_node
            self.all_status_maps[dev_num] = status_map
            self.all_rawctx_maps[dev_num] = rawctx_map
        del self.cross_node

    def form_candidates(self, dev_num):
        cur_search_space = {}
        cur_dev_cands = {}
        for node, cands in self.global_search_space.items():
            new_cands = [cand for cand in cands[1] if cand.dev_num <= dev_num]
            cur_search_space[node] = (cands[0], new_cands)
        for key, value in self.global_device_candidates.items():
            cur_level_worker_num = [None, min(self.level_worker_num[1], key)]
            cur_level_worker_num.append(key // cur_level_worker_num[1])
            max_dev_id = cur_level_worker_num[1]
            valid_hosts = list(self.workers.keys())[:cur_level_worker_num[2]]

            def is_valid(ctx):
                return ctx.hostname in valid_hosts and ctx.device_id < max_dev_id
            new_cands = []
            for cand in value:
                ctxs = cand.workers[0]
                if not isinstance(ctxs, tuple):
                    ctxs = (ctxs,)
                if cand.mp_dev_num <= dev_num and all([is_valid(ctx) for ctx in ctxs]):
                    new_cands.append(cand)
            cur_dev_cands[key] = new_cands
        return cur_search_space, cur_dev_cands

    def get_group_comm_time(self, start_index, middle_index, ending_index, key, cross_hosts=False):
        all_comm_time = 0.
        cross_node = self.all_cross_nodes[key]
        status_map = self.all_status_maps[key]
        rawctx_map = self.all_rawctx_maps[key]
        shape_map = self.all_node_to_shape_map[key]
        if key <= self.level_worker_num[1]:
            if cross_hosts:
                hostnames = list(self.workers.keys())[:2]
                from_devices = [rgpu(hostnames[0], i) for i in range(key)]
                to_devices = [rgpu(hostnames[1], i) for i in range(key)]
            else:
                from_devices = [gpu(i) for i in range(key)]
                to_devices = [gpu(i+key) for i in range(key)]
        else:
            num_hosts_per_part = key // self.level_worker_num[1]
            assert cross_hosts
            hostnames = list(self.workers.keys())[:2*num_hosts_per_part]
            from_devices = [rgpu(host, i) for host in hostnames[:num_hosts_per_part]
                            for i in range(self.level_worker_num[1])]
            to_devices = [rgpu(host, i) for host in hostnames[num_hosts_per_part:]
                          for i in range(self.level_worker_num[1])]
        original_ctx_group = from_devices
        for lind in range(start_index - 1, middle_index):
            for rind in range(middle_index, ending_index):
                pre_backbone_node = self.coarse_topo_order[lind]
                tar_backbone_node = self.coarse_topo_order[rind]
                key = (pre_backbone_node, tar_backbone_node)
                if key in cross_node:
                    cur_node = cross_node[key]
                    pre_backbone_node_status = status_map[pre_backbone_node]
                    tar_backbone_node_status = status_map[tar_backbone_node]
                    pre_rawctx = rawctx_map[pre_backbone_node]
                    tar_rawctx = rawctx_map[tar_backbone_node]
                    cur_fdevices = self.remap_ctxs(
                        pre_rawctx, original_ctx_group, from_devices)
                    cur_tdevices = self.remap_ctxs(
                        tar_rawctx, original_ctx_group, to_devices)
                    with self.wrapped_complete_partial_graph(
                            pre_backbone_node, pre_backbone_node_status, self.graph_status) as pre_cur_state_map:
                        pre_status = pre_cur_state_map[cur_node]
                        with self.wrapped_complete_partial_graph(
                                tar_backbone_node, tar_backbone_node_status, self.graph_status) as tar_cur_state_map:
                            if tar_backbone_node not in tar_cur_state_map:
                                tar_cur_state_map[tar_backbone_node] = tar_backbone_node_status
                            if isinstance(pre_backbone_node, OpLayer) and pre_backbone_node.output == cur_node:
                                tar_status = tar_cur_state_map[pre_backbone_node]
                            else:
                                tar_status = tar_cur_state_map[cur_node]
                            # here we only consider the forward edges
                            # so we multiply the results with 2
                            # TODO: consider whether and how to express backward edges?
                            comm_time = self.simulator.get_general_comm_time(
                                pre_status, tar_status, cur_fdevices, cur_tdevices, shape_map[cur_node], use_nccl_collectives=self.use_nccl_collectives)
                            all_comm_time += comm_time
        return 2 * all_comm_time

    def remap_ctxs(self, dev, original_ctx_group, ctx_group):
        workers = dev.workers[0]
        if not isinstance(workers, tuple):
            workers = (workers,)
        return DeviceGroup(tuple([ctx_group[original_ctx_group.index(c)] for c in workers]))

    def save_json(self, best_cur_status, best_raw_ctx, save_path):
        import json
        contents = self.save_contents(
            best_cur_status, best_raw_ctx, self.search_space)
        contents.insert(0, self.num_parts)
        with open(save_path, 'w') as fw:
            json.dump(contents, fw, indent=4)

    def load_json(self, load_path):
        def eval_str(x):
            if isinstance(x, str):
                x = eval(x)
            return x

        import json
        with open(load_path, 'r') as fr:
            contents = json.load(fr)
        self.num_parts = eval_str(contents[0])
        return self.load_contents(contents[1:])

    def simulate_time(self, *args):
        return 0.
