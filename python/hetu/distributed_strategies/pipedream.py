from ..context import DeviceGroup, NodeStatus
from ..optimizer import OptimizerOp
from ..ndarray import gpu, rgpu
from .base import BaseSearchingStrategy


class PipeDreamSearching(BaseSearchingStrategy):
    # this class is for pipeline parallel (specifically, PipeDream partition strategy)
    # PipeDream only consider using data parallel in lower level
    def __init__(self, feed_shapes, batch_num_factor=1, **kargs):
        super().__init__(feed_shapes, **kargs)
        self.batch_num = batch_num_factor * len(self.all_devices)
        self.debugging = False

    def searching(self, graph_status, memory_pool):
        self.init_pipeline_states(graph_status)
        node_raw_ctx_map, result = self.pipedream_simple()
        print('The simulated optimized execution time is: {}ms.'.format(result))
        self.simulator.write_cache()
        return {node.name: NodeStatus(1, partial_or_node=node) for node in self.search_space}, {node.name: node_raw_ctx_map[node] for node in self.search_space}

    def pipedream(self):
        # DEPRECATED! Bugs may exist; not considering bubbles.
        num_group = len(self.coarse_topo_order)
        level_worker_num = self.check_valid_topo()

        # we assume there's only 2 levels: devices in the same machine, and multiple machines
        time_level1 = {}
        time_level2 = {}
        result_level1 = {}
        result_level2 = {}
        best_split_level1 = {}
        best_split_level2 = {}
        all_level = 2  # TODO: rewrite to support any number of levels

        # level 1
        worker_num = level_worker_num[1]
        for i in range(1, num_group + 1):
            for j in range(i, num_group + 1):
                main_compute = self.accum_time[j] - self.accum_time[i-1]
                cur_worker_num = 1
                time_level1[(i, j, cur_worker_num)] = main_compute
                cur_worker_num += 1
                while cur_worker_num <= worker_num:
                    if self.overlap:
                        time_level1[(i, j, cur_worker_num)] = max(
                            main_compute / cur_worker_num, self.get_range_allreduce_time(i, j, (cur_worker_num,)))
                    else:
                        time_level1[(i, j, cur_worker_num)] = main_compute / cur_worker_num + \
                            self.get_range_allreduce_time(
                                i, j, (cur_worker_num,))
                    cur_worker_num += 1
        cur_worker_num = 1
        while cur_worker_num <= worker_num:
            for i in range(1, num_group + 1):
                for j in range(i, num_group + 1):
                    key = (i, j, cur_worker_num)
                    cur_result = time_level1[key]
                    best_split_level1[key] = (i - 1, cur_worker_num)
                    for s in range(i, j):
                        for m in range(1, cur_worker_num):
                            temp_result = max(result_level1[(
                                i, s, cur_worker_num - m)], self.get_group_comm_time(i, s, j), time_level1[(s + 1, j, m)])
                            if temp_result < cur_result:
                                cur_result = temp_result
                                best_split_level1[key] = (s, m)
                    result_level1[key] = cur_result
            cur_worker_num += 1

        # level 2
        worker_num = level_worker_num[2]
        for i in range(1, num_group + 1):
            for j in range(i, num_group + 1):
                main_compute = result_level1[(i, j, level_worker_num[1])]
                cur_worker_num = 1
                time_level2[(i, j, cur_worker_num)] = main_compute
                cur_worker_num += 1
                while cur_worker_num <= worker_num:
                    if self.overlap:
                        time_level2[(i, j, cur_worker_num)] = max(main_compute / cur_worker_num, self.get_range_allreduce_time(
                            i, j, (level_worker_num[1], cur_worker_num)))
                    else:
                        time_level2[(i, j, cur_worker_num)] = main_compute / cur_worker_num + self.get_range_allreduce_time(
                            i, j, (level_worker_num[1], cur_worker_num))
                    cur_worker_num += 1
        cur_worker_num = 1
        while cur_worker_num <= worker_num:
            for i in range(1, num_group + 1):
                for j in range(i, num_group + 1):
                    key = (i, j, cur_worker_num)
                    cur_result = time_level2[key]
                    best_split_level2[key] = (i - 1, cur_worker_num)
                    for s in range(i, j):
                        for m in range(1, cur_worker_num):
                            temp_result = max(result_level2[(
                                i, s, cur_worker_num - m)], self.get_group_comm_time(i, s, j, True), time_level2[(s + 1, j, m)])
                            if temp_result < cur_result:
                                cur_result = temp_result
                                best_split_level2[key] = (s, m)
                    result_level2[key] = cur_result
            cur_worker_num += 1

        result = result_level2[(1, num_group, level_worker_num[2])]

        # assign devices according to best splits
        assigned_devices = self.assign_ctxs(
            best_split_level1, best_split_level2, level_worker_num, num_group)
        node_raw_ctx_map = dict()
        for key, value in assigned_devices.items():
            lind, rind = key
            for i in range(lind - 1, rind):
                node_raw_ctx_map[self.coarse_topo_order[i]] = value
        return node_raw_ctx_map, result

    def pipedream_simple(self):
        # ! currently DP cannot be applied for communication

        # simplified version of pipedream search
        # use same degree for data parallel
        num_group = len(self.coarse_topo_order)
        level_worker_num = self.check_valid_topo()

        # we assume there's only 2 levels: devices in the same machine, and multiple machines
        keys = []
        m1 = 1
        while m1 <= level_worker_num[1]:
            m2 = 1
            while m2 <= level_worker_num[2]:
                keys.append((m1, m2))
                m2 *= 2
            m1 *= 2
        num_all_workers = level_worker_num[1] * level_worker_num[2]

        time_level = {}
        for i in range(1, num_group + 1):
            for j in range(i, num_group + 1):
                time_level[(i, j)] = self.accum_time[j] - self.accum_time[i-1]

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
            if self.overlap:
                cur_time_level = {(i, j): max(time_level[(i, j)], self.get_range_allreduce_time(
                    i, j, key)) for i in range(1, num_group+1) for j in range(i, num_group+1)}
            else:
                cur_time_level = {(i, j): time_level[(i, j)] + self.get_range_allreduce_time(
                    i, j, key) for i in range(1, num_group+1) for j in range(i, num_group+1)}
            cur_coeff = self.batch_num // cur_worker_num - 1
            maxcomp_dp = {}
            comm_dp = {}
            cur_split = {}
            for p in range(parts):
                for j in range(1 + p, num_group + 1):
                    if p == 0:
                        maxcomp_dp[(p, j)] = cur_time_level[(1, j)]
                        comm_dp[(p, j)] = 0.
                    else:
                        cur_ans = -1
                        cur_res = None
                        res_max_comp = None
                        res_comm = None
                        for s in range(p, j):
                            cur_comm = self.get_group_comm_time(
                                1, s, j, p % per_machine_part == 0) + comm_dp[(p-1, s)]
                            cur_max_comp = max(
                                maxcomp_dp[(p-1, s)], cur_time_level[(s+1, j)], cur_comm)
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
                parts-1, num_group)] + cur_time_level[(1, num_group)]
            # consider bubbles
            if best_result is None or cur_best_result < best_result:
                best_result = cur_best_result
                best_split = cur_split
                best_key = key

        cur_worker_num = best_key[0] * best_key[1]
        parts = num_all_workers // cur_worker_num
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
        for p, (lind, rind) in enumerate(layers):
            value = DeviceGroup([rgpu(hostname, i) for hostname in hosts[host_lind:host_rind]
                                 for i in range(gpu_lind, gpu_rind)])
            gpu_lind = gpu_rind
            if gpu_lind == level_worker_num[1]:
                gpu_lind = 0
                host_lind = host_rind
                host_rind = host_lind + best_key[1]
            gpu_rind = gpu_lind + best_key[0]
            for i in range(lind - 1, rind):
                node_raw_ctx_map[self.coarse_topo_order[i]] = value
            final_splits[(lind, rind)] = value
        print('Pipeline partition:', final_splits)
        return node_raw_ctx_map, best_result

    def get_range_allreduce_time(self, start_index, ending_index, num_devs):
        if num_devs in ((1,), (1, 1)):
            return 0.
        all_allreduce_time = 0.
        for ind in range(start_index - 1, ending_index):
            if len(num_devs) == 1:
                devices = [gpu(i) for i in range(num_devs[0])]
            else:
                hostnames = list(self.workers.keys())[:num_devs[1]]
                devices = [rgpu(hostname, i)
                           for hostname in hostnames for i in range(num_devs[0])]
            group_topo = self.group_topo[self.coarse_topo_order[ind]]
            cur_allreduce_time = 0.
            opt_node = group_topo[-1]
            if isinstance(opt_node, OptimizerOp):
                for i, n in enumerate(opt_node.inputs):
                    if n.use_indexed_slices:
                        ind_node, val_node = n.inputs[1], n.inputs[0]
                        indices_shape = self.node_to_shape_map[ind_node]
                        values_shape = self.node_to_shape_map[val_node]
                        cur_time = self.simulator.get_allgather_time(
                            indices_shape, values_shape, devices)
                    else:
                        cur_time = self.simulator.get_allreduce_time(
                            self.node_to_shape_map[n], devices)
                    if self.debugging:
                        with open('pipedream.txt', 'a') as fw:
                            print('allreduce', cur_time, n, file=fw, flush=True)
                    cur_allreduce_time += cur_time
            all_allreduce_time += cur_allreduce_time
        return all_allreduce_time

    def get_group_comm_time(self, start_index, middle_index, ending_index, cross_hosts=False):
        all_comm_time = 0.
        if cross_hosts:
            hostnames = list(self.workers.keys())[:2]
            from_device, to_device = rgpu(
                hostnames[0], 0), rgpu(hostnames[1], 0)
        else:
            from_device, to_device = gpu(0), gpu(1)
        for lind in range(start_index - 1, middle_index):
            for rind in range(middle_index, ending_index):
                key = (self.coarse_topo_order[lind],
                       self.coarse_topo_order[rind])
                if key in self.cross_shape:
                    cur_comm_time = self.simulator.get_comm_time(
                        from_device, to_device, self.cross_shape[key])
                    all_comm_time += cur_comm_time
                    if self.debugging:
                        with open('pipedream.txt', 'a') as fw:
                            print('comm', cur_comm_time, key[0], key[1], file=fw, flush=True)
        return 2 * all_comm_time

    def assign_ctxs(self, best_split_level1, best_split_level2, level_worker_num, num_group):
        def get_level1_device(lind, rind, num_workers):
            if num_workers == 1:
                return {(lind, rind): 1}
            elif num_workers == 0:
                return {}
            layer_id, cur_worker_num = best_split_level1[(
                lind, rind, num_workers)]
            cur_dict = get_level1_device(
                lind, layer_id, num_workers - cur_worker_num)
            cur_dict[(layer_id + 1, rind)] = cur_worker_num
            return cur_dict

        ending = num_group
        cur_split_point = best_split_level2[(1, ending, level_worker_num[2])]
        ori_worker_num = level_worker_num[2]
        all_devices = {}
        hosts = list(self.workers.keys())[::-1]
        host_ind = 0
        while cur_split_point is not None:
            layer_id, cur_worker_num = cur_split_point
            cur_level1_device = get_level1_device(
                layer_id + 1, ending, level_worker_num[1])
            gpu_ind = 0
            host_rind = host_ind + cur_worker_num
            for key in cur_level1_device:
                gpu_rind = gpu_ind + cur_level1_device[key]
                all_devices[key] = DeviceGroup(
                    [rgpu(hostname, i) for hostname in hosts[host_ind:host_rind] for i in range(gpu_ind, gpu_rind)])
                gpu_ind = gpu_rind
            assert gpu_ind == level_worker_num[1]
            host_ind = host_rind
            ending = layer_id
            rest_worker_num = ori_worker_num - cur_worker_num
            if rest_worker_num <= 1:
                cur_split_point = None
            else:
                cur_split_point = best_split_level2[(
                    1, ending, rest_worker_num)]
            ori_worker_num = cur_worker_num
        if rest_worker_num == 1:
            cur_level1_device = get_level1_device(
                1, ending, level_worker_num[1])
            gpu_ind = 0
            host_rind = host_ind + 1
            for key in cur_level1_device:
                gpu_rind = gpu_ind + cur_level1_device[key]
                all_devices[key] = DeviceGroup(
                    [rgpu(hostname, i) for hostname in hosts[host_ind:host_rind] for i in range(gpu_ind, gpu_rind)])
                gpu_ind = gpu_rind
            assert gpu_ind == level_worker_num[1]
            host_ind = host_rind
        else:
            assert rest_worker_num == 0 and layer_id == 0
        assert host_ind == len(hosts)
        print('Pipeline partition:', sorted(
            all_devices.items(), key=lambda x: x[0]))
        return all_devices

    def simulate_time(self, graph_status, best_cur_status, best_raw_ctx):
        # now only check level 1 time; we consider level 2 is totally the same
        self.init_pipeline_states(graph_status)
        all_devices = dict()
        prev_devs = None
        cur_start = None
        cur_ending = None
        cur_worker_num = None
        for i, cur_devs in enumerate(best_raw_ctx.values()):
            if cur_devs != prev_devs:
                cur_ending = i
                all_devices[(cur_start, cur_ending)] = prev_devs
                cur_start = i + 1
                prev_devs = cur_devs
        cur_ending = len(best_raw_ctx)
        all_devices[(cur_start, cur_ending)] = prev_devs
        all_devices.pop((None, 0))

        max_time = 0.
        all_comp_time = 0.
        all_comm_time = 0.
        prev = None
        for (start, end), devs in all_devices.items():
            cur_time = self.accum_time[end] - self.accum_time[start - 1]
            num_devs = len(devs)
            assert cur_worker_num in (None, num_devs)
            cur_worker_num = num_devs
            if num_devs > 1:
                all_reduce_time = self.get_range_allreduce_time(
                    start, end, (num_devs,))
                if self.overlap:
                    cur_time = max(cur_time, all_reduce_time)
                else:
                    cur_time += all_reduce_time
            all_comp_time += cur_time
            max_time = max(max_time, cur_time)
            if prev is not None:
                comm_time = self.get_group_comm_time(prev, start-1, end)
                all_comm_time += comm_time
            prev = start
        return all_comp_time + all_comm_time + max_time * (self.batch_num // cur_worker_num - 1)
