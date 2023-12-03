from ..context import DeviceGroup, NodeStatus
from ..ndarray import rgpu
from .base import BaseSearchingStrategy


class GPipeSearching(BaseSearchingStrategy):
    # this class is for pipeline parallel (specifically, GPipe partition strategy)
    def searching(self, graph_status, memory_pool):
        self.init_pipeline_states(graph_status)
        num_group = len(self.coarse_topo_order)

        # ALL CODES ABOVE ARE THE SAME WITH PIPEDREAM

        self.all_workers = []
        for key, value in self.simulator.nccl_profiler.workers.items():
            for i in range(value):
                self.all_workers.append(rgpu(key, i))
        num_workers = len(self.all_workers)
        assert num_group >= num_workers, 'Number of layers must be larger than number of workers! Got {} layers and {} workers.'.format(
            num_group, num_workers)

        # use dp to get least variance
        # each elements represent 0 -> j layers in (i+1) devices
        # deduction: dp[i][j] = min_k (dp[i-1][k] + (k+1 ~ j)^2)
        dp = [[None for _ in range(num_group)] for _ in range(num_workers)]
        place = [[0 for _ in range(num_group)] for _ in range(num_workers)]
        # initialize
        for j in range(num_group):
            # in 1 device
            dp[0][j] = (self.accum_time[j+1] - self.accum_time[0]) ** 2
        for i in range(1, num_workers):
            # now try (i+1) devices
            for j in range(i, num_group):
                min_result = None
                for k in range(i-1, j):
                    cur_result = dp[i-1][k] + \
                        (self.accum_time[j+1] - self.accum_time[k+1]) ** 2
                    if min_result is None or cur_result < min_result:
                        min_result = cur_result
                        place[i][j] = k  # record split point
                dp[i][j] = min_result
        final_result = dp[num_workers-1][num_group-1]
        points = [num_group - 1]
        for i in range(num_workers-1, 0, -1):
            points.insert(0, place[i][points[0]])

        # with open('test_strategy/test.txt', 'w') as fw:
        #     print(' '.join([str(x) for x in self.accum_time]), file=fw, flush=True)
        #     print(num_workers, num_group, file=fw, flush=True)
        #     print(' '.join([str(x) for x in points]), file=fw, flush=True)
        #     print(final_result, file=fw, flush=True)

        node_raw_ctx_map = dict()
        cur_start = 0
        for i, point in enumerate(points):
            # cur_start -> point in a device
            cur_dev = DeviceGroup(self.all_workers[i])
            cur_ending = point + 1
            for j in range(cur_start, cur_ending):
                node_raw_ctx_map[self.coarse_topo_order[j]] = cur_dev
            cur_start = cur_ending
        print('Pipeline ending partition:', points)
        return {node.name: NodeStatus(1, partial_or_node=node) for node in self.search_space}, {node.name: node_raw_ctx_map[node] for node in self.search_space}
