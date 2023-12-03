""" library to take autodiff and execute a computation graph """
from __future__ import absolute_import

from ..optimizer import OptimizerOp
from .PipelineSend import PipelineSendOp
from .PipelineReceive import PipelineReceiveOp
from .ReduceCommunicate import ReduceCommunicateOp
from .BroadcastCommunicate import BroadcastCommunicateOp

from .executor import SubExecutor


class SubExecutor4Pipe(SubExecutor):
    def __init__(self, name, eval_node_list, config, reserve_opt=True):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        topo_order: list of nodes in topological order
        node_to_shape_map: dict from node to shape of the node
        node_to_arr_maps: a list of [dict from node to ndarray.NDArray allocated for node]
        feed_shapes: shapes of feed_dict from last run(...)
        """
        super().__init__(name, eval_node_list, config)
        assert self.use_p2p
        self.get_partitions(reserve_opt)
        self.get_schedule_for_different_dp()

    def get_partitions(self, reserve_opt=True):
        # naive partition
        layer_indices = self.config.layer_indices
        opt = None
        loss_node = self.config.graph_status.opt.optimizer.loss
        partitions = []
        cur_part = []
        prev_node = None
        pivot = None
        for node in self.computing_nodes[::-1]:
            if isinstance(node, OptimizerOp):
                assert opt is None, 'Optimizer must be unique.'
                opt = node
                loss_node = opt.optimizer.loss
                if reserve_opt:
                    cur_part.append(node)
            else:
                if node is loss_node or \
                    (isinstance(prev_node, PipelineReceiveOp)
                        and (isinstance(node, PipelineSendOp) or node in self.config.all_forward_nodes)
                        and layer_indices[node] != layer_indices[prev_node]) or \
                    (isinstance(prev_node, BroadcastCommunicateOp)
                        and isinstance(node, ReduceCommunicateOp)
                        and layer_indices[node] != layer_indices[prev_node]):
                    partitions.append(cur_part[::-1])
                    cur_part = []
                cur_part.append(node)
            prev_node = node
        partitions.append(cur_part[::-1])
        partitions = partitions[::-1]

        self.opt = opt
        self.partitions = partitions
        # reduce to 2 partitions
        # workaround for strange topology; e.g. bert gpipe 8 workers / gpt2 gpipe 4 or 8 workers
        if len(self.partitions) > 2:
            pivot = None
            for i, part in enumerate(self.partitions):
                cur_forward = False
                for node in part:
                    if node in self.config.all_forward_nodes:
                        cur_forward = True
                        break
                if not cur_forward:
                    pivot = i
                    break
            # print('pivot is', pivot)
            new_partitions = [sum(self.partitions[:pivot], []), sum(
                self.partitions[pivot:], [])]
            self.partitions = new_partitions
        # with open('parts{}.txt'.format(self.config.rank), 'w') as fw:
        #     for parts in self.partitions:
        #         for node in parts:
        #             print(node, node.inputs, file=fw, flush=True)
        #         print(file=fw, flush=True)
        assert len(self.partitions) == 2 or (len(self.partitions) == 1 and self.partitions[0] == []), 'Now only support 2 partitions; got {}.'.format(
            len(self.partitions))

    def get_schedule_for_different_dp(self):
        # we assume that feed values and dataloaders all in the stages with minimum data parallelism degree
        # TODO: re-design dataloaders to release the constraint above
        # TODO: carefully design the loop of schedule to enable validation!!!!
        schedule = [True]
        dp_rank = self.config.dp_rank
        dp_nrank = self.config.dp_nrank
        min_dp_nrank = self.config.min_dp_nrank
        if dp_nrank > min_dp_nrank:
            num1, num2 = dp_nrank, min_dp_nrank
            while num2 != 0:
                num1, num2 = num2, num1 % num2
            num_cycle = dp_nrank // num1
            index = dp_rank
            schedule = []
            while len(schedule) < num_cycle:
                if index < min_dp_nrank:
                    schedule.append(True)
                    index = (index + dp_nrank) - min_dp_nrank
                else:
                    schedule.append(False)
                    index -= min_dp_nrank
        self.schedule = schedule
        self.execution_index = 0

    def step_index(self, steps=None):
        if steps is None:
            # step once
            result = self.schedule[self.execution_index]
            self.execution_index = (
                self.execution_index + 1) % len(self.schedule)
        else:
            assert steps % len(self.schedule) == 0, \
                'Number of steps {} should be the multiple of the length of schedule {}, so that the AllReduce op can be valid.'.format(
                    steps, len(self.schedule))
            # return a schedule list
            result = []
            len_sched = len(self.schedule)
            for _ in range(steps):
                result.append(self.schedule[self.execution_index])
                self.execution_index = (self.execution_index + 1) % len_sched
        return result

    def memory_plan(self):
        raise NotImplementedError

    def run(self, eval_node_list, feed_dicts_list, convert_to_numpy_ret_vals, batch_num=None):
        raise NotImplementedError
