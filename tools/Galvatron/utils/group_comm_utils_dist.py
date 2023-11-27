import torch
from .allgather_utils import gather_from_tensor_model_parallel_region_group
from .parallel_utils import relocate_activations

class CommGroup(object):
    def __init__(self, ranks):
        assert isinstance(ranks, list) or isinstance(ranks, range), 'Rank list or range should be provided to create a CommGroup!'
        self.ranks = sorted(list(set(list(ranks))))
        self.size = len(self.ranks)
        self.group = torch.distributed.new_group(self.ranks)
    def has_rank(self, rank):
        if rank in self.ranks:
            self.intra_group_id = self.ranks.index(rank)
            return True
        return False
    def allgather(self, input):
        return gather_from_tensor_model_parallel_region_group(input, self.group)
    def print(self):
        print(self.ranks, end = ' ')

class SliceFunc(object):
    def __init__(self, slice_num, local_rank):
        self.n = slice_num
        self.local_rank = local_rank
        assert(local_rank < slice_num)
    def __call__(self, input):
        length = len(input)
        step = int(length // self.n)
        return input[int(self.local_rank*step):int((self.local_rank+1)*step)]
    def print(self):
        print("%d/%d"%(self.local_rank, self.n), end = ' ')

def show_groups(groups):
    for group in groups:
        if group is None:
            print('None', end = ' ')
        else:
            group.print()
    print()

def gen_tp_group_dist(tp_size, pp_size, to_print = True, consecutive = True):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    all_tp_groups = []
    dp_size = world_size // tp_size // pp_size
    num_pp_groups = world_size // pp_size
    num_tp_groups = world_size // tp_size

    if consecutive:
        for i in range(num_tp_groups):
            ranks = range(i * tp_size, (i+1) * tp_size)
            group = CommGroup(ranks)
            all_tp_groups.append(group)
            if group.has_rank(rank):
                tp_group = group
    else:
        for i in range(pp_size):
            start_rank = i * num_pp_groups
            end_rank = (i + 1) * num_pp_groups
            for j in range(dp_size):
                ranks = range(start_rank + j, end_rank, dp_size)
                group = CommGroup(ranks)
                all_tp_groups.append(group)
                if group.has_rank(rank):
                    tp_group = group
    
    if rank == 0 and to_print:
        print("TP groups:", end = ' ')
        show_groups(all_tp_groups)
    return tp_group

def gen_dp_group_dist(tp_size, pp_size, to_print = True, consecutive = False):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    all_dp_groups = []
    dp_size = world_size // tp_size // pp_size
    num_pp_groups = world_size // pp_size
    num_dp_groups = world_size // dp_size

    if not consecutive:
        for i in range(pp_size):
            start_rank = i * num_pp_groups
            end_rank = (i + 1) * num_pp_groups
            for j in range(tp_size):
                ranks = range(start_rank + j, end_rank, tp_size)
                group = CommGroup(ranks)
                all_dp_groups.append(group)
                if group.has_rank(rank):
                    dp_group = group
    else:
        for i in range(num_dp_groups):
            ranks = range(i * dp_size, (i+1) * dp_size)
            group = CommGroup(ranks)
            all_dp_groups.append(group)
            if group.has_rank(rank):
                dp_group = group
    
    if rank == 0 and to_print:
        print("DP groups:", end = ' ')
        show_groups(all_dp_groups)
    return dp_group

def gen_pp_group_dist(pp_size, to_print = True):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    all_pp_groups = []
    num_pp_groups = world_size // pp_size
    for i in range(num_pp_groups):
        ranks = range(i, world_size, num_pp_groups)
        group = CommGroup(ranks)
        all_pp_groups.append(group)
        if group.has_rank(rank):
            pp_group = group

    if rank == 0 and to_print:
        print("PP groups:", end = ' ')
        show_groups(all_pp_groups)
    return pp_group

def gen_allgather_group_dist(tp_size_old, tp_size_new, tp_consec_old, tp_consec_new, tp_group_new, pp_size):
    world_size = torch.distributed.get_world_size()
    world_size_per_stage = world_size // pp_size
    case0 = (tp_size_new > tp_size_old)
    case1 = (tp_size_new == tp_size_old and tp_consec_new != tp_consec_old)
    case2 = (tp_size_new < tp_size_old and tp_consec_new != tp_consec_old and tp_size_new > 1 and tp_size_old < world_size_per_stage)
    if case0 or case1 or case2:
        return tp_group_new
    return None

def gen_slice_func_dist(tp_size_old, tp_size_new, tp_consec_old, tp_consec_new, tp_group_old, pp_size):
    world_size = torch.distributed.get_world_size()
    world_size_per_stage = world_size // pp_size
    case0 = (tp_size_new > tp_size_old)
    case1 = (tp_size_new == tp_size_old and tp_consec_new != tp_consec_old)
    case2 = (tp_size_new < tp_size_old and tp_consec_new != tp_consec_old and tp_size_new > 1 and tp_size_old < world_size_per_stage)
    if case0 or case1 or case2:
        slice_num = tp_size_old
        local_rank = tp_group_old.intra_group_id
        return SliceFunc(slice_num, local_rank)
    elif tp_size_new < tp_size_old:
        slice_num = tp_size_old // tp_size_new
        if tp_consec_new and tp_consec_old:
            local_rank = (torch.distributed.get_rank() % world_size_per_stage // tp_size_new) % slice_num
        else:
            local_rank = (torch.distributed.get_rank() % world_size_per_stage % (world_size // pp_size // tp_size_new)) // (world_size // pp_size // tp_size_old)
        return SliceFunc(slice_num, local_rank)
    return None


def get_tp_group_dict_dist(all_tp_sizes, pp_size, consecutive = True):
    tp_sizes_set = list(set(all_tp_sizes))
    tp_group_dict={}
    for tp_size in tp_sizes_set:
        tp_group_dict[tp_size] = gen_tp_group_dist(tp_size, pp_size, to_print=False, consecutive=consecutive)
    return tp_group_dict

def get_dp_group_dict_dist(all_tp_sizes, pp_size, consecutive = False):
    tp_sizes_set = list(set(all_tp_sizes))
    dp_group_dict={}
    for tp_size in tp_sizes_set:
        dp_group_dict[tp_size] = gen_dp_group_dist(tp_size, pp_size, to_print=False, consecutive=consecutive)
    return dp_group_dict

def gen_groups_dist(all_tp_sizes, pp_size, tp_consecutive_flags, show_rank = -1):
    world_size = torch.distributed.get_world_size()
    world_size_per_stage = world_size // pp_size
    for i in range(len(all_tp_sizes)):
        tp_consec = tp_consecutive_flags[i]
        assert tp_consec == 0 or tp_consec == 1
        if all_tp_sizes[i] in [1, world_size_per_stage]:
            tp_consecutive_flags[i] = 1
    tp_groups = []
    dp_groups = []
    allgather_groups = [None]
    slice_funcs = [None]
    pp_group = gen_pp_group_dist(pp_size, to_print=False)
    tp_group_dict_consec = get_tp_group_dict_dist(all_tp_sizes, pp_size, True)
    tp_group_dict_inconsec = get_tp_group_dict_dist(all_tp_sizes, pp_size, False)
    dp_group_dict_consec = get_dp_group_dict_dist(all_tp_sizes, pp_size, True)
    dp_group_dict_inconsec = get_dp_group_dict_dist(all_tp_sizes, pp_size, False)
    for i in range(len(all_tp_sizes)):
        if tp_consecutive_flags[i]:
            tp_groups.append(tp_group_dict_consec[all_tp_sizes[i]])
            dp_groups.append(dp_group_dict_inconsec[all_tp_sizes[i]])
        else:
            tp_groups.append(tp_group_dict_inconsec[all_tp_sizes[i]])
            dp_groups.append(dp_group_dict_consec[all_tp_sizes[i]])
    for i in range(1, len(all_tp_sizes)):
        allgather_groups.append(gen_allgather_group_dist(all_tp_sizes[i-1], all_tp_sizes[i], tp_consecutive_flags[i-1], tp_consecutive_flags[i], tp_groups[i], pp_size))
    for i in range(1, len(all_tp_sizes)):
        slice_funcs.append(gen_slice_func_dist(all_tp_sizes[i-1], all_tp_sizes[i], tp_consecutive_flags[i-1], tp_consecutive_flags[i], tp_groups[i-1], pp_size))
    if show_rank >= 0 and torch.distributed.get_rank() == show_rank:
        print('====================== Galvatron Communication Group ===========================')
        print("TP groups for rank %d (all layers):"%show_rank)
        show_groups(tp_groups)
        print("DP groups for rank %d (all layers):"%show_rank)
        show_groups(dp_groups)
        # print("AllGather groups for rank %d:"%show_rank)
        # show_groups(allgather_groups)
        # print("Slice Funcs for rank %d:"%show_rank)
        # show_groups(slice_funcs)
        print('================================================================================')
    return pp_group, tp_groups, dp_groups, allgather_groups, slice_funcs