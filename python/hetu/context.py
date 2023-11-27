from __future__ import annotations

from .ndarray import rcpu, rgpu, DLContext, is_gpu_ctx
import contextlib
import yaml
import socket
import psutil
import re
import numpy as np
from collections import defaultdict
from copy import copy, deepcopy
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple, Union, Iterator, Dict, List, Set, DefaultDict, Any
    from .gpu_ops.Node import Op
    from .gpu_ops.AllReduceCommunicate import AllReduceCommunicateP2POp
    from .gpu_ops.AllGatherCommunicate import AllGatherCommunicateOp
    from .gpu_ops.ReduceScatterCommunicate import ReduceScatterCommunicateOp
    from .gpu_ops.BroadcastCommunicate import BroadcastCommunicateOp
    from .gpu_ops.ReduceCommunicate import ReduceCommunicateOp
    from .communicator.mpi_nccl_comm import NCCL_Communicator
    ContextRepr = Union[str, DLContext]
    ContextUnitRepr = Union[ContextRepr, Tuple[ContextRepr, ...]]
    ContextReprList = Union[ContextUnitRepr, List[ContextUnitRepr]]
    ContextUnit = Union[DLContext, Tuple[DLContext, ...]]


class DeviceGroup(object):
    def __init__(
        self,
        ctxs: Union[DeviceGroup, str, ContextReprList],
    ) -> None:
        self._contexts = self.parse_contexts(ctxs)
        self.get_servers_n_workers()
        if self.worker_num > 0:
            self._is_mp = isinstance(self._workers[0], tuple)
            self._mp_dev_num = len(self._workers[0]) if self._is_mp else 1
            for worker in self._workers[1:]:
                assert self._is_mp == isinstance(worker, tuple)
                if self._is_mp:
                    assert self._mp_dev_num == len(worker), \
                        'Now only support same model parallel in data parallel.'
        else:
            self._is_mp = False
            self._mp_dev_num = None

    @classmethod
    def parse_contexts(
        cls,
        ctxs: Union[DeviceGroup, str, ContextReprList],
    ) -> List[ContextUnit]:
        if isinstance(ctxs, DeviceGroup):
            return ctxs
        if isinstance(ctxs, str):
            ctxs = re.split(';|,| +', ctxs.lower())
        if not isinstance(ctxs, list):
            ctxs = [ctxs]
        new_ctxs = []
        for c in ctxs:
            if isinstance(c, tuple):
                c = tuple([ccc for ccc in [cls.str2ctx(cc)
                                           for cc in c] if ccc is not None])
                if len(c) == 1:
                    c = c[0]
            else:
                c = cls.str2ctx(c)
            if c is not None:
                new_ctxs.append(c)
        return new_ctxs

    @classmethod
    def str2ctx(
        cls,
        c: ContextRepr,
    ) -> DLContext:
        if isinstance(c, str):
            c = c.lower().split(':')
            assert c[-2] in ('cpu', 'gpu'), 'Context invalid: %s' % c
            hostname = 'localhost' if len(c) == 2 else c[0]
            idx = int(c[-1])
            c = rcpu(hostname, idx) if c[-2] == 'cpu' else rgpu(hostname, idx)
        assert isinstance(c, DLContext), 'Context invalid: %s' % c
        return c

    def index(
        self,
        ctx: ContextUnit,
    ) -> int:
        return self._contexts.index(ctx)

    def set_index(
        self,
        ctx: DLContext,
    ) -> None:
        mp_index = -1
        dp_index = -1
        for i, c in enumerate(self.workers):
            if isinstance(c, tuple):
                if ctx in c:
                    mp_index = c.index(ctx)
                    dp_index = i
            elif ctx == c:
                dp_index = i
        self.mp_index = mp_index
        self.dp_index = dp_index
        self.local_mp = mp_index >= 0
        self.local_dp = dp_index >= 0

    def relocalize(self) -> None:
        for c in self._contexts:
            if isinstance(c, tuple):
                for cc in c:
                    cc.relocalize()
            else:
                c.relocalize()

    def __getitem__(self, key: int) -> ContextUnit:
        return self._workers[key]

    def __iter__(self) -> Iterator[ContextUnit]:
        return iter(self._contexts)

    def __len__(self) -> int:
        return len(self._contexts)

    def check_mp_num(self, mp_dev_num: int) -> None:
        assert mp_dev_num == self._mp_dev_num

    @property
    def is_mp(self) -> bool:
        return self._is_mp

    @property
    def mp_dev_num(self) -> int:
        return self._mp_dev_num

    @property
    def worker_num(self) -> int:
        return len(self._workers)

    @property
    def server_num(self) -> int:
        return len(self._servers)

    @property
    def workers(self) -> Tuple[ContextUnit]:
        return self._workers

    @property
    def servers(self) -> Tuple[DLContext]:
        return self._servers

    def get_servers_n_workers(self) -> None:
        workers = []
        servers = []
        for ctx in self._contexts:
            if isinstance(ctx, tuple) or is_gpu_ctx(ctx):
                workers.append(ctx)
            else:
                servers.append(ctx)
        self._workers: Tuple[ContextUnit] = tuple(workers)
        self._servers: Tuple[DLContext] = tuple(servers)

    def get_target_workers(
        self,
        other_ctx: DeviceGroup,
        mp_index: Optional[int] = None,
    ) -> List[ContextUnit]:
        # get target workers that will be used in send/receive
        # handle the case with different degrees of data parallelisms
        rank, nrank = other_ctx.dp_index, other_ctx.worker_num
        worker_num = self.worker_num
        workers = self.workers
        num1, num2 = max(nrank, worker_num), min(nrank, worker_num)
        while num2 != 0:
            num1, num2 = num2, num1 % num2
        gcd = num1
        num_cycle = worker_num // gcd
        index = rank % worker_num
        new_workers = []
        # we traverse workers in this case
        while len(new_workers) < num_cycle:
            new_worker = workers[index]
            if mp_index is not None:
                new_worker = new_worker[mp_index]
            new_workers.append(new_worker)
            index = (index + nrank) % worker_num
        return new_workers

    def __repr__(self) -> str:
        result = 'DeviceGroup('
        for c in self._contexts:
            result += '%s, ' % c
        result += ')'
        return result

    def full_repr(self) -> str:
        return str([tuple(cc.full_repr() for cc in c) if isinstance(c, tuple) else c.full_repr() for c in self._contexts])

    def __hash__(self) -> int:
        if not hasattr(self, 'hash'):
            self.hash = hash(tuple(self._contexts))
        return self.hash

    def __eq__(self, other: DeviceGroup) -> bool:
        return hash(self) == hash(other)

    def get_sorted(self) -> DeviceGroup:
        # this function is only for make group allreduce communicator
        # each device in self._contexts MUST NOT be tuple
        return DeviceGroup(sorted(self._contexts, key=lambda x: '{}:{}:{}'.format(x.hostname, x.device_type, x.device_id)))

    def cur_worker(self) -> ContextUnit:
        if self.dp_index:
            return None
        else:
            return self._contexts[self.dp_index]

    def get_only(self) -> DLContext:
        assert self.server_num + self.worker_num == 1
        if self.server_num == 1:
            return self.servers[0]
        else:
            res = self.workers[0]
            if isinstance(res, tuple):
                assert len(res) == 1
                res = res[0]
            return res


class ContextStack(object):
    def __init__(self) -> None:
        self._stack: List[DeviceGroup] = []

    def peek(self) -> Optional[DeviceGroup]:
        return self._stack[-1] if self._stack else None

    def push(self, ctx: DeviceGroup) -> None:
        return self._stack.append(ctx)

    def pop(self) -> DeviceGroup:
        self._stack.pop()


_default_ctx_stack = ContextStack()


class NodeStatus(object):
    def __init__(
        self,
        dev_num: int,
        state: Union[None, Dict[int, int], Tuple[int, ...]] = None,
        partial_or_node: Union[Op, bool] = False
    ) -> None:
        assert isinstance(dev_num, int), 'Device number should an integer.'
        enable_partial = partial_or_node
        if not isinstance(partial_or_node, bool):
            enable_partial = partial_or_node.enable_distributed_partial()
        assert isinstance(
            enable_partial, bool), 'Partial or not should be specified!'
        if state is not None:
            if not isinstance(state, dict):
                state = {i: v for i, v in enumerate(state) if v != 1}
            else:
                state = {i: v for i, v in sorted(state.items()) if v != 1}
        self._enable_partial = enable_partial
        self._state = state
        self._duplicate = None
        self._partial = None
        self._order = None
        self._device_num = dev_num
        self._valid_state = False
        self._valid_all = False
        self.try_get_duplicate()
        if self._device_num == 1:
            self.set_one()

    def get(self) -> Tuple[Dict[int, int], int]:
        return self.state, self.duplicate

    def get_all(self) -> Tuple[Dict[int, int], int, Tuple[int, ...]]:
        return self.state, self.duplicate, self.order

    def is_dist(self) -> bool:
        return self._device_num > 1

    @ property
    def dev_num(self) -> int:
        return self._device_num

    @ property
    def state(self) -> Dict[int, int]:
        return self._state

    @ property
    def duplicate(self) -> int:
        return self._duplicate

    @ property
    def order(self) -> Tuple[int, ...]:
        return self._order

    @ property
    def partial(self) -> int:
        return self._partial

    @ property
    def enable_partial(self) -> bool:
        return self._enable_partial

    def get_dim(self, ind: int) -> int:
        if ind == -1:
            return self.duplicate
        elif ind == -2:
            return self.partial
        else:
            return self.state.get(ind, 1)

    def try_get_duplicate(self) -> None:
        if self._valid_state:
            return

        def try_set_duplicate(duplicate: int) -> None:
            assert self_duplicate in (None, duplicate)
            self._duplicate = duplicate
            self._valid_state = True

        def try_set_partial(partial: int) -> None:
            assert self._partial in (None, partial)
            if not self.enable_partial:
                assert partial in (1, None)
            self._partial = partial
            self._valid_state = True
        self_state, self_duplicate = self.get()
        if self_state is not None:
            temp = int(np.prod(list(self_state.values()), dtype=int))
            assert self._device_num % temp == 0
            temp_duplicate = self._device_num // temp
            if not self._enable_partial:
                try_set_duplicate(temp_duplicate)
            else:
                if self._partial is not None:
                    temp_duplicate //= self._partial
                    try_set_duplicate(temp_duplicate)
                elif self._duplicate is not None:
                    temp_partial = temp_duplicate // self._duplicate
                    try_set_partial(temp_partial)
                elif temp_duplicate == 1:
                    try_set_partial(1)
                    try_set_duplicate(1)

    def set_from_combine(
        self,
        other: NodeStatus,
        deduce_order: bool,
        *src2dst: Tuple[Union[int, List[int]], int]
    ):
        if deduce_order:
            self.set_order(other.combine_order(*src2dst))
        else:
            self.set_state(*other.combine_state(*src2dst))

    def set_state(
        self,
        state: Union[None, Dict[int, int], Tuple[int, ...]] = None,
        duplicate: Optional[int] = None,
        partial: Optional[int] = None,
    ) -> None:
        self_state = self._state
        if state is not None:
            if not isinstance(state, dict):
                state = {i: v for i, v in enumerate(state) if v != 1}
            else:
                state = {i: v for i, v in sorted(state.items()) if v != 1}
            assert self_state in (state, None)
            if self_state is None:
                self._state = state
        if duplicate is not None:
            self.set_duplicate(duplicate)
        if partial is not None:
            self.set_partial(partial)
        if duplicate is None and partial is None:
            self.try_get_duplicate()

    def set_duplicate(self, duplicate: Optional[int] = None) -> None:
        if duplicate is not None:
            assert self._duplicate in (None, duplicate)
            self._duplicate = duplicate
            if self._enable_partial:
                self.try_get_duplicate()

    def set_partial(self, partial: Optional[int] = None) -> None:
        if not self.enable_partial:
            assert partial == 1
        elif partial is not None:
            assert self._partial in (None, partial)
            self._partial = partial
            self.try_get_duplicate()

    def set_order(self, order: Optional[Tuple[int, ...]] = None) -> None:
        if order is not None:
            order = list(order)
            for i, o in list(enumerate(order))[::-1]:
                if self.get_dim(o) in (1, None):
                    order.pop(i)
            order = tuple(order)
            assert self._order in (order, None)
            self._order = order

    def set_one(self) -> None:
        self_state, self_duplicate, self_order = self.get_all()
        assert self_state is None or all(
            [s == 1 for s in self_state.values()])
        assert self_duplicate in (None, 1)
        assert self_order in (None, ())
        self._state = {}
        self._duplicate = 1
        self._order = ()
        if self._enable_partial:
            self._partial = 1
        self._valid_state = True
        self._valid_all = True

    def copy_state_from(self, other: NodeStatus) -> None:
        self.set_state(other.state)
        self.set_duplicate(other.duplicate)
        if self.enable_partial:
            # TODO: check carefully
            self.set_partial(other.partial)

    def copy_order_from(self, other: NodeStatus) -> None:
        self.set_order(other.order)

    def copy_from(self, other: NodeStatus, copy_order: bool) -> None:
        if copy_order:
            self.copy_order_from(other)
        else:
            self.copy_state_from(other)

    def valid_state(self) -> bool:
        self_state, self_duplicate = self.get()
        if not self._enable_partial:
            self._partial = 1
        if self._valid_state:
            return True
        if self_state is None \
                or self_duplicate is None \
                or (self._enable_partial and self._partial is None):
            return False
        else:
            self._valid_state = True
            return True

    def valid_all(self) -> bool:
        if self._valid_all:
            # the one-device case has already set _valid_all to True
            return True
        if not self.valid_state():
            return False
        self_state, self_duplicate, self_order = self.get_all()
        has_dup = self_duplicate > 1
        has_par = self._enable_partial and self._partial > 1
        if not self._enable_partial:
            self._partial = 1
        if self_order is None:
            dims = len(self_state) + has_dup + has_par
            if dims == 0:
                self_order = ()
            elif dims == 1:
                self_order = tuple(self_state.keys()) + \
                    (-1,) * has_dup + (-2,) * has_par
            else:
                return False
            self._order = self_order
            return True
        else:
            err_msg = 'Status not valid: {}'.format(self)
            assert len(self_order) == len(self_state) + \
                has_dup + has_par, err_msg
            for key in self_state:
                assert key in self_order, err_msg
            if has_dup:
                assert -1 in self_order, err_msg
            if has_par:
                assert -2 in self_order, err_msg
            self._valid_all = True
            return True

    def valid(self, include_order: bool) -> bool:
        if include_order:
            return self.valid_all()
        else:
            return self.valid_state()

    def get_default_order(self) -> None:
        self_state, self_duplicate = self.get()
        order = tuple(sorted(self_state.keys()))
        if self_duplicate > 1:
            order = (-1,) + order
        if self._enable_partial and self._partial > 1:
            order = (-2,) + order
        self._order = order

    def check_state(self, max_dim: int, check_order: bool) -> None:
        # only allow dimensions lower than max_dim to split
        if check_order:
            assert (self.dev_num == 1 and self.order == ()) or all(
                [o < max_dim for o in self.order])
        else:
            assert (self.dev_num == 1 and self.state == {}) or all(
                [o < max_dim for o in self.state])

    def map_dev_to_index(self, global_index: int, containing_duplicate: bool = False) -> Dict[int, int]:
        cur_state_index = {}
        for cur_order in self.order[::-1]:
            ts = self.get_dim(cur_order)
            if cur_order >= 0 or containing_duplicate:
                cur_state_index[cur_order] = global_index % ts
            global_index //= ts
        return cur_state_index

    def get_loop_sizes(self) -> Tuple[int, ...]:
        loop_sizes = [1]
        for rord in self.order[::-1]:
            temp_size = loop_sizes[0] * self.get_dim(rord)
            loop_sizes.insert(0, temp_size)
        loop_sizes.pop(0)
        return loop_sizes

    def exchange_state(self, n1: int, n2: int) -> Tuple[Optional[Dict[int, int]], Optional[int], Optional[int]]:
        state = copy(self.state)
        duplicate = self.duplicate
        partial = self.partial

        if n1 == -1:
            v1 = duplicate
            duplicate = 1
        elif n1 == -2:
            if partial is None:
                v1 = 1
            else:
                v1 = partial
                partial = 1
        else:
            v1 = state.pop(n1, 1)

        if n2 == -1:
            v2 = duplicate
            duplicate = v1
        elif n2 == -2:
            if partial is None:
                v2 = 1
            else:
                v2 = partial
                partial = v1
        else:
            v2 = state.pop(n2, 1)
            if v1 > 1:
                state[n2] = v1

        if n1 == -1:
            duplicate = v2
        elif n1 == -2:
            partial = v2
        else:
            if v2 > 1:
                state[n1] = v2

        return state, duplicate, partial

    def exchange_order(self, n1: int, n2: int) -> Tuple[int]:
        ori_order = self.order
        new_order = list(self.order)
        if n1 in ori_order:
            i1 = ori_order.index(n1)
            new_order[i1] = n2
        if n2 in ori_order:
            i2 = ori_order.index(n2)
            new_order[i2] = n1
        return tuple(new_order)

    @staticmethod
    def static_combine_state(
        state: Dict[int],
        duplicate: int,
        partial: Optional[int],
        *src2dst: Tuple[Union[int, List[int]], int],
    ) -> Tuple[Optional[Dict[int, int]], Optional[int], Optional[int]]:
        for src, dst in src2dst:
            value = 1
            if isinstance(src, int):
                src = [src]
            for s in src:
                assert s != dst
                if s == -2:
                    if partial is not None:
                        value *= partial
                    partial = None
                elif s == -1:
                    value *= duplicate
                    duplicate = 1
                else:
                    value *= state.pop(s, 1)
                    for k in sorted(list(state.keys())):
                        if k > s:
                            state[k-1] = state.pop(k)
            if dst == -2:
                if partial is None:
                    partial = value
                else:
                    partial *= value
            elif dst == -1:
                duplicate *= value
            else:
                for s in src:
                    if s >= 0 and dst > s:
                        dst -= 1
                if dst not in state:
                    state[dst] = value
                else:
                    state[dst] *= value
        return state, duplicate, partial

    def combine_state(
        self,
        *src2dst: Tuple[Union[int, List[int]], int],
    ) -> Tuple[Optional[Dict[int, int]], Optional[int], Optional[int]]:
        state = copy(self.state)
        duplicate = self.duplicate
        partial = self.partial
        return self.static_combine_state(state, duplicate, partial, *src2dst)

    @staticmethod
    def static_combine_order(
        self_order: Tuple[int, ...],
        *src2dst: Tuple[Union[int, List[int]], int],
    ) -> Tuple[int]:
        def safe_index(dim: int) -> Optional[int]:
            if dim in self_order:
                return self_order.index(dim)
        self_order = list(self_order)
        for src, dst in src2dst:
            if isinstance(src, int):
                src = [src]
            inds = [safe_index(s) for s in src] + [safe_index(dst)]
            inds = sorted([ind for ind in inds if ind is not None])
            if len(inds) > 0:
                for i, ind in enumerate(inds[1:]):
                    assert ind == inds[0] + i + \
                        1, 'Cannot combine dimensions not adjacent! Got dims {} in order {}.'.format(
                            inds, self_order)
                for ind in inds[1:][::-1]:
                    self_order.pop(ind)
                self_order[inds[0]] = dst
                for i, o in enumerate(self_order):
                    if o > 0:
                        for s in src:
                            if s >= 0 and o > s:
                                o -= 1
                        self_order[i] = o
        return tuple(self_order)

    def combine_order(
        self,
        *src2dst: Tuple[Union[int, List[int]], int],
    ) -> Tuple[int]:
        return self.static_combine_order(self.order, *src2dst)

    def reduce_state(self, dim: int) -> Tuple[Optional[Dict[int, int]], Optional[int], Optional[int]]:
        state = copy(self.state)
        duplicate = self.duplicate
        partial = self.partial
        if dim == -1:
            duplicate = 1
        elif dim == -2:
            if partial is not None:
                partial = 1
        else:
            state.pop(dim, 1)
        return state, duplicate, partial

    def reduce_order(self, dim: int) -> Tuple[int]:
        order = list(self.order)
        if dim in order:
            order.remove(dim)
        return tuple(order)

    def get_combine_from(
        self,
        status: NodeStatus,
        deduce_order: bool,
        *src2dst: Tuple[Union[int, List[int]], int],
    ):
        if deduce_order:
            if status.valid_all():
                self.set_order(status.combine_order(*src2dst))
        else:
            if status.valid_state():
                self.set_state(*status.combine_state(*src2dst))

    def remove_partial(self) -> NodeStatus:
        # return a node status without partial
        # here the new status is a copy,
        if self._enable_partial:
            new_status = NodeStatus(self.dev_num)
            valid_state = self.valid_state()
            valid_order = self.valid_all()
            if valid_state:
                new_status.set_state(*self.combine_state((-2, -1)))
                if valid_order:
                    new_status.set_order(self.combine_order((-2, -1)))
            return new_status
        else:
            return self

    def __repr__(self) -> str:
        if self._enable_partial:
            partial = '({})'.format(self.partial)
        else:
            partial = ''
        return '({}, {}{}, {})'.format(self.state, self.duplicate, partial, self.order)

    def __eq__(self, other: NodeStatus) -> bool:
        if self is other:
            return True
        assert self.valid_all() and other.valid_all(), 'Cannot check equal if not valid.'
        return self.enable_partial == other.enable_partial \
            and self.state == other.state \
            and self.duplicate == other.duplicate \
            and (not self.enable_partial or self.partial == other.partial) \
            and self.order == other.order

    def effect_equal(self, other: Optional[NodeStatus]) -> bool:
        if self is other:
            return True
        if other is None:
            return False
        assert self.valid_all() and other.valid_all(), 'Cannot check equal if not valid.'
        return self.state == other.state \
            and self.duplicate == other.duplicate \
            and (self.partial == other.partial or ({None, 1} == set([self.partial, other.partial]))) \
            and self.order == other.order

    def value_equal(self, state: Dict[int, int], duplicate: int, partial: int, order: Tuple[int, ...]) -> bool:
        return self.state == state \
            and self.duplicate == duplicate \
            and (self.partial == partial or set([self.partial, partial]) == {1, None}) \
            and self.order == order

    def __hash__(self) -> int:
        return hash(id(self))

    def content_hash(self) -> int:
        assert self.valid_all()
        if not hasattr(self, '_content_hash'):
            self._content_hash = hash(str(self))
        return self._content_hash

    def check_combine(self, other: NodeStatus, *src2dst: Tuple[Union[int, List[int]], int]) -> bool:
        state, duplicate, partial = self.combine_state(*src2dst)
        order = self.combine_order(*src2dst)
        return other.value_equal(state, duplicate, partial, order)

    def check_reduce_dim(self, other: NodeStatus, dim: int) -> bool:
        state, duplicate, partial = self.reduce_state(dim)
        order = self.reduce_order(dim)
        return other.value_equal(state, duplicate, partial, order)

    def check_allreduce(self, other: NodeStatus) -> bool:
        return self.get_dim(-2) > 1 and self.check_combine(other, (-2, -1))

    def check_allgather(self, other: NodeStatus) -> bool:
        return self.get_dim(0) > 1 and self.check_combine(other, (0, -1))

    def check_reducescatter(self, other: NodeStatus) -> bool:
        return self.get_dim(-2) > 1 and self.check_combine(other, (-2, 0))

    def check_broadcast(self, other: NodeStatus) -> bool:
        return other.get_dim(-1) > 1 and other.check_reduce_dim(self, -1)

    def check_reduce(self, other: NodeStatus) -> bool:
        return self.get_dim(-2) > 1 and self.check_reduce_dim(other, -2)

    def get_devices_by_dim(
        self,
        context: DeviceGroup,
        dim: int,
        dp_index: Optional[int] = None,
        mp_index: Optional[int] = None,
        rdc_ctx: Optional[DeviceGroup] = None,
        return_device_group: bool = True,
    ) -> Union[Optional[DeviceGroup], Tuple[Optional[DeviceGroup], DLContext]]:
        if dp_index is None:
            dp_index = context.dp_index
        if mp_index is None:
            mp_index = context.mp_index
        devices = []
        if dim in self.order:
            idx = self.order.index(dim)
            interval = 1
            for cur_order in self.order[idx+1:]:
                interval *= self.get_dim(cur_order)
            macro_interval = interval * self.get_dim(dim)
            start = mp_index - mp_index % macro_interval + mp_index % interval
            worker = context.workers[dp_index]
            for ind in range(start, start + macro_interval, interval):
                devices.append(worker[ind])
            if rdc_ctx is not None:
                rdc_worker = rdc_ctx.workers[dp_index]
                if isinstance(rdc_worker, DLContext):
                    assert mp_index // macro_interval == 0
                    rdc_dev = rdc_worker
                else:
                    rdc_dev = rdc_worker[mp_index // macro_interval]
        if len(devices) <= 1:
            devices = None
        elif return_device_group:
            devices = DeviceGroup(devices)
        if rdc_ctx is not None:
            return devices, rdc_dev
        else:
            return devices


def get_current_context() -> Optional[DeviceGroup]:
    return _default_ctx_stack.peek()


@ contextlib.contextmanager
def context(ctx):
    try:
        ctx = DeviceGroup(ctx)
        _default_ctx_stack.push(ctx)
        yield ctx
    finally:
        _default_ctx_stack.pop()


def get_launch_config_by_traverse_nodes(
    node_list: List[Op],
    default_ctx: Union[DLContext, DeviceGroup],
    pipeline: Optional[str]
) -> Tuple[bool, bool, Dict[Op, Optional[str]], Set[DLContext], int, int]:
    def gcd(a: int, b: int) -> int:
        if a < b:
            return gcd(b, a)
        if b == 0 or a == b:
            return a
        return gcd(b, a % b)

    def traverse_dfs(node: Op) -> None:
        from .gpu_ops.Dispatch import DispatchOp, DispatchGradientOp
        from .optimizer import OptimizerOp
        from .dataloader import DataloaderOp
        if node in node_strategy:
            return
        strategy = None
        cur_worker_num = 1 if node.raw_ctx is None else node.raw_ctx.worker_num
        if not isinstance(node, DataloaderOp):
            nonlocal min_worker_num
            nonlocal lcm_worker_num
            if pipeline:
                min_worker_num = min(cur_worker_num, min_worker_num)
                gcd_worker_num = gcd(cur_worker_num, lcm_worker_num)
                lcm_worker_num = cur_worker_num * lcm_worker_num // gcd_worker_num
            else:
                # different data parallel degree only allowed in pipeline parallelism
                assert cur_worker_num == min_worker_num
        if node.raw_ctx is not None and node.raw_ctx.server_num > 0 and cur_worker_num > 0:
            strategy = 'PS'
        elif node.raw_ctx is not None and cur_worker_num > 1:
            strategy = 'AllReduce'
        node_strategy[node] = strategy
        if not isinstance(node, (DispatchOp, DispatchGradientOp, OptimizerOp)):
            for ctx in node.raw_ctx:
                if isinstance(ctx, tuple):
                    devices.update(ctx)
                else:
                    devices.add(ctx)
        for n in node.inputs:
            traverse_dfs(n)

    node_strategy = dict()
    devices = set()
    for ctx in default_ctx:
        if isinstance(ctx, tuple):
            devices.update(ctx)
        else:
            devices.add(ctx)
    min_worker_num = default_ctx.worker_num
    lcm_worker_num = min_worker_num  # for pipeline parallelism
    launchPS = default_ctx.server_num > 0
    launchMPI = (not launchPS) and min_worker_num > 1
    for node in node_list:
        traverse_dfs(node)
    launchPS = launchPS or any([x == 'PS' for x in node_strategy.values()])
    launchMPI = launchMPI or any(
        [x == 'AllReduce' for x in node_strategy.values()])
    return launchMPI, launchPS, node_strategy, devices, min_worker_num, lcm_worker_num


class GraphStatus(object):
    def __init__(self, node_list: List[Op]) -> None:
        # node lists and optimizer
        from .optimizer import OptimizerOp
        self.node_list: List[Op] = node_list
        opt = None
        for node in self.node_list:
            if isinstance(node, OptimizerOp) and hasattr(node.optimizer, 'backward2forward'):
                opt = node.optimizer
                break
        self.opt = opt
        if opt is not None:
            self.bf_map = self.opt.backward2forward
        self.forward_node_list: List[Op] = self.node_list.copy() if opt is None \
            else [opt.loss]

        # node status
        # in mp if parse_dispatch or complete_graph are called
        self.model_parallel: bool = False
        # save nodes' current state
        self.node_cur_state_map: Dict[Op, NodeStatus] = {}
        # save nodes' target state
        self.node_tar_state_map: DefaultDict[Op, Dict[Op, NodeStatus]] = \
            defaultdict(dict)

        # op layers
        self.init_oplayers()
        self.extended: bool = False
        # self.extended_opt = None

    def parse_graph_with_dispatch(self) -> Tuple[Dict[Op, NodeStatus], DefaultDict[Op, Dict[Op, NodeStatus]]]:
        # DEPRECATED! Not maintained; bugs exist.
        # return node-state map, state count
        # the dispatch ops will be removed
        from .dataloader import DataloaderOp
        from .optimizer import OptimizerOp
        from .gpu_ops.Dispatch import DispatchOp, DispatchGradientOp
        from .gpu_ops.Variable import PlaceholderOp

        def remove_dispatch(node: Op) -> None:
            # TODO: handle consecutive dispatch ops
            if node in visited:
                return
            visited.add(node)
            dev_num = node.raw_ctx.mp_dev_num
            if not isinstance(node, (DataloaderOp, OptimizerOp, DispatchOp, DispatchGradientOp)):
                # placeholder op with only dispatch output will not get here
                self.node_cur_state_map[node] = NodeStatus(
                    dev_num, partial_or_node=node)
                if not self.node_cur_state_map[node].valid_all():
                    invalid_states[self.node_cur_state_map[node]] = node
            for i, n in enumerate(node.inputs):
                if isinstance(n, DispatchOp):
                    real_node = n.inputs[0]
                    if n not in dispatch_to_state_map:
                        dispatch_to_state_map[n] = NodeStatus(dev_num, n.parts)
                        if not dispatch_to_state_map[n].valid_all():
                            invalid_states[dispatch_to_state_map[n]] = None
                    if isinstance(real_node, (PlaceholderOp, DataloaderOp)):
                        if real_node not in self.node_cur_state_map:
                            # placeholder op use the first met status as default status
                            # the raw ctx should be corresponding to current state
                            self.node_cur_state_map[real_node] = dispatch_to_state_map[n]
                            visited.add(real_node)
                        elif dispatch_to_state_map[n] != self.node_cur_state_map[real_node]:
                            # dataloader op only support one output now
                            assert isinstance(real_node, PlaceholderOp)
                            self.node_tar_state_map[real_node][node] = dispatch_to_state_map[n]
                    else:
                        remove_dispatch(real_node)
                        self.node_tar_state_map[real_node][node] = dispatch_to_state_map[n]
                    node.inputs[i] = real_node
                elif isinstance(n, DispatchGradientOp):
                    real_node = n.inputs[0]
                    assert not isinstance(
                        real_node, PlaceholderOp), 'Should not get here. Please debug!'
                    remove_dispatch(real_node)
                    remove_dispatch(n.inputs[1])
                    if n not in dispatch_to_state_map:
                        dispatch_to_state_map[n] = self.node_cur_state_map[n.inputs[1]].remove_partial(
                        )
                    if not isinstance(node, OptimizerOp):
                        # if DispatchGradientOp appears before OptimizerOp,
                        # the parameter should set the current state instead of target state,
                        # so we ignore the DispatchGradientOp before OptimizerOp.
                        self.node_tar_state_map[real_node][node] = dispatch_to_state_map[n]
                    node.inputs[i] = real_node
                else:
                    remove_dispatch(n)
                    if n in self.node_cur_state_map \
                            and self.node_cur_state_map[n].enable_partial \
                            and not isinstance(node, (DispatchOp, DispatchGradientOp)):
                        self.node_tar_state_map[n][node] = \
                            self.node_cur_state_map[n].remove_partial()

        self.model_parallel = True
        visited: Set[Op] = set()
        invalid_states: Dict[NodeStatus, Optional[Op]] = {}
        dispatch_to_state_map: Dict[Union[DispatchOp,
                                          DispatchGradientOp], NodeStatus] = {}
        for node in self.node_list:
            remove_dispatch(node)

        self.infer_states(invalid_states)
        return self.node_cur_state_map, self.node_tar_state_map

    def complete_state_map_with_partial_information(self, prune=True) -> None:
        # given current state of the node,
        # add target state to the forward graph
        from .dataloader import DataloaderOp
        from .gpu_ops.Variable import PlaceholderOp
        from .gpu_ops.EmbeddingLookUp import EmbeddingLookUp

        def determine_state(node: Op) -> None:
            if node in visited:
                return
            visited.add(node)
            dev_num = node.raw_ctx.mp_dev_num
            if not isinstance(node, DataloaderOp):
                if node not in self.node_cur_state_map:
                    self.node_cur_state_map[node] = NodeStatus(
                        dev_num, partial_or_node=node)
                    for n in node.inputs:
                        if isinstance(n, DataloaderOp):
                            if n not in self.node_cur_state_map:
                                self.node_cur_state_map[n] = NodeStatus(
                                    dev_num)
                            else:
                                node.raw_ctx.check_mp_num(
                                    self.node_cur_state_map[n].dev_num)
                        elif n in self.node_cur_state_map:
                            # here means n is backbone node
                            # handle partial -> duplicate in linear/conv -> relu
                            prev_node_status = self.node_cur_state_map[n]
                            cur_node_status = self.node_cur_state_map[node]
                            if prev_node_status.partial != cur_node_status.partial:
                                self.node_tar_state_map[n][node] = prev_node_status.remove_partial(
                                )
                else:
                    # already specified
                    for n in node.inputs:
                        new_state = NodeStatus(dev_num)
                        if not new_state.valid_all():
                            invalid_states[new_state] = None
                        if isinstance(n, PlaceholderOp) and n not in self.node_cur_state_map \
                                and (isinstance(node, EmbeddingLookUp) or not n.is_embed):
                            assert n.raw_ctx == node.raw_ctx
                            self.node_cur_state_map[n] = new_state
                        elif isinstance(n, DataloaderOp):
                            if n not in self.node_cur_state_map:
                                self.node_cur_state_map[n] = new_state
                            else:
                                assert new_state.dev_num == self.node_cur_state_map[n].dev_num
                        else:
                            self.node_tar_state_map[n][node] = new_state
                if not self.node_cur_state_map[node].valid_all():
                    invalid_states[self.node_cur_state_map[node]] = node
            for n in node.inputs:
                determine_state(n)

        self.model_parallel = True
        visited = set()
        invalid_states = {}

        # specify all status of nodes in forward phase
        for node in self.forward_node_list:
            determine_state(node)

        # make all forward status valid
        self.infer_states(invalid_states, self.forward_node_list)

        # deduce status of nodes in backward phase
        self.deduce_backward_status(visited)

        # validation
        self.infer_states({}, self.node_list)  # validation check

        if prune:
            # improve status
            self.prune_status()

        # from .gpu_ops.executor import find_topo_sort
        # with open('allstatus.txt', 'w') as fw:
        #     for node in find_topo_sort(self.node_list):
        #         print(node, node.inputs, self.node_cur_state_map.get(
        #             node, None), self.node_tar_state_map.get(node, None), node.raw_ctx, file=fw, flush=True)

    def deduce_backward_status(self, visited):
        from .optimizer import OptimizerOp
        from .gpu_ops.Sum import SumOp

        def dfs(node: Op) -> None:
            if node in visited:
                return
            visited.add(node)
            if isinstance(node, OptimizerOp):
                for n in node.inputs:
                    dfs(n)
            else:
                fnode, i = bf_map[node]
                self.node_cur_state_map[node] = fnode.deduce_generated_backward_nodes_states(
                    self.get_input_statuses(fnode), self.node_cur_state_map[fnode], i)
                for n in node.inputs:
                    dfs(n)
                    if n in bf_map:
                        # backward node
                        forward_node = bf_map[n][0]
                        if forward_node is not fnode:
                            self.node_tar_state_map[n][node] = \
                                self.node_cur_state_map[fnode].remove_partial()
                    elif fnode in self.node_tar_state_map[n]:
                        self.node_tar_state_map[n][node] = self.node_tar_state_map[n][fnode]

        bf_map = self.bf_map
        for node in self.node_list:
            dfs(node)

    def prune_status(self):
        from .gpu_ops.Sum import SumOp, sum_op
        # this function make following two improvement:
        # 1. remove useless target status
        # 2. combine nodes of the same status in the inputs of SumOp to reduce communication

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            input_statuses = []
            status = self.node_cur_state_map.get(node, None)
            for n in node.inputs:
                dfs(n)
                if n in sumop_status:
                    cur_input_st = self.node_cur_state_map[n]
                    target = self.node_tar_state_map[n].get(node, None)
                    if target is not None and cur_input_st.effect_equal(target) and n.raw_ctx == node.raw_ctx:
                        self.node_tar_state_map[n].pop(node)
                    elif target is None:
                        self.node_tar_state_map[n][node] = sumop_status[n].remove_partial(
                        )
                    input_statuses.append(cur_input_st)
                else:
                    cur_input_st = self.node_cur_state_map[n]
                    target = self.node_tar_state_map[n].get(node, None)
                    if target is not None and cur_input_st.effect_equal(target) and n.raw_ctx == node.raw_ctx:
                        self.node_tar_state_map[n].pop(node)
                    input_statuses.append(cur_input_st)
            if isinstance(node, SumOp):
                st_map = {}
                need_new_op = False
                for n, st in zip(node.inputs, input_statuses):
                    key = str(st) + ' ' + str(n.raw_ctx)
                    if key not in st_map:
                        st_map[key] = [n]
                    else:
                        st_map[key].append(n)
                        if not st.effect_equal(status):
                            need_new_op = True
                if need_new_op:
                    if len(st_map) > 1:
                        new_inputs = []
                        for nodes in st_map.values():
                            if len(nodes) == 1 or node not in self.node_tar_state_map[nodes[0]]:
                                new_inputs.extend(nodes)
                            else:
                                new_node = sum_op(nodes)
                                self.node_cur_state_map[new_node] = self.node_cur_state_map[nodes[0]]
                                self.node_tar_state_map[new_node][node] = status.remove_partial(
                                )
                                new_node.raw_ctx = nodes[0].raw_ctx
                                new_node.ctx = nodes[0].ctx
                                new_inputs.append(new_node)
                                for n in nodes:
                                    self.node_tar_state_map[n].pop(node)
                        node.inputs = new_inputs
                    else:
                        # modify current sumop's status
                        nodes = node.inputs
                        sumop_status[node] = status
                        self.node_cur_state_map[node] = self.node_cur_state_map[nodes[0]]
                        node.raw_ctx = nodes[0].raw_ctx
                        node.ctx = nodes[0].ctx
                        for n in nodes:
                            self.node_tar_state_map[n].pop(node)

        visited = set()
        sumop_status = {}
        for node in self.node_list:
            dfs(node)

    def reset_status(self) -> None:
        def dfs_reset_status(node: Op) -> None:
            if node in visited:
                return
            visited.add(node)
            node.reset_status()
            for n in node.inputs:
                dfs_reset_status(n)
        visited = set()
        for node in self.node_list:
            dfs_reset_status(node)

    def get_input_statuses(self, node: Op) -> List[NodeStatus]:
        input_statuses = []
        for n in node.inputs:
            node_status = self.node_tar_state_map[n].get(
                node, self.node_cur_state_map[n])
            input_statuses.append(node_status)
        return input_statuses

    def infer_states(self, invalid_states: Dict[NodeStatus, Optional[Op]], node_list: List[Op] = None) -> None:
        from .dataloader import DataloaderOp
        from .optimizer import OptimizerOp

        def infer_node_states(node: Op, infer_order: bool) -> None:
            if node in visited:
                return
            nonlocal chance
            visited.add(node)
            if isinstance(node, DataloaderOp):
                pass
            elif isinstance(node, OptimizerOp):
                for n in node.inputs:
                    infer_node_states(n, infer_order)
            else:
                # if node.raw_ctx.is_mp:
                if infer_order and chance and not self.node_cur_state_map[node].valid_all():
                    self.node_cur_state_map[node].get_default_order()
                    chance = False
                input_statuses = self.get_input_statuses(node)
                node.backward_deduce_states(
                    self.node_cur_state_map[node], input_statuses, deduce_order=infer_order)
                for n in node.inputs:
                    infer_node_states(n, infer_order)
                node.forward_deduce_states(
                    input_statuses, self.node_cur_state_map[node], deduce_order=infer_order)
                # else:
                #     for n in node.inputs:
                #         infer_node_states(n, infer_order)

        if node_list is None:
            node_list = self.node_list
        # first infer state and duplicate
        invalid_order = invalid_states.copy()
        while True:
            visited = set()
            for node in node_list:
                infer_node_states(node, infer_order=False)
            progress = False
            for st in list(invalid_states.keys()):
                if st.valid_state():
                    invalid_states.pop(st)
                    progress = True
            if invalid_states == {}:
                break
            assert progress, "Not enough information for model parallel."
        chance = False
        # next infer order
        while True:
            visited = set()
            for node in node_list:
                infer_node_states(node, infer_order=True)
            progress = False
            for st in list(invalid_order.keys()):
                if st.valid_all():
                    invalid_order.pop(st)
                    progress = True
            if invalid_order == {}:
                break
            if not progress:
                chance = True

    def complete_partial_graph(
        self,
        subgraph: List[Op],
        backbone_node: Op,
        other_backbones: List[Op],
        backbone_node_status: NodeStatus,
    ) -> Dict[Op, NodeStatus]:
        from .layers.base import OpLayer
        dev_num = backbone_node_status.dev_num
        if isinstance(backbone_node, OpLayer):
            assert len(other_backbones) == 0
            return self.complete_partial_graph_oplayer(
                subgraph, backbone_node, backbone_node_status, dev_num)
        else:
            return self.complete_partial_graph_normal(
                subgraph, backbone_node, other_backbones, backbone_node_status, dev_num)

    def complete_partial_graph_oplayer(
        self,
        subgraph: List[Op],
        backbone_node: Op,
        backbone_node_status: NodeStatus,
        dev_num: int
    ) -> Dict[Op, NodeStatus]:
        if not backbone_node_status.valid_all():
            backbone_node_status.get_default_order()
        node_cur_state_map = {}
        reserved_keys = subgraph.copy()
        for oplayer in [backbone_node] + backbone_node.grad_layers:
            if oplayer is not None:
                node_cur_state_map[oplayer] = backbone_node_status
                for n in oplayer.inputs:
                    if n not in node_cur_state_map:
                        node_cur_state_map[n] = NodeStatus(
                            dev_num, partial_or_node=n)
                input_statuses = [node_cur_state_map[n]
                                  for n in oplayer.inputs]
                oplayer.backward_deduce_states(
                    backbone_node_status, input_statuses, False)
                oplayer.backward_deduce_states(
                    backbone_node_status, input_statuses, True)
                reserved_keys.extend(oplayer.inputs)
        backbone_node.modify_current_states(node_cur_state_map)
        # following step is to remove oplayers and other nodes that not in computation graph
        # such as the nodes generated in gradients but not in use since outputs not trainable
        node_cur_state_map = {
            k: v for k, v in node_cur_state_map.items() if k in reserved_keys}
        return node_cur_state_map

    def complete_partial_graph_normal(
        self,
        subgraph: List[Op],
        backbone_node: Op,
        other_backbones: List[Op],
        backbone_node_status: NodeStatus,
        dev_num: int
    ) -> Dict[Op, NodeStatus]:
        # we assume that all the nodes in subgraph use the same raw context
        # no targets
        from .dataloader import DataloaderOp
        from .optimizer import OptimizerOp
        from .gpu_ops.Sum import SumOp

        def infer_node_states(node: Op, infer_order: bool) -> None:
            if isinstance(node, (DataloaderOp, OptimizerOp)):
                pass
            else:
                nonlocal chance
                if infer_order and chance and not local_node_cur_state_map[node].valid_all():
                    local_node_cur_state_map[node].get_default_order()
                    chance = False
                input_statuses = [local_node_tar_state_map[n].get(node, local_node_cur_state_map[n])
                                  for n in node.inputs]
                node.backward_deduce_states(
                    local_node_cur_state_map[node], input_statuses, deduce_order=infer_order)
                node.forward_deduce_states(
                    input_statuses, local_node_cur_state_map[node], deduce_order=infer_order)

        local_node_cur_state_map = {}  # we don't use self.node_cur_state_map
        local_node_tar_state_map = defaultdict(
            dict)  # considering partial to duplicate
        bf_map = self.bf_map
        forward_nodes = []
        backward_nodes = []
        flag = False
        for node in subgraph:
            cur_list = backward_nodes
            cur_flag = node in bf_map or isinstance(node, OptimizerOp)
            if flag:
                assert cur_flag
            elif cur_flag:
                flag = True
            else:
                cur_list = forward_nodes
            if not isinstance(node, OptimizerOp):
                cur_list.append(node)
        invalid_states = {}
        for node in forward_nodes:
            if node is backbone_node:
                if not backbone_node_status.valid_all():
                    backbone_node_status.get_default_order()
                local_node_cur_state_map[node] = backbone_node_status
            elif node in other_backbones:
                local_node_cur_state_map[node] = backbone_node_status
            else:
                if node not in local_node_cur_state_map:
                    cur_node_status = NodeStatus(dev_num, partial_or_node=node)
                    local_node_cur_state_map[node] = cur_node_status
                    if not cur_node_status.valid_all():
                        invalid_states[cur_node_status] = node
            for n in node.inputs:
                if n not in local_node_cur_state_map:
                    cur_node_status = NodeStatus(dev_num, partial_or_node=n)
                    local_node_cur_state_map[n] = cur_node_status
                    if not cur_node_status.valid_all():
                        invalid_states[cur_node_status] = n
                prev_node_status = local_node_cur_state_map[n]
                if n in [backbone_node] + other_backbones and prev_node_status.partial != local_node_cur_state_map[node].partial:
                    local_node_tar_state_map[n][node] = prev_node_status.remove_partial(
                    )

        # first infer state and duplicate
        invalid_order = invalid_states.copy()
        while True:
            for node in forward_nodes:
                infer_node_states(node, infer_order=False)
            progress = False
            for st in list(invalid_states.keys()):
                if st.valid_state():
                    invalid_states.pop(st)
                    progress = True
            if invalid_states == {}:
                break
            if not progress:
                for st, node in invalid_states.items():
                    print(st, node)
            assert progress, "Not enough information for model parallel."
        chance = False
        # next infer order
        while True:
            for node in forward_nodes:
                infer_node_states(node, infer_order=True)
            progress = False
            for st in list(invalid_order.keys()):
                if st.valid_all():
                    invalid_order.pop(st)
                    progress = True
            if invalid_order == {}:
                break
            if not progress:
                chance = True
        for node in backward_nodes:
            fnode, i = bf_map[node]
            for n in node.inputs:
                if n in bf_map:
                    # backward node
                    forward_node = bf_map[n][0]
                    if forward_node is not fnode:
                        local_node_tar_state_map[n][node] = \
                            local_node_cur_state_map[fnode].remove_partial()
                elif fnode in local_node_tar_state_map[n]:
                    local_node_tar_state_map[n][node] = local_node_tar_state_map[n][fnode]
            input_statuses = [local_node_tar_state_map[n].get(node, local_node_cur_state_map[n])
                              for n in fnode.inputs]
            local_node_cur_state_map[node] = fnode.deduce_generated_backward_nodes_states(
                input_statuses, local_node_cur_state_map[fnode], i)
            if isinstance(node, SumOp) and i == -1:
                for n in node.inputs:
                    if n not in local_node_cur_state_map:
                        local_node_cur_state_map[n] = local_node_cur_state_map[node]
            else:
                input_statuses = []
                for n in node.inputs:
                    if n not in local_node_cur_state_map:
                        if node in local_node_tar_state_map[n]:
                            local_node_cur_state_map[n] = local_node_tar_state_map[n].pop(
                                node)
                        else:
                            local_node_cur_state_map[n] = NodeStatus(
                                dev_num, partial_or_node=n)
                    if node in local_node_tar_state_map[n]:
                        input_statuses.append(
                            local_node_tar_state_map[n][node])
                    else:
                        input_statuses.append(local_node_cur_state_map[n])
                # deduction and validation
                node.backward_deduce_states(
                    local_node_cur_state_map[node], input_statuses, deduce_order=False)
                node.forward_deduce_states(
                    input_statuses, local_node_cur_state_map[node], deduce_order=False)
                node.backward_deduce_states(
                    local_node_cur_state_map[node], input_statuses, deduce_order=True)
                node.forward_deduce_states(
                    input_statuses, local_node_cur_state_map[node], deduce_order=True)
        return local_node_cur_state_map

    def assign_context_by_traverse_nodes(
        self,
        ctx: DLContext,
        mpi_comm: NCCL_Communicator,
        use_nccl_collectives: bool = True,
    ) -> Tuple[List[Op], Dict[Union[Op, str], NCCL_Communicator], Dict[Op, int]]:
        # for variable data parallelism degree, we have following implementation:
        # take turns to send/receive in only ONE operation
        # this requires that every model parallel in each worker is the same
        # which does NOT support heterogeneous settings
        from .dataloader import DataloaderOp
        from .optimizer import OptimizerOp
        from .gpu_ops.PipelineSend import pipeline_send_op
        from .gpu_ops.PipelineReceive import pipeline_receive_op
        from .gpu_ops.Variable import PlaceholderOp
        from .gpu_ops.Concatenate import concatenate_op
        from .gpu_ops.Split import split_op
        from .gpu_ops.Sum import sum_op
        from .gpu_ops.EmbeddingLookUp import EmbeddingLookUp
        from .gpu_ops.executor import new_group_comm

        def general_receiving(
            from_ctx: DeviceGroup,
            to_ctx: DeviceGroup,
            key: Op,
            mp_index: Optional[int] = None,
        ) -> Op:
            if mp_index is not None and not from_ctx.is_mp:
                assert mp_index == 0
                mp_index = None
            if from_ctx.worker_num == to_ctx.worker_num:
                # normal data parallelism
                from_dev = from_ctx[to_ctx.dp_index]
                if mp_index is not None:
                    from_dev = from_dev[mp_index]
                if from_dev == ctx:
                    return self_buffer[key]
                else:
                    result = pipeline_receive_op(
                        mpi_comm.getRankFromDevice(from_dev), mpi_comm, use_indexed_slices=key.use_indexed_slices, ctx=ctx)
                    layer_indices[result] = layer_id
                    return result
            else:
                # round-robin data parallelism
                from_devs = from_ctx.get_target_workers(to_ctx, mp_index)
                ranks = [mpi_comm.getRankFromDevice(
                    fctx) for fctx in from_devs]
                result = pipeline_receive_op(
                    ranks, mpi_comm, use_indexed_slices=key.use_indexed_slices, ctx=ctx)
                layer_indices[result] = layer_id
                return result

        def general_sending(
            node: Op,
            from_ctx: DeviceGroup,
            to_ctx: DeviceGroup,
            key: Optional[Op] = None,
            mp_index: Optional[int] = None
        ) -> None:
            if mp_index is not None and not to_ctx.is_mp:
                assert mp_index == 0
                mp_index = None
            if from_ctx.worker_num == to_ctx.worker_num:
                # normal data parallelism
                to_dev = to_ctx[from_ctx.dp_index]
                if mp_index is not None:
                    to_dev = to_dev[mp_index]
                if ctx == to_dev:
                    self_buffer[key] = node
                else:
                    target_rank = mpi_comm.getRankFromDevice(to_dev)
                    result = pipeline_send_op(
                        node, target_rank, mpi_comm, ctx=ctx)
                    layer_indices[result] = layer_id
                    my_eval_nodes.append(result)
            else:
                # round-robin data parallelism
                to_devs = to_ctx.get_target_workers(from_ctx, mp_index)
                ranks = [mpi_comm.getRankFromDevice(tctx) for tctx in to_devs]
                result = pipeline_send_op(
                    node, ranks, mpi_comm, ctx=ctx)
                layer_indices[result] = layer_id
                my_eval_nodes.append(result)

        def make_allreduce(prev_input: Op, node: Op) -> AllReduceCommunicateP2POp:
            from .gpu_ops.AllReduceCommunicate import allreduceCommunicatep2p_op
            prev_ctx, cur_ctx = prev_input.raw_ctx, node.raw_ctx
            assert prev_ctx == cur_ctx
            assert cur_ctx.local_dp, 'The receive node must be on local device.'
            key = (self.node_tar_state_map[prev_input]
                   [node].content_hash(), cur_ctx)
            if key not in recv_src[prev_input]:
                allreduce_devices = get_allreduce_devices(
                    prev_ctx, self.node_cur_state_map[prev_input], reduce_dp=False)
                comm = add_to_comm_group(allreduce_devices)
                res = allreduceCommunicatep2p_op(
                    prev_input, comm)
                layer_indices[res] = layer_id
                recv_src[prev_input][key] = res
            return recv_src[prev_input][key]

        def make_allgather(prev_input: Op, node: Op) -> AllGatherCommunicateOp:
            from .gpu_ops.AllGatherCommunicate import allgatherCommunicate_op
            prev_ctx, cur_ctx = prev_input.raw_ctx, node.raw_ctx
            assert prev_ctx == cur_ctx
            assert cur_ctx.local_dp, 'The receive node must be on local device.'
            key = (self.node_tar_state_map[prev_input]
                   [node].content_hash(), cur_ctx)
            if key not in recv_src[prev_input]:
                devices = self.node_cur_state_map[prev_input].get_devices_by_dim(
                    prev_ctx, 0)
                comm = add_to_comm_group(devices)
                res = allgatherCommunicate_op(
                    prev_input, comm)
                layer_indices[res] = layer_id
                recv_src[prev_input][key] = res
            return recv_src[prev_input][key]

        def make_reducescatter(prev_input: Op, node: Op) -> ReduceScatterCommunicateOp:
            from .gpu_ops.ReduceScatterCommunicate import reducescatterCommunicate_op
            prev_ctx, cur_ctx = prev_input.raw_ctx, node.raw_ctx
            assert prev_ctx == cur_ctx
            assert cur_ctx.local_dp, 'The receive node must be on local device.'
            key = (self.node_tar_state_map[prev_input]
                   [node].content_hash(), cur_ctx)
            if key not in recv_src[prev_input]:
                devices = self.node_cur_state_map[prev_input].get_devices_by_dim(
                    prev_ctx, -2)
                comm = add_to_comm_group(devices)
                res = reducescatterCommunicate_op(
                    prev_input, comm)
                layer_indices[res] = layer_id
                recv_src[prev_input][key] = res
            return recv_src[prev_input][key]

        def make_broadcast(prev_input: Op, node: Op) -> BroadcastCommunicateOp:
            from .gpu_ops.BroadcastCommunicate import broadcastCommunicate_op
            prev_ctx, cur_ctx = prev_input.raw_ctx, node.raw_ctx
            assert cur_ctx.local_dp, 'The receive node must be on local device.'
            key = (self.node_tar_state_map[prev_input]
                   [node].content_hash(), cur_ctx)
            if key not in recv_src[prev_input]:
                devices, rdc_dev = self.node_tar_state_map[prev_input][node].get_devices_by_dim(
                    cur_ctx, -1, rdc_ctx=prev_ctx)
                assert rdc_dev in devices
                assert ctx in devices
                comm = add_to_comm_group(devices)
                res = broadcastCommunicate_op(
                    prev_input, comm, comm.getRankFromDevice(rdc_dev), ctx)
                layer_indices[res] = layer_id
                recv_src[prev_input][key] = res
            return recv_src[prev_input][key]

        def make_reduce(prev_input: Op, node: Op) -> ReduceCommunicateOp:
            from .gpu_ops.ReduceCommunicate import reduceCommunicate_op
            prev_ctx, cur_ctx = prev_input.raw_ctx, node.raw_ctx
            assert prev_ctx.local_dp, 'The receive node must be on local device.'
            key = (self.node_tar_state_map[prev_input]
                   [node].content_hash(), cur_ctx)
            if key not in recv_src[prev_input]:
                devices, rdc_dev = self.node_cur_state_map[prev_input].get_devices_by_dim(
                    prev_ctx, -2, rdc_ctx=cur_ctx)
                assert rdc_dev in devices
                assert ctx in devices
                comm = add_to_comm_group(devices)
                res = reduceCommunicate_op(
                    prev_input, comm, comm.getRankFromDevice(rdc_dev))
                layer_indices[res] = layer_id
                recv_src[prev_input][key] = res
            return recv_src[prev_input][key]

        def receive_model_parallel(prev_input: Op, node: Op) -> Op:
            prev_ctx, cur_ctx = prev_input.raw_ctx, node.raw_ctx
            key = (self.node_tar_state_map[prev_input]
                   [node].content_hash(), cur_ctx)
            assert cur_ctx.local_dp, 'The receive node must be on local device.'
            # here the prev input and the node are both in model parallel, with different states
            if key not in recv_src[prev_input]:
                prev_ns = self.node_cur_state_map[prev_input]
                target_ns = self.node_tar_state_map[prev_input][node]
                prev_state, prev_duplicate, prev_order = prev_ns.get_all()
                prev_partial = prev_ns.partial
                target_state, target_duplicate = target_ns.get()
                assert not target_ns.enable_partial, 'target must not be partial'
                loop_sizes = prev_ns.get_loop_sizes()
                cur_state_index = target_ns.map_dev_to_index(
                    max(0, cur_ctx.mp_index), containing_duplicate=True)
                device_index = 0

                def cross_receive(depth: int) -> Op:
                    nonlocal device_index
                    if depth == len(prev_order):
                        res = general_receiving(
                            prev_ctx, cur_ctx, prev_input, device_index)
                        device_index += 1
                    else:
                        cur_dim = prev_order[depth]
                        if cur_dim == -2:
                            res = sum_op([cross_receive(depth+1)
                                          for _ in range(prev_partial)], ctx=ctx)
                            layer_indices[res] = layer_id
                        elif cur_dim == -1:
                            # TODO: consider how to choose the copy with minimal communication
                            # now we use following rules:
                            # if prev_duplicate < target_duplicate, then each prev send to some targets
                            # else, each target receive from the first duplicate in the group
                            cur_st = cur_state_index.get(cur_dim, 0)
                            if prev_duplicate % target_duplicate == 0:
                                multiple = prev_duplicate // target_duplicate
                                device_index += cur_st * \
                                    multiple * loop_sizes[depth]
                                res = cross_receive(depth+1)
                                device_index += ((target_duplicate - cur_st)
                                                 * multiple - 1) * loop_sizes[depth]
                            elif target_duplicate % prev_duplicate == 0:
                                multiple = target_duplicate // prev_duplicate
                                device_index += cur_st // multiple * \
                                    loop_sizes[depth]
                                res = cross_receive(depth+1)
                                device_index += (target_duplicate - 1 -
                                                 cur_st) // multiple * loop_sizes[depth]
                            else:
                                assert False
                        else:
                            tar_st = target_state.get(cur_dim, 1)
                            cur_st = cur_state_index.get(cur_dim, 0)
                            if prev_state[cur_dim] % tar_st == 0:
                                # at `cur_dim` dimension we need to concat some inputs
                                multiple = prev_state[cur_dim] // tar_st
                                device_index += cur_st * \
                                    multiple * loop_sizes[depth]
                                if multiple == 1:
                                    res = cross_receive(depth+1)
                                else:
                                    res = concatenate_op(
                                        [cross_receive(depth+1) for _ in range(multiple)], axis=cur_dim, ctx=ctx)
                                    layer_indices[res] = layer_id
                                device_index += (tar_st - 1 - cur_st) * \
                                    multiple * loop_sizes[depth]
                            elif tar_st % prev_state[cur_dim] == 0:
                                # at `cur_dim` dimension we need to specify one input
                                multiple = tar_st // prev_state[cur_dim]
                                device_index += cur_st // multiple * \
                                    loop_sizes[depth]
                                res = cross_receive(depth+1)
                                device_index += (tar_st - 1 -
                                                 cur_st) // multiple * loop_sizes[depth]
                            else:
                                assert False, 'The dispatch state (%d, %d) at dimension %d is invalid.' % (
                                    prev_state[cur_dim], tar_st, cur_dim)
                    return res
                recv_src[prev_input][key] = cross_receive(0)
                assert device_index == prev_ctx.mp_dev_num
            return recv_src[prev_input][key]

        def send_model_parallel(prev_input: Op, node: Op) -> None:
            prev_ctx, cur_ctx = prev_input.raw_ctx, node.raw_ctx
            key = (self.node_tar_state_map[prev_input]
                   [node].content_hash(), cur_ctx)
            assert prev_ctx.local_dp, 'The send node must be on local device.'
            if key not in send_dst[prev_input]:
                send_dst[prev_input][key] = True
                prev_ns = self.node_cur_state_map[prev_input]
                target_ns = self.node_tar_state_map[prev_input][node]
                prev_state, prev_duplicate = prev_ns.get()
                prev_partial = prev_ns.partial
                target_state, target_duplicate, target_order = target_ns.get_all()
                assert not target_ns.enable_partial, 'target must not be partial'
                cur_state_index = prev_ns.map_dev_to_index(
                    max(0, prev_ctx.mp_index), containing_duplicate=True)
                loop_sizes = target_ns.get_loop_sizes()
                device_index = 0

                def cross_send(
                    split_cur_state: Dict[int, int],
                    split_target_state: Dict[int, int],
                    depth: int,
                    need_split: bool
                ):
                    nonlocal device_index
                    if depth == len(target_order):
                        if need_split:
                            keys = list(split_target_state.keys())
                            indices = [split_cur_state[k] for k in keys]
                            splits = [split_target_state[k] for k in keys]
                            cur_node = split_op(
                                prev_input, keys, indices, splits, ctx=ctx)
                            layer_indices[cur_node] = layer_id
                        else:
                            cur_node = prev_input
                        general_sending(cur_node, prev_ctx,
                                        cur_ctx, prev_input, device_index)
                        device_index += 1
                    else:
                        cur_dim = target_order[depth]
                        if cur_dim < 0:
                            assert cur_dim == -1, 'Target node status must not enable partial.'
                            cur_st = cur_state_index.get(cur_dim, 0)
                            if prev_duplicate % target_duplicate == 0:
                                # at `cur_dim` dimension we need to send one output
                                multiple = prev_duplicate // target_duplicate
                                assert cur_st % multiple == 0
                                device_index += cur_st // multiple * \
                                    loop_sizes[depth]
                                cross_send(split_cur_state,
                                           split_target_state, depth+1, need_split)
                                device_index += (prev_duplicate - 1 -
                                                 cur_st) // multiple * loop_sizes[depth]
                            elif target_duplicate % prev_duplicate == 0:
                                # at `cur_dim` dimension we need to split and send some outputs
                                multiple = target_duplicate // prev_duplicate
                                device_index += cur_st * \
                                    multiple * loop_sizes[depth]
                                for index in range(multiple):
                                    cross_send(split_cur_state,
                                               split_target_state, depth+1, True)
                                device_index += (prev_duplicate - 1 -
                                                 cur_st) * multiple * loop_sizes[depth]
                            else:
                                assert False
                        else:
                            pre_st = prev_state.get(cur_dim, 1)
                            cur_st = cur_state_index.get(cur_dim, 0)
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
                                    multiple * loop_sizes[depth]
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
                if prev_partial == 1 and prev_duplicate > target_duplicate and cur_state_index.get(-1, 0) % (prev_duplicate // target_duplicate) != 0:
                    # TODO: consider whether to eliminate redundant duplicate and how
                    my_eval_nodes.append(prev_input)
                else:
                    # TODO: consider how to choose the copy with minimal communication
                    # now we use naive method, choose the first
                    cross_send({}, {}, 0, False)
                    assert device_index == cur_ctx.mp_dev_num

        def get_allreduce_devices(context: DeviceGroup, status: NodeStatus, reduce_dp: bool = True) -> Optional[DeviceGroup]:
            # only_reduce_partial: if True, only reduce tensors in the same model parallel
            # if False, reduce all the tensors across workers
            target_state, target_duplicate, target_order = status.get_all()
            target_partial = status.partial
            mp_index = context.mp_index
            interval = 1
            allreduce_devices = []
            reduce_mp = (-2 in target_order)
            if reduce_mp:
                par_dim = target_order.index(-2)
                for cur_order in target_order[par_dim+1:]:
                    interval *= target_state[cur_order]
            macro_interval = interval * target_partial
            start = mp_index - mp_index % macro_interval + mp_index % interval
            for worker in context.workers:
                if reduce_dp or ctx in worker:
                    if reduce_mp:
                        for ind in range(start, start + interval * target_partial, interval):
                            allreduce_devices.append(
                                worker[ind])
                    else:
                        allreduce_devices.append(worker[mp_index])
            allreduce_devices = None if len(
                allreduce_devices) <= 1 else DeviceGroup(allreduce_devices)
            return allreduce_devices

        def add_to_comm_group(allreduce_devices: DeviceGroup) -> None:
            if allreduce_devices not in comm_groups:
                if len(allreduce_devices) == mpi_comm.nrank:
                    comm_groups[allreduce_devices] = mpi_comm
                else:
                    comm_groups[allreduce_devices] = new_group_comm(
                        allreduce_devices)
            return comm_groups[allreduce_devices]

        def assign_ctx(node: Op) -> None:
            nonlocal layer_id
            cur_ctx = node.raw_ctx
            if node in layer_indices:
                return
            if isinstance(node, DataloaderOp):
                layer_indices[node] = layer_id
                layer_id += 1
                return
            elif isinstance(node, PlaceholderOp):
                cur_ctx.check_mp_num(
                    self.node_cur_state_map[node].dev_num)
                cur_ctx.set_index(ctx)
                if cur_ctx.local_mp:
                    node_status = self.node_cur_state_map[node]
                    node.reshape_in_mp(node_status.map_dev_to_index(
                        cur_ctx.mp_index), node_status.state)
                layer_indices[node] = layer_id
                layer_id += 1
                if cur_ctx.local_dp:
                    node.ctx = ctx
                    if node in self.node_list:
                        my_eval_nodes.append(node)
                    if node.trainable:
                        trainable_params.append(node)
            elif isinstance(node, OptimizerOp):
                nonlocal opt
                assert opt is None, 'Multiple optimizer is invalid.'
                opt = node
                grads = []
                original_params = node.optimizer.params
                for ind, param in enumerate(original_params):
                    ori_grad = node.inputs[ind]
                    assign_ctx(ori_grad)
                    grad_ctx, param_ctx = ori_grad.raw_ctx, param.raw_ctx
                    assert grad_ctx.worker_num == param_ctx.worker_num, \
                        'Worker number of gradient and parameter should be equal!'
                    assert grad_ctx.mp_index == param_ctx.mp_index, \
                        'Model parallel state of gradient and parameter should be the same!'
                    if not param_ctx.local_mp:
                        # this branch is pipeline + data parallel
                        assert param_ctx.local_dp == (
                            param in trainable_params), 'Bug appears!'
                        # handle sending
                        if grad_ctx.local_dp:
                            if param_ctx not in send_dst[ori_grad]:
                                send_dst[ori_grad][param_ctx] = True
                                general_sending(
                                    ori_grad, grad_ctx, param_ctx, ori_grad)
                        # handle receiving
                        if param_ctx.local_dp:
                            if -1 not in recv_src[ori_grad]:
                                recv_src[ori_grad][-1] = general_receiving(
                                    grad_ctx, param_ctx, ori_grad)
                            grads.append(recv_src[ori_grad][-1])
                    else:
                        # here in the same model parallel
                        assert grad_ctx == param_ctx
                        grads.append(ori_grad)
                    layer_id += 2
                if trainable_params:
                    node.optimizer.params = trainable_params
                    node.inputs = grads
                    node.ctx = ctx
                    my_eval_nodes.append(node)
                    layer_indices[node] = layer_id
                    assert len(trainable_params) == len(grads)
                    for param, grad in zip(trainable_params, grads):
                        # here handle the nodes that need allreduce
                        grad_ctx = grad.raw_ctx
                        if grad_ctx.server_num == 0:
                            allreduce_devices = None
                            if not grad_ctx.local_mp and grad_ctx.worker_num > 1:
                                allreduce_devices = grad_ctx
                            elif grad_ctx.local_mp:
                                if -2 in self.node_cur_state_map[grad].order or grad_ctx.worker_num > 1:
                                    allreduce_devices = get_allreduce_devices(
                                        grad_ctx, self.node_cur_state_map[grad])
                            if allreduce_devices is not None:
                                param_allreduce_group[param] = add_to_comm_group(
                                    allreduce_devices)
                # here we establish a group comm for loss
                # in case users wants to reduce loss and accuracy to one worker
                loss_node_ctx = node.optimizer.loss.raw_ctx
                if loss_node_ctx.mp_dev_num > 1:
                    new_loss_node_ctx = []
                    for cc in loss_node_ctx:
                        new_loss_node_ctx.extend(cc)
                    loss_node_ctx = DeviceGroup(new_loss_node_ctx)
                if loss_node_ctx.worker_num > 1 and loss_node_ctx.server_num == 0:
                    param_allreduce_group['loss'] = add_to_comm_group(
                        loss_node_ctx)
            else:
                # now we only support SAME model parallel in data parallel
                # and 1 context can only appear once
                cur_ctx.set_index(ctx)
                for i, n in enumerate(node.inputs):
                    prev_ctx = n.raw_ctx
                    if isinstance(n, DataloaderOp):
                        if n not in layer_indices:
                            layer_indices[n] = layer_id
                            layer_id += 1
                        if cur_ctx.local_dp:
                            n.set_dp_rank(cur_ctx.dp_index, cur_ctx.worker_num)
                            if cur_ctx.local_mp:
                                node_status = self.node_cur_state_map[n]
                                n.set_mp_parts(node_status.map_dev_to_index(
                                    cur_ctx.mp_index), node_status.state)
                            if n in self.node_list and n not in my_eval_nodes:
                                my_eval_nodes.append(n)
                        continue
                    # we assume that in pipeline + data parallel mode,
                    # devices number of each stage is equal
                    # the device in correspondent place will communicate with each other

                    assign_ctx(n)

                    n_cur_stat = self.node_cur_state_map[n]
                    n_tar_stat = self.node_tar_state_map[n]

                    if node in n_tar_stat \
                            and (n_cur_stat != n_tar_stat[node] or prev_ctx != cur_ctx) \
                            and (n_cur_stat.is_dist() or n_tar_stat[node].is_dist()):
                        # here in every context each device appear only once
                        # TODO: consider whether or not release the constraint above?
                        tar_stat = n_tar_stat[node]
                        prev_worker = prev_ctx.cur_worker()
                        cur_worker = cur_ctx.cur_worker()
                        same_ctxs = (prev_ctx.dp_index == cur_ctx.dp_index and prev_worker ==
                                     cur_worker and isinstance(prev_worker, tuple))
                        bcast_ctxs = False
                        rdc_ctxs = False
                        if isinstance(cur_worker, tuple):
                            previous_devs = prev_ctx[cur_ctx.dp_index]
                            if isinstance(previous_devs, DLContext):
                                previous_devs = (previous_devs,)
                            num_prev = len(previous_devs)
                            num_cur = len(cur_worker)
                            if num_prev < num_cur and num_cur % num_prev == 0:
                                bcast_ctxs = True
                                last_ind = None
                                gap = num_cur // num_prev
                                for dev in previous_devs:
                                    if dev not in cur_worker:
                                        bcast_ctxs = False
                                        break
                                    else:
                                        ind = cur_worker.index(dev)
                                        if last_ind is None:
                                            last_ind = ind
                                        elif ind - last_ind != gap:
                                            bcast_ctxs = False
                                            break
                        if isinstance(prev_worker, tuple):
                            target_devs = cur_ctx[prev_ctx.dp_index]
                            if isinstance(target_devs, DLContext):
                                target_devs = (target_devs,)
                            num_prev = len(prev_worker)
                            num_cur = len(target_devs)
                            if num_cur < num_prev and num_prev % num_cur == 0:
                                rdc_ctxs = True
                                last_ind = None
                                gap = num_prev // num_cur
                                for dev in target_devs:
                                    if dev not in prev_worker:
                                        rdc_ctxs = False
                                        break
                                    else:
                                        ind = prev_worker.index(dev)
                                        if last_ind is None:
                                            last_ind = ind
                                        elif ind - last_ind != gap:
                                            rdc_ctxs = False
                                            break
                        processed = False
                        if use_nccl_collectives:
                            processed = True
                            if same_ctxs and n_cur_stat.check_allreduce(tar_stat):
                                # here we use allreduce instead of all2all
                                if prev_ctx.local_dp:
                                    node.inputs[i] = make_allreduce(n, node)
                            elif same_ctxs and n_cur_stat.check_allgather(tar_stat):
                                # here we use allgather instead of all2all
                                if prev_ctx.local_dp:
                                    node.inputs[i] = make_allgather(n, node)
                            elif same_ctxs and n_cur_stat.check_reducescatter(tar_stat):
                                # here we use reducescatter instead of all2all
                                if prev_ctx.local_dp:
                                    node.inputs[i] = make_reducescatter(
                                        n, node)
                            elif bcast_ctxs and n_cur_stat.check_broadcast(tar_stat):
                                # here we use broadcast instead of all2all
                                if cur_ctx.local_dp:
                                    node.inputs[i] = make_broadcast(n, node)
                            elif rdc_ctxs and n_cur_stat.check_reduce(tar_stat):
                                # here we use reduce instead of all2all
                                if prev_ctx.local_dp:
                                    cur_res = make_reduce(n, node)
                                    if cur_ctx.local_dp:
                                        node.inputs[i] = cur_res
                                    else:
                                        my_eval_nodes.append(cur_res)
                            else:
                                processed = False
                        if not processed:
                            if prev_ctx.local_dp:
                                send_model_parallel(n, node)
                            if cur_ctx.local_dp:
                                node.inputs[i] = receive_model_parallel(
                                    n, node)
                    else:
                        assert prev_ctx.mp_index == cur_ctx.mp_index
                        if not cur_ctx.local_mp:
                            # handle sending
                            if prev_ctx.local_dp:
                                if cur_ctx not in send_dst[n]:
                                    send_dst[n][cur_ctx] = True
                                    general_sending(
                                        n, prev_ctx, cur_ctx, n)
                            # handle receiving
                            if cur_ctx.local_dp:
                                if -1 not in recv_src[n]:
                                    recv_src[n][-1] = general_receiving(
                                        prev_ctx, cur_ctx, n)
                                node.inputs[i] = recv_src[n][-1]
                        else:
                            # here in the same model parallel
                            assert cur_ctx == prev_ctx
                    layer_id += 1

                layer_indices[node] = layer_id
                layer_id += 1

                if cur_ctx.local_dp:
                    if isinstance(node, EmbeddingLookUp) and cur_ctx.local_mp:
                        status = self.node_cur_state_map[node]
                        index = status.map_dev_to_index(
                            cur_ctx.mp_index, containing_duplicate=True).get(-2, 0)
                        if status.partial > 1:
                            length = node.inputs[0].shape[0]
                            offset = length * index
                            node.inputs[1].embedding_offsets = (offset, length)
                            if not node.inputs[1].reshaped:
                                node.inputs[1].reshaped = True
                                node.inputs[1].parts = {}
                    node.ctx = ctx
                    if node in self.node_list:
                        my_eval_nodes.append(node)

        opt = None
        trainable_params = []
        comm_groups = {}
        param_allreduce_group = {}
        # in send_dst and recv_src we use node state as part of the key
        # this ignore following cases: the node use the same node state, but into different devices;
        # or the node has two node state, but part of the split is the same on some devices.
        # TODO: solve the problem above
        send_dst = defaultdict(dict)
        recv_src = defaultdict(dict)
        self_buffer = {}  # send and receive from self device
        layer_indices = {}
        layer_id = 0
        my_eval_nodes = []

        for node in self.node_list:
            assign_ctx(node)

        return my_eval_nodes, param_allreduce_group, layer_indices

    def get_state_maps(self) -> Tuple[Dict[Op, NodeStatus], DefaultDict[Op, Dict[Op, NodeStatus]]]:
        return self.node_cur_state_map, self.node_tar_state_map

    def init_oplayers(self) -> None:
        from .layers.base import OpLayer

        def dfs(node: Op) -> None:
            if node in visited:
                return
            visited.add(node)
            for i, n in enumerate(node.inputs):
                dfs(n)
                if isinstance(n, OpLayer):
                    self.oplayers[n].append((node, i))
            if isinstance(node, OpLayer):
                self.oplayers[node] = []

        visited = set()
        self.oplayers = {}
        for node in self.node_list:
            dfs(node)

    def extend_oplayers(self) -> None:
        if not self.extended:
            from .layers.base import OpLayerGradient

            for layer in self.oplayers:
                for node, i in self.oplayers[layer]:
                    node.inputs[i] = layer.output
                if not isinstance(layer, OpLayerGradient):
                    layer.reset_inputs(
                        self.node_cur_state_map, self.node_tar_state_map, self.model_parallel)
            self.extended = True

    def shrink_oplayers(self) -> None:
        if self.extended:
            for layer in self.oplayers:
                for node, i in self.oplayers[layer]:
                    node.inputs[i] = layer
            self.extended = False

    def set_grad_ctx(self, none_ctx: DeviceGroup) -> None:
        for2back = self.opt.forward2backward
        for grad in for2back.pop(None):
            grad.raw_ctx = none_ctx
        for node, grads in for2back.items():
            for grad in grads:
                grad.raw_ctx = node.raw_ctx
        # self.opt.raw_ctx = none_ctx

    def assert_opt(self) -> None:
        assert self.opt is not None

    def determine_by_forward_current_states(self, none_ctx: DeviceGroup) -> None:
        # set context for backward nodes using forward nodes
        self.set_grad_ctx(none_ctx)
        # infer states using partial information
        self.complete_state_map_with_partial_information()

    def copy_cur_state_to(self) -> Dict[Op, NodeStatus]:
        new_node_cur_state_map = {
            k: deepcopy(v) for k, v in self.node_cur_state_map.items()
        }
        return new_node_cur_state_map

    def copy_cur_state_from(self, node_cur_state_map: Dict[Op, NodeStatus]) -> None:
        self.node_cur_state_map = {
            k: deepcopy(v) for k, v in node_cur_state_map.items()
        }
        self.node_tar_state_map.clear()


class DistConfig(object):
    def __init__(
        self,
        file: Optional[str] = None,
        num_local_servers: int = 0,
        num_local_workers: int = 1
    ) -> None:
        if file is None:
            assert num_local_workers > 0, \
                'Please specify the configuration file or set the number of local workers.'
            self.settings = {'nodes': [{
                'host': socket.gethostname(),
                'servers': num_local_servers,
                'workers': num_local_workers,
                'chief': True,
            }]}
        else:
            self.settings = yaml.load(
                open(file).read(), Loader=yaml.FullLoader)
        attributes = set(['host', 'servers', 'workers', 'chief'])
        hosts = []
        servers, workers = {}, {}
        chief = None
        self.chief_address = socket.gethostbyname(socket.gethostname())
        for node in self.settings['nodes']:
            assert set(node.keys(
            )) <= attributes, 'Attributes of nodes invalid, %s / %s.' % (set(node.keys()), attributes)
            hosts.append(node['host'])
            if node.get('servers', 0):
                servers[node['host']] = node['servers']
            if node.get('workers', 0):
                workers[node['host']] = node['workers']
            if node.get('chief', False):
                assert chief is None, 'There should be only one chief.'
                chief = node['host']
        assert chief, 'There should be one chief.'
        self.num_servers = sum(servers.values())
        self.num_workers = sum(workers.values())
        self.enable_PS = (self.num_servers > 0)
        self.servers = servers
        self.workers = workers
        self.chief = chief
        self.hosts = hosts
        self.chief_address = socket.gethostbyname(socket.gethostname())

    def __str__(self) -> str:
        return '\n'.join([
            'Cluster: {',
            '  Chief: %s,' % self.chief,
            '  Servers(%d): %s,' % (self.num_servers, self.servers),
            '  Workers(%d): %s,' % (self.num_workers, self.workers),
            '}',
        ])

    def __iter__(self) -> Iterator[List[Dict[Any]]]:
        return iter(self.settings['nodes'])

    def save(self, path: str) -> None:
        with open(path, 'w') as fw:
            yaml.dump(self.settings, fw)

    def make_ps_config(self) -> Dict[str, Union[str, int]]:
        port = self.get_available_port(self.chief_address)
        return {
            'DMLC_PS_ROOT_URI': self.chief_address,
            'DMLC_PS_ROOT_PORT': port,
            'DMLC_NUM_WORKER': self.num_workers,
            'DMLC_NUM_SERVER': self.num_servers,
            'DMLC_PS_VAN_TYPE': 'p3'
        }

    def get_available_port(self, localhost: str) -> int:
        ports = set()
        for conn in psutil.net_connections():
            la = conn.laddr
            ra = conn.raddr
            if len(la) == 2 and la.ip in (localhost, '127.0.0.1'):
                ports.add(la.port)
            if len(ra) == 2 and ra.ip in (localhost, '127.0.0.1'):
                ports.add(ra.port)
        for p in range(13100, 13200):
            if p not in ports:
                return p
