from __future__ import absolute_import
import numpy as np
from collections import Iterable

from . import ndarray
from .random import get_np_rand
from .gpu_ops.Node import Op


class BatchIndices(object):
    def __init__(self, batch_num, need_shuffle=False):
        self.batch_num = batch_num
        self.all_batch_indices = np.arange(self.batch_num)
        self.last_key = len(self.all_batch_indices) - 1
        self.need_shuffle = need_shuffle

    def shuffle(self):
        nprs = get_np_rand(1)
        nprs.shuffle(self.all_batch_indices)

    def assert_attr(self, dataloader):
        assert self.batch_num == dataloader.batch_num
        assert self.need_shuffle == dataloader.shuffle

    def __getitem__(self, key):
        if key == 0 and self.last_key != key:
            assert self.last_key == len(self.all_batch_indices) - 1
            if self.need_shuffle:
                self.shuffle()
        self.last_key = key
        return self.all_batch_indices[key]


class RawData(object):
    def __init__(self, raw_data, dtype, func=None) -> None:
        self.dtype = dtype
        self.func = func if func else lambda x: x
        if isinstance(raw_data, (list, tuple)):
            raw_data = [self._init_array(data) for data in raw_data]
        else:
            raw_data = [self._init_array(raw_data)]
        self.raw_data = raw_data
        self._shape = list(raw_data[0].shape)
        self._offsets = [0, self._shape[0]]
        for d in raw_data[1:]:
            assert list(d.shape[1:]) == self._shape[1:]
            self._shape[0] += d.shape[0]
            self._offsets.append(self._shape[0])

    def _init_array(self, raw_data):
        raw_data = self.func(raw_data)
        if not isinstance(raw_data, (np.memmap, np.ndarray)):
            raw_data = np.array(raw_data, dtype=self.dtype)
        elif raw_data.dtype != self.dtype:
            raw_data = raw_data.astype(self.dtype)
        return raw_data

    def __len__(self):
        return self._shape[0]

    @property
    def shape(self):
        return self._shape

    def _get_arr_index(self, index):
        # binary search
        if index < 0:
            index = index % len(self)
        elif index >= len(self):
            index = len(self) - 1
        l, r = 0, len(self.raw_data)
        while l + 1 < r:
            mid = (l + r) // 2
            if self._offsets[mid] == index:
                l = mid
                break
            elif self._offsets[mid] < index:
                l = mid
            else:
                r = mid
        arr_ind = l
        arr_offset = index - self._offsets[arr_ind]
        return arr_ind, arr_offset

    def __getitem__(self, key):
        # directly return array; not good for dp split, need further support
        if isinstance(key, int):
            arr_ind, arr_offset = self._get_arr_index(key)
            return self.raw_data[arr_ind][arr_offset]
        else:
            assert isinstance(key, slice) and key.step is None
            start, stop = key.start, key.stop
            if start is None:
                start = 0
            if stop is None:
                stop = len(self)
            start_arr_ind, start_arr_offset = self._get_arr_index(start)
            stop_arr_ind, stop_arr_offset = self._get_arr_index(stop - 1)
            if start_arr_ind == stop_arr_ind:
                result = np.array(
                    self.raw_data[start_arr_ind][start_arr_offset:stop_arr_offset + 1], dtype=self.dtype)
            else:
                cands = [self.raw_data[start_arr_ind][start_arr_offset:]]
                for i in range(start_arr_ind+1, stop_arr_ind):
                    cands.append(self.raw_data[i])
                cands.append(self.raw_data[stop_arr_ind][:stop_arr_offset + 1])
                result = np.concatenate(cands)
            return result

    def __iter__(self):
        for d in self.raw_data:
            for dd in d:
                yield dd

    def reshape(self, *new_shape):
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        else:
            new_shape = tuple(new_shape)
        assert new_shape[0] == -1
        return RawData([d.reshape(new_shape) for d in self.raw_data], dtype=self.dtype, func=None)


# Multi-Process not useful now, since we don't have memory to CPU bottleneck
class Dataloader(object):
    def __init__(self, raw_data, batch_size, name='default', func=None, batch_func=None, shuffle=False, drop_last=True, offset=0, dtype=np.float32):
        self.func = func if func else lambda x: x
        self.dtype = dtype
        self.raw_data = RawData(raw_data, self.dtype, self.func)
        self.batch_size = batch_size
        self.drop_last = drop_last
        if batch_func is None:
            def batch_func(x): return x
        self.batch_func = batch_func
        self.shuffle = shuffle
        if isinstance(name, str):
            self.name = name
        else:
            assert isinstance(name, Iterable)
            self.name = tuple(name)
        self.dp_nrank = None
        self.parts = None
        self.slices = None
        self.batch_num = None
        self.batch_index = offset

    def set_batch_index(self, offset):
        if offset >= self.batch_num:
            offset = 0
        self.batch_index = offset

    def init_states(self):
        if self.dp_nrank is not None:
            # this part is only for data parallel
            cur_size = self.raw_data.shape[0] // self.dp_nrank
            start = cur_size * self.dp_rank
            ending = start + cur_size
            self.raw_data = self.raw_data[start:ending]
        self.samples_num = len(self.raw_data)
        self.batch_size = int(self.batch_size)
        assert self.batch_size > 0, 'Batch size %d invalid.' % self.batch_size
        self.batch_num = self.get_batch_num(self.samples_num)
        if self.batch_index >= self.batch_num:
            self.batch_index = 0
        self.sample_shape = list(self.raw_data.shape[1:])
        self.shape = tuple([self.batch_size] + self.sample_shape)
        self.arr = ndarray.empty(
            self.shape, ctx=ndarray.cpu(0), dtype=self.dtype)
        # in case the last batch's shape is different, pre-allocate an array
        self.rest_arr = None
        if not self.drop_last:
            assert self.parts is None, 'Model parallel cannot use dataloader without drop_last.'
            res_num = self.samples_num % self.batch_size
            if res_num > 0 and res_num != self.batch_size:
                self.rest_arr = ndarray.empty(
                    tuple([res_num] + self.sample_shape), ctx=ndarray.cpu(0), dtype=self.dtype)
        self.set_slices()

    def _get_arr(self, batchind):
        batchind = self.all_batch_indices[batchind]
        index = batchind * self.batch_size
        res = self.raw_data[index:index+self.batch_size]
        if res.shape[0] != self.batch_size:
            self.rest_arr[:] = res
            return self.rest_arr
        else:
            self.arr[:] = res
            return self.arr

    def get_arr(self):
        # step forward in this function
        res = self._get_arr(self.batch_index)
        self.last_batch_size = res.shape[0]
        self.batch_index = (self.batch_index + 1) % self.batch_num
        return res

    def get_next_arr(self):
        res = self._get_arr(self.batch_index)
        self.last_batch_size = res.shape[0]
        return res

    def set_dp_rank(self, dp_rank, dp_nrank):
        # in data parallel
        if self.dp_nrank is not None:
            assert self.dp_rank == dp_rank
            assert self.dp_nrank == dp_nrank
        self.dp_rank = dp_rank
        self.dp_nrank = dp_nrank

    def set_mp_parts(self, cur_part, parts):
        # in model parallel
        if self.parts is not None:
            assert self.parts == parts
            assert self.cur_part == cur_part
        self.cur_part = cur_part
        self.parts = parts

    def set_slices(self):
        if self.parts is None:
            return
        # in model parallel
        ori_shape = self.shape
        new_shape = []
        slcs = []
        for i in range(len(ori_shape)):
            if i in self.parts:
                part = ori_shape[i] // self.parts[i]
                st = part * self.cur_part[i]
                en = st + part
            else:
                st = 0
                en = ori_shape[i]
            slcs.append(slice(st, en))
            new_shape.append(en - st)
        self.slices = tuple(slcs)
        self.shape = tuple(new_shape)

    def reshape_tensor(self, tensor):
        if self.slices is None:
            return self.batch_func(tensor)
        return self.batch_func(tensor[self.slices])

    def get_batch_num(self, samples_num=None):
        if samples_num is None:
            samples_num = len(self.raw_data)
        return int(np.ceil(samples_num / self.batch_size)) \
            if not self.drop_last else samples_num // self.batch_size

    def get_cur_shape(self):
        return tuple(self.arrs[self.arr_map[self.batch_index]].shape)


class GNNDataLoaderOp(Op):
    graph = None
    nxt_graph = None

    def __init__(self, handler, ctx=ndarray.cpu(0)):
        super().__init__(DataloaderOp, [], ctx)
        self.on_gpu = True
        self.on_cpu = False
        self.handler = handler
        self.name = "GNNDataloaderOp"

    @ property
    def desc(self):
        return self.name

    def get_batch_num(self, name):
        return None

    def get_arr(self, name):
        return self.handler(self.graph)

    def get_next_arr(self, name):
        return self.handler(self.nxt_graph)

    def gradient(self, output_grad):
        return None

    def infer_shape(self, input_shapes):
        raise NotImplementedError

    @ classmethod
    def step(cls, graph):
        cls.graph = cls.nxt_graph
        cls.nxt_graph = graph


class DataloaderOp(Op):
    def __init__(self, dataloaders, dtype=np.float32):
        super().__init__(DataloaderOp, [], ndarray.cpu(0))
        self.on_gpu = False
        self.on_cpu = True
        self.dataloaders = {}
        for dl in dataloaders:
            if isinstance(dl.name, tuple):
                self.dataloaders.update({name: dl for name in dl.name})
            else:
                self.dataloaders[dl.name] = dl
            assert dl.dtype == dtype
        self.name = "DataloaderOp%d(%s)" % (
            self.id, '_'.join(self.dataloaders.keys()))
        self.dtype = dtype

    @ property
    def desc(self):
        return self.name

    def set_batch_index(self, name, offset):
        self.dataloaders[name].set_batch_index(offset)

    def set_dp_rank(self, dp_rank, dp_nrank):
        for dataloader in self.dataloaders.values():
            dataloader.set_dp_rank(dp_rank, dp_nrank)

    def set_mp_parts(self, cur_part, parts):
        for dataloader in self.dataloaders.values():
            dataloader.set_mp_parts(cur_part, parts)

    def get_batch_num(self, name):
        if self.dataloaders[name].batch_num is None:
            return self.dataloaders[name].get_batch_num()
        else:
            return self.dataloaders[name].batch_num

    def get_arr(self, name):
        return self.dataloaders[name].get_arr()

    def get_next_arr(self, name):
        return self.dataloaders[name].get_next_arr()

    def gradient(self, output_grad):
        return None

    def infer_shape(self, input_shapes):
        # actually this function can never be called
        raise NotImplementedError

    def forward_hook(self, config):
        pass

    def backward_hook(self, config):
        if hasattr(config, 'min_dp_nrank'):
            min_dp_nrank = config.min_dp_nrank
        else:
            min_dp_nrank = None
        if not hasattr(config, 'dataloader_states'):
            config.dataloader_states = {}
        for d in self.dataloaders.values():
            if min_dp_nrank is not None:
                # now we enforce stages with dataloaders to have the minimum data parallel degree
                assert d.dp_nrank == min_dp_nrank
            d.init_states()
            if d.name not in config.dataloader_states:
                config.dataloader_states[d.name] = BatchIndices(
                    d.batch_num, need_shuffle=d.shuffle)
            else:
                config.dataloader_states[d.name].assert_attr(d)
            d.all_batch_indices = config.dataloader_states[d.name]


def dataloader_op(dataloaders, dtype=np.float32):
    '''
    dataloaders: list of dataloaders
    '''
    temp_dataloaders = []
    for dl in dataloaders:
        if isinstance(dl, Dataloader):
            temp_dataloaders.append(dl)
        elif isinstance(dl, list):
            temp_dataloaders.append(Dataloader(*dl, dtype=dtype))
        elif isinstance(dl, dict):
            temp_dataloaders.append(Dataloader(**dl, dtype=dtype))
        else:
            assert False, 'Dataloader parameter invalid.'
    return DataloaderOp(temp_dataloaders, dtype)
