from __future__ import absolute_import
import os
import numpy as np
import multiprocessing as mp

from . import ndarray
from .gpu_ops.Node import Op


# Multi-Process not useful now, since we don't have memory to CPU bottleneck
class Dataloader(object):
    def __init__(self, raw_data, batch_size, name='default', func=None, drop_last=True):
        self.func = func if func else lambda x: x
        self.raw_data = np.array(self.func(raw_data), np.float32)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.name = str(name)

    def init_states(self, rank=None, nrank=None):
        if rank is not None:
            cur_size = self.raw_data.shape[0] // nrank
            start = cur_size * rank
            ending = start + cur_size
            self.raw_data = self.raw_data[start:ending]
        self.samples_num = len(self.raw_data)
        self.queue_size = 3  # if use prefetch, needs 3; if only current batch, needs 2
        self.batch_size = min(int(self.batch_size),
                              self.samples_num // self.queue_size)
        assert self.batch_size > 0, 'Batch size %d invalid.' % self.batch_size
        self.batch_num = int(np.ceil(self.samples_num / self.batch_size)) if not self.drop_last else \
            self.samples_num // self.batch_size
        self.shape = tuple([self.batch_size] + list(self.raw_data.shape[1:]))
        self.seq = np.arange(self.samples_num)

        self.index = 0
        self.arrs = []
        self.arr_map = {}
        # prefetch to fill up the queue
        for i in range(self.queue_size):
            next_index = self.index + self.batch_size
            self.arrs.append(ndarray.array(
                self.raw_data[self.seq[self.index:next_index]], ctx=ndarray.cpu(0)))
            self.index = next_index
            self.arr_map[i] = i
        self.max_key = self.queue_size - 1

        # in case the last batch's shape is different, pre-allocate an array
        if not self.drop_last:
            res_num = self.samples_num % self.batch_size
            if res_num > 0:
                self.arrs.append(ndarray.empty(
                    tuple([res_num] + list(self.shape[1:])), ctx=ndarray.cpu(0)))
            self.rest = self.queue_size

        self.batch_index = 0

    def _get_arr(self, batchind):
        # get specific batch
        # if the batch to be fetched is the newest one, replace the oldest with new batch
        assert batchind in self.arr_map
        res = self.arrs[self.arr_map[batchind]]
        if batchind == self.max_key:
            self.max_key = (self.max_key + 1) % self.samples_num
            min_key = (self.max_key - self.queue_size) % self.samples_num
            if self.index >= self.samples_num or (self.drop_last and self.index + self.batch_size > self.samples_num):
                self.index = 0
            next_index = self.index + self.batch_size
            if next_index <= self.samples_num:
                temp_ind = self.arr_map.pop(min_key)
                if temp_ind == self.queue_size and not self.drop_last:
                    temp_ind = self.rest
                    self.rest = self.queue_size
                self.arr_map[self.max_key] = temp_ind
                self.arrs[temp_ind][:] = self.raw_data[self.seq[self.index:next_index]]
            else:
                assert not self.drop_last
                self.arrs[-1][:] = self.raw_data[self.seq[self.index:next_index]]
                self.rest = self.arr_map.pop(min_key)
                self.arr_map[self.max_key] = self.queue_size
            self.index = next_index
        return res

    def get_arr(self):
        # step forward in this function
        res = self._get_arr(self.batch_index)
        self.last_batch_size = res.shape[0]
        self.batch_index = (self.batch_index + 1) % self.samples_num
        return res

    def get_next_arr(self):
        res = self._get_arr(self.batch_index)
        return res

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
        self.desc = self.name

    def get_batch_num(self, name):
        return None

    def get_arr(self, name):
        return self.handler(self.graph)

    def get_next_arr(self, name):
        return self.handler(self.nxt_graph)

    def get_cur_shape(self, name):
        return self.handler(self.graph).shape

    def gradient(self, output_grad):
        return None

    def infer_shape(self, input_shapes):
        raise NotImplementedError

    @classmethod
    def step(cls, graph):
        cls.graph = cls.nxt_graph
        cls.nxt_graph = graph


class DataloaderOp(Op):
    def __init__(self, dataloaders):
        super().__init__(DataloaderOp, [], ndarray.cpu(0))
        self.on_gpu = False
        self.on_cpu = True
        self.dataloaders = {
            dl.name: dl for dl in dataloaders
        }
        self.name = "DataloaderOp%d(%s)" % (
            self.id, '_'.join(self.dataloaders.keys()))
        self.desc = self.name

    def get_batch_num(self, name):
        return self.dataloaders[name].batch_num

    def get_arr(self, name):
        return self.dataloaders[name].get_arr()

    def get_next_arr(self, name):
        return self.dataloaders[name].get_next_arr()

    def get_cur_shape(self, name):
        return self.dataloaders[name].get_cur_shape()

    def gradient(self, output_grad):
        return None

    def infer_shape(self, input_shapes):
        # actually this function can never be called
        raise NotImplementedError

    def forward_hook(self, config):
        pass

    def backward_hook(self, config):
        for d in self.dataloaders.values():
            if config.pipeline:
                d.init_states(config.pipeline_dp_rank, config.nrank // config.pipeline_nrank)
            elif config.context_launch:
                d.init_states(config.rank, config.nrank)
            else:
                d.init_states()


def dataloader_op(dataloaders):
    '''
    dataloaders: list of dataloaders
    '''
    temp_dataloaders = []
    for dl in dataloaders:
        if isinstance(dl, Dataloader):
            temp_dataloaders.append(dl)
        elif isinstance(dl, list):
            temp_dataloaders.append(Dataloader(*dl))
        elif isinstance(dl, dict):
            temp_dataloaders.append(Dataloader(**dl))
        else:
            assert False, 'Dataloader parameter invalid.'
    return DataloaderOp(temp_dataloaders)
