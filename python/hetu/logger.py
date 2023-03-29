from .ndarray import NDArray, array
from .communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t

import numpy as np
import argparse
from collections.abc import Iterable
try:
    import wandb
    WANDB_IMPORT = True
except:
    WANDB_IMPORT = False


class HetuLogger(object):
    def __init__(self, rank, nrank, ctx, comm, handle):
        self._rank = rank
        self._nrank = nrank
        self._ctx = ctx
        self._comm = comm
        self._handle = handle
        self._dtype = ncclDataType_t.ncclFloat32
        self._reduce_op = ncclRedOp_t.ncclSum
        self._buffer = {}

    @property
    def rank(self):
        return self._rank

    @property
    def nrank(self):
        return self._nrank

    @property
    def need_log(self):
        return self.nrank is None or self.rank == 0

    def item(self, value):
        if isinstance(value, NDArray):
            value = value.asnumpy()
        if isinstance(value, np.ndarray):
            value = value.item()
        elif isinstance(value, Iterable):
            while isinstance(value, Iterable):
                assert len(value) == 1
                value = value[0]
        return value

    def log(self, name, value):
        assert name not in self._buffer, f'{name} already exists in log buffer!'
        value = self.item(value)
        self._buffer[name] = value

    def dist_log(self, name, value):
        if isinstance(value, NDArray):
            assert np.prod(value.shape) == 1
        elif isinstance(value, np.ndarray):
            assert np.prod(value.shape) == 1
            value = array(value)
        else:
            if isinstance(value, Iterable):
                assert len(value) == 1
            else:
                value = [value]
            value = array(np.array(value))
        dtype = value.dtype
        self._comm.dlarrayNcclReduce(
            value, value, 0, self._dtype, self._reduce_op, self._handle)
        if self.need_log:
            value = (value.asnumpy() / self.nrank).astype(dtype).item()
            self.log(name, value)

    def wrapped_log(self, name, value):
        if self.nrank is None:
            self.log(name, value)
        else:
            self.dist_log(name, value)

    def step(self):
        # logging here
        self._buffer.clear()

    def set_config(self, attrs):
        print(attrs)

    def __del__(self):
        if self._buffer != {}:
            self.step()


class WandbLogger(HetuLogger):
    def __init__(self, project, name, id, rank, nrank, ctx, comm, handle):
        # use single process to log finally
        assert WANDB_IMPORT, 'Import error, please install wandb first.'
        super().__init__(rank, nrank, ctx, comm, handle)
        self._project = project
        self._name = name
        if self.need_log:
            self.logger = wandb.init(
                project=project,
                name=name,
                id=id,
                resume="allow",
            )

    @property
    def name(self):
        return self._name

    def step(self):
        if self.need_log:
            wandb.log(self._buffer)
        self._buffer.clear()

    def set_config(self, attrs):
        print(attrs)
        if isinstance(attrs, argparse.Namespace):
            attrs = vars(attrs)
        assert isinstance(attrs, dict)
        if self.need_log:
            wandb.config.update(attrs, allow_val_change=True)

    def __del__(self):
        super().__del__()
        if self.need_log:
            self.logger.finish()
