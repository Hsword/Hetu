import numpy as np
import pickle
from collections import defaultdict
import contextlib

from ..context import DeviceGroup, DistConfig
from ..ndarray import gpu, rgpu
from ..gpu_ops.Variable import PlaceholderOp


class Strategy(object):
    def __init__(self, save_path=None):
        # TODO: modify executor's logic to use communicators
        self.settings = DistConfig('/tmp/hetu_config.yml')
        self.save_path = save_path

    def set_raw_ctxs_n_states(self, node_list, memory_pool):
        raise NotImplementedError

    def set_overlap(self, overlap):
        self.overlap = overlap
