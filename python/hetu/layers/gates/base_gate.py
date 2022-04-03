"""
Base gate
"""
import hetu as ht

class BaseGate(object):
    def __init__(self, num_expert, world_size):
        super.__init__()
        self.world_size = world_size;
        self.num_expert = num_expert;
        self.tot_expert = world_size * num_expert;
        self.loss = None

    def __call__(self):
        raise NotImplementedError("Base gate cannot be directly used.")

