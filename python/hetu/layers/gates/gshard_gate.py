from .naive_gate import NaiveGate
import hetu as ht

class GshardGate(NaiveGate):
    def __init__(self, d_model, num_expert, world_size, topk=2, capacity=(1.2, 2.4), random_routing=True):
        assert topk==2, 'topk should be 2 in gshard'
        super().__init__(d_model, num_expert, world_size, topk=2)
        self.capa
