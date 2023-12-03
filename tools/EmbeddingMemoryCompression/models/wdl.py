import hetu as ht
import hetu.layers as htl
from .base import CTRModel_Head


class WDL_Head(CTRModel_Head):
    def __init__(
        self,
        embed_dim=16,
        sparse_slot=26,
        dense_slot=13,
    ) -> None:
        super().__init__(embed_dim, sparse_slot, dense_slot)
        input_dim = embed_dim * sparse_slot + dense_slot
        self.reshape = htl.Reshape((-1, sparse_slot * embed_dim))
        self.deep_fc = self.create_mlp(
            [input_dim, 256, 256, 1], sigmoid_layer=2, name='deep')
        self.wide_fc = htl.Linear(
            input_dim, 1, initializer=ht.init.GenXavierNormal(), bias=False, name="wide")

    def __call__(self, sparse_input, dense_input, label):
        sparse_input = self.reshape(sparse_input)
        if dense_input is None:
            all_input = sparse_input
        else:
            all_input = ht.concat_op(sparse_input, dense_input, axis=1)
        deep_logit = self.deep_fc(all_input)
        wide_logit = self.wide_fc(all_input)
        logit = ht.add_op(deep_logit, wide_logit)
        return self.output(logit, label)
