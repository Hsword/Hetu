import hetu as ht
import hetu.layers as htl


class WDLHead(object):
    def __init__(self, num_dim, sparse_slot, dense_slot) -> None:
        input_dim = num_dim * sparse_slot + dense_slot
        initializer = ht.init.GenNormal(stddev=0.01)
        self.reshape = htl.Reshape((-1, sparse_slot * num_dim))
        self.deep_fc = htl.Sequence(
            htl.Linear(input_dim, 256, activation=ht.relu_op,
                       initializer=initializer, name="deep1"),
            htl.Linear(256, 256, activation=ht.relu_op,
                       initializer=initializer, name="deep2"),
            htl.Linear(256, 1, initializer=initializer, name="deep3"),
        )
        self.wide_fc = htl.Linear(
            input_dim, 1, initializer=initializer, bias=False, name="wide")

    def __call__(self, sparse_input, dense_input):
        sparse_input = self.reshape(sparse_input)
        all_input = ht.concat_op(sparse_input, dense_input, axis=1)
        deep_logit = self.deep_fc(all_input)
        wide_logit = self.wide_fc(all_input)
        logit = ht.add_op(deep_logit, wide_logit)
        logit = ht.sigmoid_op(logit)
        return logit
