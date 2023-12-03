import hetu as ht
import hetu.layers as htl


class CTRModel_Head(object):
    # model without embedding layer
    # by default, use kaggle criteo dataset
    def __init__(
        self,
        embed_dim=16,
        sparse_slot=26,
        dense_slot=13,
    ):
        self.sparse_slot = sparse_slot
        self.dense_slot = dense_slot
        self.embed_dim = embed_dim
        self.loss_fn = htl.BCEWithLogitsLoss()

    def create_mlp(self, ln, sigmoid_layer=-1, name='mlp'):
        layers = []
        for i in range(len(ln) - 1):
            n = ln[i]
            m = ln[i + 1]

            if i == sigmoid_layer:
                # use bce loss with logits, no sigmoid here
                act = None
            else:
                act = ht.relu_op
            LL = htl.Linear(int(n), int(m), initializer=ht.init.GenXavierNormal(
            ), activation=act, name=f'{name}_{i*2}')
            layers.append(LL)

        return htl.Sequence(*layers)

    def __call__(self, sparse_input, dense_input, label):
        # here the sparse_input is the output of embedding layer
        raise NotImplementedError

    def output(self, y, label):
        return self.loss_fn(y, label), ht.sigmoid_op(y)
