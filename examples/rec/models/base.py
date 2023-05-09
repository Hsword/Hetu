import hetu as ht
import hetu.layers as htl


class RatingModel_Head(object):
    # model without embedding layer
    # by default, use NCF
    def __init__(
        self,
        embed_dim=160,
        nsparse=2,
        ndense=0,
    ):
        self.embed_dim = embed_dim
        self.loss_fn = htl.MSELoss()
        self.loss_fn_mae = htl.MAELoss()

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

    def __call__(self, user_input, item_input, label):
        # here the user_input and the item_input are the output of embedding layer
        raise NotImplementedError

    def output(self, y, label):
        return self.loss_fn(y, label), self.loss_fn_mae(y,label),y
