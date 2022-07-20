import hetu as ht
import hetu.layers as htl


class DLRM_Head(object):
    # DLRM model without embedding layer
    # by default, use kaggle criteo dataset
    def __init__(
        self,
        embed_dim=16,
        sparse_slot=26,
        dense_slot=13,
        ln_bot=[512, 256, 64],
        ln_top=[512, 256],
    ):
        self.sparse_slot = sparse_slot
        self.dense_slot = dense_slot
        self.embed_dim = embed_dim
        if dense_slot > 0:
            # for criteo, criteo-TBs
            ln_bot = [dense_slot, ] + ln_bot + [embed_dim, ]
            self.bot_mlp = self.create_mlp(ln_bot, name='bot_mlp')
            ln_top = [embed_dim + sparse_slot *
                      (sparse_slot + 1) // 2, ] + ln_top + [1, ]
        else:
            # for avazu
            ln_top = [sparse_slot * (sparse_slot - 1) // 2, ] + ln_top + [1, ]
        self.top_mlp = self.create_mlp(ln_top, len(ln_top) - 2, name='top_mlp')
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

    def interact_features(self, sparse_vec, dense_vec):
        # dense: (bs, dim); sparse: (bs, sp_slot, dim)
        assert self.dense_slot > 0 or dense_vec is None
        if dense_vec is None:
            all_features = sparse_vec
            # (bs, sp_slot + 1, dim)
        else:
            temp_dense_vec = ht.array_reshape_op(
                dense_vec, (-1, 1, self.embed_dim))
            # (bs, 1, dim)
            all_features = ht.concatenate_op(
                [temp_dense_vec, sparse_vec], axis=1)
            # (bs, sp_slot + 1, dim)
        interact = ht.batch_matmul_op(
            all_features, ht.transpose_op(all_features, perm=(0, 2, 1)))
        zflat = ht.tril_lookup_op(interact, -1)
        if dense_vec is None:
            result = zflat
        else:
            result = ht.concatenate_op([dense_vec, zflat], axis=1)
        return result

    def __call__(self, sparse_input, dense_input, label):
        # here the sparse_input is the output of embedding layer
        sparse_input = ht.array_reshape_op(
            sparse_input, (-1, self.sparse_slot, self.embed_dim))
        if self.dense_slot > 0:
            x = self.bot_mlp(dense_input)
            self.bot = x
            x = self.interact_features(sparse_input, x)
        else:
            x = self.interact_features(sparse_input, None)
        self.inter = x
        x = self.top_mlp(x)
        return self.loss_fn(x, label), ht.sigmoid_op(x)
