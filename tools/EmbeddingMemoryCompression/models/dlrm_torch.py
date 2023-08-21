from __future__ import absolute_import, division, print_function, unicode_literals

# numpy
import numpy as np

# pytorch
import torch
import torch.nn as nn


class DLRM_Head(nn.Module):
    def __init__(
        self,
        embed_dim=16,
        sparse_slot=26,
        dense_slot=13,
        ln_bot=[512, 256, 64],
        ln_top=[512, 256],
    ):
        super(DLRM_Head, self).__init__()
        self.sparse_slot = sparse_slot
        self.dense_slot = dense_slot
        self.embed_dim = embed_dim
        if dense_slot > 0:
            # for criteo, criteo-TBs
            ln_bot = [dense_slot, ] + ln_bot + [embed_dim, ]
            self.bot_mlp = self.create_mlp(ln_bot)
            ln_top = [embed_dim + sparse_slot *
                      (sparse_slot + 1) // 2, ] + ln_top + [1, ]
        else:
            # for avazu
            ln_top = [sparse_slot * (sparse_slot - 1) // 2, ] + ln_top + [1, ]
        self.top_mlp = self.create_mlp(ln_top, len(ln_top) - 2)

    def create_mlp(self, ln, sigmoid_layer=-1):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, len(ln) - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            mean = 0.0
            std_dev = np.sqrt(2 / (m + n))
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                # use bce loss with logits, no sigmoid here
                # layers.append(nn.Sigmoid())
                ...
            else:
                layers.append(nn.ReLU())

        return torch.nn.Sequential(*layers)

    def apply_mlp(self, x, layers):
        return layers(x)

    def interact_features(self, sparse_vec, dense_vec):
        if dense_vec is None:
            T = sparse_vec
        else:
            # concatenate dense and sparse features
            (batch_size, d) = dense_vec.shape
            T = torch.cat(
                [dense_vec.view((batch_size, -1, d)), sparse_vec], dim=1)
        # perform a dot product
        Z = torch.bmm(T, torch.transpose(T, 1, 2))
        # append dense feature with the interactions (into a row vector)
        _, ni, nj = Z.shape
        li, lj = torch.tril_indices(ni, nj, offset=-1)
        Zflat = Z[:, li, lj]
        if dense_vec is None:
            R = Zflat
        else:
            # concatenate dense features and interactions
            R = torch.cat([dense_vec, Zflat], dim=1)
        return R

    def forward(self, sparse_input, dense_input):
        if self.dense_slot > 0:
            # process dense features (using bottom mlp), resulting in a row vector
            x = self.apply_mlp(dense_input, self.bot_mlp)
            # interact features (dense and sparse)
            z = self.interact_features(sparse_input, x)
        else:
            # interact features (dense and sparse)
            z = self.interact_features(sparse_input, None)

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_mlp)

        return p
