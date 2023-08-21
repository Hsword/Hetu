import hetu as ht
from hetu.layers import Embedding
import numpy as np
import os.path as osp


class DeepHashEmbedding(Embedding):
    def __init__(self, embedding_dim, mlp_dim, num_buckets, num_hash, nprs, dist='uniform', initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        assert dist in ('uniform', 'normal')
        self.distribution = dist
        self.embedding_dim = embedding_dim
        self.num_buckets = num_buckets
        self.num_hash = num_hash
        self.name = name
        self.ctx = ctx
        self.mlp_dim = mlp_dim
        prime_path = osp.join(osp.dirname(osp.abspath(
            __file__)), 'primes.npy')
        allprimes = np.load(prime_path)
        for i, p in enumerate(allprimes):
            if p >= num_buckets:
                break
        self.allprimes = allprimes[i:]
        self.slopes = self.make_random(nprs, 'slopes')
        self.biases = self.make_random(nprs, 'biases')
        self.primes = self.make_primes(nprs, 'primes')
        self.layers = self.make_layers(initializer)

    def make_layers(self, initializer):
        from hetu.layers.linear import Linear
        from hetu.layers.normalization import BatchNorm
        from hetu.layers.mish import Mish
        from hetu.layers.sequence import Sequence
        layers = [
            Linear(self.num_hash, self.mlp_dim,
                   initializer=initializer, name='linear1'),
            BatchNorm(self.mlp_dim, name='bn1'),
            Mish(),
        ]
        for i in range(4):
            layers.extend([
                Linear(self.mlp_dim, self.mlp_dim,
                       initializer=initializer, name=f'linear{i+2}'),
                BatchNorm(self.mlp_dim, name=f'bn{i+2}'),
                Mish(),
            ])
        layers.append(Linear(
            self.mlp_dim, self.embedding_dim, initializer=initializer, name='linear6'))
        return Sequence(*layers)

    def make_primes(self, nprs, name):
        primes = ht.placeholder_op(name=name, value=nprs.choice(
            self.allprimes, size=self.num_hash).astype(np.int32), dtype=np.int32, trainable=False)
        return primes

    def make_random(self, nprs, name):
        randoms = ht.placeholder_op(name=name, value=nprs.randint(
            1, self.num_buckets, size=self.num_hash).astype(np.int32), dtype=np.int32, trainable=False)
        return randoms

    def __call__(self, x):
        # KDD21, DHE
        x = ht.learn_hash_op(x, self.slopes, self.biases,
                             self.primes, self.num_buckets, self.distribution)
        x = ht.array_reshape_op(x, (-1, self.num_hash))
        x = self.layers(x)
        return x

    def __repr__(self):
        return f'{self.name}({self.mlp_dim})'
