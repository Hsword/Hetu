import hetu as ht
from .sparse import SparseEmbedding


class DeepLightEmbedding(SparseEmbedding):
    def __init__(self, num_embeddings, embedding_dim, prune_rate, form, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        assert form in ('coo', 'csr')
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(self.num_embeddings, self.embedding_dim), name=self.name, ctx=ctx)
        self.prune_rate = prune_rate
        self.form = form

    def __call__(self, x):
        with ht.context(self.ctx):
            return ht.embedding_lookup_op(self.embedding_table, x)

    def make_adaptive_rate(self, batch_num):
        ignore_iter = 0 * batch_num

        def updater(n_iter):
            if n_iter <= ignore_iter:
                adaptive_sparse = 0
            else:
                real_niter = n_iter - ignore_iter
                if real_niter % 10 == 0 or real_niter % batch_num == 0:
                    adaptive_sparse = self.prune_rate * \
                        (1 - 0.99**(real_niter / 100.))
                else:
                    adaptive_sparse = 0
            return adaptive_sparse
        return updater

    def make_prune_op(self, y_, buffer_conf):
        batch_num = y_.get_batch_num('train')
        rate_updater = self.make_adaptive_rate(batch_num)
        return ht.prune_low_magnitude_op(self.embedding_table, rate_updater, buffer_conf=buffer_conf)
