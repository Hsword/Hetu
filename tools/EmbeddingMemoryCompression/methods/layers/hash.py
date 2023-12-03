import hetu as ht
from hetu.layers import Embedding


class HashEmbedding(Embedding):
    def __call__(self, x):
        # ref MLSys20, HierPS
        with ht.context(self.ctx):
            sparse_input = ht.mod_hash_op(x, self.num_embeddings)
            return ht.embedding_lookup_op(self.embedding_table, sparse_input)
