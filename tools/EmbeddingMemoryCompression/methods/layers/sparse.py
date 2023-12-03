import hetu as ht
from hetu.layers import Embedding


class SparseEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, form, name='embedding', ctx=None):
        # only for inference
        self.form = form
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        from hetu.ndarray import ND_Sparse_Array
        embeddings = ND_Sparse_Array(
            self.num_embeddings, self.embedding_dim, ctx=self.ctx)
        self.sparse_embedding_table = ht.placeholder_op(
            f'{self.name}_sparse', value=embeddings)

    def __call__(self, x):
        with ht.context(self.ctx):
            return ht.sparse_embedding_lookup_op(self.sparse_embedding_table, x)

    def make_inference(self, embed_input):
        # here from train to inference
        with ht.context(self.ctx):
            # not for validate; convert to csr format for inference
            from hetu.ndarray import dense_to_sparse
            embeddings = dense_to_sparse(
                self.embedding_table.tensor_value, form=self.form)
            self.sparse_embedding_table = ht.placeholder_op(
                f'{self.name}_sparse', value=embeddings)
            return ht.sparse_embedding_lookup_op(self.sparse_embedding_table, embed_input)
