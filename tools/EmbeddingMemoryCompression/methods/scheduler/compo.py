from .base import EmbeddingTrainer
from ..layers import CompositionalEmbedding
import math


class CompoEmbTrainer(EmbeddingTrainer):
    def _get_single_memory(self, nemb, divisor):
        return math.ceil(nemb / divisor) + math.ceil(divisor)

    def _get_collision(self):

        def multi_evaluate(x):
            memory = 0
            for nemb in self.num_embed_separate:
                if nemb > threshold:
                    newmem = self._get_single_memory(nemb, x)
                    memory += min(nemb, newmem)
                else:
                    memory += nemb
            return target_nemb - memory

        def single_evaluate(x):
            memory = self._get_single_memory(self.num_embed, x)
            return target_nemb - memory

        threshold = self.embedding_args['threshold']
        if self.use_multi:
            evaluate = multi_evaluate
        else:
            evaluate = single_evaluate
        target_nemb = self.num_embed * self.compress_rate
        res = self.binary_search(1, math.sqrt(self.num_embed), evaluate)
        res = math.ceil(res[0])
        self.log_func(
            f'Collision {res} given compression rate {self.compress_rate}.')
        return res

    def _decompo(self, num, collision):
        if num <= collision:
            return num
        another = math.ceil(num / collision)
        if num <= another + collision:
            return num
        return (another, collision)

    def get_single_embed_layer(self, nemb, collision, name):
        return CompositionalEmbedding(
            math.ceil(nemb / collision),
            collision,
            self.embedding_dim,
            aggregator=self.embedding_args['aggregator'],
            initializer=self.initializer,
            name=name,
            ctx=self.ectx
        )

    def get_embed_layer(self):
        collision = self._get_collision()
        all_size = 0
        if self.use_multi:
            emb = []
            threshold = self.embedding_args['threshold']
            for i, nemb in enumerate(self.num_embed_separate):
                cur_compo = self._decompo(nemb, collision)
                if nemb > threshold and isinstance(cur_compo, tuple):
                    all_size += self._get_single_memory(nemb, collision)
                    emb.append(self.get_single_embed_layer(
                        nemb, collision, f'CompoEmb({self.compress_rate})_{i}'))
                else:
                    all_size += nemb
                    emb.append(super().get_single_embed_layer(
                        nemb, f'Embedding_{i}'))
        else:
            all_size = self._get_single_memory(self.num_embed, collision)
            emb = self.get_single_embed_layer(
                self.num_embed, collision, f'CompoEmb({self.compress_rate})')
        self.log_func(
            f'Real compress rate: {all_size / self.num_embed}')
        return emb
