from .base import EmbeddingTrainer
from ..layers import AdaptiveEmbedding
import math


class AdaptEmbTrainer(EmbeddingTrainer):
    def get_data(self):
        top_percent = self.embedding_args['top_percent']
        embed_input, dense_input, y_ = super().get_data()
        if self.use_multi:
            remap_indices = self.dataset.get_separate_remap(
                [op.dataloaders[self.train_name].raw_data for op in embed_input], top_percent)
        else:
            remap_indices = self.dataset.get_whole_remap(
                embed_input.dataloaders[self.train_name].raw_data, top_percent)
        self.remap_indices = remap_indices
        return embed_input, dense_input, y_

    def get_single_embed_layer(self, nfreq, nrare, remap_indices, name):
        return AdaptiveEmbedding(
            nfreq,
            nrare,
            remap_indices,
            self.embedding_dim,
            initializer=self.initializer,
            name=name,
            ctx=self.ectx,
        )

    def _split_freq_rare(self, nemb):
        nemb = math.ceil(nemb * self.compress_rate)
        nfreq_emb = math.ceil(nemb * self.embedding_args['high_freq_ratio'])
        nrare_emb = nemb - nfreq_emb
        return nfreq_emb, nrare_emb

    def get_embed_layer(self):
        if self.use_multi:
            emb = []
            for i, nemb in enumerate(self.num_embed_separate):
                nfreq, nrare = self._split_freq_rare(nemb)
                if nrare > 0:
                    emb.append(self.get_single_embed_layer(
                        nfreq, nrare, self.remap_indices[i], f'AdaptEmb_{i}'))
                else:
                    emb.append(super().get_single_embed_layer(
                        nemb, f'Embedding_{i}'))
        else:
            nfreq, nrare = self._split_freq_rare(self.num_embed)
            assert nrare > 0
            emb = self.get_single_embed_layer(
                nfreq, nrare, self.remap_indices, 'AdaptEmb')
        return emb