# only for DLRMs now

from .base import BaseTrainer
from .switchinference import SwitchInferenceTrainer
from .autodim import AutoDimTrainer


def get_trainer(layer):
    from ..layers.compressed_embedding import \
        DPQEmbedding, \
        AutoDimEmbedding, \
        DeepLightEmbedding

    trainer_mapping = {
        # Embedding: BaseTrainer,
        # MultipleEmbedding: BaseTrainer,
        # RobeEmbedding: BaseTrainer,
        # HashEmbedding: BaseTrainer,
        # MultipleHashEmbedding: BaseTrainer,
        # CompositionalEmbedding: BaseTrainer,
        # DeepHashEmbedding: BaseTrainer,
        # MDEmbedding: BaseTrainer,
        # QuantizedEmbedding: BaseTrainer,
        DPQEmbedding: SwitchInferenceTrainer,
        DeepLightEmbedding: SwitchInferenceTrainer,
        AutoDimEmbedding: AutoDimTrainer,
    }

    return trainer_mapping.get(type(layer), BaseTrainer)
