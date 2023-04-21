# only for DLRMs now

from .base import EmbeddingTrainer
from .hash import HashEmbTrainer
from .compo import CompoEmbTrainer
from .tensortrain import TTEmbTrainer
from .dhe import DHETrainer
from .robe import ROBETrainer
from .dpq import DPQTrainer
from .mgqe import MGQETrainer
from .adapt import AdaptEmbTrainer
from .md import MDETrainer
from .autodim import AutoDimTrainer
from .optembed import OptEmbedTrainer
from .deeplight import DeepLightTrainer
from .pep import PEPEmbTrainer
from .autosrh import AutoSrhTrainer
from .quantize import QuantizeEmbTrainer
from .alpt import ALPTEmbTrainer
from ..layers import Embedding, HashEmbedding, \
    CompositionalEmbedding, TensorTrainEmbedding, \
    DeepHashEmbedding, RobeEmbedding, \
    DPQEmbedding, MGQEmbedding, AdaptiveEmbedding, \
    MDEmbedding, AutoDimEmbedding, OptEmbedding, \
    DeepLightEmbedding, PEPEmbedding, AutoSrhEmbedding, \
    QuantizedEmbedding, ALPTEmbedding


_layer2trainer_mapping = {
    Embedding: EmbeddingTrainer,
    HashEmbedding: HashEmbTrainer,
    CompositionalEmbedding: CompoEmbTrainer,
    TensorTrainEmbedding: TTEmbTrainer,
    DeepHashEmbedding: DHETrainer,
    RobeEmbedding: ROBETrainer,
    DPQEmbedding: DPQTrainer,
    MGQEmbedding: MGQETrainer,
    AdaptiveEmbedding: AdaptEmbTrainer,
    MDEmbedding: MDETrainer,
    AutoDimEmbedding: AutoDimTrainer,
    OptEmbedding: OptEmbedTrainer,
    DeepLightEmbedding: DeepLightTrainer,
    PEPEmbedding: PEPEmbTrainer,
    AutoSrhEmbedding: AutoSrhTrainer,
    QuantizedEmbedding: QuantizeEmbTrainer,
    ALPTEmbedding: ALPTEmbTrainer,
}

_trainer2layer_mapping = {value: key for key,
                          value in _layer2trainer_mapping.items()}


def get_trainer(layer_type):
    trainer = _layer2trainer_mapping[layer_type]
    return trainer


def get_layer_type(trainer):
    layer_type = _trainer2layer_mapping[trainer]
    return layer_type
