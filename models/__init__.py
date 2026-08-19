"""
EEG Conformer Models Package.
"""

from models.blocks import (
    CausalTransformerEncoderBlock,
    Conv1DClassifierBackbone,
    ConvFrontEnd,
    PositionalEncoding,
)
from models.conformer import CausalEEGConformer, FullPipelineModel
from models.heads import (
    NextTokenPredictionHead,
    SeizureOccurrenceAndTimingHead,
    SeizurePreictalClassifierHead,
    SeizureTypeClassifierHead,
)
from models.simplified_conformer import SimplifiedEEGConformer

__all__ = [
    "ConvFrontEnd",
    "PositionalEncoding",
    "CausalTransformerEncoderBlock",
    "Conv1DClassifierBackbone",
    "NextTokenPredictionHead",
    "SeizureOccurrenceAndTimingHead",
    "SeizureTypeClassifierHead",
    "SeizurePreictalClassifierHead",
    "CausalEEGConformer",
    "SimplifiedEEGConformer",
    "FullPipelineModel",
]
