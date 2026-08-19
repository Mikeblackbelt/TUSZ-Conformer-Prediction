"""
Causal EEG-Conformer model definitions.

Backward-compatibility wrapper re-exporting all symbols from the `models` package.
"""

from models import *
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
    SeizureTypeClassifierHead,
)

if __name__ == "__main__":
    import torch

    B, C, T = 2, 19, 1000
    dummy_eeg = torch.randn(B, C, T)

    model = CausalEEGConformer(n_channels=C, num_classes=3, default_horizon_tokens=8)
    outputs = model(dummy_eeg, use_horizon_context=False)
    print(f"Input EEG shape: {dummy_eeg.shape}")
    print(f"Preictal patch tokens shape: {outputs['preictal_tokens'].shape}")
    print(f"Next-token prediction shape: {outputs['pred_next_tokens'].shape}")
    print(f"Generated horizon tokens shape: {outputs['generated_horizon_tokens'].shape}")
    print(f"Full sequence features shape (gate closed): {outputs['full_sequence_features'].shape}")

    outputs_gated_open = model(dummy_eeg, use_horizon_context=True)
    print(f"Full sequence features shape (gate open):   {outputs_gated_open['full_sequence_features'].shape}")
    print(f"Seizure occurrence logits shape (IF): {outputs['occurrence_logits'].shape}")
    print(f"Seizure onset predictions shape (WHEN): {outputs['onset_preds'].shape}")
    print(f"Seizure type logits shape (TYPE): {outputs['type_logits'].shape}")