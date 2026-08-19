"""
Simplified EEG-Conformer model definitions.

Backward-compatibility wrapper re-exporting all symbols from the `models` package.
"""

from models import *
from models.heads import SeizurePreictalClassifierHead
from models.simplified_conformer import SimplifiedEEGConformer

if __name__ == "__main__":
    import torch

    B, C, T = 2, 19, 1000
    dummy_eeg = torch.randn(B, C, T)

    model = SimplifiedEEGConformer(n_channels=C, num_classes=4, default_horizon_tokens=8)
    outputs = model(dummy_eeg, use_horizon_context=False)
    print("--- SimplifiedEEGConformer Standalone Test ---")
    print(f"Input EEG shape: {dummy_eeg.shape}")
    print(f"Preictal patch tokens shape: {outputs['preictal_tokens'].shape}")
    print(f"Next-token prediction shape (Head 1): {outputs['pred_next_tokens'].shape}")
    print(f"Generated horizon tokens shape: {outputs['generated_horizon_tokens'].shape}")
    print(f"Full sequence features shape: {outputs['full_sequence_features'].shape}")
    print(f"Occurrence logits shape (Head 2 IF): {outputs['occurrence_logits'].shape}")
    print(f"Onset predictions shape (Head 2 WHEN): {outputs['onset_preds'].shape}")
    print(f"Preictal vs Not-Preictal logits shape (Head 3): {outputs['preictal_logits'].shape}")
    print(f"Preictal seizure type logits shape (Head 4): {outputs['preictal_type_logits'].shape}")
