"""
Simplified EEG-Conformer for Preictal Horizon Generation, Seizure Occurrence & Timing,
Binary Preictal Classification, and Preictal Seizure Type Classification.

Architecture:
1. ConvFrontEnd & PositionalEncoding: Converts (B, 1, C, T) raw EEG windows into sequence patch tokens.
2. Main Causal Transformer Encoder Stack: Standard causal attention over sequence patches.
3. Head 1 (Horizon Next-Token Generator): Autoregressively generates EEG tokens representing the horizon between preictal phase and seizure onset.
4. Head 2 (Seizure Occurrence & Timing Classifier): Predicts IF a seizure occurs (occurrence logit) and WHEN it occurs (onset offset).
5. Head 3 (Preictal vs. Not-Preictal Classifier): Binary classification head predicting whether the input window is preictal vs not preictal.
6. Additional Transformer Layer & Head 4 (Preictal Seizure Type Classifier): An extra Causal Transformer layer specifically refining sequence representations, followed by a multi-class head classifying preictal seizure types.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from conformer import (
    ConvFrontEnd,
    PositionalEncoding,
    CausalTransformerEncoderBlock,
    NextTokenPredictionHead,
    Conv1DClassifierBackbone,
    SeizureOccurrenceAndTimingHead,
    SeizureTypeClassifierHead,
)


class SeizurePreictalClassifierHead(nn.Module):
    """
    Head 3: 1D CNN-based binary classification head predicting whether the window/sequence
    is Preictal vs Not Preictal.
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.backbone = Conv1DClassifierBackbone(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout)
        feat_dim = hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, seq_len, embed_dim)
        Returns: preictal_logits: (B, 1)
        """
        feat = self.backbone(sequence_features)
        return self.classifier(feat)


class SimplifiedEEGConformer(nn.Module):
    """
    Simplified EEG-Conformer Architecture:
    - Shared Patch Embedding & Causal Transformer Backbone.
    - Head 1: Horizon Next-Token Generator (pred_head).
    - Head 2: Seizure Occurrence (IF) and Timing (WHEN) Classifier (occurrence_timing_head).
    - Head 3: Binary Classification Head for Preictal vs Not Preictal (preictal_head).
    - Dedicated Layer: Additional CausalTransformerEncoderBlock for preictal representation refinement.
    - Head 4: Classification Head for different Preictal Seizure Types (preictal_type_head).
    """

    def __init__(
        self,
        n_channels: int,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 512,
        temp_kernel: int = 25,
        stride: int = 10,
        dropout: float = 0.3,
        max_len: int = 2000,
        num_classes: int = 2,
        default_horizon_tokens: int = 10,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.stride = stride
        self.default_horizon_tokens = default_horizon_tokens
        self.num_classes = num_classes

        # 1. Front-end Patch Embedding & Positional Encoding
        self.front_end = ConvFrontEnd(
            n_channels=n_channels,
            embed_dim=embed_dim,
            temp_kernel=temp_kernel,
            stride=stride,
            dropout=dropout,
        )
        self.pos_encoder = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)

        # Causal Transformer Stack
        self.blocks = nn.ModuleList([
            CausalTransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)

        # Head 1: Next-Token Prediction / Horizon Generation Head
        self.pred_head = NextTokenPredictionHead(embed_dim=embed_dim, target_dim=embed_dim, dropout=dropout)

        # Head 2: Seizure Occurrence (IF) and Timing (WHEN) Classifier
        self.occurrence_timing_head = SeizureOccurrenceAndTimingHead(
            embed_dim=embed_dim, hidden_dim=ffn_dim // 2, dropout=dropout
        )

        # Head 3: Preictal vs. Not Preictal Binary Classifier
        self.preictal_head = SeizurePreictalClassifierHead(
            embed_dim=embed_dim, hidden_dim=ffn_dim // 2, dropout=dropout
        )

        # Additional Layer specifically for refining preictal features before classification
        self.preictal_type_layer = CausalTransformerEncoderBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.ln_type = nn.LayerNorm(embed_dim)

        # Head 4: Multi-class Preictal Seizure Type Classifier Head
        self.preictal_type_head = SeizureTypeClassifierHead(
            embed_dim=embed_dim, num_classes=num_classes, dropout=dropout
        )

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Passes a token sequence through Positional Encoding and standard Causal Transformer blocks.
        """
        seq_len = tokens.shape[1]
        h = self.pos_encoder(tokens)
        causal_mask = self._generate_causal_mask(seq_len, device=tokens.device)
        for block in self.blocks:
            h = block(h, causal_mask)
        return self.ln_f(h)

    def generate_horizon(
        self,
        seed_tokens: torch.Tensor,
        horizon_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Head 1: Autoregressively generates 'horizon_steps' patch tokens from preictal window tokens.
        """
        steps = horizon_steps if horizon_steps is not None else self.default_horizon_tokens
        curr_tokens = seed_tokens.clone()

        generated_list = []
        for _ in range(steps):
            h_encoded = self.encode_tokens(curr_tokens)
            next_token = self.pred_head(h_encoded[:, -1:, :])  # (B, 1, embed_dim)
            generated_list.append(next_token)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

        if generated_list:
            gen_tokens = torch.cat(generated_list, dim=1)
        else:
            gen_tokens = torch.empty(seed_tokens.shape[0], 0, self.embed_dim, device=seed_tokens.device)

        return gen_tokens

    def forward(
        self,
        x: torch.Tensor,
        horizon_steps: Optional[int] = None,
        generate_horizon_tokens: bool = False,
        use_horizon_context: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Input x: Preictal window EEG of shape (B, n_channels, n_samples)

        Returns dictionary containing:
            - 'pred_next_tokens': Next token predictions for ground-truth preictal sequence (B, num_patches, embed_dim)
            - 'preictal_tokens': Ground truth preictal patch tokens extracted by ConvFrontEnd (B, num_patches, embed_dim)
            - 'generated_horizon_tokens': Autoregressively generated horizon tokens (B, horizon_steps, embed_dim)
            - 'full_sequence_features': Encoded features used by classifier heads (B, total_seq_len, embed_dim)
            - 'occurrence_logits': Seizure occurrence logits (B, 1) - Head 2 IF
            - 'onset_preds': Seizure onset offset predictions (B, 1) - Head 2 WHEN
            - 'preictal_logits': Preictal vs Not Preictal logits (B, 1) - Head 3
            - 'preictal_type_logits': Preictal seizure type classification logits (B, num_classes) - Head 4
        """
        steps = horizon_steps if horizon_steps is not None else self.default_horizon_tokens

        # 1. Front-end patch extraction of preictal phase window
        preictal_tokens = self.front_end(x)  # (B, N_pre, embed_dim)

        # 2. Causal encoder pass over the preictal window
        preictal_features = self.encode_tokens(preictal_tokens)
        pred_next_tokens = self.pred_head(preictal_features)  # (B, N_pre, embed_dim)

        # 3. Horizon tokens: Autoregressive generation if explicitly requested, otherwise parallel slice
        if generate_horizon_tokens:
            generated_horizon_tokens = self.generate_horizon(preictal_tokens, horizon_steps=steps)
        else:
            if pred_next_tokens.shape[1] >= steps:
                generated_horizon_tokens = pred_next_tokens[:, -steps:, :]
            else:
                generated_horizon_tokens = pred_next_tokens

        # 4. Sequence feature formulation for downstream heads
        if use_horizon_context:
            combined_tokens = torch.cat([preictal_tokens, generated_horizon_tokens.detach()], dim=1)
            full_sequence_features = self.encode_tokens(combined_tokens)
        else:
            full_sequence_features = preictal_features

        # Head 2: Seizure Occurrence (IF) and Timing (WHEN)
        occurrence_logits, onset_preds = self.occurrence_timing_head(full_sequence_features)

        # Head 3: Binary Classification (Preictal vs Not Preictal)
        preictal_logits = self.preictal_head(full_sequence_features)

        # Additional Transformer Layer for refining sequence features prior to preictal type head
        seq_len = full_sequence_features.shape[1]
        causal_mask = self._generate_causal_mask(seq_len, device=full_sequence_features.device)
        preictal_type_features = self.ln_type(
            self.preictal_type_layer(full_sequence_features, causal_mask)
        )

        # Head 4: Multi-Class Preictal Seizure Type Classification
        preictal_type_logits = self.preictal_type_head(preictal_type_features)

        return {
            "pred_next_tokens": pred_next_tokens,
            "preictal_tokens": preictal_tokens,
            "generated_horizon_tokens": generated_horizon_tokens,
            "full_sequence_features": full_sequence_features,
            "occurrence_logits": occurrence_logits,
            "onset_preds": onset_preds,
            "preictal_logits": preictal_logits,
            "preictal_type_logits": preictal_type_logits,
        }


if __name__ == "__main__":
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
