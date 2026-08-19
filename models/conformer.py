"""
Causal EEG Conformer Architecture implementation.
"""

from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn

from models.blocks import (
    CausalTransformerEncoderBlock,
    ConvFrontEnd,
    PositionalEncoding,
)
from models.heads import (
    NextTokenPredictionHead,
    SeizureOccurrenceAndTimingHead,
    SeizureTypeClassifierHead,
)


class CausalEEGConformer(nn.Module):
    """Complete 3-Stage Model Architecture:
    1. Preictal next-token prediction head (horizon generation).
    2. Seizure Occurrence & Timing Classifier (IF and WHEN).
    3. Seizure Type Classifier (TYPE).
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

        # Task 1: Next-Token Prediction / Horizon Generation Head
        self.pred_head = NextTokenPredictionHead(embed_dim=embed_dim, target_dim=embed_dim, dropout=dropout)

        # Task 2: Seizure Occurrence (IF) and Timing (WHEN) Classifier
        self.occurrence_timing_head = SeizureOccurrenceAndTimingHead(
            embed_dim=embed_dim, hidden_dim=ffn_dim // 2, dropout=dropout
        )

        # Task 3: Seizure Type Classifier
        self.type_head = SeizureTypeClassifierHead(
            embed_dim=embed_dim, num_classes=num_classes, dropout=dropout
        )

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
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
        steps = horizon_steps if horizon_steps is not None else self.default_horizon_tokens
        curr_tokens = seed_tokens.clone()

        generated_list = []
        for _ in range(steps):
            h_encoded = self.encode_tokens(curr_tokens)
            next_token = self.pred_head(h_encoded[:, -1:, :])
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
        steps = horizon_steps if horizon_steps is not None else self.default_horizon_tokens

        preictal_tokens = self.front_end(x)
        preictal_features = self.encode_tokens(preictal_tokens)
        pred_next_tokens = self.pred_head(preictal_features)

        if generate_horizon_tokens:
            generated_horizon_tokens = self.generate_horizon(preictal_tokens, horizon_steps=steps)
        else:
            if pred_next_tokens.shape[1] >= steps:
                generated_horizon_tokens = pred_next_tokens[:, -steps:, :]
            else:
                generated_horizon_tokens = pred_next_tokens

        if use_horizon_context:
            combined_tokens = torch.cat([preictal_tokens, generated_horizon_tokens.detach()], dim=1)
            full_sequence_features = self.encode_tokens(combined_tokens)
        else:
            full_sequence_features = preictal_features

        occurrence_logits, onset_preds = self.occurrence_timing_head(full_sequence_features)
        type_logits = self.type_head(full_sequence_features)

        return {
            "pred_next_tokens": pred_next_tokens,
            "preictal_tokens": preictal_tokens,
            "generated_horizon_tokens": generated_horizon_tokens,
            "full_sequence_features": full_sequence_features,
            "occurrence_logits": occurrence_logits,
            "onset_preds": onset_preds,
            "type_logits": type_logits,
        }


# Backwards compatibility alias
FullPipelineModel = CausalEEGConformer
