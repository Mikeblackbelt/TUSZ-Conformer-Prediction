"""
Causal EEG-Conformer for Preictal Generation, Seizure Detection & Timing, and Seizure Type Classification.

Architecture:
1. ConvFrontEnd & PositionalEncoding: Converts (B, 1, C, T) EEG preictal windows into sequence patch tokens.
2. Horizon Next-Token Generator: Autoregressively generates EEG tokens representing the horizon
   between the end of the preictal phase and seizure onset (longest distance / SPH).
3. Seizure Occurrence & Timing Classifier: Predicts IF a seizure occurs (occurrence logit)
   and WHEN it occurs (relative onset time/token offset).
4. Seizure Type Classifier: Multi-class classifier predicting seizure type downstream of detection.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFrontEnd(nn.Module):
    """
    EEG Spatial-Temporal Convolutional Front-End.
    Transforms (B, 1, n_channels, n_samples) raw window into a sequence of patch tokens.
    """

    def __init__(
        self,
        n_channels: int,
        embed_dim: int = 128,
        temp_kernel: int = 25,
        temp_filters: int = 32,
        stride: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.stride = stride

        # Temporal conv: (B, 1, C, T) -> (B, temp_filters, C, T)
        self.temp_conv = nn.Conv2d(
            in_channels=1,
            out_channels=temp_filters,
            kernel_size=(1, temp_kernel),
            padding=(0, temp_kernel // 2),
            bias=False,
        )
        self.bn_temp = nn.BatchNorm2d(temp_filters)

        # Spatial conv across channels: (B, temp_filters, C, T) -> (B, embed_dim, 1, T)
        self.spatial_conv = nn.Conv2d(
            in_channels=temp_filters,
            out_channels=embed_dim,
            kernel_size=(n_channels, 1),
            bias=False,
        )
        self.bn_spatial = nn.BatchNorm2d(embed_dim)

        # Temporal downsampling / stride patch projection
        self.pool = nn.AvgPool2d(kernel_size=(1, stride), stride=(1, stride))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, n_channels, n_samples) or (B, 1, n_channels, n_samples)
        Output: (B, num_patches, embed_dim)
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)

        h = F.elu(self.bn_temp(self.temp_conv(x)))
        h = F.elu(self.bn_spatial(self.spatial_conv(h)))  # (B, embed_dim, 1, T)
        h = self.pool(h)                                  # (B, embed_dim, 1, num_patches)
        h = self.drop(h)

        # Reshape to token sequence: (B, num_patches, embed_dim)
        h = h.squeeze(2).permute(0, 2, 1).contiguous()
        return h


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for sequence tokens.
    """

    def __init__(self, embed_dim: int, max_len: int = 2000):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, seq_len, embed_dim)
        Output: (B, seq_len, embed_dim)
        """
        seq_len = x.shape[1]
        return x + self.pos_embed[:, :seq_len, :]


class CausalTransformerEncoderBlock(nn.Module):
    """
    Pre-norm Causal Transformer Encoder block with Multi-Head Self-Attention (MHSA).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class NextTokenPredictionHead(nn.Module):
    """
    Per-position next-token/patch prediction head.
    Predicts the feature vector at position t+1 from hidden state at position t.
    """

    def __init__(self, embed_dim: int, target_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        out_dim = target_dim if target_dim is not None else embed_dim
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class Conv1DClassifierBackbone(nn.Module):
    """
    1D Convolutional Backbone for sequence feature classification.
    Processes temporal feature maps (B, embed_dim, seq_len) with 1D convolutions,
    residual connections, BatchNorm, and dual adaptive pooling (avg + max).
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(dropout)

        # Residual shortcut if dimensions change
        self.shortcut = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1) if embed_dim != hidden_dim else nn.Identity()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, seq_len, embed_dim)
        Returns: (B, hidden_dim * 2) aggregated temporal feature representation
        """
        # Permute to (B, embed_dim, seq_len) for 1D convolution
        h = x.permute(0, 2, 1)

        res = self.shortcut(h)
        h = F.gelu(self.bn1(self.conv1(h)))
        h = self.drop(h)
        h = F.gelu(self.bn2(self.conv2(h)) + res)
        h = self.drop(h)

        # Dual pooling (Average + Max) captures both global context and peak temporal signals
        avg_feat = self.avg_pool(h).squeeze(-1)  # (B, hidden_dim)
        max_feat = self.max_pool(h).squeeze(-1)  # (B, hidden_dim)

        return torch.cat([avg_feat, max_feat], dim=-1)  # (B, hidden_dim * 2)


class SeizureOccurrenceAndTimingHead(nn.Module):
    """
    Task 2 Classifier: 1D CNN-based head that determines IF a seizure occurs (occurrence)
    and WHEN (onset timing offset).
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.backbone = Conv1DClassifierBackbone(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout)
        feat_dim = hidden_dim * 2

        self.occurrence_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.timing_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, sequence_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: (B, seq_len, embed_dim)
        Returns:
            occurrence_logits: (B, 1) - raw logit for seizure presence
            onset_preds: (B, 1) - predicted onset offset in seconds / token steps
        """
        feat = self.backbone(sequence_features)
        occ_logits = self.occurrence_head(feat)
        onset_preds = self.timing_head(feat)
        return occ_logits, onset_preds


class SeizureTypeClassifierHead(nn.Module):
    """
    Task 3 Classifier: 1D CNN-based head that classifies seizure type downstream of detection.
    """

    def __init__(self, embed_dim: int = 128, num_classes: int = 2, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.backbone = Conv1DClassifierBackbone(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout)
        feat_dim = hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, seq_len, embed_dim)
        Returns: type_logits: (B, num_classes)
        """
        feat = self.backbone(sequence_features)
        return self.classifier(feat)


class CausalEEGConformer(nn.Module):
    """
    Complete 3-Stage Model Architecture:
    1. Preictal next-token prediction head generates tokens matching the longest distance
       between end of preictal phase and start of ictal (horizon generation).
    2. Seizure Occurrence & Timing Classifier determines IF and WHEN a seizure occurs.
    3. Seizure Type Classifier classifies seizure type downstream of detection.
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
        """
        Passes a token sequence through Positional Encoding and Causal Transformer blocks.
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
        Task 1: Autoregressively generates 'horizon_steps' patch tokens from preictal window tokens.
        Matches the longest distance between end of preictal phase and start of ictal.
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

        Args:
            use_horizon_context: Curriculum gate for Tasks 2 & 3. When True, the
                (detached) generated horizon tokens are concatenated onto the
                preictal tokens before the final encoding that Tasks 2/3 pool
                from. When False, Tasks 2/3 see the preictal window only. Keep
                this False early in training -- the horizon generator's output
                is close to noise until `pred_next_tokens` has converged, and
                feeding noisy tokens into the classifiers destabilizes them.
                The horizon tokens are always detached before concatenation so
                classification gradients never flow back into the generator.

        Returns dictionary containing:
            - 'pred_next_tokens': Next token predictions for ground-truth preictal sequence (B, num_patches, embed_dim)
            - 'preictal_tokens': Ground truth preictal patch tokens extracted by ConvFrontEnd (B, num_patches, embed_dim)
            - 'generated_horizon_tokens': Autoregressively generated horizon tokens (B, horizon_steps, embed_dim)
            - 'full_sequence_features': Encoded features Tasks 2/3 pool from -- preictal-only when
              use_horizon_context=False, preictal+horizon when True (B, total_seq_len, embed_dim)
            - 'occurrence_logits': Seizure occurrence logits (B, 1) - Task 2 IF
            - 'onset_preds': Seizure onset offset predictions (B, 1) - Task 2 WHEN
            - 'type_logits': Seizure type classification logits (B, num_classes) - Task 3 TYPE
        """
        steps = horizon_steps if horizon_steps is not None else self.default_horizon_tokens

        # 1. Front-end patch extraction of preictal phase window
        preictal_tokens = self.front_end(x)  # (B, N_pre, embed_dim)

        # 2. Causal encoder pass over the preictal window ONLY. This is what
        #    Task 1's next-token loss is scored against, so it must stay scoped
        #    to preictal_tokens (not the horizon) or the teacher-forcing target
        #    alignment in train.py's gen_loss breaks.
        preictal_features = self.encode_tokens(preictal_tokens)
        pred_next_tokens = self.pred_head(preictal_features)  # (B, N_pre, embed_dim)

        # 3. Horizon tokens: Autoregressive generation if explicitly requested, otherwise parallel slice
        if generate_horizon_tokens:
            generated_horizon_tokens = self.generate_horizon(preictal_tokens, horizon_steps=steps)
        else:
            # Parallel slice for fast training & compilation (take last 'steps' predicted tokens)
            if pred_next_tokens.shape[1] >= steps:
                generated_horizon_tokens = pred_next_tokens[:, -steps:, :]
            else:
                generated_horizon_tokens = pred_next_tokens

        # 4. Full sequence for Tasks 2 & 3. Detach the horizon tokens so
        #    occurrence/timing/type gradients never flow back into the
        #    next-token generator, and gate them out entirely early in
        #    training (curriculum: caller passes use_horizon_context=False
        #    until gen_loss has converged) since a barely-trained generator's
        #    output is close to noise and destabilizes the classifiers.
        if use_horizon_context:
            combined_tokens = torch.cat([preictal_tokens, generated_horizon_tokens.detach()], dim=1)
            full_sequence_features = self.encode_tokens(combined_tokens)
        else:
            full_sequence_features = preictal_features

        # Task 2: Seizure Occurrence (IF) and Timing (WHEN)
        occurrence_logits, onset_preds = self.occurrence_timing_head(full_sequence_features)

        # Task 3: Seizure Type Classification
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


if __name__ == "__main__":
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