"""
Prediction Heads for EEG Conformer Models.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import Conv1DClassifierBackbone


class NextTokenPredictionHead(nn.Module):
    """Per-position next-token/patch prediction head."""

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


class SeizureOccurrenceAndTimingHead(nn.Module):
    """Task 2 Classifier: 1D CNN-based head predicting seizure occurrence (IF) and timing (WHEN)."""

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
        feat = self.backbone(sequence_features)
        occ_logits = self.occurrence_head(feat)
        onset_preds = self.timing_head(feat)
        return occ_logits, onset_preds


class SeizureTypeClassifierHead(nn.Module):
    """Task 3 Classifier: Multi-class seizure type classification head."""

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
        feat = self.backbone(sequence_features)
        return self.classifier(feat)


class SeizurePreictalClassifierHead(nn.Module):
    """Head 3: 1D CNN-based binary classification head predicting Preictal vs Not Preictal."""

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
        feat = self.backbone(sequence_features)
        return self.classifier(feat)
