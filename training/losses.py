"""
Loss functions for Multi-Task EEG Conformer training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from models.conformer import CausalEEGConformer


class FocalBCEWithLogitsLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017) on top of BCEWithLogitsLoss."""

    def __init__(self, pos_weight: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None, persistent=False)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        focal_term = (1.0 - p_t) ** self.gamma
        loss = focal_term * bce
        if self.pos_weight is not None:
            alpha_t = torch.where(targets > 0.5, self.pos_weight, torch.ones_like(targets))
            loss = alpha_t * loss
        return loss.mean()


class FocalCrossEntropyLoss(nn.Module):
    """Multi-class focal loss on top of CrossEntropyLoss."""

    def __init__(self, weight: torch.Tensor = None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets, weight=self.weight, label_smoothing=self.label_smoothing, reduction="none"
        )
        p_t = torch.exp(-ce)
        focal_term = (1.0 - p_t) ** self.gamma
        return (focal_term * ce).mean()


@torch.no_grad()
def _true_horizon_tokens(model: nn.Module, horizon_windows: torch.Tensor) -> torch.Tensor:
    """Token-space target for horizon_loss: runs the model's own front_end
    over the REAL ground-truth horizon signal."""
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    front_end = raw_model.front_end
    was_training = front_end.training
    front_end.eval()
    try:
        tokens = front_end(horizon_windows)
    finally:
        if was_training:
            front_end.train()
    return tokens


def _compute_horizon_loss(
    model: CausalEEGConformer,
    outputs: dict,
    targets: dict,
    device: torch.device,
    fallback: torch.Tensor,
) -> torch.Tensor:
    """MSE between generated horizon tokens and real horizon tokens."""
    horizon_windows = targets.get("horizon_window") if isinstance(targets, dict) else None
    has_horizon = targets.get("has_horizon") if isinstance(targets, dict) else None
    if horizon_windows is None or has_horizon is None:
        return fallback

    has_horizon = has_horizon.to(device, non_blocking=True)
    mask = has_horizon.bool()
    if not bool(mask.any()):
        return fallback

    horizon_windows = horizon_windows.to(device, non_blocking=True)
    gen_tokens = outputs["generated_horizon_tokens"][mask]
    true_tokens = _true_horizon_tokens(model, horizon_windows[mask])

    length = min(gen_tokens.shape[1], true_tokens.shape[1])
    if length == 0:
        return fallback
    return F.mse_loss(gen_tokens[:, :length, :], true_tokens[:, :length, :])
