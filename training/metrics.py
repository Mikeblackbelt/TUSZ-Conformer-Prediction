"""
Metrics calculation, confusion matrix generation, and logging utilities.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _macro_prf1_from_confusion(cm: torch.Tensor) -> Tuple[float, float, float]:
    """Macro-averaged precision/recall/F1 over all classes in a confusion matrix."""
    num_classes = cm.shape[0]
    precisions, recalls, f1s = [], [], []
    for i in range(num_classes):
        support = cm[i, :].sum().item()
        predicted_count = cm[:, i].sum().item()
        tp = cm[i, i].item()
        recall = tp / support if support > 0 else 0.0
        precision = tp / predicted_count if predicted_count > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return sum(precisions) / num_classes, sum(recalls) / num_classes, sum(f1s) / num_classes


@torch.no_grad()
def compute_confusion_matrix(
    model: nn.Module,
    val_loader,
    device: torch.device,
    num_classes: int,
    use_amp: bool = False,
    horizon_tokens: int = 10,
    use_horizon_context: bool = True,
) -> torch.Tensor:
    """Runs the final model over val_loader and returns a (num_classes, num_classes) confusion matrix."""
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for batch in tqdm(val_loader, desc="Confusion Matrix"):
        if isinstance(batch, (list, tuple)):
            windows, targets = batch
        else:
            windows, targets = batch, {}

        windows = windows.to(device, non_blocking=True)
        if isinstance(targets, dict):
            seizure_type_targets = targets.get("seizure_type")
            if seizure_type_targets is not None:
                seizure_type_targets = seizure_type_targets.to(device, non_blocking=True)
        else:
            seizure_type_targets = None

        # type_mask uses the dedicated seizure_type target, NOT
        # occurrence/label -- occurrence and label are the same binary flag
        # under binary_preictal, so masking by occurrence here would make
        # every included true-label a 1 by construction (this was the bug:
        # it produced a "100% accurate" confusion matrix with zero support
        # for class 0 regardless of what the model predicted).
        if seizure_type_targets is not None:
            type_mask = seizure_type_targets >= 0
        else:
            type_mask = torch.zeros(windows.shape[0], dtype=torch.bool, device=device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(
                windows,
                horizon_steps=horizon_tokens,
                use_horizon_context=use_horizon_context,
            )
            type_logits = outputs["preictal_type_logits"] if "preictal_type_logits" in outputs else outputs["type_logits"]
            type_preds = type_logits.argmax(dim=-1)

        if type_mask.any():
            for t, p in zip(seizure_type_targets[type_mask].view(-1).tolist(), type_preds[type_mask].view(-1).tolist()):
                if 0 <= t < cm.shape[0] and 0 <= p < cm.shape[1]:
                    cm[t, p] += 1

    return cm


def log_confusion_matrix(cm: torch.Tensor, label_names: Optional[list] = None) -> None:
    """Pretty-prints a confusion matrix via `logger`."""
    num_classes = cm.shape[0]
    names = [str(n) for n in label_names] if label_names else [str(i) for i in range(num_classes)]
    col_w = max(10, max(len(n) for n in names) + 2)

    logger.info("=" * 60)
    logger.info("CONFUSION MATRIX (Validation Set) - rows=True, cols=Predicted")
    logger.info("=" * 60)
    logger.info(" " * col_w + "".join(f"{n:>{col_w}}" for n in names))
    for i, name in enumerate(names):
        row = "".join(f"{cm[i, j].item():>{col_w}d}" for j in range(num_classes))
        logger.info(f"{name:>{col_w}}{row}")

    logger.info("-" * 60)
    for i, name in enumerate(names):
        support = cm[i, :].sum().item()
        recall = cm[i, i].item() / support if support > 0 else 0.0
        predicted_count = cm[:, i].sum().item()
        precision = cm[i, i].item() / predicted_count if predicted_count > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        logger.info(f"Class {name:>{col_w}}: support={support:5d} | precision={precision:.4f} | recall={recall:.4f} | F1={f1:.4f}")
    logger.info("=" * 60)