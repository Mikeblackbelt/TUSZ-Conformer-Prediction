"""
Training and Validation loops for EEG Conformer pipeline.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from training.losses import FocalBCEWithLogitsLoss, FocalCrossEntropyLoss, _compute_horizon_loss
from training.metrics import _macro_prf1_from_confusion


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler=None,
    scheduler=None,
    grad_accum_steps: int = 1,
    horizon_tokens: int = 10,
    occ_pos_weight: torch.Tensor = None,
    type_class_weights: torch.Tensor = None,
    timing_weight: float = 0.001,
    horizon_loss_weight: float = 1.0,
    max_grad_norm: float = 1.0,
    use_horizon_context: bool = True,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
):
    model.train()
    total_loss = 0.0
    total_gen_loss = 0.0
    total_occ_loss = 0.0
    total_timing_loss = 0.0
    total_type_loss = 0.0
    total_horizon_loss = 0.0

    if use_focal_loss:
        bce_criterion = FocalBCEWithLogitsLoss(pos_weight=occ_pos_weight, gamma=focal_gamma)
        ce_criterion = FocalCrossEntropyLoss(weight=type_class_weights, gamma=focal_gamma, label_smoothing=0.1)
    else:
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=occ_pos_weight)
        ce_criterion = nn.CrossEntropyLoss(weight=type_class_weights, label_smoothing=0.1)
    num_batches = len(train_loader)
    use_amp = scaler is not None and device.type == "cuda"

    grad_norm: torch.Tensor = torch.tensor(float("nan"))
    pbar = tqdm(train_loader, desc="Training")
    for i, batch in enumerate(pbar, start=1):
        if isinstance(batch, (list, tuple)):
            windows, targets = batch
        else:
            windows, targets = batch, {}

        windows = windows.to(device, non_blocking=True)
        if isinstance(targets, dict):
            labels = targets["label"].to(device, non_blocking=True)
            occ_targets = targets["occurrence"].to(device, non_blocking=True).unsqueeze(-1)
            onset_targets = targets["onset_offset"].to(device, non_blocking=True).unsqueeze(-1)
        else:
            labels = targets.to(device, non_blocking=True)
            occ_targets = (labels > 0).float().unsqueeze(-1)
            onset_targets = torch.zeros_like(occ_targets)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(
                windows, horizon_steps=horizon_tokens,
                generate_horizon_tokens=True,
                use_horizon_context=use_horizon_context,
            )

            pred_next = outputs["pred_next_tokens"]
            preictal_tokens = outputs["preictal_tokens"]
            occ_logits = outputs["occurrence_logits"]
            onset_preds = outputs["onset_preds"]
            type_logits = outputs["preictal_type_logits"] if "preictal_type_logits" in outputs else outputs["type_logits"]

            gen_loss = F.mse_loss(pred_next[:, :-1, :], preictal_tokens[:, 1:, :])
            horizon_loss = _compute_horizon_loss(
                model, outputs, targets, device, fallback=gen_loss.new_zeros(())
            )
            occ_loss = bce_criterion(occ_logits, occ_targets)
            timing_loss = F.smooth_l1_loss(onset_preds, onset_targets)

            pos_mask = (occ_targets.squeeze(-1) > 0)
            if pos_mask.any():
                type_loss = ce_criterion(type_logits[pos_mask], labels[pos_mask])
            else:
                type_loss = type_logits.sum() * 0.0

            preictal_loss = gen_loss.new_zeros(())
            if "preictal_logits" in outputs:
                preictal_targets = (labels > 0).float().unsqueeze(-1)
                preictal_loss = bce_criterion(outputs["preictal_logits"], preictal_targets)

            loss = gen_loss + horizon_loss_weight * horizon_loss + occ_loss + timing_weight * timing_loss + type_loss + preictal_loss

            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if i % grad_accum_steps == 0 or i == num_batches:
            if use_amp:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

        loss_val = loss.item() * (grad_accum_steps if grad_accum_steps > 1 else 1)
        total_loss += loss_val
        total_gen_loss += gen_loss.item()
        total_occ_loss += occ_loss.item()
        total_timing_loss += timing_loss.item()
        total_type_loss += type_loss.item()
        total_horizon_loss += horizon_loss.item()

        gnorm_val = grad_norm.item() if not torch.isnan(grad_norm) else float("nan")
        pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            gen=f"{gen_loss.item():.4f}",
            hzn=f"{horizon_loss.item():.4f}",
            occ=f"{occ_loss.item():.4f}",
            time=f"{timing_loss.item():.4f}",
            type=f"{type_loss.item():.4f}",
            gnorm=f"{gnorm_val:.2f}",
        )

    return (
        total_loss / num_batches,
        total_gen_loss / num_batches,
        total_occ_loss / num_batches,
        total_timing_loss / num_batches,
        total_type_loss / num_batches,
        total_horizon_loss / num_batches,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    device: torch.device,
    use_amp=False,
    horizon_tokens=10,
    occ_pos_weight: torch.Tensor = None,
    type_class_weights: torch.Tensor = None,
    timing_weight: float = 0.001,
    use_horizon_context: bool = True,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    num_classes: Optional[int] = None,
):
    model.eval()
    total_loss = 0.0
    correct_type = 0
    correct_occ = 0
    total_samples = 0
    total_pos_samples = 0

    cm = torch.zeros(num_classes or 2, num_classes or 2, dtype=torch.long)

    if use_focal_loss:
        ce_criterion = FocalCrossEntropyLoss(weight=type_class_weights, gamma=focal_gamma, label_smoothing=0.1)
        bce_criterion = FocalBCEWithLogitsLoss(pos_weight=occ_pos_weight, gamma=focal_gamma)
    else:
        ce_criterion = nn.CrossEntropyLoss(weight=type_class_weights, label_smoothing=0.1)
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=occ_pos_weight)

    for batch in tqdm(val_loader, desc="Validating"):
        if isinstance(batch, (list, tuple)):
            windows, targets = batch
        else:
            windows, targets = batch, {}

        windows = windows.to(device, non_blocking=True)
        if isinstance(targets, dict):
            labels = targets["label"].to(device, non_blocking=True)
            occ_targets = targets["occurrence"].to(device, non_blocking=True).unsqueeze(-1)
            onset_targets = targets["onset_offset"].to(device, non_blocking=True).unsqueeze(-1)
        else:
            labels = targets.to(device, non_blocking=True)
            occ_targets = (labels > 0).float().unsqueeze(-1)
            onset_targets = torch.zeros_like(occ_targets)

        pos_mask = (occ_targets.squeeze(-1) > 0)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(
                windows, horizon_steps=horizon_tokens,
                generate_horizon_tokens=True,
                use_horizon_context=use_horizon_context,
            )
            pred_next = outputs["pred_next_tokens"]
            preictal_tokens = outputs["preictal_tokens"]
            occ_logits = outputs["occurrence_logits"]
            onset_preds = outputs["onset_preds"]
            type_logits = outputs["preictal_type_logits"] if "preictal_type_logits" in outputs else outputs["type_logits"]

            gen_loss = F.mse_loss(pred_next[:, :-1, :], preictal_tokens[:, 1:, :])
            horizon_loss = _compute_horizon_loss(
                model, outputs, targets, device, fallback=gen_loss.new_zeros(())
            )
            occ_loss = bce_criterion(occ_logits, occ_targets)
            timing_loss = F.smooth_l1_loss(onset_preds, onset_targets)

            if pos_mask.any():
                type_loss = ce_criterion(type_logits[pos_mask], labels[pos_mask])
            else:
                type_loss = type_logits.sum() * 0.0

            preictal_loss = gen_loss.new_zeros(())
            if "preictal_logits" in outputs:
                preictal_targets = (labels > 0).float().unsqueeze(-1)
                preictal_loss = bce_criterion(outputs["preictal_logits"], preictal_targets)

            loss = gen_loss + horizon_loss + occ_loss + timing_weight * timing_loss + type_loss + preictal_loss

        total_loss += loss.item()

        type_preds = type_logits.argmax(dim=-1)
        occ_preds = (torch.sigmoid(occ_logits) >= 0.5).float()

        if pos_mask.any():
            correct_type += (type_preds[pos_mask] == labels[pos_mask]).sum().item()
            total_pos_samples += pos_mask.sum().item()
            for t, p in zip(labels[pos_mask].view(-1).tolist(), type_preds[pos_mask].view(-1).tolist()):
                if t < cm.shape[0] and p < cm.shape[1]:
                    cm[t, p] += 1

        correct_occ += (occ_preds == occ_targets).sum().item()
        total_samples += labels.size(0)

    num_batches = len(val_loader)
    type_acc = correct_type / total_pos_samples if total_pos_samples > 0 else 0.0
    occ_acc = correct_occ / total_samples if total_samples > 0 else 0.0
    macro_precision, macro_recall, macro_f1 = _macro_prf1_from_confusion(cm)
    return total_loss / num_batches, type_acc, occ_acc, macro_f1, macro_precision, macro_recall
