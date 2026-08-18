"""
Full Multi-Task Training Pipeline:
1. Loads EEG window datasets using dataset_loader.build_dataloaders.
2. Task 1: Horizon Next-Token Generator (generates tokens matching preictal -> ictal max distance).
3. Task 2: Seizure Occurrence (IF) & Timing (WHEN) Classifier.
4. Task 3: Seizure Type Classifier (classifies seizure type downstream of detection).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from tqdm import tqdm

from dataset_loader import build_dataloaders
from conformer import CausalEEGConformer

logger = logging.getLogger(__name__)


@torch.no_grad()
def _true_horizon_tokens(model: CausalEEGConformer, horizon_windows: torch.Tensor) -> torch.Tensor:
    """Token-space target for horizon_loss: runs the model's own front_end
    over the REAL ground-truth horizon signal (the actual EEG between the
    end of a preictal window and the true seizure onset -- see
    dataset_loader's `horizon_window` target) so Task 1 is supervised
    against what actually happens next, not just against its own
    within-preictal-window continuation.

    Uses front_end.eval() for this call (restoring train mode after) so
    this target-only pass doesn't perturb BatchNorm running stats, and
    torch.no_grad since it's a target, never something to backprop
    through.
    """
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
    """MSE between the model's autoregressively GENERATED horizon tokens and
    the REAL horizon tokens (from dataset_loader's `horizon_window` /
    `has_horizon` targets), masked to rows where a real horizon target is
    actually available (preictal rows with a resolvable seizure onset).
    Returns `fallback` (a zero tensor matching dtype/device) if the batch
    has no such rows or no horizon target was provided by the dataloader
    (e.g. return_dict=False)."""
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


def train_epoch(
    model: CausalEEGConformer,
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
):
    model.train()
    total_loss = 0.0
    total_gen_loss = 0.0
    total_occ_loss = 0.0
    total_timing_loss = 0.0
    total_type_loss = 0.0
    total_horizon_loss = 0.0

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
            # Relative onset time target (in seconds relative to generated data)
            onset_targets = targets["onset_offset"].to(device, non_blocking=True).unsqueeze(-1)
        else:
            labels = targets.to(device, non_blocking=True)
            occ_targets = (labels > 0).float().unsqueeze(-1)
            onset_targets = torch.zeros_like(occ_targets)

        # --- AMP forward pass ---
        with torch.amp.autocast("cuda", enabled=use_amp):
            # generate_horizon_tokens=True: we need the REAL autoregressive
            # generation (not the cheap pred_next-slice approximation) so
            # horizon_loss below actually supervises Task 1 against the
            # true continuation signal.
            outputs = model(
                windows, horizon_steps=horizon_tokens,
                generate_horizon_tokens=True,
                use_horizon_context=use_horizon_context,
            )

            pred_next = outputs["pred_next_tokens"]
            preictal_tokens = outputs["preictal_tokens"]
            occ_logits = outputs["occurrence_logits"]
            onset_preds = outputs["onset_preds"]
            type_logits = outputs["type_logits"]

            # Task 1a: within-preictal-window next-token MSE (representation
            # pretraining -- learns short-horizon preictal dynamics).
            gen_loss = F.mse_loss(pred_next[:, :-1, :], preictal_tokens[:, 1:, :])

            # Task 1b: the actual "horizon generator" objective -- MSE
            # between generated and REAL horizon tokens, masked to rows
            # with a resolvable ground-truth horizon (see dataset_loader's
            # has_horizon/horizon_window targets).
            horizon_loss = _compute_horizon_loss(
                model, outputs, targets, device, fallback=gen_loss.new_zeros(())
            )

            # Task 2 (IF): Seizure Occurrence BCE Loss
            occ_loss = bce_criterion(occ_logits, occ_targets)

            # Task 2 (WHEN): Seizure Onset Timing Smooth L1 (Huber) Loss (relative seconds)
            timing_loss = F.smooth_l1_loss(onset_preds, onset_targets)

            # Task 3 (TYPE): Seizure Type CrossEntropy Loss
            type_loss = ce_criterion(type_logits, labels)

            # Composite Multi-Task Loss (scaled timing loss to prevent dominating the gradients)
            loss = gen_loss + horizon_loss_weight * horizon_loss + occ_loss + timing_weight * timing_loss + type_loss


            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps


        # --- AMP backward pass ---
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if i % grad_accum_steps == 0 or i == num_batches:
            # Gradient clipping (unscale first when using AMP)
            if use_amp:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            # Scheduler step AFTER optimizer step (fixes PyTorch ordering warning)
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
    model: CausalEEGConformer,
    val_loader,
    device: torch.device,
    use_amp=False,
    horizon_tokens=10,
    occ_pos_weight: torch.Tensor = None,
    type_class_weights: torch.Tensor = None,
    timing_weight: float = 0.001,
    use_horizon_context: bool = True,
):
    model.eval()
    total_loss = 0.0
    correct_type = 0
    correct_occ = 0
    total_samples = 0
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
            type_logits = outputs["type_logits"]

            gen_loss = F.mse_loss(pred_next[:, :-1, :], preictal_tokens[:, 1:, :])
            horizon_loss = _compute_horizon_loss(
                model, outputs, targets, device, fallback=gen_loss.new_zeros(())
            )
            occ_loss = bce_criterion(occ_logits, occ_targets)
            timing_loss = F.smooth_l1_loss(onset_preds, onset_targets)
            type_loss = ce_criterion(type_logits, labels)

            loss = gen_loss + horizon_loss + occ_loss + timing_weight * timing_loss + type_loss

        total_loss += loss.item()

        # Metrics computation
        type_preds = type_logits.argmax(dim=-1)
        occ_preds = (torch.sigmoid(occ_logits) >= 0.5).float()

        correct_type += (type_preds == labels).sum().item()
        correct_occ += (occ_preds == occ_targets).sum().item()
        total_samples += labels.size(0)

    num_batches = len(val_loader)
    type_acc = correct_type / total_samples if total_samples > 0 else 0.0
    occ_acc = correct_occ / total_samples if total_samples > 0 else 0.0
    return total_loss / num_batches, type_acc, occ_acc


@torch.no_grad()
def compute_confusion_matrix(
    model: CausalEEGConformer,
    val_loader,
    device: torch.device,
    num_classes: int,
    use_amp: bool = False,
    horizon_tokens: int = 10,
    use_horizon_context: bool = True,
) -> torch.Tensor:
    """Runs the final model over val_loader and returns a (num_classes, num_classes)
    confusion matrix for the Task 3 (type) predictions: rows = true label, cols = predicted."""
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for batch in tqdm(val_loader, desc="Confusion Matrix"):
        if isinstance(batch, (list, tuple)):
            windows, targets = batch
        else:
            windows, targets = batch, {}

        windows = windows.to(device, non_blocking=True)
        labels = targets["label"] if isinstance(targets, dict) else targets
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(
                windows,
                horizon_steps=horizon_tokens,
                use_horizon_context=use_horizon_context,
            )
            type_preds = outputs["type_logits"].argmax(dim=-1)

        for t, p in zip(labels.view(-1).tolist(), type_preds.view(-1).tolist()):
            cm[t, p] += 1

    return cm


def log_confusion_matrix(cm: torch.Tensor, label_names: Optional[list] = None) -> None:
    """Pretty-prints a confusion matrix (rows=true, cols=predicted) via `logger`."""
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


def main():
    parser = argparse.ArgumentParser(description="Train 3-Head Causal EEG-Conformer Model.")
    parser.add_argument("--master-csv", required=True, help="Path to master CSV from build_master_file()")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing session checkpoints")
    parser.add_argument("--stage", default="proc", choices=["raw", "proc"], help="Checkpoint stage ('raw' or 'proc')")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4, lowered from 1e-3 to reduce instability)")
    parser.add_argument("--term-value", default=None, help="Value in channel column to filter by")
    parser.add_argument("--window-samples", type=int, default=1000, help="Window sample length (default: 1000)")
    parser.add_argument("--embed-dim", type=int, default=128, help="Token embedding dimension (default: 128)")
    parser.add_argument("--horizon-tokens", type=int, default=10, help="Number of horizon tokens to generate (default: 10)")
    parser.add_argument("--binary-preictal", action="store_true", help="Map labels into binary classification")
    parser.add_argument("--cache-capacity", type=int, default=128, help="LRU cache capacity (default: 128)")
    parser.add_argument("--no-session-batching", action="store_true", help="Disable SessionBatchSampler")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--output-dir", default="checkpoints", help="Directory to save training checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to specific checkpoint file")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile()")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP mixed-precision training")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--use-scheduler", action="store_true", help="Enable LR scheduler")
    parser.add_argument(
        "--scheduler-type",
        default="cosine",
        choices=["cosine", "onecycle"],
        help="LR scheduler: 'cosine' (CosineAnnealingWarmRestarts, default) or 'onecycle' (OneCycleLR)",
    )
    parser.add_argument(
        "--timing-weight",
        type=float,
        default=0.1,
        help="Loss weight for seizure onset timing loss (default: 0.1)",
    )
    parser.add_argument(
        "--timing-norm-seconds",
        type=float,
        default=300.0,
        help="Divides the WHEN target (seconds to seizure onset) by this before training, forwarded to EEGWindowDataset (default: 300.0)",
    )
    parser.add_argument(
        "--restart-period-epochs",
        type=int,
        default=2,
        help="CosineAnnealingWarmRestarts T_0, expressed as a multiple of one epoch's steps (default: 2). "
             "Raised from 1 epoch to reduce how often the LR spikes back up, which was correlated with grad-norm blowups.",
    )
    parser.add_argument(
        "--horizon-gate-threshold",
        type=float,
        default=0.01,
        help=(
            "Curriculum gate: Task 2/3 heads only start seeing the (detached) generated horizon "
            "tokens once the training gen_loss (next-token MSE) drops below this value. Before that, "
            "the horizon tokens are close to noise and destabilize the classifiers. Once opened the "
            "gate stays open (default: 0.01)."
        ),
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=256.0,
        help=(
            "Checkpoint sample rate in Hz, forwarded to EEGWindowDataset for "
            "slicing windows out of the raw arrays. MUST match Tuh-Preprocess's "
            "raw_eeg_extraction.TARGET_SFREQ (256 Hz as of this pipeline) -- a "
            "mismatch here silently mis-slices every window (default: 256.0)."
        ),
    )
    parser.add_argument(
        "--horizon-window-samples",
        type=int,
        default=None,
        help="Fixed length (in samples) of the ground-truth horizon window used to "
             "supervise Task 1 (default: same as --window-samples).",
    )
    parser.add_argument(
        "--horizon-loss-weight",
        type=float,
        default=1.0,
        help="Loss weight for the real-horizon-supervision loss (default: 1.0). "
             "See horizon_loss in train_epoch/validate.",
    )
    parser.add_argument(
        "--exclude-labels",
        type=str,
        default="bckg",
        help="Comma-separated exact labels to drop entirely from training (default: "
             "'bckg' -- it's an artifact-flagged tag, not reliable background; true "
             "background should come from explicitly-derived 'bg' rows, see "
             "Tuh-Preprocess's preictal_segment.add_background_tags). Pass '' for none.",
    )
    parser.add_argument(
        "--include-status-0",
        action="store_true",
        help=(
            "Include status-0 rows (collapsed/invalid windows -- e.g. a preictal "
            "window whose start_cutoff pushed it before recording start, so its "
            "start_time==stop_time==0.0 and it's mostly zero-padding after slicing). "
            "Excluded by default since they're degenerate, not real interictal data."
        ),
    )
    parser.add_argument(
        "--max-bg-ratio",
        type=float,
        default=3.0,
        help=(
            "Cap background (bg*) rows to at most this multiple of the non-background "
            "row count after all other filters. Prevents the model from overfitting to "
            "the majority background class. Set to 0 to disable subsampling (default: 3.0)."
        ),
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping max norm (default: 1.0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("cudnn.benchmark enabled")

    start_epoch = 1
    best_val_loss = float("inf")
    ckpt_path = None
    if args.load_checkpoint is not None:
        ckpt_path = args.load_checkpoint
    elif args.resume:
        ckpt_files = [f for f in os.listdir(args.output_dir) if f.startswith("epoch_") and f.endswith(".pt")]
        if ckpt_files:
            latest = max(ckpt_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
            ckpt_path = os.path.join(args.output_dir, latest)

    loaded_state_dict = None
    loaded_optimizer_state = None
    loaded_scheduler_state = None
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        logger.info(f"Loading checkpoint {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        loaded_state_dict = checkpoint["model_state_dict"]
        loaded_optimizer_state = checkpoint["optimizer_state_dict"]
        loaded_scheduler_state = checkpoint.get("scheduler_state_dict")
        start_epoch = checkpoint.get("epoch", 1) + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)

    exclude_labels = tuple(
        t.strip() for t in args.exclude_labels.split(",") if t.strip()
    ) if args.exclude_labels else ()

    # 1. Datasets
    logger.info("Initializing DataLoaders...")
    train_loader, val_loader, dataset = build_dataloaders(
        master_csv=args.master_csv,
        checkpoint_dir=args.checkpoint_dir,
        stage=args.stage,
        batch_size=args.batch_size,
        term_value=args.term_value,
        sampling_rate=args.sampling_rate,
        window_samples=args.window_samples,
        horizon_window_samples=args.horizon_window_samples,
        binary_preictal=args.binary_preictal,
        cache_capacity=args.cache_capacity,
        num_workers=args.num_workers,
        exclude_status={2} if args.include_status_0 else {0, 2},
        exclude_labels=exclude_labels,
        timing_norm=args.timing_norm_seconds,
        max_bg_ratio=args.max_bg_ratio if args.max_bg_ratio > 0 else None,
        use_session_batching=not args.no_session_batching,
    )

    num_classes = dataset.num_classes
    sample_window, _ = dataset[0]
    n_channels, n_samples = sample_window.shape

    logger.info(f"Dataset loaded: {len(dataset)} samples | {num_classes} classes | shape: ({n_channels}, {n_samples})")

    # 2. Build Model
    model = CausalEEGConformer(
        n_channels=n_channels,
        embed_dim=args.embed_dim,
        num_classes=num_classes,
        default_horizon_tokens=args.horizon_tokens,
    ).to(device)

    if loaded_state_dict is not None:
        model.load_state_dict(loaded_state_dict)
        logger.info("Model state loaded from checkpoint")

    if not args.no_compile and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if loaded_optimizer_state is not None:
        optimizer.load_state_dict(loaded_optimizer_state)

    use_amp = not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Compute class-balance weights from the training split indices
    from dataset_loader import split_by_column
    _subsets = split_by_column(dataset)
    _train_indices = _subsets["train"].indices if hasattr(_subsets.get("train", None), "indices") else None
    type_class_weights = dataset.class_weights_tensor(indices=_train_indices, device=device)
    occ_pos_weight = dataset.occ_pos_weight(indices=_train_indices, device=device)
    logger.info(f"Type class weights: {type_class_weights.tolist()}")
    logger.info(f"Occurrence pos_weight: {occ_pos_weight.item():.4f}")

    scheduler = None
    if args.use_scheduler:
        # Use ceil so total_steps matches the actual scheduler.step() call count:
        # the loop fires on i%grad_accum==0 (floor steps) AND on i==num_batches
        # for any leftover batch, giving ceil(len/accum) steps per epoch.
        steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
        remaining_epochs = args.epochs - start_epoch + 1
        total_steps = steps_per_epoch * remaining_epochs

        if args.scheduler_type == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=args.lr,
                total_steps=total_steps,
            )
            logger.info(f"Scheduler: OneCycleLR | total_steps={total_steps}")
        else:
            # CosineAnnealingWarmRestarts: first restart after T_0 steps,
            # then doubles each time (T_mult=2) — smooth decay with gentle restarts.
            # T_0 spans --restart-period-epochs full epochs (not just 1) so LR spikes
            # back up less often; frequent restarts were correlated with the grad-norm
            # blowups / accuracy collapses observed in earlier runs.
            t0 = steps_per_epoch * args.restart_period_epochs
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=t0,
                T_mult=2,
                eta_min=1e-6,
            )
            logger.info(f"Scheduler: CosineAnnealingWarmRestarts | T_0={t0} ({args.restart_period_epochs} epoch(s)), T_mult=2, eta_min=1e-6")

        if loaded_scheduler_state is not None:
            scheduler.load_state_dict(loaded_scheduler_state)
            logger.info("Scheduler state restored from checkpoint")

    logger.info(f"Gradient clipping max_norm={args.max_grad_norm}")

    # Curriculum gate for feeding the generated horizon tokens into Tasks 2/3 (see
    # --horizon-gate-threshold). Starts closed; opens permanently once train gen_loss
    # drops below the threshold.
    horizon_gate_open = False

    logger.info("Starting Multi-Task Training Loop...")
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"--- Epoch {epoch:02d}/{args.epochs:02d} ---")
        train_loss, gen_l, occ_l, time_l, type_l, horizon_l = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            scheduler=scheduler,
            grad_accum_steps=args.grad_accum_steps,
            horizon_tokens=args.horizon_tokens,
            occ_pos_weight=occ_pos_weight,
            type_class_weights=type_class_weights,
            timing_weight=args.timing_weight,
            horizon_loss_weight=args.horizon_loss_weight,
            max_grad_norm=args.max_grad_norm,
            use_horizon_context=horizon_gate_open,
        )

        if not horizon_gate_open and gen_l < args.horizon_gate_threshold:
            horizon_gate_open = True
            logger.info(
                f"Horizon curriculum gate OPENED (gen_loss {gen_l:.4f} < threshold "
                f"{args.horizon_gate_threshold}) — Tasks 2/3 will see horizon context from next epoch on."
            )

        val_loss, val_type_acc, val_occ_acc = validate(
            model,
            val_loader,
            device,
            use_amp=use_amp,
            horizon_tokens=args.horizon_tokens,
            occ_pos_weight=occ_pos_weight,
            type_class_weights=type_class_weights,
            timing_weight=args.timing_weight,
            use_horizon_context=horizon_gate_open,
        )

        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        ckpt_save_path = os.path.join(args.output_dir, f"epoch_{epoch:02d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_type_acc": val_type_acc,
                "val_occ_acc": val_occ_acc,
                "best_val_loss": min(best_val_loss, val_loss),
            },
            ckpt_save_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                },
                os.path.join(args.output_dir, "best.pt"),
            )

        logger.info(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} "
            f"(Gen: {gen_l:.4f}, Horizon: {horizon_l:.4f}, Occ: {occ_l:.4f}, Timing: {time_l:.4f}, Type: {type_l:.4f}) | "
            f"Val Loss: {val_loss:.4f} | Val Occ Acc: {val_occ_acc * 100:.2f}% | Val Type Acc: {val_type_acc * 100:.2f}%"
        )

    logger.info("Training complete. Computing final confusion matrix on validation set...")
    idx_to_label = {v: k for k, v in dataset.label_map.items()}
    label_names = [str(idx_to_label.get(i, i)) for i in range(num_classes)]
    cm = compute_confusion_matrix(
        model,
        val_loader,
        device,
        num_classes,
        use_amp=use_amp,
        horizon_tokens=args.horizon_tokens,
        use_horizon_context=horizon_gate_open,
    )
    log_confusion_matrix(cm, label_names=label_names)


if __name__ == "__main__":
    main()