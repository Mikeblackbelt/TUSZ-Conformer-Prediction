"""
CLI Entrypoint and Training Pipeline Orchestrator.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from typing import Optional

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

from dataset.samplers import build_dataloaders, split_by_column
from models.conformer import CausalEEGConformer
from models.simplified_conformer import SimplifiedEEGConformer
from training.metrics import compute_confusion_matrix, log_confusion_matrix
from training.trainer import train_epoch, validate

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train 3-Head Causal EEG-Conformer Model.")
    parser.add_argument(
        "--model-arch",
        type=str,
        default="causal_conformer",
        choices=["causal_conformer", "simplified_eeg_conformer", "simplified_conformer"],
        help="Model architecture: 'causal_conformer' (default) or 'simplified_eeg_conformer'",
    )
    parser.add_argument("--master-csv", required=True, help="Path to master CSV from build_master_file()")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing session checkpoints")
    parser.add_argument("--stage", default="proc", choices=["raw", "proc"], help="Checkpoint stage ('raw' or 'proc')")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
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
        help="Divides the WHEN target by this before training (default: 300.0)",
    )
    parser.add_argument(
        "--restart-period-epochs",
        type=int,
        default=2,
        help="CosineAnnealingWarmRestarts T_0 multiple (default: 2).",
    )
    parser.add_argument(
        "--horizon-gate-threshold",
        type=float,
        default=0.01,
        help="Curriculum gate threshold (default: 0.01).",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=256.0,
        help="Checkpoint sample rate in Hz (default: 256.0).",
    )
    parser.add_argument(
        "--horizon-window-samples",
        type=int,
        default=None,
        help="Fixed length of ground-truth horizon window (default: same as --window-samples).",
    )
    parser.add_argument(
        "--horizon-loss-weight",
        type=float,
        default=1.0,
        help="Loss weight for real-horizon supervision (default: 1.0).",
    )
    parser.add_argument(
        "--horizon-ramp-epochs",
        type=int,
        default=3,
        help=(
            "Number of epochs over which --horizon-loss-weight ramps linearly "
            "from ~0 to its full value after the horizon curriculum gate "
            "opens, instead of applying full weight the instant the gate "
            "opens (default: 3). Softens the loss/gradient transition when "
            "the model starts conditioning on real horizon context."
        ),
    )
    parser.add_argument(
        "--type-weight-cap",
        type=float,
        default=8.0,
        help="Max per-class weight for the seizure-type loss (default: 8.0). "
             "Lower this further if most classes saturate at the cap.",
    )
    parser.add_argument(
        "--exclude-labels",
        type=str,
        default="bckg",
        help="Comma-separated exact labels to drop (default: 'bckg').",
    )
    parser.add_argument(
        "--include-status-0",
        action="store_true",
        help="Include status-0 rows (collapsed/invalid windows).",
    )
    parser.add_argument(
        "--max-bg-ratio",
        type=float,
        default=3.0,
        help="Cap background (bg*) rows ratio (default: 3.0).",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping max norm (default: 1.0)")
    parser.add_argument(
        "--use-focal-loss",
        action="store_true",
        help="Use Focal Loss (FocalBCE and FocalCrossEntropy).",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal Loss gamma parameter (default: 2.0).",
    )
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

    # num_classes (dataset.num_classes) is the binary occurrence/preictal
    # class count (always 2) -- it is NOT the type head's output size. The
    # type head classifies seizure SUBTYPE (fnsz/gnsz/...), so it must be
    # sized off dataset.num_type_classes instead. Wiring it to num_classes
    # previously meant the type head only ever had 2 output logits, matching
    # the (buggy) binary type target it was being trained against.
    num_classes = dataset.num_classes
    num_type_classes = dataset.num_type_classes
    sample_window, _ = dataset[0]
    n_channels, n_samples = sample_window.shape

    logger.info(
        f"Dataset loaded: {len(dataset)} samples | {num_classes} occurrence classes | "
        f"{num_type_classes} seizure-type classes ({dataset.type_classes}) | shape: ({n_channels}, {n_samples})"
    )

    if args.model_arch in ("simplified_eeg_conformer", "simplified_conformer"):
        model = SimplifiedEEGConformer(
            n_channels=n_channels,
            embed_dim=args.embed_dim,
            num_classes=num_type_classes,
            default_horizon_tokens=args.horizon_tokens,
        ).to(device)
        logger.info("Instantiated SimplifiedEEGConformer model architecture")
    else:
        model = CausalEEGConformer(
            n_channels=n_channels,
            embed_dim=args.embed_dim,
            num_classes=num_type_classes,
            default_horizon_tokens=args.horizon_tokens,
        ).to(device)
        logger.info("Instantiated CausalEEGConformer model architecture")

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

    _subsets = split_by_column(dataset)
    _train_indices = _subsets["train"].indices if hasattr(_subsets.get("train", None), "indices") else None
    type_class_weights = dataset.seizure_type_class_weights_tensor(
        indices=_train_indices, device=device, cap=args.type_weight_cap
    )
    occ_pos_weight = dataset.occ_pos_weight(indices=_train_indices, device=device)
    logger.info(f"Type class weights ({dataset.type_classes}): {type_class_weights.tolist()}")
    logger.info(f"Occurrence pos_weight: {occ_pos_weight.item():.4f}")

    scheduler = None
    if args.use_scheduler:
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
    horizon_gate_open = False
    gate_open_epoch: Optional[int] = None
    best_macro_f1 = -1.0

    logger.info("Starting Multi-Task Training Loop...")
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"--- Epoch {epoch:02d}/{args.epochs:02d} ---")

        # Fix 2: ramp horizon_loss_weight in linearly over --horizon-ramp-epochs
        # after the gate opens, instead of jumping straight to full weight the
        # instant the model starts conditioning on real horizon context. While
        # the gate is still closed, horizon_loss stays small on its own, so
        # full weight is fine and left untouched.
        if not horizon_gate_open or gate_open_epoch is None:
            current_horizon_weight = args.horizon_loss_weight
        else:
            epochs_since_gate = epoch - gate_open_epoch
            ramp_fraction = min(1.0, (epochs_since_gate + 1) / max(1, args.horizon_ramp_epochs))
            current_horizon_weight = args.horizon_loss_weight * ramp_fraction
            logger.info(
                f"Horizon loss weight ramp: {current_horizon_weight:.4f} / {args.horizon_loss_weight:.4f} "
                f"(epoch {epochs_since_gate + 1}/{args.horizon_ramp_epochs} since gate opened)"
            )

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
            horizon_loss_weight=current_horizon_weight,
            max_grad_norm=args.max_grad_norm,
            use_horizon_context=horizon_gate_open,
            use_focal_loss=args.use_focal_loss,
            focal_gamma=args.focal_gamma,
        )

        if not horizon_gate_open and gen_l < args.horizon_gate_threshold:
            horizon_gate_open = True
            gate_open_epoch = epoch + 1
            logger.info(
                f"Horizon curriculum gate OPENED (gen_loss {gen_l:.4f} < threshold "
                f"{args.horizon_gate_threshold}) — Tasks 2/3 will see horizon context from next epoch on. "
                f"Horizon loss weight will ramp over the next {args.horizon_ramp_epochs} epoch(s)."
            )
            # Fix 1: decouple the LR warm-restart schedule from the curriculum
            # change. Without this, a pending CosineAnnealingWarmRestarts
            # restart (LR jumping back to max) can land on the exact same
            # epoch the gate opens -- two independent sources of instability
            # stacking on top of each other. Rebuilding the scheduler here
            # starts a fresh cosine cycle from the CURRENT (already-decayed)
            # LR instead of snapping back to max_lr at this moment.
            if scheduler is not None and args.scheduler_type == "cosine":
                current_lr = optimizer.param_groups[0]["lr"]
                steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
                t0 = steps_per_epoch * args.restart_period_epochs
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=t0,
                    T_mult=2,
                    eta_min=1e-6,
                )
                logger.info(
                    f"Scheduler reinitialized at horizon-gate-open (base_lr={current_lr:.2e}) "
                    f"to avoid restart/curriculum overlap"
                )

        (
            val_loss, val_type_acc, val_occ_acc, val_macro_f1, val_macro_precision, val_macro_recall,
            val_occ_cm, val_occ_precision, val_occ_recall, val_occ_f1,
        ) = validate(
            model,
            val_loader,
            device,
            use_amp=use_amp,
            horizon_tokens=args.horizon_tokens,
            occ_pos_weight=occ_pos_weight,
            type_class_weights=type_class_weights,
            timing_weight=args.timing_weight,
            use_horizon_context=horizon_gate_open,
            use_focal_loss=args.use_focal_loss,
            focal_gamma=args.focal_gamma,
            num_classes=num_type_classes,
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
                "val_macro_f1": val_macro_f1,
                "val_macro_precision": val_macro_precision,
                "val_macro_recall": val_macro_recall,
                "val_occ_f1": val_occ_f1,
                "val_occ_precision": val_occ_precision,
                "val_occ_recall": val_occ_recall,
                "best_val_loss": min(best_val_loss, val_loss),
                "best_macro_f1": max(best_macro_f1, val_macro_f1),
            },
            ckpt_save_path,
        )
        best_val_loss = min(best_val_loss, val_loss)

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_macro_f1": val_macro_f1,
                    "val_macro_precision": val_macro_precision,
                    "val_macro_recall": val_macro_recall,
                    "best_val_loss": best_val_loss,
                    "best_macro_f1": best_macro_f1,
                },
                os.path.join(args.output_dir, "best.pt"),
            )
            logger.info(f"New best macro-F1: {best_macro_f1:.4f} — saved best.pt")

        logger.info(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} "
            f"(Gen: {gen_l:.4f}, Horizon: {horizon_l:.4f}, Occ: {occ_l:.4f}, Timing: {time_l:.4f}, Type: {type_l:.4f}) | "
            f"Val Loss: {val_loss:.4f} | Val Occ Acc: {val_occ_acc * 100:.2f}% | Val Type Acc: {val_type_acc * 100:.2f}% | "
            f"Val Macro-F1: {val_macro_f1:.4f} (P: {val_macro_precision:.4f} / R: {val_macro_recall:.4f})"
        )
        # Occupancy accuracy alone can look fine while the head has actually
        # collapsed to majority-class prediction under imbalance -- compare
        # against the trivial majority-class baseline (support ratio in
        # occ_cm) to tell real learning apart from that. occ_f1 uses macro
        # averaging so it isn't inflated by the majority class either.
        occ_support = val_occ_cm.sum(dim=1).tolist()
        logger.info(
            f"           Occ confusion (support {occ_support}) | "
            f"Occ Macro-F1: {val_occ_f1:.4f} (P: {val_occ_precision:.4f} / R: {val_occ_recall:.4f})"
        )

    logger.info("Training complete. Computing final confusion matrix on validation set...")
    # Labeled by seizure TYPE (fnsz/gnsz/...), not by the binary
    # occurrence/preictal flag -- dataset.num_classes/label_map describe the
    # latter and would mislabel this matrix.
    cm = compute_confusion_matrix(
        model,
        val_loader,
        device,
        num_type_classes,
        use_amp=use_amp,
        horizon_tokens=args.horizon_tokens,
        use_horizon_context=horizon_gate_open,
    )
    log_confusion_matrix(cm, label_names=dataset.type_classes)