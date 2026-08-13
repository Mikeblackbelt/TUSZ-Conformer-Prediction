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
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from dataset_loader import build_dataloaders
from conformer import CausalEEGConformer

logger = logging.getLogger(__name__)


def train_epoch(
    model: CausalEEGConformer,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler=None,
    scheduler=None,
    grad_accum_steps: int = 1,
    horizon_tokens: int = 10,
):
    model.train()
    total_loss = 0.0
    total_gen_loss = 0.0
    total_occ_loss = 0.0
    total_timing_loss = 0.0
    total_type_loss = 0.0

    bce_criterion = nn.BCEWithLogitsLoss()
    ce_criterion = nn.CrossEntropyLoss()
    num_batches = len(train_loader)
    use_amp = scaler is not None and device.type == "cuda"

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

        # --- AMP forward pass ---
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(windows, horizon_steps=horizon_tokens)

            pred_next = outputs["pred_next_tokens"]
            preictal_tokens = outputs["preictal_tokens"]
            occ_logits = outputs["occurrence_logits"]
            onset_preds = outputs["onset_preds"]
            type_logits = outputs["type_logits"]

            # Task 1: Autoregressive MSE Loss on preictal patch sequence
            gen_loss = F.mse_loss(pred_next[:, :-1, :], preictal_tokens[:, 1:, :])

            # Task 2 (IF): Seizure Occurrence BCE Loss
            occ_loss = bce_criterion(occ_logits, occ_targets)

            # Task 2 (WHEN): Seizure Onset Timing MSE Loss
            timing_loss = F.mse_loss(onset_preds, onset_targets)

            # Task 3 (TYPE): Seizure Type CrossEntropy Loss
            type_loss = ce_criterion(type_logits, labels)

            # Composite Multi-Task Loss
            loss = gen_loss + 0.5 * occ_loss + 0.1 * timing_loss + 1.0 * type_loss

            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps

        # --- AMP backward pass ---
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if i % grad_accum_steps == 0 or i == num_batches:
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

        pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            gen=f"{gen_loss.item():.4f}",
            occ=f"{occ_loss.item():.4f}",
            time=f"{timing_loss.item():.4f}",
            type=f"{type_loss.item():.4f}",
        )

    return (
        total_loss / num_batches,
        total_gen_loss / num_batches,
        total_occ_loss / num_batches,
        total_timing_loss / num_batches,
        total_type_loss / num_batches,
    )


@torch.no_grad()
def validate(model: CausalEEGConformer, val_loader, device: torch.device, use_amp=False, horizon_tokens=10):
    model.eval()
    total_loss = 0.0
    correct_type = 0
    correct_occ = 0
    total_samples = 0
    ce_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCEWithLogitsLoss()

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
            outputs = model(windows, horizon_steps=horizon_tokens)
            pred_next = outputs["pred_next_tokens"]
            preictal_tokens = outputs["preictal_tokens"]
            occ_logits = outputs["occurrence_logits"]
            onset_preds = outputs["onset_preds"]
            type_logits = outputs["type_logits"]

            gen_loss = F.mse_loss(pred_next[:, :-1, :], preictal_tokens[:, 1:, :])
            occ_loss = bce_criterion(occ_logits, occ_targets)
            timing_loss = F.mse_loss(onset_preds, onset_targets)
            type_loss = ce_criterion(type_logits, labels)

            loss = gen_loss + 0.5 * occ_loss + 0.1 * timing_loss + 1.0 * type_loss

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


def main():
    parser = argparse.ArgumentParser(description="Train 3-Head Causal EEG-Conformer Model.")
    parser.add_argument("--master-csv", required=True, help="Path to master CSV from build_master_file()")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing session checkpoints")
    parser.add_argument("--stage", default="proc", choices=["raw", "proc"], help="Checkpoint stage ('raw' or 'proc')")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--term-value", default=None, help="Value in channel column to filter by")
    parser.add_argument("--window-samples", type=int, default=1000, help="Window sample length (default: 1000)")
    parser.add_argument("--embed-dim", type=int, default=128, help="Token embedding dimension (default: 128)")
    parser.add_argument("--horizon-tokens", type=int, default=10, help="Number of horizon tokens to generate (default: 10)")
    parser.add_argument("--binary-preictal", action="store_true", help="Map labels into binary classification")
    parser.add_argument("--cache-capacity", type=int, default=4, help="LRU cache capacity")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--output-dir", default="checkpoints", help="Directory to save training checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to specific checkpoint file")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile()")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP mixed-precision training")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--use-scheduler", action="store_true", help="Enable OneCycleLR scheduler")
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
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        logger.info(f"Loading checkpoint {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        loaded_state_dict = checkpoint["model_state_dict"]
        loaded_optimizer_state = checkpoint["optimizer_state_dict"]
        start_epoch = checkpoint.get("epoch", 1) + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)

    # 1. Datasets
    logger.info("Initializing DataLoaders...")
    train_loader, val_loader, dataset = build_dataloaders(
        master_csv=args.master_csv,
        checkpoint_dir=args.checkpoint_dir,
        stage=args.stage,
        batch_size=args.batch_size,
        term_value=args.term_value,
        window_samples=args.window_samples,
        binary_preictal=args.binary_preictal,
        cache_capacity=args.cache_capacity,
        num_workers=args.num_workers,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if loaded_optimizer_state is not None:
        optimizer.load_state_dict(loaded_optimizer_state)

    use_amp = not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    scheduler = None
    if args.use_scheduler:
        steps_per_epoch = len(train_loader) // args.grad_accum_steps
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs - start_epoch + 1,
            steps_per_epoch=steps_per_epoch,
        )

    logger.info("Starting Multi-Task Training Loop...")
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"--- Epoch {epoch:02d}/{args.epochs:02d} ---")
        train_loss, gen_l, occ_l, time_l, type_l = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            scheduler=scheduler,
            grad_accum_steps=args.grad_accum_steps,
            horizon_tokens=args.horizon_tokens,
        )
        val_loss, val_type_acc, val_occ_acc = validate(
            model, val_loader, device, use_amp=use_amp, horizon_tokens=args.horizon_tokens
        )

        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        ckpt_save_path = os.path.join(args.output_dir, f"epoch_{epoch:02d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
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
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                },
                os.path.join(args.output_dir, "best.pt"),
            )

        logger.info(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} "
            f"(Gen: {gen_l:.4f}, Occ: {occ_l:.4f}, Timing: {time_l:.4f}, Type: {type_l:.4f}) | "
            f"Val Loss: {val_loss:.4f} | Val Occ Acc: {val_occ_acc * 100:.2f}% | Val Type Acc: {type_acc * 100 if 'type_acc' in locals() else val_type_acc * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
