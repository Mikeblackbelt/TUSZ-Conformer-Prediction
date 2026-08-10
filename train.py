"""
Full Training Pipeline:
1. Loads EEG window datasets using `dataset_loader.build_dataloaders`.
2. Stage 1: Causal Generative EEG-Conformer (Generates next-patch representations / models EEG token dynamics).
3. Stage 2: [Placeholder] Trajectory / Phase Dynamics Head.
4. Stage 3: [Placeholder] Ictal Transition / Seizure Onset Predictor Head.
"""

from __future__ import annotations

import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset_loader import build_dataloaders
from conformer import CausalEEGConformer

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Stage 2 Model Placeholder: Temporal Dynamics / Trajectory Aggregator
# --------------------------------------------------------------------------
class Stage2DynamicsHead(nn.Module):
    """
    Stage 2 Placeholder:
    Receives token sequence / representations from Stage 1 Causal Conformer
    and models sequence-level phase trajectory (e.g. interictal vs preictal progression).
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        # TODO: Implement full Stage 2 architecture (e.g., Sequence Transformer / LSTM / Trajectory Encoder)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, stage1_tokens: torch.Tensor) -> torch.Tensor:
        """
        Input stage1_tokens: (B, seq_len, embed_dim)
        Output: (B, seq_len, embed_dim)
        """
        h = F.relu(self.fc1(stage1_tokens))
        out = self.fc2(h)
        return out


# --------------------------------------------------------------------------
# Stage 3 Model Placeholder: Ictal Transition / Seizure Onset Predictor
# --------------------------------------------------------------------------
class Stage3TransitionHead(nn.Module):
    """
    Stage 3 Placeholder:
    Receives aggregated features from Stage 2 and predicts final ictal onset / phase transition output.
    """

    def __init__(self, embed_dim: int = 128, num_classes: int = 2):
        super().__init__()
        # TODO: Implement full Stage 3 architecture (e.g., Time-to-Onset Regressor or Ictal Transition Classifier)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, stage2_features: torch.Tensor) -> torch.Tensor:
        """
        Input stage2_features: (B, seq_len, embed_dim)
        Output logits: (B, num_classes)
        """
        # Global temporal pooling: (B, embed_dim, seq_len) -> (B, embed_dim)
        h = self.pool(stage2_features.permute(0, 2, 1)).squeeze(-1)
        logits = self.classifier(h)
        return logits


# --------------------------------------------------------------------------
# Full End-to-End Pipeline Model
# --------------------------------------------------------------------------
class FullPipelineModel(nn.Module):
    """
    Unified 3-stage model:
      Stage 1: Causal EEG-Conformer
      Stage 2: Trajectory / Dynamics Head (Placeholder)
      Stage 3: Ictal Transition Head (Placeholder)
    """

    def __init__(
        self,
        n_channels: int,
        embed_dim: int = 128,
        num_classes: int = 2,
        conformer_depth: int = 4,
    ):
        super().__init__()
        # Stage 1: Causal EEG-Conformer
        self.stage1_conformer = CausalEEGConformer(
            n_channels=n_channels,
            embed_dim=embed_dim,
            depth=conformer_depth,
        )
        # Stage 2: Dynamics Head
        self.stage2_dynamics = Stage2DynamicsHead(embed_dim=embed_dim)

        # Stage 3: Transition Predictor
        self.stage3_transition = Stage3TransitionHead(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        """
        x: (B, n_channels, n_samples)
        Returns:
            pred_next_tokens: (B, seq_len, embed_dim) from Stage 1
            stage1_tokens: (B, seq_len, embed_dim) ground truth tokens
            stage2_out: (B, seq_len, embed_dim) Stage 2 trajectory output
            stage3_logits: (B, num_classes) Stage 3 classification/prediction logits
        """
        pred_next_tokens, stage1_tokens = self.stage1_conformer(x)
        stage2_out = self.stage2_dynamics(pred_next_tokens)
        stage3_logits = self.stage3_transition(stage2_out)

        return pred_next_tokens, stage1_tokens, stage2_out, stage3_logits


# --------------------------------------------------------------------------
# Training & Validation Functions
# --------------------------------------------------------------------------
def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_stage1_loss = 0.0
    total_stage3_loss = 0.0

    ce_criterion = nn.CrossEntropyLoss()
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc="Training")
    for i, (windows, labels) in enumerate(pbar, start=1):
        windows = windows.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        pred_next, tokens, stage2_out, stage3_logits = model(windows)

        # 1. Stage 1 Causal Autoregressive Next-Patch MSE Loss
        stage1_loss = F.mse_loss(pred_next[:, :-1, :], tokens[:, 1:, :])

        # 2. Stage 3 Classification/Transition CrossEntropy Loss
        stage3_loss = ce_criterion(stage3_logits, labels)

        # Combined Loss (Stage 1 + Stage 2 Placeholder + Stage 3)
        loss = stage1_loss + stage3_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_stage1_loss += stage1_loss.item()
        total_stage3_loss += stage3_loss.item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}", 
            s1=f"{stage1_loss.item():.4f}", 
            s3=f"{stage3_loss.item():.4f}"
        )

    return total_loss / num_batches, total_stage1_loss / num_batches, total_stage3_loss / num_batches


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    ce_criterion = nn.CrossEntropyLoss()

    for windows, labels in tqdm(val_loader, desc="Validating"):
        windows = windows.to(device)
        labels = labels.to(device)

        pred_next, tokens, stage2_out, stage3_logits = model(windows)

        stage1_loss = F.mse_loss(pred_next[:, :-1, :], tokens[:, 1:, :])
        stage3_loss = ce_criterion(stage3_logits, labels)
        loss = stage1_loss + stage3_loss

        total_loss += loss.item()

        preds = stage3_logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    num_batches = len(val_loader)
    acc = correct / total if total > 0 else 0.0
    return total_loss / num_batches, acc


# --------------------------------------------------------------------------
# Main CLI Entrypoint
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train 3-Stage Causal Conformer Pipeline.")
    parser.add_argument("--master-csv", required=True, help="Path to master CSV from build_master_file()")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing .npz/.npy session checkpoints")
    parser.add_argument("--stage", default="proc", choices=["raw", "proc"], help="Checkpoint stage ('raw' or 'proc')")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--term-value", default=None, help="Value in channel column to filter by (default: None, keep all rows)")
    parser.add_argument("--window-samples", type=int, default=1000, help="Window sample length (default: 1000)")
    parser.add_argument("--embed-dim", type=int, default=128, help="Token embedding dimension (default: 128)")
    parser.add_argument("--binary-preictal", action="store_true", help="Map labels into binary classification (1: preictal p*, 0: interictal)")
    parser.add_argument("--cache-capacity", type=int, default=128, help="LRU cache capacity for session arrays (default: 128)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes (default: 0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Datasets
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
    sample_windows, _ = next(iter(train_loader))
    _, n_channels, n_samples = sample_windows.shape

    logger.info(f"Dataset loaded: {len(dataset)} samples | {num_classes} classes")
    logger.info(f"EEG shape: channels={n_channels}, samples={n_samples}")

    # 2. Build Model
    model = FullPipelineModel(
        n_channels=n_channels,
        embed_dim=args.embed_dim,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 3. Train Loop
    logger.info("Starting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"--- Epoch {epoch:02d}/{args.epochs:02d} ---")
        train_loss, s1_loss, s3_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)

        logger.info(
            f"Epoch {epoch:02d} Summary | "
            f"Train Loss: {train_loss:.4f} (Stage1: {s1_loss:.4f}, Stage3: {s3_loss:.4f}) | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
