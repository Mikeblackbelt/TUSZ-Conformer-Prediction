"""
Full Multi-Task Training Pipeline.

Backward-compatibility wrapper re-exporting all symbols from the `training` package.
"""

from training import *
from training.cli import main
from training.losses import FocalBCEWithLogitsLoss, FocalCrossEntropyLoss, _compute_horizon_loss
from training.metrics import _macro_prf1_from_confusion, compute_confusion_matrix, log_confusion_matrix
from training.trainer import train_epoch, validate

if __name__ == "__main__":
    main()