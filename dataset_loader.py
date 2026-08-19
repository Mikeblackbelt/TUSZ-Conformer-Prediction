"""
EEG window Dataset/DataLoader for the tuh-preprocess pipeline.

Backward-compatibility wrapper re-exporting all symbols from the `dataset` package.
"""

from dataset import *
from dataset.dataset import EEGWindowDataset
from dataset.io import (
    SessionCache,
    _default_patient_id_fn,
    _default_session_key_fn,
    _load_checkpoint_array,
    _load_npz_or_npy_array,
    _load_parquet_array,
    _normalize_dir_path,
)
from dataset.label_utils import (
    ALLOWED_LABEL_CATEGORIES,
    BACKGROUND_LABELS,
    EXCLUSION_TAGS,
    KNOWN_SEIZURE_TYPES,
    LABEL_CATEGORY_BACKGROUND,
    LABEL_CATEGORY_EXCLUDED,
    LABEL_CATEGORY_ICTAL,
    LABEL_CATEGORY_PREICTAL,
    _build_valid_ictal_mask,
    _is_background_label,
    classify_label,
)
from dataset.reporting import _bar, _print_dataset_summary
from dataset.samplers import (
    SessionBatchSampler,
    _check_split_collisions,
    build_dataloaders,
    split_by_column,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the EEG window dataloader.")
    parser.add_argument("master_csv")
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--stage", default="raw")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    train_loader, val_loader, dataset = build_dataloaders(
        args.master_csv, args.checkpoint_dir, stage=args.stage, batch_size=args.batch_size
    )
    print(f"classes: {dataset.label_map}")
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    windows, targets = next(iter(train_loader))
    if isinstance(targets, dict):
        print(f"batch windows: {tuple(windows.shape)}, labels: {tuple(targets['label'].shape)}")
    else:
        print(f"batch windows: {tuple(windows.shape)}, labels: {tuple(targets.shape)}")
    print(f"windows resized (crop/pad) so far: {dataset.n_resized}")