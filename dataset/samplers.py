"""
Batch samplers, dataset split utilities, and build_dataloaders entrypoint.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
from torch.utils.data import DataLoader, Sampler, Subset

try:
    from rich.panel import Panel
    from rich.console import Console
    _console = Console()
    _RICH_AVAILABLE = True
except ImportError:
    _console = None
    _RICH_AVAILABLE = False

if TYPE_CHECKING:
    from dataset.dataset import EEGWindowDataset


def _check_split_collisions(
    dataset: EEGWindowDataset,
    subsets: dict,
    train_split_name: str,
    val_split_name: str,
) -> None:
    """Warn if any patient IDs appear in *both* the train and val subsets."""
    def _patient_set(subset) -> set:
        indices = subset.indices if isinstance(subset, Subset) else list(range(len(subset)))
        return {dataset.patient_id_fn(dataset.df.iloc[i]["session_key"]) for i in indices}

    train_patients = _patient_set(subsets[train_split_name]) if train_split_name in subsets else set()
    val_patients   = _patient_set(subsets[val_split_name])   if val_split_name   in subsets else set()
    collisions = sorted(train_patients & val_patients)

    if not collisions:
        if _RICH_AVAILABLE and _console is not None:
            _console.print(
                Panel(
                    f"[green]✓ No patient-level collisions detected between "
                    f"'{train_split_name}' and '{val_split_name}' splits.[/green]",
                    title="[bold]Split Collision Check[/bold]",
                    border_style="green",
                )
            )
        else:
            print(f"[OK] No patient-level collisions between '{train_split_name}' and '{val_split_name}'.")
        return

    msg = (
        f"{len(collisions)} patient(s) appear in BOTH '{train_split_name}' and "
        f"'{val_split_name}' splits — this is a data-leakage risk!\n"
        f"Colliding patient IDs: {', '.join(collisions[:20])}"
        + (f" … and {len(collisions) - 20} more" if len(collisions) > 20 else "")
    )
    if _RICH_AVAILABLE and _console is not None:
        _console.print(
            Panel(
                f"[bold red]⚠  COLLISION WARNING[/bold red]\n{msg}",
                title="[bold red]Split Collision Check[/bold red]",
                border_style="red",
            )
        )
    else:
        warnings.warn(f"SPLIT COLLISION: {msg}", UserWarning, stacklevel=4)


def split_by_column(dataset: EEGWindowDataset, split_map: Optional[dict] = None):
    """Use the CSV's own `split` column rather than re-splitting.

    Returns a dict of {split_name: Subset}, e.g. {"train": ..., "dev": ..., "eval": ...}.
    """
    col = dataset.df[dataset.split_col]
    if split_map:
        col = col.map(lambda v: split_map.get(v, v))

    subsets = {}
    for split_name in col.unique():
        idx = np.where(col.values == split_name)[0].tolist()
        subsets[split_name] = Subset(dataset, idx)
    return subsets


class SessionBatchSampler(Sampler):
    """BatchSampler that groups sample indices by session_key so all samples in a batch
    come from the same session checkpoint(s).
    """

    def __init__(
        self,
        dataset: EEGWindowDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        indices: Optional[list] = None,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        target_indices = indices if indices is not None else list(range(len(dataset)))
        df_sub = dataset.df.iloc[target_indices]

        # Group indices by session_key
        self.session_groups: list[list[int]] = []
        for _, group in df_sub.groupby("session_key", sort=False):
            self.session_groups.append(group.index.tolist())

    def __iter__(self):
        groups = [list(g) for g in self.session_groups]
        if self.shuffle:
            np.random.shuffle(groups)
            for g in groups:
                np.random.shuffle(g)

        batch = []
        for g in groups:
            for idx in g:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        total = sum(len(g) for g in self.session_groups)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size


def build_dataloaders(
    master_csv: str,
    checkpoint_dir: str,
    stage: str = "raw",
    batch_size: int = 32,
    num_workers: int = 0,
    split_map: Optional[dict] = None,
    train_split_name: str = "train",
    val_split_name: str = "dev",
    use_session_batching: bool = True,
    **dataset_kwargs,
):
    """Build the dataset, split it by the CSV's split column, and return (train_loader, val_loader, dataset)."""
    from dataset.dataset import EEGWindowDataset

    dataset = EEGWindowDataset(master_csv, checkpoint_dir, stage=stage, **dataset_kwargs)
    subsets = split_by_column(dataset, split_map=split_map)

    missing = [s for s in (train_split_name, val_split_name) if s not in subsets]
    if missing:
        raise ValueError(
            f"Split value(s) {missing} not found in the CSV's split column. "
            f"Found: {list(subsets.keys())}. Pass split_map to rename, or "
            f"train_split_name/val_split_name to match your actual values."
        )

    _check_split_collisions(dataset, subsets, train_split_name, val_split_name)

    persistent = num_workers > 0
    if use_session_batching:
        train_sampler = SessionBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            indices=subsets[train_split_name].indices if hasattr(subsets[train_split_name], "indices") else None,
        )
        val_sampler = SessionBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            indices=subsets[val_split_name].indices if hasattr(subsets[val_split_name], "indices") else None,
        )

        train_loader = DataLoader(
            dataset, batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True, persistent_workers=persistent,
        )
        val_loader = DataLoader(
            dataset, batch_sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True, persistent_workers=persistent,
        )
    else:
        train_loader = DataLoader(
            subsets[train_split_name], batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True,
            pin_memory=True, persistent_workers=persistent,
        )
        val_loader = DataLoader(
            subsets[val_split_name], batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
            pin_memory=True, persistent_workers=persistent,
        )
    return train_loader, val_loader, dataset
