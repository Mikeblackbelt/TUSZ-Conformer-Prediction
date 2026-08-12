"""
EEG window Dataset/DataLoader for the tuh-preprocess pipeline.

Reads the master CSV produced by `build_master_file()` in
pipeline_gpt2class.py, pulls the matching raw/proc checkpoint array for
each row's session, slices out the labeled time window, and hands back
`(window, label)` tensors shaped for `EEGConformer`
(`(n_channels, n_samples)`).

Actual master CSV schema (confirmed):
    edf_path, csv_path, split, channel, start_time, stop_time, label,
    confidence, status

ASSUMPTIONS -- please check these and adjust constructor args if wrong:
  - `session_key` isn't a column, so it's derived from `edf_path`'s
    basename (no extension) -- e.g. ".../aaaaaaaa_s001_t000.edf" ->
    "aaaaaaaa_s001_t000" -- to match `{session_key}_{stage}.parquet/.npz/.npy`
    checkpoint filenames. Override `session_key_fn` if your checkpoint
    naming derives session_key differently.
  - `channel` follows TUSZ annotation convention: either a specific
    montage channel (e.g. "FP1-F7") or "TERM", meaning the label
    applies to the whole recording across all channels. Since windows
    here are multichannel, rows are filtered to `channel == term_value`
    ("TERM" by default) -- pass `term_value=None` to disable filtering
    and keep every row regardless of channel.
  - `split` is treated as the official TUSZ train/dev/eval split and
    used directly for train/val loaders, rather than a random
    patient-level split -- pass `split_map` if your split values
    differ from {"train", "dev", "eval"}.
  - `confidence` is available for optional filtering
    (`min_confidence`) but not used by default.
  - `status` -1 marks original/background rows; pass
    `exclude_status={-1}` to drop them if you only want generated
    preictal/postictal/interictal windows.
  - Checkpoint arrays are `(n_channels, n_samples)` float arrays at a
    fixed `sampling_rate`, matching the pipeline's axis convention.
  - CHECKPOINT FORMAT: loader now prefers `.parquet` (columns =
    channels, rows = samples, transposed back to (n_channels,
    n_samples) on load) over the legacy `.npz`/`.npy` formats, since
    parquet's columnar compression cuts checkpoint size significantly
    versus raw npz -- useful if you're tight on disk quota. All three
    formats are supported side by side per session; `.parquet` wins if
    present, otherwise falls back to `.npz` then `.npy`. Requires
    `pyarrow` (`pip install pyarrow`).
  - Patient ID (for any grouping/sanity checks) is the session_key
    substring before the first underscore (TUSZ convention:
    `aaaaaaaa_s001_t000` -> patient `aaaaaaaa`).

Everything above is a constructor kwarg -- nothing past the defaults
is hardcoded.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


def _default_session_key_fn(edf_path: str) -> str:
    """basename without extension, e.g. '.../aaaaaaaa_s001_t000.edf' -> 'aaaaaaaa_s001_t000'"""
    return os.path.splitext(os.path.basename(edf_path))[0]


def _default_patient_id_fn(session_key: str) -> str:
    """TUSZ convention: patient id is the token before the first underscore."""
    return session_key.split("_")[0]


# Checked in this order -- parquet first since it's the compressed,
# disk-friendly format; npz/npy kept as fallbacks for any sessions not
# yet converted.
_CHECKPOINT_EXTENSIONS: Tuple[str, ...] = (".parquet", ".npz", ".npy")


def _load_parquet_array(path: str) -> np.ndarray:
    """Loads a (n_channels, n_samples) float32 array from a parquet
    checkpoint. Expected layout: one column per channel, one row per
    sample -- i.e. the transpose of the pipeline's (n_channels,
    n_samples) convention -- so we transpose back on load."""
    table = pq.read_table(path)
    arr = table.to_pandas().to_numpy(dtype=np.float32).T
    return np.ascontiguousarray(arr)


def _load_npz_or_npy_array(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        with np.load(path) as data:
            if "eeg" in data:
                arr = data["eeg"]
            elif "data" in data:
                arr = data["data"]
            else:
                first_key = list(data.keys())[0]
                arr = data[first_key]
    else:
        arr = np.load(path)

    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    return arr


def _load_checkpoint_array(base_path: str) -> Optional[np.ndarray]:
    """Given a base path with no extension (e.g. '.../aaaaaaaa_s001_t000_raw'),
    tries each supported checkpoint extension in `_CHECKPOINT_EXTENSIONS`
    order and loads the first one found. Returns None if nothing exists."""
    for ext in _CHECKPOINT_EXTENSIONS:
        candidate = base_path + ext
        if os.path.exists(candidate):
            if ext == ".parquet":
                return _load_parquet_array(candidate)
            return _load_npz_or_npy_array(candidate)
    return None


_DIR_INDEX_CACHE: dict[Tuple[str, str], dict[str, str]] = {}


def _get_dir_index(checkpoint_dir: str, stage: str) -> dict[str, str]:
    """Maps '_{patient}_{sess}_' tokens (and full filename stems) to base
    paths (no extension) for every checkpoint file matching `stage` in
    `checkpoint_dir`, across all supported extensions."""
    key = (checkpoint_dir, stage)
    if key not in _DIR_INDEX_CACHE:
        index = {}
        if os.path.exists(checkpoint_dir):
            suffixes = tuple(f"_{stage}{ext}" for ext in _CHECKPOINT_EXTENSIONS)
            for fname in os.listdir(checkpoint_dir):
                matched_suffix = next((s for s in suffixes if fname.endswith(s)), None)
                if matched_suffix is None:
                    continue
                full_base_path = os.path.join(checkpoint_dir, os.path.splitext(fname)[0])
                # Parse out tokens like _aaaaaaac_s001_
                parts = fname.split("_")
                for i in range(len(parts) - 1):
                    if parts[i].startswith("s") and len(parts[i]) == 4 and parts[i][1:].isdigit() and i > 0:
                        patient = parts[i - 1]
                        sess = parts[i]
                        index[f"_{patient}_{sess}_"] = full_base_path
                # Also store full filename stem
                index[os.path.splitext(fname)[0]] = full_base_path
        _DIR_INDEX_CACHE[key] = index
    return _DIR_INDEX_CACHE[key]


def _find_checkpoint_file(checkpoint_dir: str, session_key: str, stage: str) -> Optional[str]:
    """Locates the base path (no extension) for a session's checkpoint,
    across whichever of `.parquet`/`.npz`/`.npy` actually exists."""
    base = os.path.join(checkpoint_dir, f"{session_key}_{stage}")
    if any(os.path.exists(base + ext) for ext in _CHECKPOINT_EXTENSIONS):
        return base

    index = _get_dir_index(checkpoint_dir, stage)
    parts = session_key.split("_")
    if len(parts) >= 2:
        patient, sess = parts[0], parts[1]
        target_key = f"_{patient}_{sess}_"
        if target_key in index:
            return index[target_key]
    return None


class SessionCache:
    """Small LRU cache so consecutive windows from the same session don't
    re-read the checkpoint off disk every time. Rows in the master CSV
    are typically grouped/sorted by session, so this gets a lot of reuse
    even with a small capacity."""

    def __init__(self, checkpoint_dir: str, stage: str, capacity: int = 4):
        self.checkpoint_dir = checkpoint_dir
        self.stage = stage
        self.capacity = capacity
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, session_key: str) -> np.ndarray:
        if session_key in self._cache:
            self._cache.move_to_end(session_key)
            return self._cache[session_key]

        base_path = _find_checkpoint_file(self.checkpoint_dir, session_key, self.stage)
        arr = None
        if base_path is not None:
            arr = _load_checkpoint_array(base_path)

        if arr is None:
            raise FileNotFoundError(
                f"No {self.stage} checkpoint found for session '{session_key}' "
                f"in {self.checkpoint_dir} (looked for {', '.join(_CHECKPOINT_EXTENSIONS)})"
            )

        self._cache[session_key] = arr
        self._cache.move_to_end(session_key)
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return arr


class EEGWindowDataset(Dataset):
    """One example = one labeled time window from the master CSV.

    See module docstring for the full assumption list. Key args:

        master_csv: path to the CSV from `build_master_file()`.
        checkpoint_dir: directory holding `{session_key}_{stage}.parquet`
            (preferred) or `.npz`/`.npy` (legacy fallback).
        stage: "raw" or "proc" -- which checkpoint set to read from
            (use "proc" if you ran `--create-montage`).
        sampling_rate: Hz, used to convert start/stop_time seconds
            into sample indices.
        window_samples: fixed output length in samples. Longer windows
            are center-cropped, shorter ones zero-padded (both counted
            in `self.n_resized`).
        label_map: dict mapping raw `label` values to integer class
            labels. If None, inferred by sorting unique values seen --
            fine for a first pass, but pin it explicitly once you know
            your label set so class indices stay stable across runs.
        term_value: keep only rows where `channel == term_value`
            (default "TERM"). Pass None to keep all rows regardless
            of channel.
        exclude_status: rows whose `status` value is in this set are
            dropped (default: none dropped).
        min_confidence: rows with `confidence` below this are dropped
            (default: no filtering).
    """

    def __init__(
        self,
        master_csv: str,
        checkpoint_dir: str,
        stage: str = "raw",
        sampling_rate: float = 250.0,
        window_samples: int = 1000,
        label_map: Optional[dict] = None,
        edf_path_col: str = "edf_path",
        split_col: str = "split",
        channel_col: str = "channel",
        start_col: str = "start_time",
        stop_col: str = "stop_time",
        label_col: str = "label",
        confidence_col: str = "confidence",
        status_col: str = "status",
        term_value: Optional[str] = None,
        exclude_status: Optional[set] = {0, 2},
        exclude_prefix: Tuple[str, ...] = ("x",),
        min_confidence: Optional[float] = None,
        binary_preictal: bool = False,
        session_key_fn: Callable[[str], str] = _default_session_key_fn,
        patient_id_fn: Callable[[str], str] = _default_patient_id_fn,
        cache_capacity: int = 4,
        skip_missing_checkpoints: bool = True,
    ):
        self.df = pd.read_csv(master_csv)
        required = [edf_path_col, start_col, stop_col, label_col]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"master_csv is missing expected column(s) {missing}. "
                f"Available columns: {list(self.df.columns)}."
            )

        self.session_key_fn = session_key_fn
        self.df["session_key"] = self.df[edf_path_col].apply(session_key_fn)

        # 1. Channel Montage Filtering
        if term_value is not None and channel_col in self.df.columns:
            self.df = self.df[self.df[channel_col] == term_value].reset_index(drop=True)

        # 2. Status Filtering (Exclude collapsed status 0 and 2 from TUH_preprocess)
        if exclude_status and status_col in self.df.columns:
            self.df = self.df[~self.df[status_col].isin(exclude_status)].reset_index(drop=True)

        # 3. Label Exclusion Interval Filtering (Exclude x* exclusion intervals)
        if exclude_prefix and label_col in self.df.columns:
            pattern = "^(?:" + "|".join(exclude_prefix) + ")"
            self.df = self.df[~self.df[label_col].astype(str).str.contains(pattern, regex=True)].reset_index(drop=True)

        # 4. Confidence Filtering
        if min_confidence is not None and confidence_col in self.df.columns:
            self.df = self.df[self.df[confidence_col] >= min_confidence].reset_index(drop=True)

        # 5. Filter out sessions whose checkpoint files do not exist on disk
        if skip_missing_checkpoints and os.path.exists(checkpoint_dir):
            tqdm.pandas(desc="Verifying checkpoint files")
            valid_mask = self.df["session_key"].progress_apply(
                lambda k: _find_checkpoint_file(checkpoint_dir, k, stage) is not None
            )
            initial_count = len(self.df)
            self.df = self.df[valid_mask].reset_index(drop=True)
            print(f"Skipped {initial_count - len(self.df)} rows missing checkpoints. {len(self.df)} valid rows retained.")

        if len(self.df) == 0:
            raise ValueError(
                "No rows left after filtering -- check term_value/exclude_status/"
                "min_confidence against your actual data."
            )

        self.checkpoint_dir = checkpoint_dir
        self.stage = stage
        self.sampling_rate = sampling_rate
        self.window_samples = window_samples
        self.split_col = split_col
        self.start_col = start_col
        self.stop_col = stop_col
        self.label_col = label_col
        self.patient_id_fn = patient_id_fn

        # Label Mapping
        if binary_preictal:
            # Map preictal (p*) -> 1, all others -> 0
            self.df[label_col] = self.df[label_col].astype(str).apply(lambda l: 1 if l.startswith("p") else 0)
            label_map = {0: 0, 1: 1}
        elif label_map is None:
            uniq = sorted(self.df[label_col].dropna().unique().tolist())
            label_map = {lab: i for i, lab in enumerate(uniq)}

        self.label_map = label_map
        self.num_classes = len(set(label_map.values()))

        self._cache = SessionCache(checkpoint_dir, stage, capacity=cache_capacity)
        self.n_resized = 0  # counts windows that needed crop/pad, for sanity checks

    def __len__(self) -> int:
        return len(self.df)

    @property
    def patient_ids(self) -> list:
        """Patient id per row, aligned with dataset indices."""
        return [self.patient_id_fn(k) for k in self.df["session_key"]]

    def _slice_window(self, arr: np.ndarray, start_time: float, stop_time: float) -> np.ndarray:
        start_idx = int(round(start_time * self.sampling_rate))
        stop_idx = int(round(stop_time * self.sampling_rate))
        stop_idx = max(stop_idx, start_idx + 1)

        window = arr[:, start_idx:stop_idx]
        n = window.shape[1]

        if n == self.window_samples:
            return window

        self.n_resized += 1
        if n > self.window_samples:
            # center-crop
            offset = (n - self.window_samples) // 2
            return window[:, offset: offset + self.window_samples]

        # zero-pad (centered)
        pad_total = self.window_samples - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(window, ((0, 0), (pad_left, pad_right)), mode="constant")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        session_key = row["session_key"]

        arr = self._cache.get(session_key)
        window = self._slice_window(arr, float(row[self.start_col]), float(row[self.stop_col]))

        label = self.label_map[row[self.label_col]]

        window_t = torch.from_numpy(np.ascontiguousarray(window)).float()
        label_t = torch.tensor(label, dtype=torch.long)
        return window_t, label_t


def split_by_column(dataset: EEGWindowDataset, split_map: Optional[dict] = None):
    """Use the CSV's own `split` column (TUSZ's official train/dev/eval
    split) rather than re-splitting ourselves -- avoids accidentally
    contradicting the split TUSZ already validated for patient
    non-overlap.

    Returns a dict of {split_name: Subset}, e.g. {"train": ..., "dev": ...,
    "eval": ...} -- whatever distinct values exist in the split column.
    `split_map` can rename values, e.g. {"training": "train"} if your
    CSV uses different strings.
    """
    from torch.utils.data import Subset

    col = dataset.df[dataset.split_col]
    if split_map:
        col = col.map(lambda v: split_map.get(v, v))

    subsets = {}
    for split_name in col.unique():
        idx = np.where(col.values == split_name)[0].tolist()
        subsets[split_name] = Subset(dataset, idx)
    return subsets


def build_dataloaders(
    master_csv: str,
    checkpoint_dir: str,
    stage: str = "raw",
    batch_size: int = 32,
    num_workers: int = 0,
    split_map: Optional[dict] = None,
    train_split_name: str = "train",
    val_split_name: str = "dev",
    **dataset_kwargs,
):
    """Convenience wrapper: build the dataset, split it by the CSV's own
    `split` column, and return (train_loader, val_loader, dataset).
    `dataset` is returned too since it holds `label_map`/`num_classes`
    you'll want for the model, plus any other splits (e.g. "eval") via
    `split_by_column(dataset)` directly."""
    dataset = EEGWindowDataset(master_csv, checkpoint_dir, stage=stage, **dataset_kwargs)
    subsets = split_by_column(dataset, split_map=split_map)

    missing = [s for s in (train_split_name, val_split_name) if s not in subsets]
    if missing:
        raise ValueError(
            f"Split value(s) {missing} not found in the CSV's split column. "
            f"Found: {list(subsets.keys())}. Pass split_map to rename, or "
            f"train_split_name/val_split_name to match your actual values."
        )

    train_loader = DataLoader(
        subsets[train_split_name], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        subsets[val_split_name], batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader, dataset


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

    windows, labels = next(iter(train_loader))
    print(f"batch windows: {tuple(windows.shape)}, labels: {tuple(labels.shape)}")
    print(f"windows resized (crop/pad) so far: {dataset.n_resized}")