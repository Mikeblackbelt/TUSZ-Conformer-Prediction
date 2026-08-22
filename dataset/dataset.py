"""
Main EEGWindowDataset PyTorch Dataset implementation.
"""

from __future__ import annotations

import bisect
import os
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.io import (
    SessionCache,
    _default_patient_id_fn,
    _default_session_key_fn,
    _find_checkpoint_file,
    _get_session_file_offsets,
    _match_offset_entry,
    _normalize_dir_path,
)
from dataset.label_utils import (
    LABEL_CATEGORY_BACKGROUND,
    LABEL_CATEGORY_PREICTAL,
    LABEL_CATEGORY_ICTAL,
    LABEL_CATEGORY_EXCLUDED,
    SEIZURE_TYPE_CLASSES,
    _build_valid_ictal_mask,
    _is_background_label,
    classify_label,
    extract_seizure_type,
)
from dataset.reporting import _print_dataset_summary


class EEGWindowDataset(Dataset):
    """One example = one labeled time window from the master CSV."""

    def __init__(
        self,
        master_csv: str,
        checkpoint_dir: str,
        stage: str = "raw",
        sampling_rate: float = 256.0,
        window_samples: int = 1000,
        horizon_window_samples: Optional[int] = None,
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
        exclude_prefix: bool = True,
        exclude_labels: Tuple[str, ...] = ("bckg",),
        exclude_ictal_without_preictal: bool = True,
        min_confidence: Optional[float] = None,
        binary_preictal: bool = False,
        session_key_fn: Callable[[str], str] = _default_session_key_fn,
        patient_id_fn: Callable[[str], str] = _default_patient_id_fn,
        cache_capacity: int = 128,
        skip_missing_checkpoints: bool = True,
        return_dict: bool = True,
        timing_norm: float = 300.0,
        max_bg_ratio: float = 3.0,
    ):
        master_csv = _normalize_dir_path(master_csv)
        checkpoint_dir = _normalize_dir_path(checkpoint_dir)

        self.df = pd.read_csv(master_csv)
        initial_count = len(self.df)
        counts = {"initial": initial_count}

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
        counts["after_channel"] = len(self.df)

        # 2. Status Filtering
        if exclude_status and status_col in self.df.columns:
            self.df = self.df[~self.df[status_col].isin(exclude_status)].reset_index(drop=True)
        counts["after_status"] = len(self.df)

        # 2b. Exact-label exclusions
        if exclude_labels and label_col in self.df.columns:
            self.df = self.df[~self.df[label_col].astype(str).isin(exclude_labels)].reset_index(drop=True)
        counts["after_exclude_labels"] = len(self.df)

        # 3. Ictal-validity filter
        if exclude_ictal_without_preictal and label_col in self.df.columns:
            keep_mask = _build_valid_ictal_mask(self.df, label_col, start_col)
            self.df = self.df[keep_mask].reset_index(drop=True)
        counts["after_ictal_filter"] = len(self.df)

        # 4. Excluded-category filtering (postictal q*, consecutive c*,
        # artifact/exclusion x*, SOP-buffer p*_sopbuffer, ...).
   
        if exclude_prefix and label_col in self.df.columns:
            self.df = self.df[
                self.df[label_col].apply(lambda l: classify_label(l) != LABEL_CATEGORY_EXCLUDED)
            ].reset_index(drop=True)
        counts["after_label_prefix"] = len(self.df)

        # 5. Confidence Filtering
        if min_confidence is not None and confidence_col in self.df.columns:
            self.df = self.df[self.df[confidence_col] >= min_confidence].reset_index(drop=True)
        counts["after_confidence"] = len(self.df)

        # 5b. Background subsampling
        if max_bg_ratio is not None:
            bg_mask = self.df[label_col].apply(lambda l: classify_label(l) == LABEL_CATEGORY_BACKGROUND)
            n_nonbg = int((~bg_mask).sum())
            n_bg = int(bg_mask.sum())
            max_bg = int(n_nonbg * max_bg_ratio)
            if n_bg > max_bg:
                keep_bg_idx = (
                    self.df[bg_mask]
                    .sample(n=max_bg, random_state=42)
                    .index
                )
                self.df = pd.concat([
                    self.df[~bg_mask],
                    self.df.loc[keep_bg_idx],
                ]).sort_index().reset_index(drop=True)
                print(f"Background subsampled: {n_bg} -> {max_bg} rows "
                      f"(ratio {max_bg_ratio}x {n_nonbg} non-bg rows).")
        counts["after_bg_subsample"] = len(self.df)

        # 6. Filter out sessions missing checkpoints
        if skip_missing_checkpoints:
            if not os.path.exists(checkpoint_dir):
                print(f"Warning: Checkpoint directory '{checkpoint_dir}' does not exist on disk!")
            tqdm.pandas(desc="Verifying checkpoint files")
            valid_mask = self.df["session_key"].progress_apply(
                lambda k: _find_checkpoint_file(checkpoint_dir, k, stage) is not None
            )
            initial_check = len(self.df)
            self.df = self.df[valid_mask].reset_index(drop=True)
            counts["after_checkpoints"] = len(self.df)
            print(f"Skipped {initial_check - len(self.df)} rows missing checkpoints in '{checkpoint_dir}'. {len(self.df)} valid rows retained.")

        if len(self.df) == 0:
            msg = (
                f"No rows left after filtering! Filter breakdown:\n"
                f"  - Initial CSV rows: {counts.get('initial')}\n"
                f"  - After channel filter (term_value={term_value}): {counts.get('after_channel')}\n"
                f"  - After status filter (exclude_status={exclude_status}): {counts.get('after_status')}\n"
                f"  - After ictal-validity filter (exclude_ictal_without_preictal={exclude_ictal_without_preictal}): {counts.get('after_ictal_filter')}\n"
                f"  - After label prefix filter (exclude_prefix={exclude_prefix}): {counts.get('after_label_prefix')}\n"
                f"  - After confidence filter (min_confidence={min_confidence}): {counts.get('after_confidence')}\n"
                f"  - After checkpoint verification in '{checkpoint_dir}': {counts.get('after_checkpoints', 0)}\n"
                f"Check your filter options or WSL path formats."
            )
            raise ValueError(msg)

        self.checkpoint_dir = checkpoint_dir
        self.stage = stage
        self.sampling_rate = sampling_rate
        self.window_samples = window_samples
        self.horizon_window_samples = horizon_window_samples if horizon_window_samples is not None else window_samples
        self.edf_path_col = edf_path_col
        self.split_col = split_col
        self.start_col = start_col
        self.stop_col = stop_col
        self.label_col = label_col
        self.patient_id_fn = patient_id_fn

        self.binary_preictal = binary_preictal
        self.df["_raw_label"] = self.df[label_col].astype(str)

        # Label Mapping
        if binary_preictal:
            self.df[label_col] = self.df["_raw_label"].apply(
                lambda l: 1 if classify_label(l) == LABEL_CATEGORY_PREICTAL else 0
            )
            label_map = {0: 0, 1: 1}
        elif label_map is None:
            uniq = sorted(self.df[label_col].dropna().unique().tolist())
            label_map = {lab: i for i, lab in enumerate(uniq)}

        self.label_map = label_map
        self.num_classes = len(set(label_map.values()))

        # Seizure-TYPE target, independent of the binary occurrence/preictal
        # flag above. `label`/`occurrence` answer "is a seizure imminent";
        # `seizure_type` answers "which kind" (fnsz/gnsz/...), so the type
        # classification head has a real, distinct task to learn instead of
        # duplicating occurrence. See extract_seizure_type() in label_utils.
        self.df["_seizure_type_raw"] = self.df["_raw_label"].apply(extract_seizure_type)
        self.type_classes = SEIZURE_TYPE_CLASSES
        self.type_label_map = {t: i for i, t in enumerate(self.type_classes)}
        self.num_type_classes = len(self.type_classes)
        self.df["_seizure_type_idx"] = self.df["_seizure_type_raw"].apply(
            lambda t: self.type_label_map.get(t, -1)
        )

        self.status_col = status_col
        self.return_dict = return_dict
        self.timing_norm = timing_norm
        self._cache = SessionCache(checkpoint_dir, stage, capacity=cache_capacity)
        self.n_resized = 0
        self._n_channels: Optional[int] = None
        self._warned_missing_offset: set = set()

        self._session_ictal_onsets: dict = {}
        is_ictal = self.df["_raw_label"].apply(
            lambda l: classify_label(l) == LABEL_CATEGORY_ICTAL
        )
        for session_key, grp in self.df[is_ictal].groupby("session_key"):
            self._session_ictal_onsets[session_key] = sorted(grp[start_col].astype(float).tolist())

        split_series = self.df[split_col] if split_col in self.df.columns else None
        _print_dataset_summary(
            counts=counts,
            label_map=label_map,
            label_col_values=self.df[label_col],
            split_col=split_col,
            split_series=split_series,
            binary_preictal=binary_preictal,
        )

    def __len__(self) -> int:
        return len(self.df)

    @property
    def patient_ids(self) -> list:
        """Patient id per row, aligned with dataset indices."""
        return [self.patient_id_fn(k) for k in self.df["session_key"]]

    def get_class_counts(self, indices: Optional[list] = None) -> dict:
        col = self.df[self.label_col]
        if indices is not None:
            col = col.iloc[indices]
        counts: dict = {}
        for raw_val in col:
            cls_idx = self.label_map.get(raw_val, raw_val)
            counts[cls_idx] = counts.get(cls_idx, 0) + 1
        return counts

    def class_weights_tensor(
        self,
        indices: Optional[list] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        counts = self.get_class_counts(indices)
        total = sum(counts.values())
        num_cls = self.num_classes
        weights = []
        for c in range(num_cls):
            cnt = counts.get(c, 1)
            raw_w = total / (num_cls * max(1, cnt))
            w = min(raw_w, 25.0)
            weights.append(w)
        t = torch.tensor(weights, dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return t

    def seizure_type_class_weights_tensor(
        self,
        indices: Optional[list] = None,
        device: Optional[torch.device] = None,
        cap: float = 8.0,
    ) -> torch.Tensor:
        """Inverse-frequency class weights for the seizure-TYPE head, computed
        only over rows with a valid (>= 0) seizure type (i.e. actual
        preictal/ictal windows -- background rows have no type and are
        excluded from this computation, matching how the type loss/metric
        are masked to occ==1 samples during training/validation).

        `cap` bounds how large any single class's weight can get. Lowered
        from an earlier 25.0 default: with several seizure subtypes this rare
        in the TUSZ split, a 25x weight on a handful of examples was
        dominating the joint loss and producing very large, noisy gradients
        (visible as elevated `gnorm` and near-zero Val Type Acc). If most
        classes still saturate at `cap`, consider merging the rarest types
        into an explicit "other" bucket instead of raising the cap further.
        """
        col = self.df["_seizure_type_idx"]
        if indices is not None:
            col = col.iloc[indices]
        valid = col[col >= 0]
        counts = valid.value_counts().to_dict()
        total = sum(counts.values())
        num_cls = self.num_type_classes
        weights = []
        for c in range(num_cls):
            cnt = counts.get(c, 1)
            raw_w = (total / (num_cls * max(1, cnt))) if total > 0 else 1.0
            w = min(raw_w, cap)
            weights.append(w)
        t = torch.tensor(weights, dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return t

    def occ_pos_weight(
        self,
        indices: Optional[list] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if getattr(self, "binary_preictal", False):
            col = self.df[self.label_col]
            if indices is not None:
                col = col.iloc[indices]
            positives = int((col == 1).sum())
            negatives = int((col == 0).sum())
        else:
            raw_col = self.df["_raw_label"] if "_raw_label" in self.df.columns else self.df[self.label_col]
            if indices is not None:
                raw_col = raw_col.iloc[indices]

            negatives = 0
            positives = 0
            for raw_val in raw_col:
                if _is_background_label(raw_val):
                    negatives += 1
                else:
                    positives += 1

        negatives = max(1, negatives)
        positives = max(1, positives)
        ratio = float(negatives / positives)
        ratio = min(ratio, 20.0)
        pw = torch.tensor([ratio], dtype=torch.float32)
        if device is not None:
            pw = pw.to(device)
        return pw

    def _slice_window(
        self,
        arr: np.ndarray,
        start_time: float,
        stop_time: float,
        offset_samples: int = 0,
        target_samples: Optional[int] = None,
    ) -> np.ndarray:
        target_samples = target_samples if target_samples is not None else self.window_samples

        start_idx = int(round(start_time * self.sampling_rate)) + offset_samples
        stop_idx = int(round(stop_time * self.sampling_rate)) + offset_samples
        stop_idx = max(stop_idx, start_idx + 1)

        n_total = arr.shape[1]
        start_idx = max(0, min(start_idx, n_total))
        stop_idx = max(start_idx, min(stop_idx, n_total))

        window = arr[:, start_idx:stop_idx]
        n = window.shape[1]

        if n == target_samples:
            return window

        self.n_resized += 1
        if n > target_samples:
            offset = (n - target_samples) // 2
            return window[:, offset: offset + target_samples]

        pad_total = target_samples - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(window, ((0, 0), (pad_left, pad_right)), mode="constant")

    def _get_offset_samples(self, session_key: str, edf_path: str) -> int:
        entries = _get_session_file_offsets(self.checkpoint_dir, self.stage, session_key)
        if not entries:
            return 0
        match = _match_offset_entry(entries, edf_path)
        if match is None:
            if session_key not in self._warned_missing_offset:
                self._warned_missing_offset.add(session_key)
                print(
                    f"Warning: session '{session_key}' has an offsets file but no entry "
                    f"matched edf_path={edf_path!r} -- using offset 0 for this row. Check path formatting."
                )
            return 0
        return int(match.get("start_sample", 0))

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        session_key = row["session_key"]

        try:
            arr = self._cache.get(session_key)
        except FileNotFoundError:
            # Corrupted or missing checkpoint — return a zero-filled sample
            # so the DataLoader worker doesn't crash mid-epoch.
            import logging as _log
            _log.getLogger(__name__).warning(
                "Skipping corrupted/missing checkpoint for session '%s' (row %d)",
                session_key, idx,
            )
            nc = self._n_channels if self._n_channels is not None else 1
            window_t = torch.zeros((nc, self.window_samples), dtype=torch.float32)
            label_t = torch.tensor(0, dtype=torch.long)
            if not self.return_dict:
                return window_t, label_t
            return window_t, {
                "label": label_t,
                "occurrence": torch.tensor(0.0, dtype=torch.float32),
                "seizure_type": torch.tensor(-1, dtype=torch.long),
                "onset_offset": torch.tensor(0.0, dtype=torch.float32),
                "status": torch.tensor(1, dtype=torch.long),
                "horizon_window": torch.zeros((nc, self.horizon_window_samples), dtype=torch.float32),
                "has_horizon": torch.tensor(0.0, dtype=torch.float32),
            }

        # Cache the channel count from the first successful load so fallback
        # paths can produce tensors with a matching shape.
        if self._n_channels is None:
            self._n_channels = arr.shape[0]

        start_time = float(row[self.start_col])
        stop_time = float(row[self.stop_col])
        offset_samples = self._get_offset_samples(session_key, row[self.edf_path_col])
        window = self._slice_window(arr, start_time, stop_time, offset_samples=offset_samples)

        raw_label = row["_raw_label"] if "_raw_label" in row else row[self.label_col]
        label = self.label_map[row[self.label_col]] if row[self.label_col] in self.label_map else row[self.label_col]
        status_val = int(row[self.status_col]) if (self.status_col in row and pd.notna(row[self.status_col])) else 1

        window_t = torch.from_numpy(np.ascontiguousarray(window)).float()
        label_t = torch.tensor(label, dtype=torch.long)

        if not self.return_dict:
            return window_t, label_t

        raw_str = str(raw_label)
        cat = classify_label(raw_label)
        if getattr(self, "binary_preictal", False):
            has_seizure = 1.0 if label == 1 else 0.0
        else:
            has_seizure = 0.0 if cat == LABEL_CATEGORY_BACKGROUND else 1.0
        occurrence_t = torch.tensor(has_seizure, dtype=torch.float32)

        seizure_type_idx = int(row["_seizure_type_idx"]) if "_seizure_type_idx" in row else -1
        seizure_type_t = torch.tensor(seizure_type_idx, dtype=torch.long)

        onset_time: Optional[float] = None
        has_horizon = False
        if cat == LABEL_CATEGORY_PREICTAL:
            onsets = self._session_ictal_onsets.get(session_key, [])
            pos = bisect.bisect_left(onsets, stop_time)
            if pos < len(onsets):
                onset_time = onsets[pos]
                relative_onset = max(0.0, onset_time - stop_time)
                has_horizon = onset_time > stop_time
            else:
                relative_onset = max(0.0, float(stop_time - start_time))
        elif has_seizure > 0:
            relative_onset = 0.0
        else:
            relative_onset = 0.0

        relative_onset_clamped = min(relative_onset, self.timing_norm)
        onset_offset_t = torch.tensor(relative_onset_clamped / self.timing_norm, dtype=torch.float32)
        status_t = torch.tensor(status_val, dtype=torch.long)

        if has_horizon and onset_time is not None:
            horizon_window = self._slice_window(
                arr, stop_time, onset_time,
                offset_samples=offset_samples,
                target_samples=self.horizon_window_samples,
            )
            horizon_window_t = torch.from_numpy(np.ascontiguousarray(horizon_window)).float()
            has_horizon_t = torch.tensor(1.0, dtype=torch.float32)
        else:
            horizon_window_t = torch.zeros((window_t.shape[0], self.horizon_window_samples), dtype=torch.float32)
            has_horizon_t = torch.tensor(0.0, dtype=torch.float32)

        targets = {
            "label": label_t,
            "occurrence": occurrence_t,
            "seizure_type": seizure_type_t,
            "onset_offset": onset_offset_t,
            "status": status_t,
            "horizon_window": horizon_window_t,
            "has_horizon": has_horizon_t,
        }

        return window_t, targets