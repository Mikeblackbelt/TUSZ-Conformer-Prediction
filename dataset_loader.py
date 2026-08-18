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
    here are multichannel, rows CAN be filtered to `channel ==
    term_value` by passing e.g. `term_value="TERM"` -- the default is
    `term_value=None`, which disables this filter and keeps every row
    regardless of channel.
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
  - MULTI-FILE SESSIONS: a session's checkpoint array is a
    concatenation of every .edf file in that session (see
    Tuh-Preprocess's raw_eeg_extraction.concatenate_session_eeg), but
    each row's start_time/stop_time are local to its own file. If a
    `{composed_session_key}_offsets.json` file (written by
    checkpoint_io.save_offsets) is found next to the checkpoint, each
    row's slice is automatically shifted by that file's offset within
    the concatenated array. Sessions without an offsets file (e.g.
    single-file sessions, or checkpoints written before this was
    tracked) fall back to offset 0, matching the previous behavior.
  - `sampling_rate` MUST match the checkpoint's actual sample rate
    (Tuh-Preprocess's raw_eeg_extraction.TARGET_SFREQ, 256 Hz as of
    this pipeline) -- it is not a free tuning knob, since every
    start_time/stop_time -> sample-index conversion depends on it.
  - `exclude_labels` drops exact-match labels regardless of prefix
    filtering (default: `("bckg",)`). True background/negative-class
    windows should come from explicitly-derived "bg" rows (see
    Tuh-Preprocess's preictal_segment.add_background_tags), not the
    "bckg" tag, which is artifact-flagged and not reliable clean
    background.
  - `has_horizon`/`horizon_window` in the returned targets dict give
    the REAL ground-truth EEG spanning from the end of a preictal
    window to the actual seizure onset, for rows where that's
    resolvable -- used to supervise the model's horizon generator
    against reality (see train.py's horizon_loss) rather than only
    against its own within-window continuation.

Everything above is a constructor kwarg -- nothing past the defaults
is hardcoded.
"""

from __future__ import annotations

import bisect
import json
import os
from collections import OrderedDict
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

_console = Console() if _RICH_AVAILABLE else None


def _default_session_key_fn(edf_path: str) -> str:
    """basename without extension, e.g. '.../aaaaaaaa_s001_t000.edf' -> 'aaaaaaaa_s001_t000'"""
    # Normalize backslashes to forward slashes for Linux/WSL compatibility
    normalized_path = str(edf_path).replace("\\", "/")
    return os.path.splitext(os.path.basename(normalized_path))[0]


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


def _composed_session_key_from_base_path(base_path: str, stage: str) -> str:
    """base_path is '.../<checkpoint_dir>/<actual_checkpoint_session_key>_<stage>'
    (no extension) -- e.g. the checkpoint that ACTUALLY got saved for a
    whole session (see pipeline/session_index.py in Tuh-Preprocess), which
    is not necessarily the same string as this dataset's per-row
    `session_key` (derived from the individual .edf file's basename -- see
    `_default_session_key_fn`). Strips the directory and the trailing
    "_<stage>" to recover the actual session key used for files that
    aren't stage-specific, like the per-file sample offsets JSON."""
    fname = os.path.basename(base_path)
    suffix = f"_{stage}"
    if fname.endswith(suffix):
        fname = fname[: -len(suffix)]
    return fname


_OFFSETS_CACHE: dict[Tuple[str, str], Optional[list]] = {}


def _get_session_file_offsets(checkpoint_dir: str, stage: str, session_key: str) -> Optional[list]:
    """Loads the `{composed_session_key}_offsets.json` file (written by
    Tuh-Preprocess's `checkpoint_io.save_offsets()`) for the session a row
    belongs to, if one exists. A session's raw/proc checkpoint array is a
    CONCATENATION of every .edf file in that session -- when a session has
    more than one file (t000, t001, ... -- common in TUSZ), each file's
    annotation start_time/stop_time are local to that individual file, not
    the concatenated array, so they must be shifted by that file's sample
    offset within the array before slicing. Returns None (rather than an
    empty list) when no offsets file is found, e.g. single-file sessions or
    checkpoints written before this was tracked -- callers should treat
    that as "no correction available/needed", not an error.
    """
    base_path = _find_checkpoint_file(checkpoint_dir, session_key, stage)
    if base_path is None:
        return None
    composed_key = _composed_session_key_from_base_path(base_path, stage)
    cache_key = (checkpoint_dir, composed_key)
    if cache_key not in _OFFSETS_CACHE:
        offsets_path = os.path.join(checkpoint_dir, f"{composed_key}_offsets.json")
        entries = None
        if os.path.exists(offsets_path):
            try:
                with open(offsets_path) as f:
                    entries = json.load(f)
            except Exception:
                entries = None
        _OFFSETS_CACHE[cache_key] = entries
    return _OFFSETS_CACHE[cache_key]


def _match_offset_entry(entries: list, edf_path: str) -> Optional[dict]:
    """Finds this row's file within a session's offset-entry list. Matched
    by normalized path first, falling back to basename, since the offsets
    JSON was written from whatever OS/path-separator convention the
    preprocessing run used, which may differ from the CSV's."""
    if not entries:
        return None
    target_norm = os.path.normpath(str(edf_path).replace("\\", "/"))
    target_base = os.path.basename(target_norm)
    for e in entries:
        e_norm = os.path.normpath(str(e.get("edf_path", "")).replace("\\", "/"))
        if e_norm == target_norm:
            return e
    for e in entries:
        e_norm = os.path.normpath(str(e.get("edf_path", "")).replace("\\", "/"))
        if os.path.basename(e_norm) == target_base:
            return e
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


def _normalize_dir_path(path: str) -> str:
    r"""Normalizes path strings across OS environments (e.g. C:\... -> /mnt/c/... under WSL)."""
    if not path:
        return path

    path_str = str(path).strip()
    if os.name == "posix" and len(path_str) >= 2 and path_str[1] == ":":
        drive_letter = path_str[0].lower()
        rest = path_str[2:].replace("\\", "/")
        return f"/mnt/{drive_letter}{rest}"
    return path_str


# ---------------------------------------------------------------------------
# Terminal dashboard helpers
# ---------------------------------------------------------------------------

def _bar(value: float, total: float, width: int = 28, fill: str = "█", empty: str = "░") -> str:
    """Render a Unicode progress bar scaled to *total*."""
    if total <= 0:
        frac = 0.0
    else:
        frac = max(0.0, min(1.0, value / total))
    filled = int(round(frac * width))
    return fill * filled + empty * (width - filled)


def _print_dataset_summary(
    counts: dict,
    label_map: dict,
    label_col_values: "pd.Series",
    split_col: Optional[str],
    split_series: Optional["pd.Series"],
    binary_preictal: bool,
) -> None:
    """Print a live graphical summary to the terminal using *rich*.
    Falls back to plain print() when rich is not installed."""

    # --- collect stats -------------------------------------------------------
    initial       = counts.get("initial", 0)
    after_channel = counts.get("after_channel", initial)
    after_status  = counts.get("after_status", after_channel)
    after_exlabels = counts.get("after_exclude_labels", after_status)
    after_ictal   = counts.get("after_ictal_filter", after_exlabels)
    after_prefix  = counts.get("after_label_prefix", after_ictal)
    after_conf    = counts.get("after_confidence", after_prefix)
    after_ckpt    = counts.get("after_checkpoints", after_conf)
    final         = after_ckpt

    # Class counts
    class_counts: dict = {}
    for raw_val, mapped in label_map.items():
        mask = label_col_values == raw_val
        class_counts[mapped] = class_counts.get(mapped, 0) + int(mask.sum())

    majority = max(class_counts.values()) if class_counts else 1

    # Split counts
    split_counts: dict = {}
    if split_series is not None:
        for sp in split_series.unique():
            split_counts[sp] = int((split_series == sp).sum())

    # --- render with rich if available ---------------------------------------
    if _RICH_AVAILABLE and _console is not None:
        _console.rule("[bold cyan]EEGWindowDataset — Build Summary[/bold cyan]")

        # 1. Filter funnel
        funnel = Table(title="Filter Funnel", box=box.SIMPLE_HEAD, show_lines=False)
        funnel.add_column("Stage",   style="dim",    no_wrap=True)
        funnel.add_column("Rows",    justify="right")
        funnel.add_column("Dropped", justify="right", style="red")
        funnel.add_column("Bar",     no_wrap=True)

        stages = [
            ("Initial CSV rows",                 initial),
            ("After channel filter",              after_channel),
            ("After status filter",               after_status),
            ("After exact-label exclusion",       after_exlabels),
            ("After ictal-validity filter",       after_ictal),
            ("After label-prefix filter",         after_prefix),
            ("After confidence filter",           after_conf),
            ("After checkpoint check",            after_ckpt),
        ]
        prev = initial
        for label, count in stages:
            dropped = prev - count
            bar = _bar(count, initial, width=24)
            drop_str = f"-{dropped}" if dropped else "—"
            funnel.add_row(label, str(count), drop_str, f"[green]{bar}[/green]")
            prev = count
        _console.print(funnel)

        # 2. Class distribution
        cls_table = Table(title="Class Distribution", box=box.SIMPLE_HEAD, show_lines=False)
        cls_table.add_column("Class",   style="magenta", no_wrap=True)
        cls_table.add_column("Count",   justify="right")
        cls_table.add_column("Pct",     justify="right")
        cls_table.add_column("Bar",     no_wrap=True)

        label_names = {v: k for k, v in label_map.items()} if not binary_preictal else {0: "non-preictal (0)", 1: "preictal (1)"}
        for cls_idx in sorted(class_counts.keys()):
            cnt  = class_counts[cls_idx]
            pct  = 100.0 * cnt / final if final else 0.0
            bar  = _bar(cnt, majority, width=24)
            name = str(label_names.get(cls_idx, cls_idx))
            cls_table.add_row(f"{cls_idx}  {name}", str(cnt), f"{pct:.1f}%", f"[cyan]{bar}[/cyan]")
        _console.print(cls_table)

        # 3. Split breakdown
        if split_counts:
            sp_table = Table(title="Split Breakdown", box=box.SIMPLE_HEAD, show_lines=False)
            sp_table.add_column("Split",  style="yellow", no_wrap=True)
            sp_table.add_column("Count",  justify="right")
            sp_table.add_column("Pct",    justify="right")
            sp_table.add_column("Bar",    no_wrap=True)
            for sp, cnt in sorted(split_counts.items()):
                pct = 100.0 * cnt / final if final else 0.0
                bar = _bar(cnt, final, width=24)
                sp_table.add_row(str(sp), str(cnt), f"{pct:.1f}%", f"[yellow]{bar}[/yellow]")
            _console.print(sp_table)

        _console.rule(f"[bold green]✓ Dataset ready — {final:,} windows — {len(class_counts)} class(es)[/bold green]")

    else:
        # ---- plain-text fallback -------------------------------------------
        W = 50
        print("\n" + "═" * W)
        print(" EEGWindowDataset — Build Summary ".center(W, "═"))
        print("═" * W)
        print("  Filter Funnel:")
        prev = initial
        for label, count in [
            ("Initial CSV rows",                 initial),
            ("After channel filter",              after_channel),
            ("After status filter",               after_status),
            ("After exact-label exclusion",       after_exlabels),
            ("After ictal-validity filter",       after_ictal),
            ("After label-prefix filter",         after_prefix),
            ("After confidence filter",           after_conf),
            ("After checkpoint check",            after_ckpt),
        ]:
            dropped = prev - count
            bar = _bar(count, initial, width=20)
            drop_str = f" (-{dropped})" if dropped else ""
            print(f"    {label:<28} {count:>6}{drop_str}  {bar}")
            prev = count
        print("  Class distribution:")
        label_names = {v: k for k, v in label_map.items()} if not binary_preictal else {0: "non-preictal", 1: "preictal"}
        for cls_idx in sorted(class_counts.keys()):
            cnt  = class_counts[cls_idx]
            pct  = 100.0 * cnt / final if final else 0.0
            bar  = _bar(cnt, majority, width=20)
            name = str(label_names.get(cls_idx, cls_idx))
            print(f"    [{cls_idx}] {name:<20} {cnt:>6}  {pct:5.1f}%  {bar}")
        if split_counts:
            print("  Split breakdown:")
            for sp, cnt in sorted(split_counts.items()):
                pct = 100.0 * cnt / final if final else 0.0
                bar = _bar(cnt, final, width=20)
                print(f"    {sp:<10} {cnt:>6}  {pct:5.1f}%  {bar}")
        print("═" * W + "\n")


def _is_background_label(raw_val) -> bool:
    """Return True if raw_val represents a background / non-seizure window."""
    s = str(raw_val).strip().lower()
    if s in ("0", "0.0", "-1", "none", "nan", "null", "") or s.startswith("b") or s.startswith("0"):
        return True
    return False


# Labels that are NOT ictal (used to identify pure ictal rows).
# Background (b*), preictal (p*), postictal (q*), continuing (c*), exclusion (x*).
_NON_ICTAL_PREFIXES: Tuple[str, ...] = ("b", "p", "q", "c", "x")


def _build_valid_ictal_mask(
    df: "pd.DataFrame",
    label_col: str,
    start_col: str,
) -> "pd.Series":
    """Return a boolean Series (same index as *df*) that is **False** for any
    ictal row whose immediately preceding annotation (within the same session,
    sorted by start_time) is **not** a preictal label (``p*``).

    Non-ictal rows (background, preictal, postictal, continuing, exclusion)
    are always True — they are handled by separate filters.

    The check scans ALL preceding labels including ``c*``/``q*`` that will be
    dropped later, so the pattern ``c{type} → {type}`` is caught correctly
    before those rows disappear.

    Rules enforced:
    * Keep ictal if and only if the immediately preceding event is ``p*``.
    * Ictals with no preceding event (first event in session) are dropped.
    """
    keep = pd.Series(True, index=df.index)

    for _, grp in df.groupby("session_key", sort=False):
        grp_sorted = grp.sort_values(start_col)
        lbls = grp_sorted[label_col].astype(str).tolist()
        orig_idx = grp_sorted.index.tolist()

        prev_lbl: Optional[str] = None
        for idx, lbl in zip(orig_idx, lbls):
            is_ictal = not any(lbl.startswith(pf) for pf in _NON_ICTAL_PREFIXES)
            if is_ictal:
                # Only valid if immediately preceded by a preictal (p*) window
                if prev_lbl is None or not prev_lbl.startswith("p"):
                    keep[idx] = False
            prev_lbl = lbl  # track ALL labels so c*→ictal is detected

    return keep


class EEGWindowDataset(Dataset):
    """One example = one labeled time window from the master CSV.

    See module docstring for the full assumption list.
    """

    def __init__(
        self,
        master_csv: str,
        checkpoint_dir: str,
        stage: str = "raw",
        # NOTE: must match Tuh-Preprocess's raw_eeg_extraction.TARGET_SFREQ
        # (256 Hz as of this pipeline) -- checkpoints are resampled to that
        # rate regardless of the source .edf's native rate, so this is NOT
        # a per-dataset tuning knob, it's a fixed fact about the checkpoint
        # files. If your pipeline's TARGET_SFREQ ever changes, override
        # this explicitly (e.g. via --sampling-rate in train.py) rather
        # than relying on this default.
        sampling_rate: float = 256.0,
        window_samples: int = 1000,
        # Fixed-length window used for the ground-truth "horizon" signal
        # (see `has_horizon`/`horizon_window` below) -- defaults to
        # `window_samples` if not given.
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
        # status 0 = collapsed/invalid window (start_cutoff pushed the
        # window before recording start, or insufficient baseline -- see
        # Tuh-Preprocess's preictal_segment.py docstrings), NOT
        # "interictal data". status 2 = collapsed consecutive/postictal
        # window. Both are degenerate (near-zero-length, mostly
        # zero-padded after slicing) and should be excluded by default --
        # pass an empty set to keep them if you specifically want to
        # inspect/debug them.
        exclude_status: Optional[set] = {0, 2},
        exclude_prefix: Tuple[str, ...] = ("x", "q", "c"),
        # Exact-label exclusions (distinct from exclude_prefix, which
        # matches by prefix). Default excludes "bckg": per project
        # findings this tag is artifact-flagged, not reliable clean
        # background, so it must not be used as the negative/background
        # class. True background should come from explicitly-derived "bg"
        # rows (see Tuh-Preprocess's preictal_segment.add_background_tags),
        # which are generated from stretches of the recording with NO
        # reliable label at all, not from the "bckg" tag.
        exclude_labels: Tuple[str, ...] = ("bckg",),
        exclude_ictal_without_preictal: bool = True,
        min_confidence: Optional[float] = None,
        binary_preictal: bool = False,
        session_key_fn: Callable[[str], str] = _default_session_key_fn,
        patient_id_fn: Callable[[str], str] = _default_patient_id_fn,
        cache_capacity: int = 4,
        skip_missing_checkpoints: bool = True,
        return_dict: bool = True,
        timing_norm: float = 300.0,  # seconds; onset_offset target is divided by this so its
                                      # loss scale is O(1) instead of O(hundreds of seconds)
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

        # 2. Status Filtering (Exclude collapsed status 0 and 2 from TUH_preprocess)
        if exclude_status and status_col in self.df.columns:
            self.df = self.df[~self.df[status_col].isin(exclude_status)].reset_index(drop=True)
        counts["after_status"] = len(self.df)

        # 2b. Exact-label exclusions (e.g. "bckg" -- artifact-flagged, not
        #     reliable background; see exclude_labels docstring above).
        if exclude_labels and label_col in self.df.columns:
            self.df = self.df[~self.df[label_col].astype(str).isin(exclude_labels)].reset_index(drop=True)
        counts["after_exclude_labels"] = len(self.df)

        # 3. Ictal-validity filter: drop ictals not preceded by p* (preictal).
        #    MUST run before exclude_prefix removes c*/q* rows so that the
        #    pattern c{type}→{type} is still visible in the label sequence.
        if exclude_ictal_without_preictal and label_col in self.df.columns:
            keep_mask = _build_valid_ictal_mask(self.df, label_col, start_col)
            self.df = self.df[keep_mask].reset_index(drop=True)
        counts["after_ictal_filter"] = len(self.df)

        # 4. Label Exclusion Prefix Filtering
        #    Default excludes x* (exclusion intervals), q* (postictal), c* (continuing).
        if exclude_prefix and label_col in self.df.columns:
            pattern = "^(?:" + "|".join(exclude_prefix) + ")"
            self.df = self.df[~self.df[label_col].astype(str).str.contains(pattern, regex=True)].reset_index(drop=True)
        counts["after_label_prefix"] = len(self.df)

        # 5. Confidence Filtering
        if min_confidence is not None and confidence_col in self.df.columns:
            self.df = self.df[self.df[confidence_col] >= min_confidence].reset_index(drop=True)
        counts["after_confidence"] = len(self.df)

        # 6. Filter out sessions whose checkpoint files do not exist on disk
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

        # Keep original raw labels in a separate column before mapping
        self.df["_raw_label"] = self.df[label_col].astype(str)

        # Label Mapping
        if binary_preictal:
            # Map preictal (p*) -> 1, all others -> 0
            self.df[label_col] = self.df["_raw_label"].apply(lambda l: 1 if l.startswith("p") else 0)
            label_map = {0: 0, 1: 1}
        elif label_map is None:
            uniq = sorted(self.df[label_col].dropna().unique().tolist())
            label_map = {lab: i for i, lab in enumerate(uniq)}

        self.label_map = label_map
        self.num_classes = len(set(label_map.values()))

        self.status_col = status_col
        self.return_dict = return_dict
        self.timing_norm = timing_norm
        self._cache = SessionCache(checkpoint_dir, stage, capacity=cache_capacity)
        self.n_resized = 0  # counts windows that needed crop/pad, for sanity checks
        self._warned_missing_offset: set = set()

        # Per-session sorted list of ictal (seizure) onset times, from the final
        # filtered rows. Used in __getitem__ to compute the real "WHEN" target for
        # preictal windows: how long from this window's end until the seizure
        # actually starts -- not just how long the window itself is.
        self._session_ictal_onsets: dict = {}
        is_ictal = ~self.df["_raw_label"].astype(str).apply(
            lambda l: any(l.startswith(pf) for pf in _NON_ICTAL_PREFIXES)
        )
        for session_key, grp in self.df[is_ictal].groupby("session_key"):
            self._session_ictal_onsets[session_key] = sorted(grp[start_col].astype(float).tolist())

        # --- Live terminal dashboard -----------------------------------------
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
        """Return {class_idx: count} for the given row indices (or all rows).

        Useful for computing inverse-frequency loss weights from the training
        subset without re-reading the CSV.
        """
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
        """Inverse-frequency weight tensor for CrossEntropyLoss.

        Returns a 1-D float tensor of length ``num_classes`` where
        ``weights[c] = total / (num_classes * count[c])``, which is the
        standard sklearn-style balanced weight.  Missing classes get weight 1.
        """
        counts = self.get_class_counts(indices)
        total = sum(counts.values())
        num_cls = self.num_classes
        weights = []
        for c in range(num_cls):
            cnt = counts.get(c, 1)  # avoid div-by-zero for unseen classes
            raw_w = (total / (num_cls * max(1, cnt))) ** 0.5  # smooth square-root scaling
            w = min(raw_w, 3.0)  # cap max weight to 3.0 to prevent minority-class collapse
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
        """Scalar ``pos_weight`` for BCEWithLogitsLoss on the occurrence head.

        ``pos_weight = num_negatives / num_positives`` where positives are
        preictal/ictal seizure windows and negatives are background windows.
        """
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
        # Cap pos_weight to a reasonable range (e.g. max 20.0) to prevent gradient explosion
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

        # Defensive clamp to the array's actual bounds -- guards against a
        # bad/stale offset or an annotation that runs past the end of a
        # (possibly truncated) checkpoint array, rather than raising or
        # silently reading garbage via negative-index wraparound.
        n_total = arr.shape[1]
        start_idx = max(0, min(start_idx, n_total))
        stop_idx = max(start_idx, min(stop_idx, n_total))

        window = arr[:, start_idx:stop_idx]
        n = window.shape[1]

        if n == target_samples:
            return window

        self.n_resized += 1
        if n > target_samples:
            # center-crop
            offset = (n - target_samples) // 2
            return window[:, offset: offset + target_samples]

        # zero-pad (centered)
        pad_total = target_samples - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(window, ((0, 0), (pad_left, pad_right)), mode="constant")

    def _get_offset_samples(self, session_key: str, edf_path: str) -> int:
        """Sample offset (in this dataset's `sampling_rate` units, which
        must match the checkpoint's actual rate -- see the constructor's
        `sampling_rate` docstring) to add to a row's start_time/stop_time
        before indexing into the session's checkpoint array. See
        `_get_session_file_offsets` for why this is needed: a session's
        checkpoint is a concatenation of potentially several .edf files,
        but each row's start_time/stop_time are local to its own file.
        Returns 0 (no correction) if no offsets file is found for this
        session, e.g. single-file sessions or legacy checkpoints."""
        entries = _get_session_file_offsets(self.checkpoint_dir, self.stage, session_key)
        if not entries:
            return 0
        match = _match_offset_entry(entries, edf_path)
        if match is None:
            if session_key not in self._warned_missing_offset:
                self._warned_missing_offset.add(session_key)
                print(
                    f"Warning: session '{session_key}' has an offsets file but no entry "
                    f"matched edf_path={edf_path!r} -- using offset 0 for this row (and any "
                    f"others in this session that also fail to match). Check path formatting."
                )
            return 0
        return int(match.get("start_sample", 0))

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        session_key = row["session_key"]

        arr = self._cache.get(session_key)
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

        # Multi-task ground truth targets:
        # Occurrence (IF): 0 for non-seizure/background, 1 for preictal/ictal
        raw_str = str(raw_label)
        if getattr(self, "binary_preictal", False):
            has_seizure = 1.0 if label == 1 else 0.0
        else:
            has_seizure = 0.0 if _is_background_label(raw_label) else 1.0
        occurrence_t = torch.tensor(has_seizure, dtype=torch.float32)

        # Timing offset (WHEN): Relative onset time (in seconds) relative to the generated data window
        # For preictal windows (p*), relative onset is the distance/duration from preictal end to seizure start
        onset_time: Optional[float] = None
        has_horizon = False
        if raw_str.startswith("p"):
            onsets = self._session_ictal_onsets.get(session_key, [])
            pos = bisect.bisect_left(onsets, stop_time)
            if pos < len(onsets):
                onset_time = onsets[pos]
                relative_onset = max(0.0, onset_time - stop_time)
                has_horizon = onset_time > stop_time
            else:
                # Shouldn't normally happen (the ictal-validity filter requires every
                # kept ictal event be immediately preceded by a p* window), but fall
                # back to the window's own duration rather than crashing.
                relative_onset = max(0.0, float(stop_time - start_time))
        elif has_seizure > 0:
            relative_onset = 0.0
        else:
            relative_onset = 0.0

        relative_onset_clamped = min(relative_onset, self.timing_norm)
        onset_offset_t = torch.tensor(relative_onset_clamped / self.timing_norm, dtype=torch.float32)
        status_t = torch.tensor(status_val, dtype=torch.long)

        # Ground-truth "horizon" signal (the real EEG spanning from the end of
        # this preictal window to the actual seizure onset). Used to
        # supervise the model's horizon generator against the real
        # continuation rather than only next-token prediction within the
        # preictal window itself (see train.py's horizon_loss). Only
        # available for preictal rows with a resolvable onset -- everything
        # else gets a zeroed placeholder + has_horizon=0 so batches can
        # still collate, and the loss is masked to has_horizon==1 rows.
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
            "onset_offset": onset_offset_t,
            "status": status_t,
            "horizon_window": horizon_window_t,
            "has_horizon": has_horizon_t,
        }

        return window_t, targets



def _check_split_collisions(
    dataset: "EEGWindowDataset",
    subsets: dict,
    train_split_name: str,
    val_split_name: str,
) -> None:
    """Warn if any patient IDs appear in *both* the train and val subsets
    (data-leakage / patient-overlap collision)."""
    from torch.utils.data import Subset

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
        import warnings
        warnings.warn(f"SPLIT COLLISION: {msg}", UserWarning, stacklevel=4)


def split_by_column(dataset: "EEGWindowDataset", split_map: Optional[dict] = None):
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

    # Patient-level collision check (train vs val data-leakage guard)
    _check_split_collisions(dataset, subsets, train_split_name, val_split_name)

    persistent = num_workers > 0
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