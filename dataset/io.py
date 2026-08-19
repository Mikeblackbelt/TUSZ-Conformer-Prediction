"""
Checkpoint array loading, directory indexing, session key parsing, and SessionCache.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
import pyarrow.parquet as pq


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
    checkpoint. Direct PyArrow -> NumPy conversion (bypasses Pandas)."""
    table = pq.read_table(path)
    arr = np.vstack([col.to_numpy() for col in table.columns]).astype(np.float32, copy=False)
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
    (no extension). Strips the directory and the trailing
    "_<stage>" to recover the actual session key used for files that
    aren't stage-specific, like the per-file sample offsets JSON."""
    fname = os.path.basename(base_path)
    suffix = f"_{stage}"
    if fname.endswith(suffix):
        fname = fname[: -len(suffix)]
    return fname


_OFFSETS_CACHE: dict[Tuple[str, str], Optional[dict[str, dict]]] = {}


def _get_session_file_offsets(checkpoint_dir: str, stage: str, session_key: str) -> Optional[dict[str, dict]]:
    """Loads the `{composed_session_key}_offsets.json` file for a session,
    caching a fast O(1) path/basename lookup dict for row offset indexing."""
    base_path = _find_checkpoint_file(checkpoint_dir, session_key, stage)
    if base_path is None:
        return None
    composed_key = _composed_session_key_from_base_path(base_path, stage)
    cache_key = (checkpoint_dir, composed_key)
    if cache_key not in _OFFSETS_CACHE:
        offsets_path = os.path.join(checkpoint_dir, f"{composed_key}_offsets.json")
        offsets_map = None
        if os.path.exists(offsets_path):
            try:
                with open(offsets_path) as f:
                    entries = json.load(f)
                if isinstance(entries, list):
                    offsets_map = {}
                    for e in entries:
                        p_norm = os.path.normpath(str(e.get("edf_path", "")).replace("\\", "/"))
                        p_base = os.path.basename(p_norm)
                        offsets_map[p_norm] = e
                        if p_base not in offsets_map:
                            offsets_map[p_base] = e
            except Exception:
                offsets_map = None
        _OFFSETS_CACHE[cache_key] = offsets_map
    return _OFFSETS_CACHE[cache_key]


def _match_offset_entry(offsets_map: Optional[dict[str, dict]], edf_path: str) -> Optional[dict]:
    """Finds this row's file within a session's offset dictionary in O(1) time."""
    if not offsets_map:
        return None
    target_norm = os.path.normpath(str(edf_path).replace("\\", "/"))
    if target_norm in offsets_map:
        return offsets_map[target_norm]
    target_base = os.path.basename(target_norm)
    return offsets_map.get(target_base)


class SessionCache:
    """Small LRU cache so consecutive windows from the same session don't
    re-read the checkpoint off disk every time."""

    def __init__(self, checkpoint_dir: str, stage: str, capacity: int = 4):
        self.checkpoint_dir = checkpoint_dir
        self.stage = stage
        self.capacity = capacity
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

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
