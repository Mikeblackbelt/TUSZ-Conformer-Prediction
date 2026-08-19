"""
Label category constants and classification functions for EEG window datasets.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

# Label category constants
LABEL_CATEGORY_BACKGROUND = "background"
LABEL_CATEGORY_PREICTAL = "preictal"
LABEL_CATEGORY_ICTAL = "ictal"
LABEL_CATEGORY_EXCLUDED = "excluded"

ALLOWED_LABEL_CATEGORIES = {
    LABEL_CATEGORY_BACKGROUND,
    LABEL_CATEGORY_PREICTAL,
    LABEL_CATEGORY_ICTAL,
    LABEL_CATEGORY_EXCLUDED,
}

# Known base seizure types from TUH EEG dataset / preictal preprocessing engine
KNOWN_SEIZURE_TYPES = {
    "fnsz", "gnsz", "cpsz", "spsz", "tnsz", "tcsz", "absz", "mysz", "seiz", "nesz"
}

# Background raw labels emitted by preprocessing or raw CSVs
BACKGROUND_LABELS = {
    "bg", "bckg", "background", "0", "0.0", "-1", "none", "nan", "null", ""
}

# Artifact / exclusion tags
EXCLUSION_TAGS = {
    "artf", "eyem", "chew", "cero", "eloh", "elec", "gspd", "pled"
}


def classify_label(raw_val: Any) -> str:
    """Classify raw annotation label into one of four explicit categories:
    {'background', 'preictal', 'ictal', 'excluded'}.

    Raises ValueError if raw_val is unrecognized (fail-loud).
    """
    if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
        return LABEL_CATEGORY_BACKGROUND

    s = str(raw_val).strip().lower()

    # 1. Exact background match
    if s in BACKGROUND_LABELS or s.startswith("0"):
        return LABEL_CATEGORY_BACKGROUND

    # 2. Exact ictal match
    if s in KNOWN_SEIZURE_TYPES:
        return LABEL_CATEGORY_ICTAL

    # 3. Preictal variants: starts with 'p' followed by known seizure type or is 'p' / 'preictal'
    if s in ("p", "preictal"):
        return LABEL_CATEGORY_PREICTAL
    if s.startswith("p") or s.startswith("p_"):
        suffix = s[2:] if s.startswith("p_") else s[1:]
        if suffix in KNOWN_SEIZURE_TYPES or not suffix:
            return LABEL_CATEGORY_PREICTAL

    # 4. Excluded / Postictal / Continuing / Artifact variants
    if s in ("x", "q", "c", "postictal", "continuing", "exclusion") or s in EXCLUSION_TAGS:
        return LABEL_CATEGORY_EXCLUDED
    if s.startswith("q") or s.startswith("q_") or s.startswith("c") or s.startswith("c_") or s.startswith("x") or s.startswith("x_"):
        suffix = s[2:] if (s.startswith("q_") or s.startswith("c_") or s.startswith("x_")) else s[1:]
        if suffix in KNOWN_SEIZURE_TYPES or not suffix:
            return LABEL_CATEGORY_EXCLUDED

    # 5. Fail-loud on anything unrecognized
    raise ValueError(
        f"Unrecognized dataset label '{raw_val}' (normalized: '{s}'). "
        f"Label must be explicitly allow-listed into one of {{background, preictal, ictal, excluded}}."
    )


def _is_background_label(raw_val: Any) -> bool:
    """Return True if raw_val represents a background / non-seizure window."""
    return classify_label(raw_val) == LABEL_CATEGORY_BACKGROUND


def _build_valid_ictal_mask(
    df: pd.DataFrame,
    label_col: str,
    start_col: str,
) -> pd.Series:
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
            is_ictal = (classify_label(lbl) == LABEL_CATEGORY_ICTAL)
            if is_ictal:
                # Only valid if immediately preceded by a preictal (p*) window
                if prev_lbl is None or classify_label(prev_lbl) != LABEL_CATEGORY_PREICTAL:
                    keep[idx] = False
            prev_lbl = lbl  # track ALL labels so c*→ictal is detected

    return keep
