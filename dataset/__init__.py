"""
EEG Window Dataset and Data Loading Package.
"""

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
    SEIZURE_TYPE_CLASSES,
    classify_label,
    extract_seizure_type,
)
from dataset.reporting import _bar, _print_dataset_summary
from dataset.samplers import (
    SessionBatchSampler,
    _check_split_collisions,
    build_dataloaders,
    split_by_column,
)

__all__ = [
    "EEGWindowDataset",
    "SessionCache",
    "SessionBatchSampler",
    "build_dataloaders",
    "split_by_column",
    "classify_label",
    "extract_seizure_type",
    "SEIZURE_TYPE_CLASSES",
    "LABEL_CATEGORY_BACKGROUND",
    "LABEL_CATEGORY_PREICTAL",
    "LABEL_CATEGORY_ICTAL",
    "LABEL_CATEGORY_EXCLUDED",
    "ALLOWED_LABEL_CATEGORIES",
    "KNOWN_SEIZURE_TYPES",
    "BACKGROUND_LABELS",
    "EXCLUSION_TAGS",
]
