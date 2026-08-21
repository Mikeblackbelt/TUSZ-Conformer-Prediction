"""
Unit tests for TUSZ label classification and seizure type extraction.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.label_utils import (
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


def test_tusz_official_10_seizure_types_supported():
    """Verify all 10 official TUSZ seizure types are recognized as ICTAL."""
    official_10 = [
        "fnsz",  # Focal Non-Specific
        "gnsz",  # Generalized Non-Specific
        "spsz",  # Simple Partial
        "cpsz",  # Complex Partial
        "absz",  # Absence
        "tnsz",  # Tonic
        "cnsz",  # Clonic
        "tcsz",  # Tonic-Clonic
        "atsz",  # Atonic
        "mysz",  # Myoclonic
    ]
    for stype in official_10:
        assert stype in KNOWN_SEIZURE_TYPES, f"{stype} missing from KNOWN_SEIZURE_TYPES"
        assert classify_label(stype) == LABEL_CATEGORY_ICTAL, f"{stype} failed classify_label"
        assert extract_seizure_type(stype) == stype, f"{stype} failed extract_seizure_type"


def test_preictal_and_exclusion_prefixes_for_all_tusz_types():
    """Verify preictal (p*) and exclusion (x*, q*, c*) prefixes work for all TUSZ types."""
    official_10 = ["fnsz", "gnsz", "spsz", "cpsz", "absz", "tnsz", "cnsz", "tcsz", "atsz", "mysz"]
    for stype in official_10:
        pre_label = f"p{stype}"
        assert classify_label(pre_label) == LABEL_CATEGORY_PREICTAL
        assert extract_seizure_type(pre_label) == stype

        ex_label = f"x{stype}"
        assert classify_label(ex_label) == LABEL_CATEGORY_EXCLUDED
        assert extract_seizure_type(ex_label) == LABEL_CATEGORY_BACKGROUND


def test_background_and_artifacts():
    """Verify background labels and artifact tags are properly classified."""
    for bg in ["bckg", "bg", "background", "0", 0]:
        assert classify_label(bg) == LABEL_CATEGORY_BACKGROUND
        assert extract_seizure_type(bg) == LABEL_CATEGORY_BACKGROUND

    for artf in ["artf", "eyem", "eybl", "chew", "cero", "eloh", "elec", "gspd", "pled", "spsw", "gped"]:
        assert classify_label(artf) == LABEL_CATEGORY_EXCLUDED
        assert extract_seizure_type(artf) == LABEL_CATEGORY_BACKGROUND


def test_reexport_via_dataset_loader():
    """Verify symbols are accessible via dataset_loader wrapper."""
    import dataset_loader
    assert hasattr(dataset_loader, "KNOWN_SEIZURE_TYPES")
    assert hasattr(dataset_loader, "SEIZURE_TYPE_CLASSES")
    assert hasattr(dataset_loader, "extract_seizure_type")
    assert hasattr(dataset_loader, "classify_label")
    assert "cnsz" in dataset_loader.KNOWN_SEIZURE_TYPES
    assert "atsz" in dataset_loader.KNOWN_SEIZURE_TYPES


if __name__ == "__main__":
    test_tusz_official_10_seizure_types_supported()
    test_preictal_and_exclusion_prefixes_for_all_tusz_types()
    test_background_and_artifacts()
    test_reexport_via_dataset_loader()
    print("All TUSZ label tests passed successfully!")

