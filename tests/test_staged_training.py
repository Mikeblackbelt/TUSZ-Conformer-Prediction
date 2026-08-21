"""
Unit tests for staged layer-by-layer training.
"""

import sys
import os
import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.simplified_conformer import SimplifiedEEGConformer
from training.trainer import set_model_stage
from training.cli import get_training_stage


def test_get_training_stage():
    """Verify get_training_stage partitions epochs correctly."""
    args = SimpleNamespace(stage1_epochs=4, stage2_epochs=4, no_staged_training=False)
    assert get_training_stage(1, args) == 1
    assert get_training_stage(4, args) == 1
    assert get_training_stage(5, args) == 2
    assert get_training_stage(8, args) == 2
    assert get_training_stage(9, args) == 3
    assert get_training_stage(20, args) == 3

    # Test disabled staged training
    args_no_stage = SimpleNamespace(stage1_epochs=4, stage2_epochs=4, no_staged_training=True)
    assert get_training_stage(1, args_no_stage) == 3


def test_set_model_stage_parameter_freezing():
    """Verify parameter trainable status (requires_grad) per stage."""
    model = SimplifiedEEGConformer(n_channels=19, embed_dim=64, depth=2, num_classes=10)

    # Stage 1: Only backbone + pred_head active
    set_model_stage(model, stage=1)
    for name, param in model.named_parameters():
        if any(h in name for h in ("occurrence_timing_head", "preictal_head", "preictal_type_layer", "preictal_type_head")):
            assert not param.requires_grad, f"{name} should be frozen in Stage 1"
        else:
            assert param.requires_grad, f"{name} should be trainable in Stage 1"

    # Stage 2: Backbone + occurrence + timing active; type head frozen
    set_model_stage(model, stage=2)
    for name, param in model.named_parameters():
        if any(h in name for h in ("preictal_type_layer", "preictal_type_head")):
            assert not param.requires_grad, f"{name} should be frozen in Stage 2"
        elif any(h in name for h in ("occurrence_timing_head", "preictal_head")):
            assert param.requires_grad, f"{name} should be trainable in Stage 2"

    # Stage 3: All active
    set_model_stage(model, stage=3)
    for name, param in model.named_parameters():
        assert param.requires_grad, f"{name} should be trainable in Stage 3"


if __name__ == "__main__":
    test_get_training_stage()
    test_set_model_stage_parameter_freezing()
    print("All staged training tests passed successfully!")
