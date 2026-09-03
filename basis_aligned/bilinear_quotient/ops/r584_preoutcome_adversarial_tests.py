"""Adversarial pre-outcome contract tests for the R584 implementation.

This file is deliberately not named ``test_*.py``: it is a review artifact that
can be run explicitly while the owner repairs the model-facing runner.  The
tests encode R582 evidence and fail-closed requirements without loading a model.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


RUNNER = Path(__file__).with_name("numbered_list_cached_value_downstream_use_rung584.py")
SPEC = importlib.util.spec_from_file_location("r584_review_target", RUNNER)
r584 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(r584)


def _synthetic_raw(rows, *, target_damage: float = 1.0, copy_damage: float = 0.05):
    raw = []
    for row in rows:
        if row["split"] != "FIT":
            continue
        common = {
            "row_id": row["row_id"],
            "group_id": row["group_id"],
            "split": "FIT",
            "representation": row["representation"],
            "source_level": row["source_level"],
            "condition": row["condition"],
            "action": row["action"],
            "intervention_vector_norm": 1.0,
            "full_vocabulary_logit_rms": 1.0 if row["action"] == "successor" else 0.1,
            "null_donor_row_id": None,
        }
        if row["condition"] == "step_two":
            common.update({
                "native": {"arithmetic_minus_structural": 1.0},
                "intervened": {"arithmetic_minus_structural": 0.5},
                "preference_sign_preserved": True,
            })
        else:
            damage = target_damage if row["action"] == "successor" else copy_damage
            common.update({
                "native": {"margin": 2.0, "ce": 1.0, "answer_best": True},
                "intervened": {
                    "margin": 2.0 - damage,
                    "ce": 2.0 if row["action"] == "successor" else 1.0,
                    "answer_best": True,
                },
                "margin_damage": damage,
                "ce_increase": 1.0 if row["action"] == "successor" else 0.0,
            })
        raw.append(common)
    return raw


def _nonfinite_paths(value, path="root"):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(_nonfinite_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_nonfinite_paths(item, f"{path}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        found.append((path, value))
    return found


def test_missing_complete_semantic_group_is_an_integrity_error():
    """Dropping a whole group must not quietly reduce every cell from n=16 to n=15."""
    rows = r584.load_authority()
    raw = _synthetic_raw(rows)
    missing_group = raw[0]["group_id"]
    incomplete = [row for row in raw if row["group_id"] != missing_group]
    with pytest.raises(RuntimeError, match="membership|missing|count"):
        r584.score_candidate(incomplete, cell_prefix="review:missing_group")


def test_scientific_null_report_contains_only_finite_json_numbers():
    """A negative action gap is a normal null and must not emit -Infinity."""
    rows = r584.load_authority()
    report = r584.score_candidate(
        _synthetic_raw(rows, target_damage=-1.0),
        cell_prefix="review:negative_action_gap",
    )
    assert _nonfinite_paths(report) == []


def test_saved_row_schema_contains_authoritative_positions_and_source_value():
    """Every intervention row must be independently joinable and position-auditable."""
    row = next(row for row in r584.load_authority() if row["split"] == "FIT")
    before = torch.zeros(50_304)
    after = before.clone()
    record = r584.intervention_record(row, before, after, 1.0)
    required = {"token_ids", "query_position", "source_position", "source_value"}
    assert required <= set(record), f"missing row-evidence fields: {sorted(required - set(record))}"


def test_capture_schema_and_provenance_are_audit_complete():
    """R582 requires per-row/site exactness, deletion RMS, and code/test hashes."""
    capture_source = inspect.getsource(r584.capture_split)
    main_source = inspect.getsource(r584.main)
    for required in (
        "native_replay_relative_squared_error_by_row",
        "bilinear_response_relative_squared_error_by_site",
        "source_deleted_logit_difference_squared_sum",
        "source_deleted_logit_vocabulary_count",
    ):
        assert required in capture_source, f"capture_split does not save {required}"
    assert '"implementation_sha256"' in main_source
    assert '"test_sha256"' in main_source


def test_reciprocal_product_rescaling_preserves_cross_and_self_terms():
    """The second R582 gauge claim is independent of the existing L/R-swap test."""
    generator = torch.Generator().manual_seed(58_404)
    left = nn.Linear(7, 13, bias=False)
    right = nn.Linear(7, 13, bias=False)
    down = nn.Linear(13, 5, bias=False)
    with torch.no_grad():
        left.weight.copy_(torch.randn(left.weight.shape, generator=generator))
        right.weight.copy_(torch.randn(right.weight.shape, generator=generator))
        down.weight.copy_(torch.randn(down.weight.shape, generator=generator))
    mlp = type("MLP", (), {"Left": left, "Right": right, "Down": down})()
    x0 = torch.randn(11, 7, generator=generator)
    x1 = torch.randn(11, 7, generator=generator)
    reference = r584.torch_bilinear_response(mlp, x0, x1)

    scales = torch.exp(torch.linspace(-2.0, 2.0, 13))
    left_scaled = nn.Linear(7, 13, bias=False)
    right_scaled = nn.Linear(7, 13, bias=False)
    down_scaled = nn.Linear(13, 5, bias=False)
    with torch.no_grad():
        left_scaled.weight.copy_(left.weight * scales[:, None])
        right_scaled.weight.copy_(right.weight / scales[:, None])
        down_scaled.weight.copy_(down.weight)
    scaled = type(
        "MLP", (), {"Left": left_scaled, "Right": right_scaled, "Down": down_scaled}
    )()
    observed = r584.torch_bilinear_response(scaled, x0, x1)
    for component in ("background_cross", "contrast_self"):
        difference = (reference[component].float() - observed[component].float()).detach()
        relative_squared = float(difference.square().sum()) / max(
            float(reference[component].float().detach().square().sum()), 1e-30
        )
        assert relative_squared <= r584.EXACT_BAR


def test_metadata_container_types_are_unambiguous():
    plan = r584.dryrun(r584.load_authority())
    assert isinstance(plan["selection_order"], list)
    assert isinstance(plan["opened_splits"], list)
    assert isinstance(plan["input_sha256"], dict)
    assert isinstance(plan["literal_executable_maximum_forwards"], int)
    assert all(isinstance(value, str) for value in plan["input_sha256"].values())
