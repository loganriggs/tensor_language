"""CPU tests for the immutable rung-522 optimizer-health summary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


OPS = Path(__file__).parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))


def _load(name: str):
    path = OPS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GUARD = _load("attention8_selective_shared_projector_rung522_state_guard")
ANALYSIS = _load("attention8_selective_shared_projector_rung522_health_analysis")


def _record(spec, *, healthy: bool, failures: tuple[str, ...], shift: float):
    return SimpleNamespace(
        spec=spec,
        healthy=healthy,
        health_failures=failures,
        health_record_payload={
            "initial_validation_objective": 2.0,
            "final_validation_objective": 2.0 + shift,
            "initial_window_mean": 3.0,
            "final_window_mean": 3.0 + shift,
            "projector_distance_from_initialization": 0.5,
            "orthonormality_error": 1e-7,
        },
    )


def test_summary_separates_families_targets_and_failure_mechanisms():
    records = {}
    for index, spec in enumerate(GUARD.EXPECTED_FRAME_SPECS.values()):
        unhealthy = index % 3 == 0
        failures = (
            ("validation_not_better_than_initialization",) if unhealthy else ()
        )
        shift = 0.25 if unhealthy else -0.25
        records[spec.frame_id] = _record(
            spec, healthy=not unhealthy, failures=failures, shift=shift
        )
    loaded = SimpleNamespace(
        records=records,
        file_sha256="a" * 64,
        content_sha256="b" * 64,
    )
    result = ANALYSIS.summarize_archive_health(loaded)
    assert result["overall"]["frame_count"] == 103
    assert result["overall"]["healthy_count"] == 68
    assert result["overall"]["failure_reason_counts"] == {
        "validation_not_better_than_initialization": 35
    }
    assert result["by_family"]["real_leave_one_out"]["frame_count"] == 15
    assert result["by_family"]["label_null"]["frame_count"] == 48
    assert result["by_family_and_target"][
        "target_oracle:r.2.0.1"
    ]["frame_count"] == 5
    assert result["scientific_scope"].startswith("optimizer-health diagnosis only")


def test_lower_objective_change_is_reported_as_improvement():
    spec = next(iter(GUARD.EXPECTED_FRAME_SPECS.values()))
    summary = ANALYSIS._summarize_group([
        _record(spec, healthy=True, failures=(), shift=-0.5),
        _record(spec, healthy=True, failures=(), shift=-0.25),
    ])
    assert summary["median_validation_change_final_minus_initial"] == pytest.approx(-0.375)
    assert summary["median_training_window_change_final_minus_initial"] == pytest.approx(-0.375)
