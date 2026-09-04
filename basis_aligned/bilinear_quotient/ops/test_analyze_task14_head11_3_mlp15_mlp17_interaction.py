from __future__ import annotations

import copy

import pytest

import analyze_task14_head11_3_mlp15_mlp17_interaction as run


def _planted_rows_and_corners():
    rows, corners, _error = run._load()
    return rows, copy.deepcopy(corners)


def test_source_identity_and_zero_gpu_dryrun():
    dryrun = run.compile_dryrun()
    assert dryrun["row_count"] == 128
    assert dryrun["source_empty_max_abs_error"] == 0.0
    assert dryrun["model_loaded"] is False
    assert dryrun["gpu_accessed"] is False
    assert dryrun["maximum_new_execution_price"]["forward_calls"] == 0


def test_exactly_additive_pair_passes_additive_bar():
    rows, corners = _planted_rows_and_corners()
    for values in corners.values():
        values.update(empty=0.0, mlp15=-0.02, mlp17=-0.03, both=-0.05)
    score = run._score(rows, corners)
    assert score["interaction_rms"] == pytest.approx(0.0)
    assert score["additive"] is True
    assert score["nonlinear"] is False


def test_planted_pair_interaction_passes_nonlinear_bar():
    rows, corners = _planted_rows_and_corners()
    for row in rows:
        values = corners[str(row["row_id"])]
        interaction = -0.10 if row["transform_id"] in {"A1", "A2"} else 0.0
        values.update(empty=0.0, mlp15=-0.02, mlp17=-0.03, both=-0.05 + interaction)
    score = run._score(rows, corners)
    assert score["interaction_rms"] == pytest.approx(0.10)
    assert score["additive"] is False
    assert score["nonlinear"] is True


def test_control_failure_blocks_both_scientific_terminals():
    rows, corners = _planted_rows_and_corners()
    for row in rows:
        values = corners[str(row["row_id"])]
        control = 0.20 if row["transform_id"] in {"P", "C"} else 0.0
        values.update(empty=control, mlp15=control, mlp17=control, both=control)
    score = run._score(rows, corners)
    assert score["control_ok"] is False
    assert score["additive"] is False
    assert score["nonlinear"] is False
