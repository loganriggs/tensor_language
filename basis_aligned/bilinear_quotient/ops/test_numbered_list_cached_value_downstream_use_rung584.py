from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn


PATH = Path(__file__).with_name("numbered_list_cached_value_downstream_use_rung584.py")
SPEC = importlib.util.spec_from_file_location("r584", PATH)
r584 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(r584)


@pytest.fixture(scope="module")
def rows():
    return r584.load_authority()


def test_authority_and_fixed_selection_order(rows):
    assert len(rows) == 1440
    assert r584.SELECTION == tuple(
        (site, component) for site in (8, 10, 12, 14)
        for component in ("background_cross", "contrast_self", "joint_response"))
    assert all(row["model_outcome_opened"] is False for row in rows)


def test_executable_price_is_literal_and_below_r582_ceiling(rows):
    result = r584.price(rows)
    assert result == {
        "split_batches": {"FIT": 27, "SELECT": 14},
        "null_eligible_batches": {"FIT": 20, "SELECT": 10},
        "fit_maximum_forwards": 419,
        "conditional_select_maximum_forwards": 91,
        "literal_executable_maximum_forwards": 510,
        "r582_conservative_maximum_forwards": 530,
    }


def test_subprocess_dryrun_is_import_safe_outcome_closed_and_writes_receipt(rows):
    environment = dict(os.environ, BQLIB_DRYRUN="1", CUDA_VISIBLE_DEVICES="")
    completed = subprocess.run([sys.executable, str(PATH)], env=environment,
                               check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    assert result["literal_executable_maximum_forwards"] == 510
    assert result["model_loaded"] is False and result["model_forwards"] == 0
    assert result["opened_splits"] == []
    assert result["FINAL_TEST_or_OOD_opened"] is False
    assert json.loads(r584.DRYRUN_OUT.read_text()) == result


def test_torch_bilinear_response_matches_direct_difference_and_gauge():
    generator = torch.Generator().manual_seed(584)
    left = nn.Linear(7, 13, bias=False); right = nn.Linear(7, 13, bias=False)
    down = nn.Linear(13, 5, bias=False)
    for layer in (left, right, down):
        with torch.no_grad():
            layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator))
    mlp = SimpleNamespace(Left=left, Right=right, Down=down)
    x0 = torch.randn(11, 7, generator=generator)
    x1 = torch.randn(11, 7, generator=generator)
    response = r584.torch_bilinear_response(mlp, x0, x1)
    torch.testing.assert_close(response["joint_response"], response["direct_response"], atol=2e-5, rtol=2e-6)
    assert response["relative_squared_error"] < 1e-12
    swapped = SimpleNamespace(Left=right, Right=left, Down=down)
    other = r584.torch_bilinear_response(swapped, x0, x1)
    torch.testing.assert_close(response["background_cross"], other["background_cross"])
    torch.testing.assert_close(response["contrast_self"], other["contrast_self"])


def _synthetic_raw(rows, *, target_damage=1.0, copy_damage=.05):
    raw = []
    for row in rows:
        if row["split"] != "FIT":
            continue
        common = {
            "row_id": row["row_id"], "group_id": row["group_id"], "split": "FIT",
            "representation": row["representation"], "source_level": row["source_level"],
            "condition": row["condition"], "action": row["action"],
            "intervention_vector_norm": 1.0,
            "full_vocabulary_logit_rms": 1.0 if row["action"] == "successor" else .1,
            "null_donor_row_id": None,
        }
        if row["condition"] == "step_two":
            common.update({
                "native": {"arithmetic_minus_structural": 1.0},
                "intervened": {"arithmetic_minus_structural": .5},
                "preference_sign_preserved": True,
            })
        else:
            damage = target_damage if row["action"] == "successor" else copy_damage
            common.update({
                "native": {"margin": 2.0, "ce": 1.0, "answer_best": True},
                "intervened": {"margin": 2.0 - damage, "ce": 2.0 if row["action"] == "successor" else 1.0,
                               "answer_best": True},
                "margin_damage": damage,
                "ce_increase": 1.0 if row["action"] == "successor" else 0.0,
            })
        raw.append(common)
    return raw


def test_candidate_scoring_implements_action_copy_surface_source_and_conflict_gates(rows):
    raw = _synthetic_raw(rows)
    report = r584.score_candidate(raw, cell_prefix="test:passing")
    assert report["passed_without_nulls"] is True
    assert report["all_representations_pass"] is True
    assert all(item["passed"] for item in report["action_gaps"].values())
    assert all(item["passed"] for item in report["copies"].values())
    assert all(item["passed"] for item in report["active_relation_and_conflict_controls"].values())
    damaged = _synthetic_raw(rows, copy_damage=.6)
    failed = r584.score_candidate(damaged, cell_prefix="test:copy_failure")
    assert failed["passed_without_nulls"] is False
    assert any(not item["passed"] for item in failed["copies"].values())


def test_select_uses_frozen_fit_scales(rows):
    raw = _synthetic_raw(rows)
    fit = r584.score_candidate(raw, cell_prefix="test:fit")
    tiny_scales = {key: {"margin_damage": .01, "logit_rms": .01} for key in fit["fit_scales"]}
    report = r584.score_candidate(raw, cell_prefix="test:select", frozen_scales=tiny_scales)
    assert report["passed_without_nulls"] is False
    assert report["fit_scales"] == tiny_scales


def test_null_gate_requires_active_norm_match_and_lower_action_gap(rows):
    real = _synthetic_raw(rows)
    null = [dict(item, margin_damage=0.0,
                 intervention_vector_norm=1.0,
                 intervened={**item.get("intervened", {}), "margin": 2.0, "ce": 1.0,
                              "answer_best": True})
            for item in real if item["condition"] in {
                "factorial_copy", "factorial_successor", "surface_copy", "surface_successor"}]
    report = r584.score_null(real, null, cell_prefix="test:null")
    assert report["passed"] is True
    dead = [dict(item, intervention_vector_norm=.01) for item in null]
    assert r584.score_null(real, dead, cell_prefix="test:dead_null")["passed"] is False


def test_interaction_records_use_exact_two_factor_mobius(rows):
    template = _synthetic_raw(rows)[:1]
    row = template[0]
    if row["condition"] == "step_two":
        pytest.skip("unexpected first synthetic row")
    def arm(margin):
        return [{**row, "intervened": {**row["intervened"], "margin": margin}}]
    raw = {
        "mlp8_background_cross": arm(7.0),
        "mlp8_contrast_self": arm(8.0),
        "mlp8_joint_response": arm(4.0),
    }
    raw["mlp8_background_cross"][0]["native"] = {**row["native"], "margin": 10.0}
    raw["mlp8_contrast_self"][0]["native"] = {**row["native"], "margin": 10.0}
    raw["mlp8_joint_response"][0]["native"] = {**row["native"], "margin": 10.0}
    result = r584.interaction_records(raw, 8)[0]
    assert {key: result[key] for key in ("cross", "self", "cross_x_self")} == {
        "cross": -3.0, "self": -2.0, "cross_x_self": -1.0}


def test_runner_has_explicit_scientific_null_and_split_tripwires():
    source = PATH.read_text()
    assert '"downstream_use_decomposition_null"' in source
    assert 'opened = ["FIT"]' in source
    assert 'opened.append("SELECT")' in source
    assert '"forbidden_splits_opened": []' in source
    assert 'model_backwards": 0' in source
    assert 'model_weights_updated": False' in source
