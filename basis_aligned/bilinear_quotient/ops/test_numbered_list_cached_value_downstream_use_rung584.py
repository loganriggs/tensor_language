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
            "source_value": row["source_value"], "condition": row["condition"],
            "action": row["action"], **r584.row_coordinates(row),
            "site": 8, "component": "background_cross", "arm": "mlp8_background_cross",
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
        common.update({
            "source_deleted": dict(common["native"]),
            "source_deleted_logit_difference_squared_sum": 0.0,
            "source_deleted_logit_vocabulary_count": 50_304,
            "source_deleted_full_vocabulary_logit_rms": 0.0,
            "source_deleted_evidence_reason": None,
        })
        raw.append(common)
    return raw


def _synthetic_capture(rows):
    capture = []
    for row in rows:
        if row["split"] != "FIT":
            continue
        capture.append({
            "row_id": row["row_id"], "group_id": row["group_id"], "split": "FIT",
            "representation": row["representation"], "source_level": row["source_level"],
            "source_value": row["source_value"], "condition": row["condition"],
            "action": row["action"], **r584.row_coordinates(row),
            "arm": "source_present_and_deleted_capture", "sites": list(r584.SITES),
            "native": {}, "source_deleted": {}, "r576_term_norm": 1.0,
            "source_deleted_logit_difference_squared_sum": 0.0,
            "source_deleted_logit_vocabulary_count": 50_304,
            "source_deleted_full_vocabulary_logit_rms": 0.0,
            "component_norms": {str(site): {component: 1.0 for component in r584.COMPONENTS}
                                for site in r584.SITES},
            "bilinear_response_relative_squared_error": 0.0,
            "bilinear_response_relative_squared_error_by_site": {
                str(site): 0.0 for site in r584.SITES},
            "native_replay_relative_squared_error_by_row": {
                "source_present": 0.0, "source_deleted": 0.0, "maximum": 0.0},
        })
    return capture


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
    real_report = r584.score_candidate(real, cell_prefix="test:null:real", authority_rows=rows)
    donors = r584.r582.deterministic_null_maps(rows, "FIT")["different_group_same_cell"]
    null = [dict(item, margin_damage=0.0,
                 intervention_vector_norm=1.0,
                 null_donor_row_id=donors[item["row_id"]],
                 arm="null:different_group_same_cell",
                 intervened={**item.get("intervened", {}), "margin": 2.0, "ce": 1.0,
                              "answer_best": True})
            for item in real if item["condition"] in {
                "factorial_copy", "factorial_successor", "surface_copy", "surface_successor"}]
    report = r584.score_null(
        real, null, cell_prefix="test:null", real_report=real_report,
        null_name="different_group_same_cell", authority_rows=rows)
    assert report["passed"] is True
    dead = [dict(item, intervention_vector_norm=.01) for item in null]
    assert r584.score_null(
        real, dead, cell_prefix="test:dead_null", real_report=real_report,
        null_name="different_group_same_cell", authority_rows=rows)["passed"] is False


def test_planted_scientific_null_uses_json_null_and_named_reason(rows):
    negative = r584.score_candidate(
        _synthetic_raw(rows, target_damage=-1.0),
        cell_prefix="test:planted_negative", authority_rows=rows)
    cell = negative["stability"]["surface_recovery"]["list:source0"]
    assert cell["mean_gap_ratio"] is None
    assert cell["mean_gap_ratio_reason"] == "nonpositive_ordinary_action_gap"
    json.dumps(negative, allow_nan=False)

    zero_scale = r584.score_candidate(
        _synthetic_raw(rows, target_damage=0.0, copy_damage=0.0),
        cell_prefix="test:planted_zero_scale", authority_rows=rows)
    copy_cell = zero_scale["copies"]["list:source0:factorial"]
    assert copy_cell["median_absolute_margin_fraction"] is None
    assert copy_cell["median_absolute_margin_fraction_reason"] == \
        "nonpositive_successor_margin_scale"
    json.dumps(zero_scale, allow_nan=False)


def test_planted_dead_null_is_finite_typed_and_reuses_real_bounds(rows):
    real = [dict(item, intervention_vector_norm=0.0) for item in _synthetic_raw(rows)]
    real_report = r584.score_candidate(
        real, cell_prefix="test:dead_real", authority_rows=rows)
    donors = r584.r582.deterministic_null_maps(rows, "FIT")["different_group_same_cell"]
    null = [dict(item, margin_damage=0.0, intervention_vector_norm=1.0,
                 null_donor_row_id=donors[item["row_id"]],
                 arm="null:different_group_same_cell")
            for item in real if item["condition"] in {
                "factorial_copy", "factorial_successor", "surface_copy", "surface_successor"}]
    report = r584.score_null(
        real, null, cell_prefix="test:dead_real:null", real_report=real_report,
        null_name="different_group_same_cell", authority_rows=rows)
    cell = report["representation_cells"]["list:source0:factorial"]
    assert cell["median_null_norm_over_median_real_norm"] is None
    assert cell["norm_ratio_reason"] == "nonpositive_real_intervention_norm"
    assert cell["real_gap_lower95_reused"] == real_report[
        "action_gaps"]["list:source0:factorial"]["bootstrap95_lower_mean_gap"]
    assert report["real_bounds_reused_without_redraw"] is True
    json.dumps(report, allow_nan=False)


def test_null_rule_is_conservative_across_representations_not_merely_cellwise(rows, monkeypatch):
    real = _synthetic_raw(rows)
    real_report = r584.score_candidate(
        real, cell_prefix="test:conservative:real", authority_rows=rows)
    real_report["action_gaps"]["list:source0:factorial"][
        "bootstrap95_lower_mean_gap"] = 0.3
    real_report["action_gaps"]["digit:source0:factorial"][
        "bootstrap95_lower_mean_gap"] = 0.9
    real_report["action_gaps"]["word:source0:factorial"][
        "bootstrap95_lower_mean_gap"] = 0.9
    donors = r584.r582.deterministic_null_maps(rows, "FIT")["different_group_same_cell"]
    null = [dict(item, margin_damage=0.0,
                 null_donor_row_id=donors[item["row_id"]],
                 arm="null:different_group_same_cell")
            for item in real if item["condition"] in {
                "factorial_copy", "factorial_successor", "surface_copy", "surface_successor"}]

    original = r584._bootstrap_lower
    def planted(cells, key, cell_id):
        if "source0:factorial:null" in cell_id:
            return {"list": 0.2, "digit": 0.8, "word": 0.8}[
                next(rep for rep in ("list", "digit", "word") if f":{rep}:" in cell_id)
            ]
        return original(cells, key, cell_id)
    monkeypatch.setattr(r584, "_bootstrap_lower", planted)
    report = r584.score_null(
        real, null, cell_prefix="test:conservative:null", real_report=real_report,
        null_name="different_group_same_cell", authority_rows=rows)
    comparison = report["source_surface_comparisons"]["source0:factorial"]
    assert comparison["minimum_real_gap_lower95_across_representations"] == 0.3
    assert comparison["maximum_null_gap_lower95_across_representations"] == 0.8
    assert comparison["strict_real_exceeds_null"] is False
    assert report["passed"] is False


def test_null_donor_ids_and_capture_evidence_fail_closed(rows):
    real = _synthetic_raw(rows)
    donors = r584.r582.deterministic_null_maps(rows, "FIT")["different_group_same_cell"]
    null = [dict(item, null_donor_row_id=donors[item["row_id"]],
                 arm="null:different_group_same_cell")
            for item in real if item["condition"] in {
                "factorial_copy", "factorial_successor", "surface_copy", "surface_successor"}]
    null[0]["null_donor_row_id"] = null[0]["row_id"]
    with pytest.raises(RuntimeError, match="null donor disagrees"):
        r584.validate_null_raw(null, rows, "FIT", "different_group_same_cell")

    capture = _synthetic_capture(rows)
    r584.validate_capture_raw(capture, rows, "FIT")
    del capture[0]["bilinear_response_relative_squared_error_by_site"]["14"]
    with pytest.raises(RuntimeError, match="per-site C/Q exactness is incomplete"):
        r584.validate_capture_raw(capture, rows, "FIT")


def test_scientific_result_uses_generic_contract_and_weight_update_field(rows):
    digest = "a" * 64
    result = {
        "rung": 584, "stage": "fixture", "provisional_fit_selection": None,
        "selected_component": None, "evaluated_splits": ["FIT"],
        "forbidden_splits_opened": [], "decision": "fixture_null",
        "next_step": "none", "execution_plan": {
            "literal_executable_maximum_forwards": 510},
        "fit_capture_raw": _synthetic_capture(rows), "select_capture_raw": None,
        "input_sha256": {"fixture": digest}, "model_forwards": 379,
        "model_backwards": 0, "model_weights_updated": False,
    }
    summary = r584.validate_scientific_result(
        result, rows, expected_forwards=379,
        expected_provenance={"fixture": digest})
    assert summary["rows"] == 576
    assert summary["model_forwards"] == 379
    tampered = dict(result, model_weights_updated=True)
    with pytest.raises(r584.result_contract.ContractError, match="expected False"):
        r584.validate_scientific_result(
            tampered, rows, expected_forwards=379,
            expected_provenance={"fixture": digest})


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
