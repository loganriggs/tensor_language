#!/usr/bin/env python3
"""Focused CPU tests for the Task14 MLP8 E/A/U/W/X factorial."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import native_capability_license as licensing
import run_task14_head11_3_fresh_matched_subject_mlp8_mlp4_7_source_factorial as run


def _exact(value=0.0):
    return {name: value for name in (
        "native_replay_max_absolute_logit_error",
        "input_state_closure_max_absolute_error",
        "input_normalized_closure_max_absolute_error",
        "M_grouping_closure_max_absolute_error",
        "V_grouping_closure_max_absolute_error",
        "hybrid_endpoint_max_absolute_error",
        "source_term_sum_max_absolute_error",
        "product_closure_max_absolute_error",
        "output_closure_max_absolute_error",
        "propagated_endpoint_max_absolute_error",
        "gauge_invariance_max_absolute_error",
        "parent_head_endpoint_max_absolute_error",
        "same_batch_native_noop_endpoint_max_absolute_error",
        "installed_head_max_absolute_error",
        "parent_raw_state_max_absolute_error",
        "parent_normalized_input_max_absolute_error",
        "parent_MLP8_output_max_absolute_error",
        "parent_propagated_slot_max_absolute_error",
        "parent_installed_head_max_absolute_error",
        "parent_downstream_outcome_max_absolute_error",
    )}


def test_plan_has_complete_factorial_frozen_price_and_scoped_license():
    plan = run.compile_plan()
    assert run.FAMILIES == ("E", "A", "U", "W", "X")
    assert len(run.SUBSETS) == len(set(run.SUBSETS)) == 31
    assert len(run.CONDITIONS) == len(set(run.CONDITIONS)) == 187
    assert set(run.CONDITIONS[1:]) == {
        f"{source}_{subset}_{component}"
        for source in run.SOURCES for subset in run.SUBSETS
        for component in run.COMPONENTS}
    assert plan["price"] == {
        "model_forwards": 4, "example_evaluations": 6080,
        "causal_interventions": 2976, "backwards": 0,
        "parameter_updates": 0, "capability_GPU_price": 0}
    license_value = licensing.validate_causal_preflight(
        run.capability.build_gate(), run.capability.CAPABILITY_RESULT, run.LICENSE,
        expected_license_sha256=run.LICENSE_SHA256,
        causal_candidate_id=run.CANDIDATE_ID)
    assert license_value["causal_candidate_id"] == run.CANDIDATE_ID


def test_no_model_dry_run_is_exact_plan():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                               check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == run.compile_plan()


def test_preflight_fails_closed_on_frozen_parent(monkeypatch):
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", "0" * 64)
    with pytest.raises(run.MLP8MLP47SourceError, match="prior-art receipt changed"):
        run.validate_preflight()
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", run._sha256(run.PRIOR_ART))
    monkeypatch.setattr(run, "PARENT_RESULT_SHA256", "0" * 64)
    with pytest.raises(run.MLP8MLP47SourceError, match="parent E/A/U/V result changed"):
        run.validate_preflight()


def test_downstream_replay_contract_keeps_every_required_parent_field():
    assert {"M0_3", "MR", "H", "HR"}.issubset(run.DOWNSTREAM_CAPTURE_REQUIRED)
    assert {f"M{i}" for i in run.downstream.LAYERS}.issubset(
        run.DOWNSTREAM_CAPTURE_REQUIRED)


def test_hybrid_uses_authoritative_same_role_aggregates_and_epsilon_follows_E():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    recipient = {"E": torch.tensor([[1., 0., 0.]]),
                 "A": torch.tensor([[0., 2., 0.]]),
                 "U": torch.tensor([[0., 0., 3.]]),
                 "W": torch.tensor([[1., 1., 0.]]),
                 "X": torch.tensor([[.5, .5, 0.]]),
                 "V_group_remainder": torch.tensor([[.01, .02, .03]]),
                 "M_group_remainder": torch.tensor([[.01, .02, .03]]),
                 "epsilon": torch.tensor([[.1, .2, .3]])}
    source = {"E": torch.tensor([[4., 0., 0.]]),
              "A": torch.tensor([[0., 5., 0.]]),
              "U": torch.tensor([[0., 0., 6.]]),
              "W": torch.tensor([[2., 2., 0.]]),
              "X": torch.tensor([[.7, .7, 0.]]),
              "V_group_remainder": torch.tensor([[.04, .05, .06]]),
              "M_group_remainder": torch.tensor([[.04, .05, .06]]),
              "epsilon": torch.tensor([[.4, .5, .6]])}
    # Deliberately make authoritative parent aggregates differ slightly from
    # fresh W+X and U+V regrouping, so this test detects the float32 mistake.
    recipient["V"] = torch.tensor([[1.6, 1.4, .04]])
    source["V"] = torch.tensor([[2.8, 2.6, .09]])
    recipient["M"] = torch.tensor([[2.7, 1.5, 3.2]])
    source["M"] = torch.tensor([[3.1, 3.3, 6.4]])
    source_raw = (source["E"] + source["A"] + source["M"]) + source["epsilon"]
    source["input"] = F.rms_norm(source_raw, (3,))
    recipient_raw = (recipient["E"] + recipient["A"] + recipient["M"]) \
        + recipient["epsilon"]
    recipient["input"] = F.rms_norm(recipient_raw, (3,))
    observed, raw, endpoint = run._hybrid_input(recipient, source, "EAUWX", F)
    assert torch.equal(raw, source_raw)
    assert torch.equal(observed, source["input"])
    assert endpoint == 0.0
    for parent_subset, child_subset in run.PARENT_TO_CHILD.items():
        parent_input, parent_raw, _ = run.parent._hybrid_input(
            recipient, source, parent_subset, F)
        child_input, child_raw, _ = run._hybrid_input(
            recipient, source, child_subset, F)
        assert torch.equal(child_raw, parent_raw)
        assert torch.equal(child_input, parent_input)

    # A genuinely mixed W/X corner constructs V and follows W's remainder.
    _, mixed_raw, _ = run._hybrid_input(recipient, source, "EW", F)
    mixed_v = (source["W"] + recipient["X"]) + source["V_group_remainder"]
    mixed_m = (recipient["U"] + mixed_v) + recipient["M_group_remainder"]
    expected_mixed = (source["E"] + recipient["A"] + mixed_m) + source["epsilon"]
    assert torch.equal(mixed_raw, expected_mixed)


def test_mobius_and_parent_aggregate_reconstruct_known_terms():
    coefficients = {subset: float(index + 1)
                    for index, subset in enumerate(run.SUBSETS)}
    values = {"": 0.0}
    for subset in run.SUBSETS:
        families = set(subset)
        values[subset] = sum(value for term, value in coefficients.items()
                             if set(term).issubset(families))
    terms = run._mobius(values)
    assert terms == pytest.approx(coefficients)
    assert run._aggregate_terms(terms, "EM") == pytest.approx(
        sum(coefficients[key] for key in ("EW", "EX", "EWX", "EUW", "EUX", "EUWX")))
    assert run._aggregate_terms(terms, "EM", "X") == pytest.approx(
        sum(coefficients[key] for key in ("EX", "EWX", "EUX", "EUWX")))
    assert run._aggregate_terms(terms, "EM", "W_only") == pytest.approx(
        coefficients["EW"] + coefficients["EUW"])


def _synthetic_evidence():
    coefficients = {subset: 0.0 for subset in run.SUBSETS}
    coefficients.update({"E": .1, "A": .1, "U": .1,
                         "W": .1, "X": .9,
                         "EW": .1, "EX": .9, "AW": .1, "AX": .9})
    evidence = []
    for row in run.build_rows():
        direction_sign = 1.0 if row["direction_id"] == "plural_to_singular" else -1.0
        cell_id = f"{row['direction_id']}__{row['template_id']}"
        for condition in run.CONDITIONS:
            item = {"row_id": row["row_id"], "cell_id": cell_id,
                    "condition": condition}
            for source in run.SOURCES:
                for metric in ("margin", "CE"):
                    item[f"{source}_target_{metric}_improvement"] = 0.0
            if condition != "recipient":
                source, subset, component = condition.split("_")
                families = set(subset)
                value = sum(weight for term, weight in coefficients.items()
                            if set(term).issubset(families))
                scale = {"cross": direction_sign,
                         "quadratic": -direction_sign, "full": 1.0}[component]
                effect = value * scale * (.01 if source == "lexical" else 1.0)
                for metric in ("margin", "CE"):
                    item[f"{source}_target_{metric}_improvement"] = effect
            evidence.append(item)
    return evidence


def test_score_identifies_mlp6_7_dominance_and_number_specificity():
    scored = run.score(_synthetic_evidence(), _exact())
    assert scored["predictions"] == {
        "pred_a_instrument_and_parent_closure": True,
        "pred_b_X_mlp6_7_dominant": True,
        "pred_c_W_mlp4_5_dominant": False,
        "pred_d_distributed_within_V": False,
        "pred_e_WX_composition": False,
        "pred_f_direction_switch": False,
        "pred_g_number_specific": True,
    }
    assert scored["direction_winners"] == {
        "plural_to_singular": "X", "singular_to_plural": "X"}
    assert scored["parent_lattice_mobius_max_absolute_error"] < 1e-12
    broken = _exact()
    broken["parent_propagated_slot_max_absolute_error"] = 1e-3
    assert not run.score(_synthetic_evidence(), broken)["predictions"][
        "pred_a_instrument_and_parent_closure"]


def test_score_rejects_incomplete_or_nonfinite_evidence():
    evidence = _synthetic_evidence()
    with pytest.raises(run.MLP8MLP47SourceError, match="187-condition"):
        run.score(evidence[:-1], _exact())
    evidence = _synthetic_evidence()
    evidence[1]["opposite_target_margin_improvement"] = float("nan")
    with pytest.raises(run.MLP8MLP47SourceError, match="non-finite"):
        run.score(evidence, _exact())
