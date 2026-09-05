#!/usr/bin/env python3
"""Focused CPU tests for the Task14 MLP8 E/A/U/V factorial."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import native_capability_license as licensing
import run_task14_head11_3_fresh_matched_subject_mlp8_mlp_depth_source_factorial as run


def _exact(value=0.0):
    return {name: value for name in (
        "native_replay_max_absolute_logit_error",
        "input_state_closure_max_absolute_error",
        "input_normalized_closure_max_absolute_error",
        "M_grouping_closure_max_absolute_error",
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
    assert run.FAMILIES == ("E", "A", "U", "V")
    assert len(run.SUBSETS) == len(set(run.SUBSETS)) == 15
    assert len(run.CONDITIONS) == len(set(run.CONDITIONS)) == 91
    assert set(run.CONDITIONS[1:]) == {
        f"{source}_{subset}_{component}"
        for source in run.SOURCES for subset in run.SUBSETS
        for component in run.COMPONENTS}
    assert plan["price"] == {
        "model_forwards": 4, "example_evaluations": 3008,
        "causal_interventions": 1440, "backwards": 0,
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
    with pytest.raises(run.MLP8MLPDepthSourceError, match="prior-art receipt changed"):
        run.validate_preflight()
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", run._sha256(run.PRIOR_ART))
    monkeypatch.setattr(run, "PARENT_RESULT_SHA256", "0" * 64)
    with pytest.raises(run.MLP8MLPDepthSourceError, match="parent E/A/M result changed"):
        run.validate_preflight()


def test_hybrid_uses_U_remainder_and_epsilon_follows_E():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    recipient = {"E": torch.tensor([[1., 0., 0.]]),
                 "A": torch.tensor([[0., 2., 0.]]),
                 "U": torch.tensor([[0., 0., 3.]]),
                 "V": torch.tensor([[1., 1., 0.]]),
                 "M_group_remainder": torch.tensor([[.01, .02, .03]]),
                 "epsilon": torch.tensor([[.1, .2, .3]])}
    source = {"E": torch.tensor([[4., 0., 0.]]),
              "A": torch.tensor([[0., 5., 0.]]),
              "U": torch.tensor([[0., 0., 6.]]),
              "V": torch.tensor([[2., 2., 0.]]),
              "M_group_remainder": torch.tensor([[.04, .05, .06]]),
              "epsilon": torch.tensor([[.4, .5, .6]])}
    recipient["M"] = (recipient["U"] + recipient["V"]) \
        + recipient["M_group_remainder"]
    source["M"] = (source["U"] + source["V"]) \
        + source["M_group_remainder"]
    source_raw = (source["E"] + source["A"] + source["M"]) + source["epsilon"]
    source["input"] = F.rms_norm(source_raw, (3,))
    recipient_raw = (recipient["E"] + recipient["A"] + recipient["M"]) \
        + recipient["epsilon"]
    recipient["input"] = F.rms_norm(recipient_raw, (3,))
    observed, raw, endpoint = run._hybrid_input(recipient, source, "EU", F)
    expected_raw = ((source["E"] + recipient["A"])
                    + (source["U"] + recipient["V"])) \
        + source["M_group_remainder"] + source["epsilon"]
    assert torch.equal(raw, expected_raw)
    assert torch.allclose(observed, F.rms_norm(expected_raw, (3,)))
    assert endpoint == 0.0
    observed, raw, endpoint = run._hybrid_input(recipient, source, "EAUV", F)
    assert torch.equal(raw, source_raw)
    assert torch.equal(observed, source["input"])
    assert endpoint == 0.0
    for parent_subset, child_subset in run.PARENT_TO_CHILD.items():
        parent_recipient = {**recipient,
                            "M": recipient["U"] + recipient["V"]
                                 + recipient["M_group_remainder"]}
        parent_source = {**source,
                         "M": source["U"] + source["V"]
                              + source["M_group_remainder"]}
        parent_input, parent_raw, _ = run.parent._hybrid_input(
            parent_recipient, parent_source, parent_subset, F)
        child_input, child_raw, _ = run._hybrid_input(
            recipient, source, child_subset, F)
        assert torch.allclose(child_raw, parent_raw)
        assert torch.allclose(child_input, parent_input)


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
        coefficients["EU"] + coefficients["EV"] + coefficients["EUV"])
    assert run._aggregate_terms(terms, "EM", "V") == pytest.approx(
        coefficients["EV"] + coefficients["EUV"])
    assert run._aggregate_terms(terms, "EM", "U_only") == pytest.approx(
        coefficients["EU"])


def _synthetic_evidence():
    coefficients = {subset: 0.0 for subset in run.SUBSETS}
    coefficients.update({"E": .1, "A": .1, "U": .1, "V": .9,
                         "EU": .1, "EV": .9, "AU": .1, "AV": .9})
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


def test_score_identifies_late_depth_dominance_and_number_specificity():
    scored = run.score(_synthetic_evidence(), _exact())
    assert scored["predictions"] == {
        "pred_a_instrument_and_parent_closure": True,
        "pred_b_V_late_dominant": True,
        "pred_c_U_early_dominant": False,
        "pred_d_distributed_depth": False,
        "pred_e_cross_depth_composition": False,
        "pred_f_direction_switch": False,
        "pred_g_number_specific": True,
    }
    assert scored["direction_winners"] == {
        "plural_to_singular": "V", "singular_to_plural": "V"}
    broken = _exact()
    broken["parent_propagated_slot_max_absolute_error"] = 1e-3
    assert not run.score(_synthetic_evidence(), broken)["predictions"][
        "pred_a_instrument_and_parent_closure"]


def test_score_rejects_incomplete_or_nonfinite_evidence():
    evidence = _synthetic_evidence()
    with pytest.raises(run.MLP8MLPDepthSourceError, match="91-condition"):
        run.score(evidence[:-1], _exact())
    evidence = _synthetic_evidence()
    evidence[1]["opposite_target_margin_improvement"] = float("nan")
    with pytest.raises(run.MLP8MLPDepthSourceError, match="non-finite"):
        run.score(evidence, _exact())
