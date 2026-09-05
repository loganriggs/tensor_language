#!/usr/bin/env python3
"""Focused CPU tests for the Task14 MLP8 input-writer response factorial."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import native_capability_license as licensing
import run_task14_head11_3_fresh_matched_subject_mlp8_input_writer_response_factorial as run


def _exact(value=0.0):
    return {name: value for name in (
        "native_replay_max_absolute_logit_error",
        "input_state_closure_max_absolute_error",
        "input_normalized_closure_max_absolute_error",
        "hybrid_endpoint_max_absolute_error",
        "source_term_sum_max_absolute_error",
        "product_closure_max_absolute_error",
        "output_closure_max_absolute_error",
        "propagated_endpoint_max_absolute_error",
        "gauge_invariance_max_absolute_error",
        "parent_head_endpoint_max_absolute_error",
        "same_batch_native_noop_endpoint_max_absolute_error",
        "installed_head_max_absolute_error",
    )}


def test_plan_has_exact_full_factorial_price_and_scoped_license():
    plan = run.compile_plan()
    assert len(run.CONDITIONS) == len(set(run.CONDITIONS)) == 43
    assert run.CONDITIONS[0] == "recipient"
    assert set(run.CONDITIONS[1:]) == {
        f"{source}_{subset}_{component}"
        for source in run.SOURCES for subset in run.SUBSETS
        for component in run.COMPONENTS
    }
    assert plan["price"] == {
        "model_forwards": 4, "example_evaluations": 1472,
        "causal_interventions": 672, "backwards": 0,
        "parameter_updates": 0, "capability_GPU_price": 0,
    }
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


def test_preflight_fails_closed_on_each_parent_hash(monkeypatch):
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", "0" * 64)
    with pytest.raises(run.MLP8InputWriterResponseError, match="prior-art receipt changed"):
        run.validate_preflight()
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", run._sha256(run.PRIOR_ART))
    monkeypatch.setattr(run, "POLARIZED_RESULT_SHA256", "0" * 64)
    with pytest.raises(run.MLP8InputWriterResponseError,
                       match="polarized parent result changed"):
        run.validate_preflight()
    monkeypatch.setattr(run, "POLARIZED_RESULT_SHA256", run._sha256(run.POLARIZED_RESULT))
    monkeypatch.setattr(run, "UPSTREAM_RESULT_SHA256", "0" * 64)
    with pytest.raises(run.MLP8InputWriterResponseError,
                       match="upstream-writer parent result changed"):
        run.validate_preflight()


def test_hybrid_normalizes_after_raw_family_swap_and_epsilon_follows_E():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    recipient = {"E": torch.tensor([[1., 0., 0.]]),
                 "A": torch.tensor([[0., 2., 0.]]),
                 "M": torch.tensor([[0., 0., 3.]]),
                 "epsilon": torch.tensor([[.1, .2, .3]])}
    source = {"E": torch.tensor([[4., 0., 0.]]),
              "A": torch.tensor([[0., 5., 0.]]),
              "M": torch.tensor([[0., 0., 6.]]),
              "epsilon": torch.tensor([[.4, .5, .6]])}
    source_raw = (source["E"] + source["A"] + source["M"]) + source["epsilon"]
    source["input"] = F.rms_norm(source_raw, (3,))
    recipient_raw = (recipient["E"] + recipient["A"] + recipient["M"]) \
        + recipient["epsilon"]
    recipient["input"] = F.rms_norm(recipient_raw, (3,))
    observed, raw, endpoint = run._hybrid_input(recipient, source, "E", F)
    expected_raw = (source["E"] + recipient["A"] + recipient["M"]) + source["epsilon"]
    assert torch.equal(raw, expected_raw)
    assert torch.allclose(observed, F.rms_norm(expected_raw, (3,)))
    assert endpoint == 0.0
    observed, raw, endpoint = run._hybrid_input(recipient, source, "EAM", F)
    assert torch.equal(raw, source_raw)
    assert torch.equal(observed, source["input"])
    assert endpoint == 0.0


def test_mobius_terms_reconstruct_full_set_function():
    values = {"": .7, "E": 1.1, "A": .4, "M": 2.0,
              "EA": 1.5, "EM": 2.9, "AM": 1.8, "EAM": 4.2}
    terms = run._mobius(values)
    assert sum(terms.values()) + values[""] == pytest.approx(values["EAM"])
    assert terms["EA"] == pytest.approx(values["EA"] - values["E"]
                                         - values["A"] + values[""])


def _synthetic_evidence():
    family_weights = {"E": .1, "A": .1, "M": .8}
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
                fraction = sum(family_weights[family] for family in subset)
                component_scale = {
                    "cross": direction_sign,
                    "quadratic": -direction_sign,
                    "full": 1.0,
                }[component]
                effect = fraction * component_scale
                if source == "lexical":
                    effect *= .01
                for metric in ("margin", "CE"):
                    item[f"{source}_target_{metric}_improvement"] = effect
            evidence.append(item)
    return evidence


def test_score_identifies_M_source_stable_number_specific_account():
    scored = run.score(_synthetic_evidence(), _exact())
    assert scored["predictions"] == {
        "pred_a_instrument_live": True,
        "pred_b_M_source_dominant": True,
        "pred_c_E_source_dominant": False,
        "pred_d_A_source_dominant": False,
        "pred_e_distributed_additive": False,
        "pred_f_source_interaction_needed": False,
        "pred_g_direction_stable": True,
        "pred_h_direction_switch": False,
        "pred_i_number_specific": True,
        "pred_j_lexical_collateral": False,
    }
    assert scored["maximum_lexical_ratio"] == pytest.approx(.01)
    assert scored["direction_component_winners"]["plural_to_singular"] == {
        component: "M" for component in run.COMPONENTS}


def test_score_rejects_incomplete_or_nonfinite_evidence():
    evidence = _synthetic_evidence()
    with pytest.raises(run.MLP8InputWriterResponseError, match="43-condition"):
        run.score(evidence[:-1], _exact())
    evidence = _synthetic_evidence()
    evidence[1]["opposite_target_margin_improvement"] = float("nan")
    with pytest.raises(run.MLP8InputWriterResponseError, match="non-finite"):
        run.score(evidence, _exact())
