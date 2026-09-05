#!/usr/bin/env python3
"""Focused CPU tests for the Task14 MLP8 E/A/U/W/Y/Z factorial."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import native_capability_license as licensing
import run_task14_head11_3_fresh_matched_subject_mlp8_mlp6_7_source_factorial as run


def _exact(value=0.0):
    return {name: value for name in (
        "native_replay_max_absolute_logit_error",
        "input_state_closure_max_absolute_error",
        "input_normalized_closure_max_absolute_error",
        "M_grouping_closure_max_absolute_error",
        "V_grouping_closure_max_absolute_error",
        "X_grouping_closure_max_absolute_error",
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
    assert run.FAMILIES == ("E", "A", "U", "W", "Y", "Z")
    assert len(run.SUBSETS) == len(set(run.SUBSETS)) == 63
    assert len(run.CONDITIONS) == len(set(run.CONDITIONS)) == 379
    assert set(run.CONDITIONS[1:]) == {
        f"{source}_{subset}_{component}"
        for source in run.SOURCES for subset in run.SUBSETS
        for component in run.COMPONENTS}
    assert plan["price"] == {
        "logical_model_forwards": 4, "physical_model_forwards": 50,
        "example_evaluations": 12224,
        "causal_interventions": 6048, "backwards": 0,
        "parameter_updates": 0, "capability_GPU_price": 0}
    assert plan["oom_batching_amendment_sha256"] == run.OOM_AMENDMENT_SHA256
    assert plan["oom_retry_amendment_sha256"] == run.OOM_RETRY_AMENDMENT_SHA256
    assert plan["batching"] == {
        "condition_chunk_rows": 256,
        "condition_chunks": [list(chunk) for chunk in run.PATCH_CHUNKS],
        "order": "contiguous_then_concatenate",
        "storage": "offload each chunk, retain subject-position logits, and score on CPU"}
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
    with pytest.raises(run.MLP8MLP67SourceError, match="prior-art receipt changed"):
        run.validate_preflight()
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", run._sha256(run.PRIOR_ART))
    monkeypatch.setattr(run, "OOM_AMENDMENT_SHA256", "0" * 64)
    with pytest.raises(run.MLP8MLP67SourceError, match="OOM batching amendment changed"):
        run.validate_preflight()
    monkeypatch.setattr(run, "OOM_AMENDMENT_SHA256", run._sha256(run.OOM_AMENDMENT))
    monkeypatch.setattr(run, "OOM_RETRY_AMENDMENT_SHA256", "0" * 64)
    with pytest.raises(run.MLP8MLP67SourceError, match="OOM retry amendment changed"):
        run.validate_preflight()
    monkeypatch.setattr(
        run, "OOM_RETRY_AMENDMENT_SHA256", run._sha256(run.OOM_RETRY_AMENDMENT))
    monkeypatch.setattr(run, "PARENT_RESULT_SHA256", "0" * 64)
    with pytest.raises(run.MLP8MLP67SourceError, match="parent E/A/U/W/X result changed"):
        run.validate_preflight()


def test_frozen_chunk_plan_slices_every_batch_axis_and_preserves_order():
    torch = pytest.importorskip("torch")
    size = 6064
    values = torch.arange(size)
    patch = {
        "tokens": values[:, None], "finals": values + 1,
        "replacement_heads": values[:, None] + 2,
        "native_reinstall_mask": values.remainder(2).bool(),
        "specs": list(range(size)),
    }
    assert run.PATCH_CHUNKS[0] == (0, 256)
    assert run.PATCH_CHUNKS[-1] == (5888, 6064)
    assert len(run.PATCH_CHUNKS) == 24
    chunks = [run._slice_patch(patch, *bounds) for bounds in run.PATCH_CHUNKS]
    assert max(len(chunk["specs"]) for chunk in chunks) == 256
    for key in ("tokens", "finals", "replacement_heads", "native_reinstall_mask"):
        assert torch.equal(torch.cat([chunk[key] for chunk in chunks]), patch[key])
    assert sum((chunk["specs"] for chunk in chunks), []) == patch["specs"]
    closures = [{"state": 2e-6} for _ in run.PATCH_CHUNKS]
    closures[-1]["state"] = 4e-6
    assert run._maximum_chunk_closure(closures, "state") == 4e-6
    with pytest.raises(run.MLP8MLP67SourceError, match="closure count"):
        run._maximum_chunk_closure(closures[:1], "state")


def test_full_logits_are_detached_and_offloaded_before_retention():
    calls = []

    class Probe:
        def detach(self):
            calls.append("detach")
            return self

        def cpu(self):
            calls.append("cpu")
            return "cpu_full_logits"

    assert run._offload_full_logits(Probe()) == "cpu_full_logits"
    assert calls == ["detach", "cpu"]


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
                 "Y": torch.tensor([[.5, .5, 0.]]),
                 "Z": torch.tensor([[.2, .2, 0.]]),
                 "X_group_remainder": torch.tensor([[.01, .01, .02]]),
                 "V_group_remainder": torch.tensor([[.01, .02, .03]]),
                 "M_group_remainder": torch.tensor([[.01, .02, .03]]),
                 "epsilon": torch.tensor([[.1, .2, .3]])}
    source = {"E": torch.tensor([[4., 0., 0.]]),
              "A": torch.tensor([[0., 5., 0.]]),
              "U": torch.tensor([[0., 0., 6.]]),
              "W": torch.tensor([[2., 2., 0.]]),
              "Y": torch.tensor([[.7, .7, 0.]]),
              "Z": torch.tensor([[.3, .3, 0.]]),
              "X_group_remainder": torch.tensor([[.02, .02, .03]]),
              "V_group_remainder": torch.tensor([[.04, .05, .06]]),
              "M_group_remainder": torch.tensor([[.04, .05, .06]]),
              "epsilon": torch.tensor([[.4, .5, .6]])}
    # Deliberately make every authoritative parent aggregate differ from fresh
    # child regrouping, so same-role corners cannot pass accidentally.
    recipient["X"] = torch.tensor([[.8, .6, .03]])
    source["X"] = torch.tensor([[1.1, .9, .05]])
    recipient["V"] = torch.tensor([[1.6, 1.4, .04]])
    source["V"] = torch.tensor([[2.8, 2.6, .09]])
    recipient["M"] = torch.tensor([[2.7, 1.5, 3.2]])
    source["M"] = torch.tensor([[3.1, 3.3, 6.4]])
    source_raw = (source["E"] + source["A"] + source["M"]) + source["epsilon"]
    source["input"] = F.rms_norm(source_raw, (3,))
    recipient_raw = (recipient["E"] + recipient["A"] + recipient["M"]) \
        + recipient["epsilon"]
    recipient["input"] = F.rms_norm(recipient_raw, (3,))
    observed, raw, endpoint = run._hybrid_input(recipient, source, "EAUWYZ", F)
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

    # A genuinely mixed Y/Z corner constructs X and follows Y's remainder,
    # then preserves the parent W/V and U/M remainder rules.
    _, mixed_raw, _ = run._hybrid_input(recipient, source, "EY", F)
    mixed_x = (source["Y"] + recipient["Z"]) + source["X_group_remainder"]
    mixed_v = (recipient["W"] + mixed_x) + recipient["V_group_remainder"]
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
    selected = lambda key: "E" in key and "A" not in key and bool(set(key) & {"Y", "Z"})
    assert run._aggregate_terms(terms, "EM") == pytest.approx(
        sum(value for key, value in coefficients.items() if selected(key)))
    assert run._aggregate_terms(terms, "EM", "Z_only") == pytest.approx(
        sum(value for key, value in coefficients.items()
            if selected(key) and "Z" in key and "Y" not in key))
    assert run._aggregate_terms(terms, "EM", "YZ") == pytest.approx(
        sum(value for key, value in coefficients.items()
            if selected(key) and {"Y", "Z"}.issubset(set(key))))


def _synthetic_evidence():
    coefficients = {subset: 0.0 for subset in run.SUBSETS}
    coefficients.update({"E": .1, "A": .1, "U": .1, "W": .1,
                         "Y": .1, "Z": .9,
                         "EY": .1, "EZ": .9, "AY": .1, "AZ": .9})
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


def test_score_identifies_mlp7_dominance_and_number_specificity():
    scored = run.score(_synthetic_evidence(), _exact())
    assert scored["predictions"] == {
        "pred_a_instrument_and_parent_closure": True,
        "pred_b_Z_mlp7_dominant": True,
        "pred_c_Y_mlp6_dominant": False,
        "pred_d_distributed_mlp6_7": False,
        "pred_e_YZ_composition": False,
        "pred_f_direction_switch": False,
        "pred_g_number_specific": True,
    }
    assert scored["direction_winners"] == {
        "plural_to_singular": "Z", "singular_to_plural": "Z"}
    assert scored["parent_lattice_mobius_max_absolute_error"] < 1e-12
    broken = _exact()
    broken["parent_propagated_slot_max_absolute_error"] = 1e-3
    assert not run.score(_synthetic_evidence(), broken)["predictions"][
        "pred_a_instrument_and_parent_closure"]


def test_score_rejects_incomplete_or_nonfinite_evidence():
    evidence = _synthetic_evidence()
    with pytest.raises(run.MLP8MLP67SourceError, match="379-condition"):
        run.score(evidence[:-1], _exact())
    evidence = _synthetic_evidence()
    evidence[1]["opposite_target_margin_improvement"] = float("nan")
    with pytest.raises(run.MLP8MLP67SourceError, match="non-finite"):
        run.score(evidence, _exact())
