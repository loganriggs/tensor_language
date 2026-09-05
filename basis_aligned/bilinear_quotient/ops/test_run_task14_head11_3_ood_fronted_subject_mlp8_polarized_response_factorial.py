"""Focused CPU tests for the prospective Task14 OOD MLP8 response split."""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import pytest

import circuit_fast_screen_candidate_task14_ood_fronted_mlp8_polarized_response as authority
import native_capability_license as licensing
import run_task14_ood_fronted_mlp8_native_capability as capability
import run_task14_head11_3_ood_fronted_subject_mlp8_polarized_response_factorial as run


def _exact(value=0.0):
    return {name: value for name in (
        "native_replay_max_absolute_logit_error",
        "state_sum_max_absolute_error",
        "normalized_state_max_absolute_error",
        "source_term_sum_max_absolute_error",
        "product_closure_max_absolute_error",
        "output_closure_max_absolute_error",
        "propagated_recipient_MLP8_max_absolute_error",
        "propagated_source_MLP8_max_absolute_error",
        "gauge_invariance_max_absolute_error",
        "parent_head_endpoint_max_absolute_error",
        "same_batch_native_noop_endpoint_max_absolute_error",
        "installed_head_max_absolute_error",
    )}


def _evidence(*, break_singular_cross=False):
    output = []
    for row in authority.build_rows():
        p_to_s = row["direction_id"] == "plural_to_singular"
        values = {}
        for background in ("standalone", "conditional"):
            values[f"{background}_recipient"] = 0.0
            values[f"{background}_full"] = 1.0
            values[f"{background}_cross"] = 2.0 if p_to_s else -.2
            values[f"{background}_quadratic"] = -1.0 if p_to_s else 1.2
        values.update(lexical_recipient=0.0, lexical_full=.1,
                      lexical_cross=.1, lexical_quadratic=0.0)
        if break_singular_cross and not p_to_s:
            values["standalone_cross"] = .2
        cell = f"{row['direction_id']}__{row['template_id']}"
        for condition in run.CONDITIONS:
            output.append({
                "row_id": row["row_id"], "cell_id": cell, "condition": condition,
                "target_margin_improvement": values[condition],
                "target_CE_improvement": values[condition],
            })
    return output


def test_authority_is_exact_balanced_token8_only_ood_reuse():
    rows = authority.build_rows()
    assert authority.validate_rows(rows) == authority.EXPECTED_AUTHORITY_SHA256
    assert [row["group_number"] for row in rows] == list(range(8)) + list(range(16, 24))
    assert set(Counter((row["direction_id"], tuple(row["attractor_state"]))
                       for row in rows).values()) == {2}
    for row in rows:
        assert row["split"] == "OOD_TEXT_REUSE_NEW_MLP8_INTERVENTION"
        recipient = row["endpoints"]["recipient"]["ids"]
        for role in authority.ROLES[1:]:
            other = row["endpoints"][role]["ids"]
            assert [i for i, pair in enumerate(zip(recipient, other))
                    if pair[0] != pair[1]] == [8]


def test_capability_gate_is_new_scoped_and_one_forward():
    plan = capability.compile_plan()
    assert plan["causal_candidate_id"] == run.CANDIDATE_ID
    assert plan["price"] == {"model_forwards": 1, "example_evaluations": 48,
                             "causal_interventions": 0, "backwards": 0,
                             "parameter_updates": 0}
    assert len(capability.build_gate().cells) == 6
    assert hashlib.sha256(Path(authority.__file__).read_bytes()).hexdigest() == \
        capability.AUTHORITY_FILE_SHA256


def test_standard_capability_result_and_license_bind_candidate(tmp_path):
    gate = capability.build_gate()
    evidence = []
    for row in authority.build_rows():
        for role in authority.ROLES:
            evidence.append({
                "example_id": f"{row['row_id']}:{role}",
                "cell_id": capability._cell_id(row, role),
                "correct": True,
                "full_vocab_CE": .1,
                "answer_minus_foil_margin": 2.0,
            })
    result_path, license_path = tmp_path / "result.json", tmp_path / "license.json"
    result, _ = licensing.finalize_native_capability(gate, evidence, result_path)
    assert result["terminal"] == "pass"
    value, digest = licensing.issue_capability_license(
        gate, result_path, license_path, causal_candidate_id=run.CANDIDATE_ID)
    assert value["causal_candidate_id"] == run.CANDIDATE_ID
    assert licensing.validate_causal_preflight(
        gate, result_path, license_path, expected_license_sha256=digest,
        causal_candidate_id=run.CANDIDATE_ID) == value


def test_causal_plan_validates_scoped_license_and_preserves_price(tmp_path, monkeypatch):
    gate = capability.build_gate()
    evidence = [{
        "example_id": f"{row['row_id']}:{role}",
        "cell_id": capability._cell_id(row, role),
        "correct": True, "full_vocab_CE": .1, "answer_minus_foil_margin": 2.0,
    } for row in authority.build_rows() for role in authority.ROLES]
    result_path, license_path = tmp_path / "result.json", tmp_path / "license.json"
    licensing.finalize_native_capability(gate, evidence, result_path)
    licensing.issue_capability_license(
        gate, result_path, license_path, causal_candidate_id=run.CANDIDATE_ID)
    monkeypatch.setattr(capability, "RESULT", result_path)
    monkeypatch.setattr(capability, "LICENSE", license_path)
    plan = run.compile_plan()
    assert plan["causal_price"] == {
        "model_forwards": 4, "example_evaluations": 480,
        "causal_interventions": 192, "backwards": 0, "parameter_updates": 0}
    assert plan["split"] == "OOD_TEXT_REUSE_NEW_MLP8_INTERVENTION"


def test_signed_direction_and_dependent_removal_predictions_hold():
    scored = run.score(_evidence(), _exact())
    assert all(scored["predictions"].values())
    assert scored["selective_removal_independence"] == \
        "algebraically dependent on the same four corners"
    for name, cell in scored["cells"].items():
        removal = cell["derived"]["selective_removal"]
        if name.startswith("plural_to_singular__"):
            assert all(value > 0 for value in removal["remove_cross"].values())
            assert all(value < 0 for value in removal["remove_quadratic"].values())
        else:
            assert all(value < 0 for value in removal["remove_cross"].values())
            assert all(value > 0 for value in removal["remove_quadratic"].values())


def test_opposing_direction_prediction_can_fail_with_live_instrument():
    scored = run.score(_evidence(break_singular_cross=True), _exact())
    assert scored["predictions"]["pred_a_instrument_live"]
    assert not scored["predictions"][
        "pred_c_singular_to_plural_cross_negative_quadratic_positive"]
    assert not scored["predictions"]["pred_d_signed_direction_pattern"]


def test_exactness_gate_can_fail():
    scored = run.score(_evidence(), _exact(1.0))
    assert not scored["predictions"]["pred_a_instrument_live"]


def test_evidence_identity_fails_closed():
    evidence = _evidence()
    evidence.pop()
    with pytest.raises(run.OODMLP8PolarizedResponseError,
                       match="exact OOD screen"):
        run.score(evidence, _exact())
