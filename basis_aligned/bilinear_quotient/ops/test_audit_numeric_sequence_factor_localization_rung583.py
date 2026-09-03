"""Focused CPU-only tests for the independent R583 audit."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


PATH = Path(__file__).with_name("audit_numeric_sequence_factor_localization_rung583.py")
SPEC = importlib.util.spec_from_file_location("audit_numeric_r583", PATH)
assert SPEC and SPEC.loader
R = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = R
SPEC.loader.exec_module(R)


def authority():
    rows, positions = R.load_authority()
    return rows, positions


def test_authority_membership_and_price_are_exact():
    rows, positions = authority()
    assert len(rows) == len(positions) == 432
    assert sum(row["split"] == "FIT" for row in rows) == 288
    assert sum(row["split"] == "SELECT" for row in rows) == 144
    assert all(len(R.expected_rows(rows, "FIT", family)) == 32 for family in R.FAMILIES)
    price = R.declared_price(rows)
    assert price == {
        "FIT": {"rows": 288, "unique_endpoint_capture_chunks": 15,
                "oriented_intervention_chunks": 27},
        "SELECT": {"rows": 144, "unique_endpoint_capture_chunks": 11,
                   "oriented_intervention_chunks": 15},
        "maximum_forwards_if_all_conditionals_open": 652,
    }


def test_seeded_bootstrap_indices_statistics_and_lower_bound_are_literal():
    values = [1., 2., 4., 8.]
    traces = {}
    observed = R.bootstrap_lower(values, 123, traces, "cell", replicates=57)
    generator = np.random.default_rng(123)
    indices = generator.integers(0, 4, size=(57, 4))
    means = np.asarray(values, dtype=np.float64)[indices].mean(1)
    assert observed == float(np.quantile(means, .025))
    assert traces["cell"]["index_matrix_sha256"] == hashlib.sha256(
        indices.astype(">u4").tobytes()).hexdigest()
    assert traces["cell"]["statistic_vector_sha256"] == hashlib.sha256(
        means.astype(">f8").tobytes()).hexdigest()


def test_held_and_active_control_null_arm_fixtures():
    rows, _ = authority()
    held = R.fixture_raw(rows, control_null=False)
    null = R.fixture_raw(rows, control_null=True)
    held_reports = {arm: R.arm_report(held, arm, R.SEED + 100 * index, {}, replicates=17)
                    for index, arm in enumerate(R.SITE_ARMS)}
    null_reports = {arm: R.arm_report(null, arm, R.SEED + 100 * index, {}, replicates=17)
                    for index, arm in enumerate(R.SITE_ARMS)}
    assert R.choose(held_reports, R.SITE_ARMS)["selected_arm"] == R.SITE_ARMS[0]
    assert R.choose(null_reports, R.SITE_ARMS)["selected_arm"] is None
    assert all(report["target_pass"] and report["relation_pass"] for report in null_reports.values())
    assert all(not report["controls_pass"] for report in null_reports.values())


def test_saved_null_recomputes_exactly_and_extracts_expected_packet():
    rows, _ = authority()
    result = json.loads(R.R577_RESULT.read_text())
    audit = R.audit_payload(result, rows)
    assert audit["audit_verdict"] == "held_independent_audit"
    assert audit["independently_recomputed_scientific_decision"] == "complete_state_site_null"
    assert audit["recomputed_site_choice"]["selected_arm"] is None
    assert not audit["factor_stage_opened"]
    assert not audit["select_opened"]
    assert audit["recomputed_observed_forwards"] == 205
    assert audit["bootstrap_cell_count"] == 56
    packet = audit["knowledge_packet"]
    assert packet["all_control_interventions_strictly_nonzero"]
    assert packet["sites"]["a8_h73_complete"]["target_direction_cells_passed"] == 6
    assert packet["sites"]["a8_all_heads_complete"]["target_direction_cells_passed"] == 6
    assert packet["sites"]["post_mlp14_state"]["target_direction_cells_passed"] == 6
    assert packet["sites"]["post_mlp14_state"]["control_direction_cells_passed"] == 0


def test_saved_raw_report_and_price_tampering_are_detected():
    rows, _ = authority()
    result = json.loads(R.R577_RESULT.read_text())
    report_tamper = copy.deepcopy(result)
    report_tamper["fit_site_reports"]["a8_h73_complete"]["target_pass"] = False
    audit = R.audit_payload(report_tamper, rows, replicates=31)
    assert audit["audit_verdict"] == "failed_independent_audit"
    assert any(item.startswith("fit_site_reports") for item in audit["audit_failures"])
    price_tamper = copy.deepcopy(result)
    price_tamper["execution_price"]["observed_forwards"] = 204
    audit = R.audit_payload(price_tamper, rows, replicates=31)
    assert "execution_price.observed_forwards:value" in audit["audit_failures"]


def test_zero_active_control_intervention_is_detected():
    rows, _ = authority()
    result = json.loads(R.R577_RESULT.read_text())
    tampered = copy.deepcopy(result)
    for cell in tampered["fit_site_raw"]["a8_h73_complete"][
            "sequence_digit_surface_preserved"]["base_to_donor"]:
        cell["intervention_vector_norm"] = 0.
    # Replacing the saved aggregate prevents a report mismatch from hiding the explicit liveness failure.
    traces = {}
    tampered["fit_site_reports"]["a8_h73_complete"] = R.arm_report(
        tampered["fit_site_raw"], "a8_h73_complete", R.SEED, traces, replicates=29)
    audit = R.audit_payload(tampered, rows, replicates=29)
    assert audit["audit_verdict"] == "failed_independent_audit"
    assert "active_controls:zero_intervention" in audit["audit_failures"]


def test_runlog_and_append_only_completion_record_bind_result():
    result = json.loads(R.R577_RESULT.read_text())
    assert R.verify_run_provenance(result) == []
    assert not any(path.name.endswith("receipt.json") and "rung577" in path.name
                   for path in R.ROOT.glob("*rung577*receipt.json"))


def test_dryrun_never_touches_result_or_runlog(monkeypatch, tmp_path):
    class Bomb:
        def __getattr__(self, name):
            raise AssertionError(f"execution artifact touched through {name}")

    monkeypatch.setattr(R, "R577_RESULT", Bomb())
    monkeypatch.setattr(R, "R577_RUNLOG", Bomb())
    monkeypatch.setattr(R, "DRYRUN", tmp_path / "dryrun.json")
    receipt = R.run_dryrun()
    assert receipt["r577_result_opened_by_dryrun"] is False
    assert receipt["held_arm_fixture_passes"]
    assert receipt["control_null_fixture_has_no_eligible_site"]
    assert receipt["future_audit_model_forwards"] == 0
