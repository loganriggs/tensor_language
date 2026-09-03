"""Focused CPU-only tests for the pre-outcome R581 auditor."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


PATH = Path(__file__).with_name(
    "audit_induction_selector_payload_native_capability_rung581.py")
SPEC = importlib.util.spec_from_file_location("audit_r581", PATH)
assert SPEC and SPEC.loader
R = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = R
SPEC.loader.exec_module(R)


def authority():
    groups, rows, _ = R.load_authority()
    return groups, rows, R.expected_sequence_specs(groups, rows)


def test_authority_and_exact_membership_census():
    groups, rows, specs = authority()
    assert len(groups) == 108
    assert len(rows) == 3240
    assert len(specs) == 3024
    assert {item["split"] for item in groups} == {"FIT", "SELECT"}
    assert len({item["group_id"] for item in groups}) == 108
    assert len({item["row_id"] for item in rows}) == 3240
    assert len({item["sequence_id"] for item in specs}) == 3024


def test_sha_cluster_draws_and_quantiles_are_literal():
    values = {"g2": [4., 6.], "g0": [0., 2.], "g1": [2., 4.]}
    traces = {}
    result = R.bootstrap(values, "literal:test", traces, two_sided=True, replicates=53)
    ordered, means, indices = sorted(values), [], []
    for replicate in range(53):
        chosen, draw_indices = [], []
        for draw in range(3):
            payload = f"{R.BOOTSTRAP_NAMESPACE}:literal:test:{replicate}:{draw}".encode()
            index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 3
            draw_indices.append(index)
            chosen.extend(values[ordered[index]])
        indices.append(draw_indices)
        means.append(sum(chosen) / len(chosen))
    assert result["ordered_group_ids"] == ordered
    assert result["lower95"] == float(np.quantile(means, .025, method="lower"))
    assert result["upper95"] == float(np.quantile(means, .975, method="higher"))
    draw_bytes = np.asarray(indices, dtype=">u2").tobytes(order="C")
    stat_bytes = np.asarray(means, dtype=">f8").tobytes(order="C")
    assert traces["literal:test"]["draw_matrix_sha256"] == hashlib.sha256(draw_bytes).hexdigest()
    assert traces["literal:test"]["statistic_vector_sha256"] == hashlib.sha256(stat_bytes).hexdigest()


def test_all_86_bootstrap_cells_and_complete_raw_evidence_exist():
    groups, rows, specs = authority()
    measurements = R.planted_measurements(specs, rows)
    raw = R.reconstruct_raw(groups, rows, specs, measurements)
    reports, traces = R.score(raw, replicates=7)
    assert reports["verdict"] == "held_capability_screen"
    assert len(raw["sequence_measurements"]) == 3024
    assert len(raw["row_measurements"]) == 3240
    assert len(raw["group_factorial_measurements"]) == 108
    assert len(raw["group_condition_effect_measurements"]) == 432
    assert len(traces) == 86
    expected = {
        f"{split}:factorial:{condition}:correct_margin"
        for split in R.SPLITS for condition in R.CONDITIONS
    } | {
        f"{split}:selector_payload_interaction" for split in R.SPLITS
    } | {
        f"{split}:control:{label}:{condition}:{endpoint}"
        for split in R.SPLITS for label in R.CONTROL_FAMILIES
        for condition in R.CONDITIONS for endpoint in ("base", "donor")
    } | {
        f"{split}:{metric}" for split in R.SPLITS
        for metric in ("selected_match_drop", "selected_vs_neutral_gap")
    } | {
        f"{split}:contrast_source:{condition}"
        for split in R.SPLITS for condition in R.CONDITIONS
    }
    assert set(traces) == expected


def test_held_and_scientific_null_fixtures_both_audit_cleanly():
    groups, rows, specs = authority()
    held = R.fixture_result(groups, rows, specs, make_null=False, replicates=11)
    null = R.fixture_result(groups, rows, specs, make_null=True, replicates=11)
    held_audit = R.audit_payload(held, groups, rows, specs, replicates=11)
    null_audit = R.audit_payload(null, groups, rows, specs, replicates=11)
    assert held_audit["audit_verdict"] == "held_independent_audit"
    assert held_audit["independently_recomputed_scientific_verdict"] == "held_capability_screen"
    assert null_audit["audit_verdict"] == "held_independent_audit"
    assert null_audit["independently_recomputed_scientific_verdict"] == "scientific_null"
    assert "factorial:SELECT:s0p0" in null_audit["independently_recomputed_failed_clauses"]


def test_raw_tamper_and_price_tamper_fail_the_audit():
    groups, rows, specs = authority()
    result = R.fixture_result(groups, rows, specs, make_null=False, replicates=7)
    raw_tamper = copy.deepcopy(result)
    raw_tamper["raw_evidence"]["row_measurements"][0]["base_margin"] += .01
    audit = R.audit_payload(raw_tamper, groups, rows, specs, replicates=7)
    assert audit["audit_verdict"] == "failed_independent_audit"
    assert any(item.startswith("raw_evidence") for item in audit["audit_failures"])
    price_tamper = copy.deepcopy(result)
    price_tamper["model_forwards"] = 94
    audit = R.audit_payload(price_tamper, groups, rows, specs, replicates=7)
    assert audit["audit_verdict"] == "failed_independent_audit"
    assert "envelope.model_forwards:value" in audit["audit_failures"]


def test_receipt_checks_result_bytes_and_full_envelope():
    groups, rows, specs = authority()
    result = R.fixture_result(groups, rows, specs, make_null=False, replicates=5)
    result_bytes = (json.dumps(result, indent=1) + "\n").encode()
    receipt = {
        "schema": "induction_selector_payload_native_capability_rung580_receipt_v1",
        "result_path": "basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_rung580_results.json",
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "implementation_sha256": R.AUTHORITY_HASHES[R.R580_SCRIPT],
        "test_sha256": R.AUTHORITY_HASHES[R.R580_TEST],
        "preregistration_sha256": R.AUTHORITY_HASHES[R.R580_PREREG],
        "input_sha256": R.R580_INPUT_HASHES,
        "checkpoint_weights_sha256": R.CHECKPOINT_SHA256,
        "verdict": result["verdict"], "model_forwards": 95, "model_backwards": 0,
        "evaluated_splits": ["FIT", "SELECT"], "forbidden_splits_opened": [],
    }
    assert R.audit_receipt(result, result_bytes, receipt) == []
    receipt["result_sha256"] = "0" * 64
    assert R.audit_receipt(result, result_bytes, receipt) == ["receipt.result_sha256:value"]


def test_dryrun_does_not_touch_future_result(monkeypatch, tmp_path):
    class Bomb:
        def __getattr__(self, name):
            raise AssertionError(f"future result was touched through {name}")

    monkeypatch.setattr(R, "R580_RESULT", Bomb())
    monkeypatch.setattr(R, "R580_RECEIPT", Bomb())
    monkeypatch.setattr(R, "DRYRUN", tmp_path / "dryrun.json")
    receipt = R.run_dryrun()
    assert receipt["future_result_opened"] is False
    assert receipt["held_fixture_audit_verdict"] == "held_independent_audit"
    assert receipt["null_fixture_scientific_verdict"] == "scientific_null"
    assert receipt["model_forwards"] == 0
