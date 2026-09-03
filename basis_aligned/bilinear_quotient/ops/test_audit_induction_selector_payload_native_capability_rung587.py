"""Focused CPU-only tests for the pre-outcome R587 auditor."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


PATH = Path(__file__).with_name(
    "audit_induction_selector_payload_native_capability_rung587.py"
)
SPEC = importlib.util.spec_from_file_location("audit_r587", PATH)
assert SPEC and SPEC.loader
R = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = R
SPEC.loader.exec_module(R)


@pytest.fixture(scope="module")
def authority():
    return R.load_authority()


@pytest.fixture(scope="module")
def fixtures(authority):
    low, groups, rows, specs = authority
    held = R.fixture_result(low, groups, rows, specs, make_null=False, replicates=11)
    null = R.fixture_result(low, groups, rows, specs, make_null=True, replicates=11)
    return held, null


def test_frozen_preoutcome_authority_and_no_future_result():
    observed = R.verify_preoutcome_authority(require_future_absent=True)
    assert observed == {
        str(path): digest for path, digest in R.PREOUTCOME_AUTHORITY_HASHES.items()
    }
    assert not R.R586_RESULT.exists()
    assert not R.R586_RECEIPT.exists()
    source = PATH.read_text()
    assert "_load_module(R586_SCRIPT" not in source
    assert "from induction_selector_payload_native_capability_rung586" not in source


def test_authority_census_membership_and_split_closure(authority):
    _, groups, rows, specs = authority
    assert (len(groups), len(rows), len(specs)) == (108, 3240, 3024)
    assert {item["split"] for item in groups} == {"FIT", "SELECT"}
    assert not ({item["split"] for item in groups} & {"FINAL_TEST", "OOD"})
    assert len({item["group_id"] for item in groups}) == 108
    assert len({item["row_id"] for item in rows}) == 3240
    assert len({item["sequence_id"] for item in specs}) == 3024
    assert R.EXPECTED_FORWARDS == 95 == (3024 + 31) // 32


def test_literal_sha_bootstrap_and_86_cell_census(authority):
    low, groups, rows, specs = authority
    values = {"g2": [4.0, 6.0], "g0": [0.0, 2.0], "g1": [2.0, 4.0]}
    traces = {}
    report = low.bootstrap(values, "literal:r587", traces,
                           two_sided=True, replicates=53)
    ordered, indices, means = sorted(values), [], []
    for replicate in range(53):
        selected, row = [], []
        for draw in range(3):
            payload = (
                f"{low.BOOTSTRAP_NAMESPACE}:literal:r587:{replicate}:{draw}"
            ).encode()
            index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 3
            row.append(index)
            selected.extend(values[ordered[index]])
        indices.append(row)
        means.append(sum(selected) / len(selected))
    assert report["lower95"] == float(np.quantile(means, .025, method="lower"))
    assert report["upper95"] == float(np.quantile(means, .975, method="higher"))
    assert traces["literal:r587"]["draw_matrix_sha256"] == hashlib.sha256(
        np.asarray(indices, dtype=">u2").tobytes(order="C")
    ).hexdigest()
    assert traces["literal:r587"]["statistic_vector_sha256"] == hashlib.sha256(
        np.asarray(means, dtype=">f8").tobytes(order="C")
    ).hexdigest()

    raw = low.reconstruct_raw(
        groups, rows, specs, low.planted_measurements(specs, rows)
    )
    _, all_traces = low.score(raw, replicates=7)
    assert len(all_traces) == R.EXPECTED_BOOTSTRAP_CELLS == 86
    assert set(all_traces) == R.expected_bootstrap_cell_ids()
    assert all(item["replicates"] == 7 for item in all_traces.values())


def test_held_and_scientific_null_reconstruct_cleanly(authority, fixtures):
    _, groups, rows, specs = authority
    held, null = fixtures
    for result, expected in (
        (held, "held_capability_screen"), (null, "scientific_null")
    ):
        audit = R.audit_payload(result, groups, rows, specs, replicates=11)
        assert audit["audit_verdict"] == "held_independent_audit"
        assert audit["independently_recomputed_scientific_verdict"] == expected
        assert audit["raw_counts"] == {
            "sequences": 3024, "rows": 3240,
            "factorial_groups": 108, "condition_effects": 432,
        }
        assert audit["bootstrap_cell_count"] == 86
        assert audit["bootstrap_replicates_per_cell"] == 11
        assert audit["generic_contract"]["model_forwards"] == 95
        assert audit["generic_contract"]["model_backwards"] == 0
        assert audit["generic_contract"]["weights_updated"] is False
    assert null["failed_scientific_clauses"]


def test_list_next_step_missing_group_and_nonfinite_fail_closed(
    authority, fixtures
):
    _, groups, rows, specs = authority
    held, _ = fixtures

    listed = copy.deepcopy(held)
    listed["next_step"] = [R.HELD_NEXT_STEP]
    audit = R.audit_payload(listed, groups, rows, specs, replicates=5)
    assert audit["audit_verdict"] == "failed_independent_audit"
    assert "result.next_step must be str" in audit["audit_failures"][0]

    missing = copy.deepcopy(held)
    group_id = groups[0]["group_id"]
    missing["raw_evidence"]["sequence_measurements"] = [
        item for item in missing["raw_evidence"]["sequence_measurements"]
        if item["group_id"] != group_id
    ]
    audit = R.audit_payload(missing, groups, rows, specs, replicates=5)
    assert audit["audit_verdict"] == "failed_independent_audit"
    assert "sequence order differs" in audit["audit_failures"][0]

    nonfinite = copy.deepcopy(held)
    nonfinite["raw_evidence"]["row_measurements"][0]["base_margin"] = float("nan")
    audit = R.audit_payload(nonfinite, groups, rows, specs, replicates=5)
    assert audit["audit_verdict"] == "failed_independent_audit"
    assert "not finite standard JSON" in audit["audit_failures"][0]


def test_early_integrity_failure_has_complete_terminal_schema(authority, fixtures):
    _, groups, rows, specs = authority
    held, _ = fixtures
    malformed = copy.deepcopy(held)
    malformed["next_step"] = [R.HELD_NEXT_STEP]
    audit = R.audit_payload(malformed, groups, rows, specs, replicates=5)
    required = {
        "audit_verdict", "audit_failures",
        "independently_recomputed_scientific_verdict",
        "independently_recomputed_failed_clauses", "raw_counts",
        "membership_hashes", "bootstrap_cell_count",
        "bootstrap_replicates_per_cell", "bootstrap_algorithm",
        "bootstrap_algorithm_sha256", "bootstrap_trace_hash",
        "bootstrap_traces", "generic_contract", "recomputed_scores",
    }
    assert set(audit) == required
    assert audit["audit_verdict"] == "failed_independent_audit"
    assert audit["independently_recomputed_scientific_verdict"] is None
    assert audit["independently_recomputed_failed_clauses"] == []


def test_exact_result_types_split_price_and_hashes(fixtures):
    held, _ = fixtures
    assert R.validate_result_envelope(held)["rows"] == 3240
    mutations = {
        "rung": True,
        "elapsed_seconds": 0,
        "model_forwards": 94,
        "model_backwards": 1,
        "model_weights_updated": True,
        "evaluated_splits": ["FIT", "SELECT", "FINAL_TEST"],
        "forbidden_splits_opened": ["OOD"],
        "implementation_sha256": "0" * 64,
    }
    for field, value in mutations.items():
        candidate = copy.deepcopy(held)
        candidate[field] = value
        with pytest.raises((TypeError, ValueError)):
            R.validate_result_envelope(candidate)


def test_strict_parser_rejects_nonfinite_and_duplicate_keys():
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        R.strict_loads(b'{"x": NaN}', "bad")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        R.strict_loads(b'{"x": 1, "x": 2}', "bad")


def test_receipt_exact_types_and_byte_binding(fixtures):
    held, _ = fixtures
    result_bytes = (json.dumps(held, indent=1, allow_nan=False) + "\n").encode()
    receipt = R.fixture_receipt(held, result_bytes)
    assert R.validate_receipt(receipt, held, result_bytes) == []
    assert type(receipt["next_step"]) is str
    listed = copy.deepcopy(receipt)
    listed["next_step"] = [receipt["next_step"]]
    assert any("receipt.next_step must be str" in item for item in R.validate_receipt(
        listed, held, result_bytes
    ))
    assert "receipt.result_sha256:value_or_type" in R.validate_receipt(
        receipt, held, result_bytes + b" "
    )
    receipt_bytes = (json.dumps(receipt, indent=1, allow_nan=False) + "\n").encode()
    digest_bytes = receipt["result_sha256"].encode()
    assert receipt_bytes.count(digest_bytes) == 1
    changed_receipt_bytes = receipt_bytes.replace(digest_bytes, b"0" * 64)
    changed_receipt = R.strict_loads(changed_receipt_bytes, "tampered receipt")
    assert "receipt.result_sha256:value_or_type" in R.validate_receipt(
        changed_receipt, held, result_bytes
    )


def test_stable_source_pair_and_missing_half_fail_closed(tmp_path, fixtures):
    held, _ = fixtures
    result_bytes = (json.dumps(held, indent=1, allow_nan=False) + "\n").encode()
    receipt = R.fixture_receipt(held, result_bytes)
    result_path, receipt_path = tmp_path / "result.json", tmp_path / "receipt.json"
    result_path.write_bytes(result_bytes)
    with pytest.raises(RuntimeError, match="must both exist"):
        R.read_stable_source_pair(result_path, receipt_path)
    receipt_bytes = (json.dumps(receipt, indent=1, allow_nan=False) + "\n").encode()
    receipt_path.write_bytes(receipt_bytes)
    assert R.read_stable_source_pair(result_path, receipt_path) == (
        result_bytes, receipt_bytes
    )


def test_dryrun_keeps_future_pair_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "R586_RESULT", tmp_path / "future_result.json")
    monkeypatch.setattr(R, "R586_RECEIPT", tmp_path / "future_receipt.json")
    monkeypatch.setattr(R, "DRYRUN", tmp_path / "r587_dryrun.json")
    receipt = R.run_dryrun()
    assert receipt["status"] == "dryrun_passed"
    assert receipt["held_fixture_audit_verdict"] == "held_independent_audit"
    assert receipt["null_fixture_scientific_verdict"] == "scientific_null"
    assert receipt["bootstrap_cells_per_fixture"] == 86
    assert receipt["real_audit_bootstrap_replicates"] == 2000
    assert receipt["list_next_step_rejected"] is True
    assert receipt["missing_group_rejected"] is True
    assert receipt["nonfinite_nested_value_rejected"] is True
    assert receipt["tampered_receipt_bytes_rejected"] is True
    assert receipt["changed_result_bytes_rejected"] is True
    assert receipt["model_loaded"] is False
    assert receipt["model_forwards"] == receipt["model_backwards"] == 0
    assert receipt["model_weights_updated"] is False
    assert not R.R586_RESULT.exists() and not R.R586_RECEIPT.exists()


def test_real_audit_is_hard_coded_to_2000_bootstraps_and_zero_model_calls():
    source = PATH.read_text()
    assert "audit_payload(result, groups, rows, specs, replicates=BOOTSTRAPS)" in source
    assert R.BOOTSTRAPS == 2000
    assert "import torch" not in source
    assert "load_bilin18" not in source
