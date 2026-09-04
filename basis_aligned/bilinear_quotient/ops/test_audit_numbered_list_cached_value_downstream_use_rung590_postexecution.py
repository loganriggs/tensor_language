"""CPU-only tests for the independent R590 postexecution audit."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


PATH = Path(__file__).with_name(
    "audit_numbered_list_cached_value_downstream_use_rung590_postexecution.py"
)
SPEC = importlib.util.spec_from_file_location("r590_postexecution_audit_test_target", PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


@pytest.fixture(scope="module")
def modules():
    return audit.load_approved_modules()


@pytest.fixture(scope="module")
def planted(modules):
    _, producer = modules
    evidence = producer.evidence_from_legacy_payload(
        producer.r588.make_fixture(held=False, replicates=8)
    )
    evidence_bytes = audit.canonical_bytes(evidence)
    result = producer.build_result(
        evidence,
        evidence_sha256=audit.digest(evidence_bytes),
        checkpoint_sha256=producer.CHECKPOINT_SHA256,
        elapsed_seconds=1.0,
        replicates=8,
    )
    result_bytes = audit.canonical_bytes(result)
    receipt = audit.expected_receipt(result, result_bytes, evidence_bytes, producer)
    receipt_bytes = audit.canonical_bytes(receipt)
    return producer, result, result_bytes, receipt_bytes, evidence, evidence_bytes


def test_actual_package_independently_reconstructs_scientific_null(modules):
    _, producer = modules
    report = audit.audit_bytes(
        audit.RESULT.read_bytes(), audit.RECEIPT.read_bytes(), audit.EVIDENCE.read_bytes(),
        producer, bind_observed_hashes=True,
    )
    assert report["audit_passed"] is True
    assert report["scientific_terminal_valid"] is True
    assert report["instrument_invalid"] is False
    assert report["independently_recomputed_decision"] == \
        "downstream_use_decomposition_null"
    assert report["independently_recomputed_model_forwards"] == 379
    assert report["independently_recomputed_opened_splits"] == ["FIT"]
    assert report["raw_counts"] == {
        "fit_capture_rows": 576,
        "fit_real_arms": 12,
        "rows_per_fit_arm": [576],
        "fit_null_arms": 0,
        "select_real_arms": 0,
        "select_null_arms": 0,
    }
    assert report["bootstrap_cell_count"] == 432


def test_planted_valid_null_reconstructs_at_small_bootstrap(planted):
    producer, _, result_bytes, receipt_bytes, _, evidence_bytes = planted
    report = audit.audit_bytes(
        result_bytes, receipt_bytes, evidence_bytes, producer,
        bind_observed_hashes=False, replicates=8,
    )
    assert report["audit_passed"] is True
    assert report["bootstrap_cell_count"] == 432


def test_correlated_terminal_and_receipt_rewrite_is_rejected(planted):
    producer, result, _, _, _, evidence_bytes = planted
    changed = copy.deepcopy(result)
    changed["decision"] = "downstream_use_component_held"
    changed["next_step"] = "invented_postexecution_claim"
    result_bytes = audit.canonical_bytes(changed)
    receipt = audit.expected_receipt(changed, result_bytes, evidence_bytes, producer)
    report = audit.audit_bytes(
        result_bytes, audit.canonical_bytes(receipt), evidence_bytes, producer,
        bind_observed_hashes=False, replicates=8,
    )
    assert report["audit_passed"] is False
    assert report["scientific_terminal_valid"] is False


def test_missing_primitive_row_fails_even_after_all_outer_hashes_change(planted):
    producer, result, _, _, evidence, _ = planted
    changed_evidence = copy.deepcopy(evidence)
    first_arm = producer.SELECTION_NAMES[0]
    changed_evidence["fit_raw"][first_arm].pop()
    evidence_bytes = audit.canonical_bytes(changed_evidence)
    changed_result = copy.deepcopy(result)
    changed_result["evidence_descriptor"]["sha256"] = audit.digest(evidence_bytes)
    result_bytes = audit.canonical_bytes(changed_result)
    receipt = audit.expected_receipt(
        changed_result, result_bytes, evidence_bytes, producer
    )
    report = audit.audit_bytes(
        result_bytes, audit.canonical_bytes(receipt), evidence_bytes, producer,
        bind_observed_hashes=False, replicates=8,
    )
    assert report["audit_passed"] is False
    assert "membership" in report["audit_failures"][0]


def test_nonfinite_and_exact_outcome_byte_tampering_fail_closed(modules, planted):
    _, producer = modules
    report = audit.audit_bytes(
        audit.RESULT.read_bytes() + b"\n", audit.RECEIPT.read_bytes(),
        audit.EVIDENCE.read_bytes(), producer, bind_observed_hashes=True,
    )
    assert report["audit_passed"] is False
    assert "landed package" in report["audit_failures"][0]

    producer, _, result_bytes, receipt_bytes, _, evidence_bytes = planted
    nonfinite = evidence_bytes.replace(b'"r576_term_norm":', b'"r576_term_norm":NaN,"old":', 1)
    report = audit.audit_bytes(
        result_bytes, receipt_bytes, nonfinite, producer,
        bind_observed_hashes=False, replicates=8,
    )
    assert report["audit_passed"] is False
    assert "finite JSON" in report["audit_failures"][0]


def test_managed_error_is_after_complete_receipt_publication():
    runlog = audit.RUNLOG.read_bytes()
    assert audit.digest(runlog) == audit.RUNLOG_SHA256
    text = runlog.decode("utf-8")
    summary_position = text.index('"decision": "downstream_use_decomposition_null"')
    error_position = text.index("R590 verified scientific entry point unexpectedly returned")
    assert summary_position < error_position
    for path, expected in audit.OUTCOME_HASHES.items():
        assert audit.digest(path.read_bytes()) == expected
    receipt = json.loads(audit.RECEIPT.read_text())
    assert receipt["result_sha256"] == audit.OUTCOME_HASHES[audit.RESULT]
    assert receipt["evidence_sha256"] == audit.OUTCOME_HASHES[audit.EVIDENCE]

