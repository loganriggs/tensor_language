"""CPU tests for staged, hash-bound native capability licensing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import native_capability_license as license_gate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate(tmp_path: Path):
    authority = tmp_path / "authority.json"
    authority.write_text('{"rows":[1,2]}\n')
    gate = license_gate.CapabilityGate(
        capability_id="example.native_capability.v1",
        authority_path=authority,
        expected_authority_file_sha256=_sha(authority),
        authority_logical_sha256="a" * 64,
        cells=(
            license_gate.CapabilityCell("A/past", 2, 1.0),
            license_gate.CapabilityCell("A/present", 2, .5),
        ),
    )
    return gate, authority


def _evidence(*, fail=False):
    rows = []
    for cell, correct in (("A/past", (True, not fail)),
                          ("A/present", (True, False))):
        for index, value in enumerate(correct):
            rows.append({"example_id": f"{cell}:{index}", "cell_id": cell,
                         "correct": value, "full_vocab_CE": .5 + index,
                         "answer_minus_foil_margin": 1.0 if value else -.1})
    return rows


def test_passing_result_emits_deterministic_bound_license_and_preflights(tmp_path):
    gate, _ = _gate(tmp_path)
    result_path, license_path = tmp_path / "result.json", tmp_path / "license.json"
    result, result_sha = license_gate.finalize_native_capability(
        gate, _evidence(), result_path)
    assert result["terminal"] == "pass"
    assert result_sha == _sha(result_path)
    license_value, license_sha = license_gate.issue_capability_license(
        gate, result_path, license_path, causal_candidate_id="causal.successor.v1")
    assert license_sha == _sha(license_path)
    assert "timestamp" not in license_value
    assert license_value["capability_result_sha256"] == result_sha
    assert license_gate.validate_causal_preflight(
        gate, result_path, license_path, expected_license_sha256=license_sha,
        causal_candidate_id="causal.successor.v1") == license_value


def test_failed_cell_finalizes_but_cannot_emit_license(tmp_path):
    gate, _ = _gate(tmp_path)
    result_path, license_path = tmp_path / "failed.json", tmp_path / "license.json"
    result, _ = license_gate.finalize_native_capability(gate, _evidence(fail=True), result_path)
    assert result["terminal"] == "fail"
    with pytest.raises(license_gate.CapabilityLicenseError, match="failed"):
        license_gate.issue_capability_license(
            gate, result_path, license_path, causal_candidate_id="causal.successor.v1")
    assert not license_path.exists()


def test_preflight_rejects_missing_or_hash_mismatched_license(tmp_path):
    gate, _ = _gate(tmp_path)
    result_path, license_path = tmp_path / "result.json", tmp_path / "license.json"
    license_gate.finalize_native_capability(gate, _evidence(), result_path)
    with pytest.raises(license_gate.CapabilityLicenseError, match="license"):
        license_gate.validate_causal_preflight(
            gate, result_path, license_path, expected_license_sha256="0" * 64,
            causal_candidate_id="causal.successor.v1")
    _, digest = license_gate.issue_capability_license(
        gate, result_path, license_path, causal_candidate_id="causal.successor.v1")
    with pytest.raises(license_gate.CapabilityLicenseError, match="hash changed"):
        license_gate.validate_causal_preflight(
            gate, result_path, license_path, expected_license_sha256="1" * 64,
            causal_candidate_id="causal.successor.v1")
    assert digest != "1" * 64


def test_changed_authority_invalidates_existing_result_and_license(tmp_path):
    gate, authority = _gate(tmp_path)
    result_path, license_path = tmp_path / "result.json", tmp_path / "license.json"
    license_gate.finalize_native_capability(gate, _evidence(), result_path)
    _, digest = license_gate.issue_capability_license(
        gate, result_path, license_path, causal_candidate_id="causal.successor.v1")
    authority.write_text('{"rows":[1,2,3]}\n')
    with pytest.raises(license_gate.CapabilityLicenseError, match="authority file hash changed"):
        license_gate.validate_causal_preflight(
            gate, result_path, license_path, expected_license_sha256=digest,
            causal_candidate_id="causal.success.v1")


def test_preflight_rejects_wrong_causal_binding_and_forged_pass(tmp_path):
    gate, _ = _gate(tmp_path)
    result_path, license_path = tmp_path / "result.json", tmp_path / "license.json"
    license_gate.finalize_native_capability(gate, _evidence(), result_path)
    _, digest = license_gate.issue_capability_license(
        gate, result_path, license_path, causal_candidate_id="causal.successor.v1")
    with pytest.raises(license_gate.CapabilityLicenseError, match="bindings"):
        license_gate.validate_causal_preflight(
            gate, result_path, license_path, expected_license_sha256=digest,
            causal_candidate_id="different.causal.v1")

    failed_path = tmp_path / "failed.json"
    failed, _ = license_gate.finalize_native_capability(gate, _evidence(fail=True), failed_path)
    failed["terminal"] = "pass"
    failed_path.unlink()
    failed_path.write_text(json.dumps(failed, sort_keys=True) + "\n")
    with pytest.raises(license_gate.CapabilityLicenseError, match="not reproducible"):
        license_gate.issue_capability_license(
            gate, failed_path, tmp_path / "forged-license.json",
            causal_candidate_id="causal.successor.v1")


def test_authority_is_verified_before_native_evaluator_runs(tmp_path):
    gate, authority = _gate(tmp_path)
    authority.write_text("changed\n")
    calls = []
    with pytest.raises(license_gate.CapabilityLicenseError, match="authority file hash changed"):
        license_gate.evaluate_and_finalize_native_capability(
            gate, lambda: calls.append(True) or _evidence(), tmp_path / "result.json")
    assert calls == []


def test_incomplete_extra_or_nonfinite_evidence_fails_closed(tmp_path):
    gate, _ = _gate(tmp_path)
    with pytest.raises(license_gate.CapabilityLicenseError, match="expected 2"):
        license_gate.finalize_native_capability(gate, _evidence()[:-1], tmp_path / "short.json")
    extra = _evidence() + [{"example_id": "x", "cell_id": "unknown", "correct": True,
                            "full_vocab_CE": 1.0, "answer_minus_foil_margin": 1.0}]
    with pytest.raises(license_gate.CapabilityLicenseError, match="outside"):
        license_gate.finalize_native_capability(gate, extra, tmp_path / "extra.json")
    bad = _evidence(); bad[0] = dict(bad[0], full_vocab_CE=float("nan"))
    with pytest.raises(license_gate.CapabilityLicenseError, match="finite"):
        license_gate.finalize_native_capability(gate, bad, tmp_path / "nan.json")


def test_gate_and_candidate_identifiers_fail_closed(tmp_path):
    gate, _ = _gate(tmp_path)
    bad_gate = license_gate.CapabilityGate(
        capability_id="contains spaces", authority_path=gate.authority_path,
        expected_authority_file_sha256=gate.expected_authority_file_sha256,
        authority_logical_sha256=gate.authority_logical_sha256, cells=gate.cells)
    with pytest.raises(license_gate.CapabilityLicenseError, match="capability_id"):
        license_gate.finalize_native_capability(bad_gate, _evidence(), tmp_path / "bad.json")

    result_path = tmp_path / "result.json"
    license_gate.finalize_native_capability(gate, _evidence(), result_path)
    with pytest.raises(license_gate.CapabilityLicenseError, match="causal candidate"):
        license_gate.issue_capability_license(
            gate, result_path, tmp_path / "license.json", causal_candidate_id="bad candidate")


def test_non_sequence_evidence_and_boolean_threshold_fail_closed(tmp_path):
    gate, _ = _gate(tmp_path)
    with pytest.raises(license_gate.CapabilityLicenseError, match="list or tuple"):
        license_gate.finalize_native_capability(gate, {}, tmp_path / "bad.json")
    bad_gate = license_gate.CapabilityGate(
        capability_id=gate.capability_id, authority_path=gate.authority_path,
        expected_authority_file_sha256=gate.expected_authority_file_sha256,
        authority_logical_sha256=gate.authority_logical_sha256,
        cells=(license_gate.CapabilityCell("A", 1, True),))
    with pytest.raises(license_gate.CapabilityLicenseError, match="cell is invalid"):
        license_gate.finalize_native_capability(bad_gate, [], tmp_path / "threshold.json")
