"""Focused model-free tests for the prospective R586 clean replication."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


PATH = Path(__file__).with_name(
    "induction_selector_payload_native_capability_rung586.py"
)
SPEC = importlib.util.spec_from_file_location("native_capability_r586", PATH)
assert SPEC and SPEC.loader
R = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = R
SPEC.loader.exec_module(R)


@pytest.fixture(scope="module")
def fixtures():
    held, r580 = R.make_fixture_result(make_null=False)
    null, _ = R.make_fixture_result(make_null=True)
    return held, null, r580


def wrong_type(expected: type):
    return {
        str: ["not-a-string"],
        int: True,
        bool: 1,
        dict: [],
        list: (),
        float: 0,
    }[expected]


def test_frozen_authority_and_literal_price_are_exact():
    observed = R.verify_authority()
    assert observed == {str(path): digest for path, digest in R.AUTHORITY_HASHES.items()}
    r580, groups, rows, specs = R.load_authority()
    assert (len(groups), len(rows), len(specs)) == (108, 3240, 3024)
    assert R.EXPECTED_FORWARDS == 95 == (3024 + 31) // 32
    assert r580.SPLITS == ("FIT", "SELECT")
    assert r580.FORBIDDEN_SPLITS == ("FINAL_TEST", "OOD")
    assert r580.BOOTSTRAPS == 2000
    assert r580.BOOTSTRAP_NAMESPACE == "a8-r580-group-bootstrap-v1"
    assert R.AUTHORITY_HASHES[R.RESULT_CONTRACT] == (
        "af8fb9557dcb77e038319b0fffa919927f3925497a0edafe27fc951125dfb272"
    )
    assert R.AUTHORITY_HASHES[R.RESULT_CONTRACT_TEST] == (
        "2f26e3125e1208b9b7e9f1b138cfc90921157143303f098f853d3f65432f0645"
    )
    assert R.AUTHORITY_HASHES[R.RESULT_CONTRACT_USAGE] == (
        "4b2ed9bc32ed5cd5e4151bc39d3a7a6a83fa8498a97b7ff1e928a82d6c8ac304"
    )


def test_new_namespace_cannot_alias_or_overwrite_old_artifacts():
    assert R.OUT.name == "induction_selector_payload_native_capability_rung586_results.json"
    assert R.OUT_RECEIPT.name == "induction_selector_payload_native_capability_rung586_receipt.json"
    assert R.DRYRUN.name == "induction_selector_payload_native_capability_rung586_dryrun.json"
    assert R.OUT not in R.AUTHORITY_HASHES
    assert R.OUT_RECEIPT not in R.AUTHORITY_HASHES
    assert {R.R580_RESULT, R.R580_RECEIPT, R.R581_AUDIT} <= set(R.AUTHORITY_HASHES)


def test_held_and_null_use_exact_r580_scientific_scores(fixtures):
    held, null, r580 = fixtures
    for result in (held, null):
        rescored = r580.score_raw_evidence(result["raw_evidence"])
        assert all(result[key] == value for key, value in rescored.items())
        summary = R.validate_result_envelope(result)
        assert summary["rows"] == 3240
        assert summary["groups"] == 108
        assert summary["opened_splits"] == ["FIT", "SELECT"]
        assert summary["model_forwards"] == 95
        assert summary["model_backwards"] == 0
        assert summary["weights_updated"] is False
    assert held["verdict"] == "held_capability_screen"
    assert held["failed_scientific_clauses"] == []
    assert null["verdict"] == "scientific_null"
    assert null["failed_scientific_clauses"]


def test_next_step_is_a_scalar_string_in_both_terminal_paths(fixtures):
    held, null, _ = fixtures
    assert type(held["next_step"]) is str
    assert held["next_step"] == R.HELD_NEXT_STEP
    assert type(null["next_step"]) is str
    assert null["next_step"] == R.NULL_NEXT_STEP
    assert type(json.loads(json.dumps(held))["next_step"]) is str
    assert type(json.loads(json.dumps(null))["next_step"]) is str


@pytest.mark.parametrize("malformed", [
    (R.HELD_NEXT_STEP,),
    [R.HELD_NEXT_STEP],
    {"value": R.HELD_NEXT_STEP},
])
def test_tuple_list_and_mapping_next_step_are_rejected(fixtures, malformed):
    held, _, _ = fixtures
    candidate = copy.deepcopy(held)
    candidate["next_step"] = malformed
    with pytest.raises(TypeError, match="result.next_step must be str"):
        R.validate_result_envelope(candidate)


@pytest.mark.parametrize("field", tuple(R.RESULT_FIELD_TYPES))
def test_every_result_envelope_field_has_an_exact_type(fixtures, field):
    held, _, _ = fixtures
    candidate = copy.deepcopy(held)
    candidate[field] = wrong_type(R.RESULT_FIELD_TYPES[field])
    with pytest.raises(TypeError, match=f"result.{field}"):
        R.validate_result_envelope(candidate)


def test_missing_and_extra_result_fields_are_rejected(fixtures):
    held, _, _ = fixtures
    missing = copy.deepcopy(held)
    missing.pop("next_step")
    with pytest.raises(TypeError, match="result fields changed"):
        R.validate_result_envelope(missing)
    extra = copy.deepcopy(held)
    extra["legacy_next_step"] = [R.HELD_NEXT_STEP]
    with pytest.raises(TypeError, match="result fields changed"):
        R.validate_result_envelope(extra)


def test_result_value_checks_cover_price_split_hash_and_decision(fixtures):
    held, _, _ = fixtures
    mutations = {
        "model_forwards": 94,
        "model_backwards": 1,
        "model_weights_updated": True,
        "unique_sequences": 3023,
        "checkpoint_weights_sha256": "0" * 64,
        "evaluated_splits": ["FIT", "SELECT", "FINAL_TEST"],
        "forbidden_splits_opened": ["OOD"],
        "verdict": "scientific_null",
        "next_step": R.NULL_NEXT_STEP,
    }
    for field, value in mutations.items():
        candidate = copy.deepcopy(held)
        candidate[field] = value
        with pytest.raises(ValueError):
            R.validate_result_envelope(candidate)


def test_generic_contract_rejects_membership_split_provenance_and_hidden_nonfinite(fixtures):
    held, _, _ = fixtures
    membership = copy.deepcopy(held)
    membership["raw_evidence"]["row_measurements"][0]["row_id"] = "not-authority"
    with pytest.raises(ValueError, match="exact row membership mismatch"):
        R.validate_result_envelope(membership)

    split = copy.deepcopy(held)
    split["raw_evidence"]["row_measurements"][0]["split"] = "FINAL_TEST"
    with pytest.raises(ValueError, match="observed split closure mismatch"):
        R.validate_result_envelope(split)

    provenance = copy.deepcopy(held)
    provenance["input_sha256"].pop(str(R.RESULT_CONTRACT))
    with pytest.raises(ValueError):
        R.validate_result_envelope(provenance)

    nonfinite = copy.deepcopy(held)
    nonfinite["raw_evidence"]["row_measurements"][0]["base_margin"] = float("nan")
    with pytest.raises(ValueError, match="not finite standard JSON"):
        R.validate_result_envelope(nonfinite)


def test_every_receipt_field_has_an_exact_type(fixtures):
    held, _, _ = fixtures
    encoded = (json.dumps(held, indent=1) + "\n").encode()
    receipt = R.make_receipt(held, encoded)
    for field, expected in R.RECEIPT_FIELD_TYPES.items():
        candidate = copy.deepcopy(receipt)
        candidate[field] = wrong_type(expected)
        with pytest.raises(TypeError, match=f"receipt.{field}"):
            R.validate_receipt_envelope(candidate, held, encoded)


def test_receipt_binds_result_and_scalar_next_step(fixtures):
    held, _, _ = fixtures
    encoded = (json.dumps(held, indent=1) + "\n").encode()
    receipt = R.make_receipt(held, encoded)
    assert type(receipt["next_step"]) is str
    assert receipt["next_step"] == held["next_step"]
    tampered = bytearray(encoded)
    tampered[-2] = ord(" ")
    with pytest.raises(ValueError, match="receipt.result_sha256 changed"):
        R.validate_receipt_envelope(receipt, held, bytes(tampered))


def test_scientific_writer_rejects_malformed_envelope_before_writing(
    fixtures, monkeypatch, tmp_path
):
    held, _, _ = fixtures
    out = tmp_path / "r586_result.json"
    receipt = tmp_path / "r586_receipt.json"
    monkeypatch.setattr(R, "OUT", out)
    monkeypatch.setattr(R, "OUT_RECEIPT", receipt)
    malformed = copy.deepcopy(held)
    malformed["next_step"] = [R.HELD_NEXT_STEP]
    with pytest.raises(TypeError, match="result.next_step must be str"):
        R.write_scientific_result(malformed)
    assert not out.exists()
    assert not receipt.exists()


def test_authority_tamper_is_a_hard_error(monkeypatch):
    changed = dict(R.AUTHORITY_HASHES)
    changed[R.R581_AUDIT] = "0" * 64
    monkeypatch.setattr(R, "AUTHORITY_HASHES", changed)
    with pytest.raises(RuntimeError, match="frozen authority mismatch"):
        R.verify_authority()


def test_dryrun_is_model_free_preserves_old_files_and_keeps_future_closed(
    monkeypatch, tmp_path
):
    old_before = {path: R.sha256(path) for path in R.AUTHORITY_HASHES}
    dryrun = tmp_path / "r586_dryrun.json"
    future_result = tmp_path / "future_result.json"
    future_receipt = tmp_path / "future_receipt.json"
    monkeypatch.setattr(R, "DRYRUN", dryrun)
    monkeypatch.setattr(R, "OUT", future_result)
    monkeypatch.setattr(R, "OUT_RECEIPT", future_receipt)
    result = R.run_dryrun()
    assert result["status"] == "dryrun_passed"
    assert result["r580_scientific_scores_exact"] is True
    assert result["generic_result_contract_held"]["rows"] == 3240
    assert result["generic_result_contract_null"]["groups"] == 108
    assert result["tuple_next_step_rejected"] is True
    assert result["list_next_step_rejected"] is True
    assert result["old_artifacts_immutable"] is True
    assert result["model_loaded"] is False
    assert result["model_forwards"] == result["model_backwards"] == 0
    assert result["model_weights_updated"] is False
    assert result["evaluated_splits"] == ["FIT", "SELECT"]
    assert result["forbidden_splits_opened"] == []
    assert result["future_result_written"] is False
    assert result["future_receipt_written"] is False
    assert dryrun.is_file()
    assert not future_result.exists()
    assert not future_receipt.exists()
    assert old_before == {path: R.sha256(path) for path in R.AUTHORITY_HASHES}


def test_no_scientific_outputs_exist_pre_run():
    assert not R.OUT.exists()
    assert not R.OUT_RECEIPT.exists()
