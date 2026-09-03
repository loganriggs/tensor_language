"""Focused regression tests for the generic CPU-only result contract."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import pytest


PATH = Path(__file__).with_name("result_contract.py")
SPEC = importlib.util.spec_from_file_location("result_contract", PATH)
assert SPEC and SPEC.loader
R = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = R
SPEC.loader.exec_module(R)

ZERO = "0" * 64
ONE = "1" * 64


def authority():
    return [
        {"row_id": "fit-a", "group_id": "group-a", "family_id": "target", "split": "FIT"},
        {"row_id": "fit-b", "group_id": "group-b", "family_id": "control", "split": "FIT"},
        {"row_id": "select-a", "group_id": "group-a", "family_id": "target", "split": "SELECT"},
        {"row_id": "test-a", "group_id": "group-a", "family_id": "target", "split": "FINAL_TEST"},
    ]


def evidence():
    return copy.deepcopy(authority()[:2])


def payload():
    return {
        "schema": "example_result_v1",
        "terminal_verdict": "fit_pass",
        "next_step": "open SELECT",
        "evaluated_splits": ["FIT"],
        "model_forwards": 12,
        "model_backwards": 0,
        "weights_updated": False,
        "metrics": {"mean_effect": 0.4, "null_effect": None},
        "input_sha256": {"rows": ZERO, "prereg": ONE},
    }


def contract():
    return R.ResultContract(
        opened_splits=("FIT",),
        allowed_splits=("FIT", "SELECT"),
        forbidden_splits=("FINAL_TEST",),
        min_model_forwards=10,
        max_model_forwards=20,
        exact_model_forwards=12,
        field_types={
            "schema": "string",
            "terminal_verdict": "scalar",
            "next_step": "string",
            "evaluated_splits": "list",
            "metrics": "dict",
            "metrics.mean_effect": "number",
            "metrics.null_effect": "null",
        },
        required_provenance=("rows", "prereg"),
        expected_provenance={"rows": ZERO, "prereg": ONE},
        group_fields=("group_id", "family_id"),
    )


def test_complete_contract_passes_without_mutating_inputs():
    result, raw, rows = payload(), evidence(), authority()
    before = copy.deepcopy((result, raw, rows))
    summary = R.validate_result_contract(result, raw, rows, contract())
    assert summary == {
        "rows": 2,
        "groups": 2,
        "opened_splits": ["FIT"],
        "model_forwards": 12,
        "model_backwards": 0,
        "weights_updated": False,
        "provenance_keys": ["prereg", "rows"],
        "canonical_payload_bytes": len(R.validate_standard_json(result).encode("utf-8")),
    }
    assert (result, raw, rows) == before


def test_r580_singleton_list_in_declared_scalar_field_is_rejected():
    result = payload()
    result["next_step"] = ["open SELECT"]
    with pytest.raises(R.ContractError, match=r"next_step: expected declared string, got list"):
        R.validate_result_contract(result, evidence(), authority(), contract())


def test_r584_missing_group_and_its_rows_are_rejected():
    missing_group_b = [evidence()[0]]
    with pytest.raises(R.ContractError, match=r"exact row membership mismatch.*fit-b"):
        R.validate_result_contract(payload(), missing_group_b, authority(), contract())


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_infinity_or_nan_hidden_in_null_path_is_rejected(bad):
    result = payload()
    result["metrics"]["null_effect"] = bad
    with pytest.raises(R.ContractError, match=r"finite standard JSON"):
        R.validate_result_contract(result, evidence(), authority(), contract())


def test_nonfinite_raw_evidence_is_also_rejected():
    raw = evidence()
    raw[0]["effect"] = float("inf")
    with pytest.raises(R.ContractError, match=r"finite standard JSON"):
        R.validate_result_contract(payload(), raw, authority(), contract())


def test_split_declaration_and_observed_split_must_both_close():
    result = payload()
    result["evaluated_splits"] = ["SELECT"]
    with pytest.raises(R.ContractError, match="opened split declaration mismatch"):
        R.validate_result_contract(result, evidence(), authority(), contract())
    wrong_split = evidence()
    wrong_split[0]["split"] = "FINAL_TEST"
    with pytest.raises(R.ContractError, match="observed split closure mismatch"):
        R.validate_result_contract(payload(), wrong_split, authority(), contract())


def test_authority_group_reassignment_and_duplicate_rows_are_rejected():
    wrong_group = evidence()
    wrong_group[0]["group_id"] = "group-b"
    with pytest.raises(R.ContractError, match="group_id disagrees with authority"):
        R.validate_result_contract(payload(), wrong_group, authority(), contract())
    duplicate = evidence() + [copy.deepcopy(evidence()[0])]
    with pytest.raises(R.ContractError, match="duplicate evidence row ID"):
        R.validate_result_contract(payload(), duplicate, authority(), contract())


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("model_forwards", 21, "outside declared envelope"),
        ("model_backwards", 1, "expected 0"),
        ("weights_updated", True, "expected False"),
    ],
)
def test_model_call_and_weight_update_envelope_is_enforced(field, bad, message):
    result = payload()
    result[field] = bad
    with pytest.raises(R.ContractError, match=message):
        R.validate_result_contract(result, evidence(), authority(), contract())


def test_required_provenance_must_be_present_well_formed_and_exact():
    missing = payload()
    del missing["input_sha256"]["prereg"]
    with pytest.raises(R.ContractError, match="missing required provenance"):
        R.validate_result_contract(missing, evidence(), authority(), contract())
    malformed = payload()
    malformed["input_sha256"]["prereg"] = "not-a-sha"
    with pytest.raises(R.ContractError, match="not lowercase SHA-256"):
        R.validate_result_contract(malformed, evidence(), authority(), contract())
    mismatch = payload()
    mismatch["input_sha256"]["prereg"] = "2" * 64
    with pytest.raises(R.ContractError, match="provenance hash mismatch: prereg"):
        R.validate_result_contract(mismatch, evidence(), authority(), contract())


def test_strict_json_rejects_tuple_and_non_string_mapping_keys():
    with pytest.raises(R.ContractError, match="tuple is not a literal JSON type"):
        R.validate_standard_json({"not_a_json_array": (1, 2)})
    with pytest.raises(R.ContractError, match="JSON object key"):
        R.validate_standard_json({1: "not a JSON object key"})
