#!/usr/bin/env python3
"""R587: independent, model-free audit of the future R586 result pair."""

# BQLANE: cpu

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Mapping, Sequence


os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
OPS = ROOT / "ops"

ROWS = ROOT / "induction_selector_payload_three_source_rows_rung578.json"
ROWS_RECEIPT = ROOT / "induction_selector_payload_three_source_rows_rung578_receipt.json"
ROWS_PREREG = POLY / "INDUCTION_SELECTOR_PAYLOAD_THREE_SOURCE_ROWS_RUNG578_PREREGISTRATION.md"
ROWS_BUILDER = OPS / "induction_selector_payload_three_source_rows_rung578.py"
ROWS_TEST = OPS / "test_induction_selector_payload_three_source_rows_rung578.py"
R580_PREREG = POLY / "INDUCTION_SELECTOR_PAYLOAD_NATIVE_CAPABILITY_RUNG580_PREREGISTRATION.md"
R580_SCRIPT = OPS / "induction_selector_payload_native_capability_rung580.py"
R580_TEST = OPS / "test_induction_selector_payload_native_capability_rung580.py"
R580_DRYRUN = ROOT / "induction_selector_payload_native_capability_rung580_dryrun.json"
R580_RESULT = ROOT / "induction_selector_payload_native_capability_rung580_results.json"
R580_RECEIPT = ROOT / "induction_selector_payload_native_capability_rung580_receipt.json"
R581_PREREG = POLY / "INDUCTION_SELECTOR_PAYLOAD_NATIVE_CAPABILITY_AUDIT_RUNG581_PREREGISTRATION.md"
R581_SCRIPT = OPS / "audit_induction_selector_payload_native_capability_rung581.py"
R581_TEST = OPS / "test_audit_induction_selector_payload_native_capability_rung581.py"
R581_DRYRUN = ROOT / "induction_selector_payload_native_capability_audit_rung581_dryrun.json"
R581_AUDIT = ROOT / "induction_selector_payload_native_capability_audit_rung581.json"
RESULT_CONTRACT = OPS / "result_contract.py"
RESULT_CONTRACT_TEST = OPS / "test_result_contract.py"
RESULT_CONTRACT_USAGE = OPS / "RESULT_CONTRACT_USAGE.md"
R586_PREREG = POLY / "INDUCTION_SELECTOR_PAYLOAD_NATIVE_CAPABILITY_RUNG586_PREREGISTRATION.md"
R586_SCRIPT = OPS / "induction_selector_payload_native_capability_rung586.py"
R586_TEST = OPS / "test_induction_selector_payload_native_capability_rung586.py"
R586_DRYRUN = ROOT / "induction_selector_payload_native_capability_rung586_dryrun.json"
R586_RESULT = ROOT / "induction_selector_payload_native_capability_rung586_results.json"
R586_RECEIPT = ROOT / "induction_selector_payload_native_capability_rung586_receipt.json"
PREREG = POLY / "INDUCTION_SELECTOR_PAYLOAD_NATIVE_CAPABILITY_AUDIT_RUNG587_PREREGISTRATION.md"
SCRIPT = Path(__file__)
TEST = SCRIPT.with_name("test_audit_induction_selector_payload_native_capability_rung587.py")
OUT = ROOT / "induction_selector_payload_native_capability_audit_rung587.json"
DRYRUN = ROOT / "induction_selector_payload_native_capability_audit_rung587_dryrun.json"

PREOUTCOME_AUTHORITY_HASHES = {
    ROWS: "8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6",
    ROWS_RECEIPT: "9e4e63ebd98503d6aa5daa27617a20fea595829c5a372f27b1ce4371d7c05b45",
    ROWS_PREREG: "276d801bbf5795e6421488dd4971b3a2d2dcb56e4fc7c4bc7ecdd2f61a73e9ce",
    ROWS_BUILDER: "d47bb3d46bd2c6061132c13b356e58ba9dfe2a56a2629f8b49a03f280d290bbd",
    ROWS_TEST: "9d795df358dfef9c5d17a539307f8e781f2a4debeb4909078858a242b3dfc512",
    R580_PREREG: "8f80926d0a90360a66ebce605732d32ff3e283a3428eb7245f4813a521d12580",
    R580_SCRIPT: "62d11395d845d663257433936773780dd4bb9ddbcb9286400c420dadd3a73249",
    R580_TEST: "9f166a61409c12d6a4a58e16640af654378151f99c05597f9c63dbb2dec64550",
    R580_DRYRUN: "3d21b62972aa0794598860228554068035af10fd743e8958bfc7a05d56d68588",
    R580_RESULT: "7c7463a95931a51cd848ff9e8033bed77a26f7889a1a5fd1a3512ec2d1224b84",
    R580_RECEIPT: "6a1ef728bca424ed27ec145adad1918923e91f190b96a9ff452b6838413b670a",
    R581_PREREG: "d2989383791cb179fecfa930742812cf8036a85bb9d2f3cfdd6555bb00640887",
    R581_SCRIPT: "812c28bd1987d0978cbf0c2b0d09f0669b159b515b8a4b3f8db5dd1a73663841",
    R581_TEST: "70782c35c4aac7089d363360de3f0365dfa19bdd9947c90ddc73ad3f096f1e93",
    R581_DRYRUN: "c6a8bb32ec0bfae17257507682b2b942be6cdf645afa93736e135a53174e14ea",
    R581_AUDIT: "8ecc1562632212ee876a794377e31966776ec15de02b5cb8d31798e438502cdb",
    RESULT_CONTRACT: "af8fb9557dcb77e038319b0fffa919927f3925497a0edafe27fc951125dfb272",
    RESULT_CONTRACT_TEST: "2f26e3125e1208b9b7e9f1b138cfc90921157143303f098f853d3f65432f0645",
    RESULT_CONTRACT_USAGE: "4b2ed9bc32ed5cd5e4151bc39d3a7a6a83fa8498a97b7ff1e928a82d6c8ac304",
    R586_PREREG: "a139948085a99a6e745d3e8bf5d08ae11b58480d30ddf5e75467b506dda3a9a5",
    R586_SCRIPT: "ab33c1afea27d624151ad68ca230fb36ae03833e95349eb6da409778e9ea271b",
    R586_TEST: "748400e0675d37d9fd7fc7ce306ac7549b73db4b50bab2fe365abdb44b4d7841",
    R586_DRYRUN: "0134f0218c3ec135abaec30d2028abae6e1da2c4f0c30bc78cf57d4d4aac0d30",
    PREREG: "1f8e51ca7dcb4c8c9bb73ba13403c098871e13b593d995ff516ed839c2a9c771",
}

# This is the exact map R586 is required to save in its result. It deliberately
# excludes R586's future result/receipt and binds the preregistration instead.
R586_INPUT_HASHES = {
    str(path): PREOUTCOME_AUTHORITY_HASHES[path]
    for path in (
        ROWS, ROWS_RECEIPT, ROWS_PREREG, ROWS_BUILDER, ROWS_TEST,
        R580_PREREG, R580_SCRIPT, R580_TEST, R580_DRYRUN, R580_RESULT,
        R580_RECEIPT, R581_PREREG, R581_SCRIPT, R581_TEST, R581_DRYRUN,
        R581_AUDIT, RESULT_CONTRACT, RESULT_CONTRACT_TEST,
        RESULT_CONTRACT_USAGE, R586_PREREG,
    )
}

CHECKPOINT_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLITS = ("FIT", "SELECT")
FORBIDDEN_SPLITS = ("FINAL_TEST", "OOD")
EXPECTED_GROUPS = 108
EXPECTED_ROWS = 3_240
EXPECTED_SEQUENCES = 3_024
EXPECTED_FORWARDS = 95
BOOTSTRAPS = 2_000
EXPECTED_BOOTSTRAP_CELLS = 86
ABS_TOLERANCE = 1e-12
HELD_NEXT_STEP = "independent_CPU_audit_then_separate_R557_R558_adaptation_preregistration"
NULL_NEXT_STEP = "preserve_scientific_null_and_do_not_search_factor_sites"
CONDITIONS = ("s0p0", "s0p1", "s1p0", "s1p1")
CONTROL_LABELS = ("neutral_source", "neutral_payload", "filler", "lag")

RESULT_FIELD_TYPES = {
    "schema": str, "rung": int, "stage": str, "instrument_passes": bool,
    "pred_a_native_factorial_and_controls": bool,
    "pred_b_selector_payload_interaction": bool,
    "pred_c_selected_match_necessity_and_neutral_selectivity": bool,
    "factorial_cells": dict, "selector_payload_interaction": dict,
    "relation_preserving_controls": dict,
    "selected_match_necessity_and_neutral_selectivity": dict,
    "contrast_source_diagnostics_not_gated": dict,
    "failed_scientific_clauses": list, "all_scientific_gates_pass": bool,
    "verdict": str, "raw_evidence": dict, "model_forwards": int,
    "model_backwards": int, "model_weights_updated": bool,
    "unique_sequences": int, "checkpoint_weights_sha256": str,
    "implementation_sha256": str, "test_sha256": str, "input_sha256": dict,
    "evaluated_splits": list, "forbidden_splits_opened": list,
    "elapsed_seconds": float, "next_step": str,
}

RECEIPT_FIELD_TYPES = {
    "schema": str, "result_path": str, "result_sha256": str,
    "implementation_sha256": str, "test_sha256": str,
    "preregistration_sha256": str, "input_sha256": dict,
    "checkpoint_weights_sha256": str, "verdict": str,
    "model_forwards": int, "model_backwards": int,
    "model_weights_updated": bool, "evaluated_splits": list,
    "forbidden_splits_opened": list, "next_step": str,
}

GENERIC_FIELD_TYPES = {
    field: {str: "string", int: "integer", bool: "boolean", dict: "dict",
            list: "list", float: "number"}[kind]
    for field, kind in RESULT_FIELD_TYPES.items()
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha256(value: object) -> str:
    encoded = json.dumps(value, allow_nan=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_preoutcome_authority(*, require_future_absent: bool = False) -> dict[str, str]:
    observed = {}
    for path, expected in PREOUTCOME_AUTHORITY_HASHES.items():
        if not path.is_file():
            raise RuntimeError(f"frozen authority missing: {path}")
        digest = sha256(path)
        if digest != expected:
            raise RuntimeError(f"frozen authority mismatch: {path}")
        observed[str(path)] = digest
    dryrun = strict_loads(R586_DRYRUN.read_bytes(), "R586 dry run")
    expected_dryrun = {
        "status": "dryrun_passed",
        "implementation_sha256": PREOUTCOME_AUTHORITY_HASHES[R586_SCRIPT],
        "test_sha256": PREOUTCOME_AUTHORITY_HASHES[R586_TEST],
        "preregistration_sha256": PREOUTCOME_AUTHORITY_HASHES[R586_PREREG],
        "model_loaded": False, "model_forwards": 0, "model_backwards": 0,
        "model_weights_updated": False, "future_result_written": False,
        "future_receipt_written": False,
    }
    if not isinstance(dryrun, dict) or any(dryrun.get(k) != v for k, v in expected_dryrun.items()):
        raise RuntimeError("R586 dry run does not bind the frozen instrument")
    if require_future_absent and (R586_RESULT.exists() or R586_RECEIPT.exists()):
        raise RuntimeError("R586 future namespace was opened before R587 freeze")
    return observed


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_low_level() -> ModuleType:
    """Load only the hash-pinned, pre-R586 R581 reconstruction mathematics."""
    verify_preoutcome_authority()
    module = _load_module(R581_SCRIPT, "r587_pinned_r581_low_level")
    expected = {
        "SPLITS": SPLITS, "BOOTSTRAPS": BOOTSTRAPS,
        "EXPECTED_GROUPS": EXPECTED_GROUPS, "EXPECTED_ROWS": EXPECTED_ROWS,
        "EXPECTED_SEQUENCES": EXPECTED_SEQUENCES,
        "EXPECTED_FORWARDS": EXPECTED_FORWARDS,
        "BOOTSTRAP_NAMESPACE": "a8-r580-group-bootstrap-v1",
    }
    for field, value in expected.items():
        if getattr(module, field) != value:
            raise RuntimeError(f"R581 low-level constant changed: {field}")
    return module


def load_result_contract() -> ModuleType:
    verify_preoutcome_authority()
    return _load_module(RESULT_CONTRACT, "r587_pinned_result_contract")


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_object(pairs: Sequence[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_loads(data: bytes, label: str) -> object:
    try:
        decoded = data.decode("utf-8", errors="strict")
        value = json.loads(decoded, parse_constant=_reject_constant,
                           object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as parser_exc:
        raise ValueError(f"{label} is not finite strict JSON: {parser_exc}") from parser_exc
    # Parsing already restricts the tree to JSON-native types; this second pass
    # is an explicit finite-standard-JSON check without importing any runner.
    json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return value


def _require_exact_types(value: object, schema: Mapping[str, type], label: str) -> None:
    if not isinstance(value, dict) or set(value) != set(schema):
        raise TypeError(f"{label} fields changed")
    for field, expected in schema.items():
        if type(value[field]) is not expected:
            raise TypeError(
                f"{label}.{field} must be {expected.__name__}, "
                f"got {type(value[field]).__name__}"
            )


def compare(expected: object, observed: object, path: str, failures: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(expected) != set(observed):
            failures.append(f"{path}:keys_or_type")
            return
        for key in expected:
            compare(expected[key], observed[key], f"{path}.{key}", failures)
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            failures.append(f"{path}:length_or_type")
            return
        for index, item in enumerate(expected):
            compare(item, observed[index], f"{path}[{index}]", failures)
    elif isinstance(expected, float):
        if type(observed) not in (int, float) or isinstance(observed, bool) or not math.isclose(
                expected, float(observed), rel_tol=0.0, abs_tol=ABS_TOLERANCE):
            failures.append(f"{path}:numeric")
    elif type(expected) is not type(observed) or expected != observed:
        failures.append(f"{path}:value_or_type")


def expected_bootstrap_cell_ids() -> set[str]:
    return (
        {f"{split}:factorial:{condition}:correct_margin"
         for split in SPLITS for condition in CONDITIONS}
        | {f"{split}:selector_payload_interaction" for split in SPLITS}
        | {f"{split}:control:{label}:{condition}:{endpoint}"
           for split in SPLITS for label in CONTROL_LABELS
           for condition in CONDITIONS for endpoint in ("base", "donor")}
        | {f"{split}:{metric}" for split in SPLITS
           for metric in ("selected_match_drop", "selected_vs_neutral_gap")}
        | {f"{split}:contrast_source:{condition}"
           for split in SPLITS for condition in CONDITIONS}
    )


def load_authority() -> tuple[ModuleType, list[dict], list[dict], list[dict]]:
    low = load_low_level()
    groups, rows, _ = low.load_authority()
    specs = low.expected_sequence_specs(groups, rows)
    if (len(groups), len(rows), len(specs)) != (
            EXPECTED_GROUPS, EXPECTED_ROWS, EXPECTED_SEQUENCES):
        raise RuntimeError("R587 authority census changed")
    if {item["split"] for item in groups} != set(SPLITS):
        raise RuntimeError("R587 FIT/SELECT closure changed")
    return low, groups, rows, specs


def validate_generic_contract(result: Mapping[str, object]) -> dict[str, object]:
    contract_module = load_result_contract()
    authority_rows = strict_loads(ROWS.read_bytes(), "R578 rows")["rows"]
    contract = contract_module.ResultContract(
        opened_splits=SPLITS, allowed_splits=SPLITS,
        forbidden_splits=FORBIDDEN_SPLITS,
        min_model_forwards=EXPECTED_FORWARDS,
        max_model_forwards=EXPECTED_FORWARDS,
        exact_model_forwards=EXPECTED_FORWARDS,
        expected_model_backwards=0, expected_weights_updated=False,
        field_types=GENERIC_FIELD_TYPES,
        required_provenance=tuple(R586_INPUT_HASHES),
        expected_provenance=R586_INPUT_HASHES,
        allow_extra_provenance=False,
        weights_updated_field="model_weights_updated",
    )
    return contract_module.validate_result_contract(
        result, result["raw_evidence"]["row_measurements"], authority_rows, contract
    )


def validate_result_envelope(result: object) -> dict[str, object]:
    _require_exact_types(result, RESULT_FIELD_TYPES, "result")
    assert isinstance(result, dict)
    load_result_contract().validate_standard_json(result)
    expected = {
        "schema": "induction_selector_payload_native_capability_rung586_result_v1",
        "rung": 586,
        "stage": "clean_replication_of_r580_induction_native_capability",
        "instrument_passes": True,
        "model_forwards": EXPECTED_FORWARDS,
        "model_backwards": 0,
        "model_weights_updated": False,
        "unique_sequences": EXPECTED_SEQUENCES,
        "checkpoint_weights_sha256": CHECKPOINT_SHA256,
        "implementation_sha256": PREOUTCOME_AUTHORITY_HASHES[R586_SCRIPT],
        "test_sha256": PREOUTCOME_AUTHORITY_HASHES[R586_TEST],
        "input_sha256": R586_INPUT_HASHES,
        "evaluated_splits": list(SPLITS),
        "forbidden_splits_opened": [],
    }
    for field, value in expected.items():
        if result[field] != value:
            raise ValueError(f"result.{field} changed")
    if result["elapsed_seconds"] < 0 or not math.isfinite(result["elapsed_seconds"]):
        raise ValueError("result.elapsed_seconds must be finite and nonnegative")
    if not all(type(item) is str for item in result["failed_scientific_clauses"]):
        raise TypeError("result.failed_scientific_clauses must contain strings")
    return validate_generic_contract(result)


def audit_payload(result: object, groups: Sequence[dict], rows: Sequence[dict],
                  specs: Sequence[dict], *, replicates: int = BOOTSTRAPS) -> dict:
    failures: list[str] = []
    contract_summary = None
    recomputed = None
    traces: dict[str, dict] = {}
    rebuilt_raw = None
    try:
        contract_summary = validate_result_envelope(result)
        assert isinstance(result, dict)
        sequence_measurements = result["raw_evidence"]["sequence_measurements"]
        expected_order = [item["sequence_id"] for item in specs]
        observed_order = [item["sequence_id"] for item in sequence_measurements]
        if observed_order != expected_order:
            raise RuntimeError("sequence order differs from frozen authority")
        low = load_low_level()
        rebuilt_raw = low.reconstruct_raw(groups, rows, specs, sequence_measurements)
        recomputed, traces = low.score(rebuilt_raw, replicates=replicates)
    except (AssertionError, KeyError, TypeError, ValueError, RuntimeError) as reconstruction_exc:
        return {
            "audit_verdict": "failed_independent_audit",
            "audit_failures": [
                "integrity_or_reconstruction:"
                f"{type(reconstruction_exc).__name__}:{reconstruction_exc}"
            ],
            "independently_recomputed_scientific_verdict": None,
            "independently_recomputed_failed_clauses": [],
            "raw_counts": None,
            "membership_hashes": None,
            "bootstrap_cell_count": 0,
            "bootstrap_replicates_per_cell": replicates,
            "bootstrap_algorithm": None,
            "bootstrap_algorithm_sha256": None,
            "bootstrap_trace_hash": None,
            "bootstrap_traces": {},
            "generic_contract": contract_summary,
            "recomputed_scores": None,
        }

    assert isinstance(result, dict) and rebuilt_raw is not None and recomputed is not None
    compare(rebuilt_raw, result["raw_evidence"], "raw_evidence", failures)
    for key, value in recomputed.items():
        compare(value, result.get(key), f"score.{key}", failures)
    held = recomputed["all_scientific_gates_pass"]
    terminal = {
        "verdict": "held_capability_screen" if held else "scientific_null",
        "next_step": HELD_NEXT_STEP if held else NULL_NEXT_STEP,
    }
    for key, value in terminal.items():
        compare(value, result.get(key), f"terminal.{key}", failures)
    if held != (recomputed["failed_scientific_clauses"] == []):
        failures.append("terminal.failed_clause_conjunction:value")
    if set(traces) != expected_bootstrap_cell_ids():
        failures.append("bootstrap.cell_census:value")
    if any(trace.get("replicates") != replicates for trace in traces.values()):
        failures.append("bootstrap.replicate_census:value")
    ordered_traces = {key: traces[key] for key in sorted(traces)}
    memberships = {
        "group_membership_sha256": content_sha256([
            [item["group_id"], item["split"]] for item in groups]),
        "row_membership_sha256": content_sha256([
            [item["row_id"], item["group_id"], item["split"], item["family_id"],
             item["family_variant"]] for item in rows]),
        "sequence_membership_sha256": content_sha256(list(specs)),
    }
    return {
        "audit_verdict": "held_independent_audit" if not failures else "failed_independent_audit",
        "audit_failures": failures,
        "independently_recomputed_scientific_verdict": recomputed["verdict"],
        "independently_recomputed_failed_clauses": recomputed["failed_scientific_clauses"],
        "raw_counts": {
            "sequences": len(rebuilt_raw["sequence_measurements"]),
            "rows": len(rebuilt_raw["row_measurements"]),
            "factorial_groups": len(rebuilt_raw["group_factorial_measurements"]),
            "condition_effects": len(rebuilt_raw["group_condition_effect_measurements"]),
        },
        "membership_hashes": memberships,
        "bootstrap_cell_count": len(traces),
        "bootstrap_replicates_per_cell": replicates,
        "bootstrap_algorithm": low.BOOTSTRAP_ALGORITHM,
        "bootstrap_algorithm_sha256": content_sha256(low.BOOTSTRAP_ALGORITHM),
        "bootstrap_trace_hash": content_sha256(ordered_traces),
        "bootstrap_traces": ordered_traces,
        "generic_contract": contract_summary,
        "recomputed_scores": recomputed,
    }


def validate_receipt(receipt: object, result: object,
                     result_bytes: bytes) -> list[str]:
    failures: list[str] = []
    try:
        _require_exact_types(receipt, RECEIPT_FIELD_TYPES, "receipt")
        assert isinstance(receipt, dict)
        load_result_contract().validate_standard_json(receipt)
    except (AssertionError, TypeError, ValueError, RuntimeError) as receipt_exc:
        return [f"receipt.integrity:{type(receipt_exc).__name__}:{receipt_exc}"]
    if not isinstance(result, Mapping):
        return ["receipt.integrity:source_result_is_not_a_mapping"]
    expected = {
        "schema": "induction_selector_payload_native_capability_rung586_receipt_v1",
        "result_path": "basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_rung586_results.json",
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "implementation_sha256": PREOUTCOME_AUTHORITY_HASHES[R586_SCRIPT],
        "test_sha256": PREOUTCOME_AUTHORITY_HASHES[R586_TEST],
        "preregistration_sha256": PREOUTCOME_AUTHORITY_HASHES[R586_PREREG],
        "input_sha256": R586_INPUT_HASHES,
        "checkpoint_weights_sha256": CHECKPOINT_SHA256,
        "verdict": result.get("verdict"),
        "model_forwards": EXPECTED_FORWARDS,
        "model_backwards": 0,
        "model_weights_updated": False,
        "evaluated_splits": list(SPLITS),
        "forbidden_splits_opened": [],
        "next_step": result.get("next_step"),
    }
    compare(expected, receipt, "receipt", failures)
    return failures


def read_stable_source_pair(result_path: Path = R586_RESULT,
                            receipt_path: Path = R586_RECEIPT) -> tuple[bytes, bytes]:
    if not result_path.is_file() or not receipt_path.is_file():
        raise RuntimeError("R586 result and receipt must both exist")
    before = (
        (result_path.stat().st_dev, result_path.stat().st_ino, result_path.stat().st_size,
         result_path.stat().st_mtime_ns),
        (receipt_path.stat().st_dev, receipt_path.stat().st_ino, receipt_path.stat().st_size,
         receipt_path.stat().st_mtime_ns),
    )
    result_first, receipt_first = result_path.read_bytes(), receipt_path.read_bytes()
    result_second, receipt_second = result_path.read_bytes(), receipt_path.read_bytes()
    after = (
        (result_path.stat().st_dev, result_path.stat().st_ino, result_path.stat().st_size,
         result_path.stat().st_mtime_ns),
        (receipt_path.stat().st_dev, receipt_path.stat().st_ino, receipt_path.stat().st_size,
         receipt_path.stat().st_mtime_ns),
    )
    if before != after or result_first != result_second or receipt_first != receipt_second:
        raise RuntimeError("R586 result/receipt pair changed during atomic read")
    # Parse before returning so an invalid pair never reaches audit-output writing.
    strict_loads(result_first, "R586 result")
    strict_loads(receipt_first, "R586 receipt")
    return result_first, receipt_first


def fixture_result(low: ModuleType, groups: Sequence[dict], rows: Sequence[dict],
                   specs: Sequence[dict], *, make_null: bool,
                   replicates: int) -> dict:
    measurements = low.planted_measurements(specs, rows, make_null=make_null)
    raw = low.reconstruct_raw(groups, rows, specs, measurements)
    scores, _ = low.score(raw, replicates=replicates)
    return {
        "schema": "induction_selector_payload_native_capability_rung586_result_v1",
        "rung": 586,
        "stage": "clean_replication_of_r580_induction_native_capability",
        "instrument_passes": True,
        **scores,
        "raw_evidence": raw,
        "model_forwards": EXPECTED_FORWARDS,
        "model_backwards": 0,
        "model_weights_updated": False,
        "unique_sequences": EXPECTED_SEQUENCES,
        "checkpoint_weights_sha256": CHECKPOINT_SHA256,
        "implementation_sha256": PREOUTCOME_AUTHORITY_HASHES[R586_SCRIPT],
        "test_sha256": PREOUTCOME_AUTHORITY_HASHES[R586_TEST],
        "input_sha256": copy.deepcopy(R586_INPUT_HASHES),
        "evaluated_splits": list(SPLITS),
        "forbidden_splits_opened": [],
        "elapsed_seconds": 0.0,
        "next_step": NULL_NEXT_STEP if make_null else HELD_NEXT_STEP,
    }


def fixture_receipt(result: Mapping[str, object], result_bytes: bytes) -> dict:
    return {
        "schema": "induction_selector_payload_native_capability_rung586_receipt_v1",
        "result_path": "basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_rung586_results.json",
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "implementation_sha256": PREOUTCOME_AUTHORITY_HASHES[R586_SCRIPT],
        "test_sha256": PREOUTCOME_AUTHORITY_HASHES[R586_TEST],
        "preregistration_sha256": PREOUTCOME_AUTHORITY_HASHES[R586_PREREG],
        "input_sha256": copy.deepcopy(R586_INPUT_HASHES),
        "checkpoint_weights_sha256": CHECKPOINT_SHA256,
        "verdict": result["verdict"],
        "model_forwards": EXPECTED_FORWARDS,
        "model_backwards": 0,
        "model_weights_updated": False,
        "evaluated_splits": list(SPLITS),
        "forbidden_splits_opened": [],
        "next_step": result["next_step"],
    }


def run_dryrun() -> dict:
    verify_preoutcome_authority(require_future_absent=True)
    low, groups, rows, specs = load_authority()
    fixture_replicates = 41
    held = fixture_result(low, groups, rows, specs, make_null=False,
                          replicates=fixture_replicates)
    null = fixture_result(low, groups, rows, specs, make_null=True,
                          replicates=fixture_replicates)
    held_audit = audit_payload(held, groups, rows, specs,
                               replicates=fixture_replicates)
    null_audit = audit_payload(null, groups, rows, specs,
                               replicates=fixture_replicates)
    if held_audit["audit_verdict"] != "held_independent_audit":
        raise RuntimeError("held fixture audit failed")
    if not (null_audit["audit_verdict"] == "held_independent_audit"
            and null_audit["independently_recomputed_scientific_verdict"] == "scientific_null"):
        raise RuntimeError("scientific-null fixture audit failed")

    malformed_next = copy.deepcopy(held)
    malformed_next["next_step"] = [HELD_NEXT_STEP]
    next_audit = audit_payload(malformed_next, groups, rows, specs,
                               replicates=fixture_replicates)
    missing_group = copy.deepcopy(held)
    missing_id = groups[0]["group_id"]
    missing_group["raw_evidence"]["sequence_measurements"] = [
        item for item in missing_group["raw_evidence"]["sequence_measurements"]
        if item["group_id"] != missing_id
    ]
    group_audit = audit_payload(missing_group, groups, rows, specs,
                                replicates=fixture_replicates)
    nonfinite = copy.deepcopy(held)
    nonfinite["raw_evidence"]["row_measurements"][0]["base_margin"] = float("nan")
    finite_audit = audit_payload(nonfinite, groups, rows, specs,
                                 replicates=fixture_replicates)
    result_bytes = (json.dumps(held, indent=1, allow_nan=False) + "\n").encode()
    source_receipt = fixture_receipt(held, result_bytes)
    if validate_receipt(source_receipt, held, result_bytes):
        raise RuntimeError("valid planted receipt failed")
    receipt_bytes = (json.dumps(source_receipt, indent=1, allow_nan=False) + "\n").encode()
    original_digest = source_receipt["result_sha256"].encode()
    if receipt_bytes.count(original_digest) != 1:
        raise RuntimeError("planted receipt hash is not uniquely byte-addressable")
    tampered_receipt_bytes = receipt_bytes.replace(original_digest, b"0" * 64)
    tampered_receipt = strict_loads(tampered_receipt_bytes, "tampered planted receipt")
    receipt_failures = validate_receipt(tampered_receipt, held, result_bytes)
    changed_result_bytes = result_bytes + b" "
    result_byte_failures = validate_receipt(source_receipt, held, changed_result_bytes)
    if not (next_audit["audit_verdict"] == group_audit["audit_verdict"]
            == finite_audit["audit_verdict"] == "failed_independent_audit"
            and receipt_failures and result_byte_failures):
        raise RuntimeError("an adversarial fixture was accepted")

    dryrun = {
        "schema": "induction_selector_payload_native_capability_audit_rung587_dryrun_v1",
        "status": "dryrun_passed",
        "preoutcome_authority_sha256": {
            str(path): digest for path, digest in PREOUTCOME_AUTHORITY_HASHES.items()
        },
        "authority_groups": len(groups), "authority_rows": len(rows),
        "authority_sequences": len(specs),
        "real_audit_bootstrap_replicates": BOOTSTRAPS,
        "fixture_bootstrap_replicates": fixture_replicates,
        "bootstrap_cells_per_fixture": held_audit["bootstrap_cell_count"],
        "held_fixture_audit_verdict": held_audit["audit_verdict"],
        "held_fixture_scientific_verdict": held_audit["independently_recomputed_scientific_verdict"],
        "null_fixture_audit_verdict": null_audit["audit_verdict"],
        "null_fixture_scientific_verdict": null_audit["independently_recomputed_scientific_verdict"],
        "null_fixture_failed_clauses": null_audit["independently_recomputed_failed_clauses"],
        "list_next_step_rejected": True, "missing_group_rejected": True,
        "nonfinite_nested_value_rejected": True,
        "tampered_receipt_bytes_rejected": True,
        "changed_result_bytes_rejected": True,
        "future_result_opened": False, "future_receipt_opened": False,
        "model_loaded": False, "model_forwards": 0, "model_backwards": 0,
        "model_weights_updated": False,
        "evaluated_splits": list(SPLITS), "forbidden_splits_opened": [],
        "script_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST) if TEST.is_file() else None,
        "preregistration_sha256": sha256(PREREG),
    }
    encoded = json.dumps(dryrun, indent=1, allow_nan=False) + "\n"
    DRYRUN.write_text(encoded)
    return dryrun


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(run_dryrun(), indent=2, allow_nan=False))
        return
    if OUT.exists():
        raise RuntimeError("R587 audit namespace already exists")
    verify_preoutcome_authority()
    low, groups, rows, specs = load_authority()
    result_bytes, receipt_bytes = read_stable_source_pair()
    result = strict_loads(result_bytes, "R586 result")
    receipt = strict_loads(receipt_bytes, "R586 receipt")
    audit = audit_payload(result, groups, rows, specs, replicates=BOOTSTRAPS)
    receipt_failures = validate_receipt(receipt, result, result_bytes)
    audit["audit_failures"].extend(receipt_failures)
    if receipt_failures:
        audit["audit_verdict"] = "failed_independent_audit"
    audit.update({
        "schema": "induction_selector_payload_native_capability_audit_rung587_v1",
        "rung": 587,
        "source_result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "source_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "preoutcome_authority_sha256": {
            str(path): digest for path, digest in PREOUTCOME_AUTHORITY_HASHES.items()
        },
        "auditor_implementation_sha256": sha256(SCRIPT),
        "auditor_test_sha256": sha256(TEST),
        "auditor_preregistration_sha256": sha256(PREREG),
        "source_pair_stable_and_receipt_bound": not receipt_failures,
        "model_loaded": False, "model_forwards": 0, "model_backwards": 0,
        "model_weights_updated": False,
        "evaluated_splits": list(SPLITS), "forbidden_splits_opened": [],
    })
    OUT.write_text(json.dumps(audit, indent=1, allow_nan=False) + "\n")
    print(json.dumps({key: audit[key] for key in (
        "audit_verdict", "audit_failures",
        "independently_recomputed_scientific_verdict",
        "independently_recomputed_failed_clauses", "bootstrap_cell_count",
        "bootstrap_trace_hash", "source_pair_stable_and_receipt_bound",
        "model_forwards",
    )}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
