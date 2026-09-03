#!/usr/bin/env python3
"""R586 prospective clean replication of R580 native capability.

The scientific definitions are delegated to the exact, hash-pinned R580
implementation.  R586 adds a new output namespace and strict envelope types so
``next_step`` cannot again become a singleton JSON list.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Mapping


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
PREREG = POLY / "INDUCTION_SELECTOR_PAYLOAD_NATIVE_CAPABILITY_RUNG586_PREREGISTRATION.md"
SCRIPT = Path(__file__)
TEST = SCRIPT.with_name("test_induction_selector_payload_native_capability_rung586.py")
OUT = ROOT / "induction_selector_payload_native_capability_rung586_results.json"
OUT_RECEIPT = ROOT / "induction_selector_payload_native_capability_rung586_receipt.json"
DRYRUN = ROOT / "induction_selector_payload_native_capability_rung586_dryrun.json"

AUTHORITY_HASHES = {
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
    PREREG: "a139948085a99a6e745d3e8bf5d08ae11b58480d30ddf5e75467b506dda3a9a5",
}

CHECKPOINT_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLITS = ("FIT", "SELECT")
FORBIDDEN_SPLITS = ("FINAL_TEST", "OOD")
BATCH = 32
EXPECTED_GROUPS = 108
EXPECTED_ROWS = 3_240
EXPECTED_SEQUENCES = 3_024
EXPECTED_FORWARDS = math.ceil(EXPECTED_SEQUENCES / BATCH)
HELD_NEXT_STEP = "independent_CPU_audit_then_separate_R557_R558_adaptation_preregistration"
NULL_NEXT_STEP = "preserve_scientific_null_and_do_not_search_factor_sites"

RESULT_FIELD_TYPES = {
    "schema": str,
    "rung": int,
    "stage": str,
    "instrument_passes": bool,
    "pred_a_native_factorial_and_controls": bool,
    "pred_b_selector_payload_interaction": bool,
    "pred_c_selected_match_necessity_and_neutral_selectivity": bool,
    "factorial_cells": dict,
    "selector_payload_interaction": dict,
    "relation_preserving_controls": dict,
    "selected_match_necessity_and_neutral_selectivity": dict,
    "contrast_source_diagnostics_not_gated": dict,
    "failed_scientific_clauses": list,
    "all_scientific_gates_pass": bool,
    "verdict": str,
    "raw_evidence": dict,
    "model_forwards": int,
    "model_backwards": int,
    "model_weights_updated": bool,
    "unique_sequences": int,
    "checkpoint_weights_sha256": str,
    "implementation_sha256": str,
    "test_sha256": str,
    "input_sha256": dict,
    "evaluated_splits": list,
    "forbidden_splits_opened": list,
    "elapsed_seconds": float,
    "next_step": str,
}

RECEIPT_FIELD_TYPES = {
    "schema": str,
    "result_path": str,
    "result_sha256": str,
    "implementation_sha256": str,
    "test_sha256": str,
    "preregistration_sha256": str,
    "input_sha256": dict,
    "checkpoint_weights_sha256": str,
    "verdict": str,
    "model_forwards": int,
    "model_backwards": int,
    "model_weights_updated": bool,
    "evaluated_splits": list,
    "forbidden_splits_opened": list,
    "next_step": str,
}

GENERIC_FIELD_TYPES = {
    field: {
        str: "string",
        int: "integer",
        bool: "boolean",
        dict: "dict",
        list: "list",
        float: "number",
    }[kind]
    for field, kind in RESULT_FIELD_TYPES.items()
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_authority() -> dict[str, str]:
    observed = {}
    for path, expected in AUTHORITY_HASHES.items():
        if not path.is_file():
            raise RuntimeError(f"frozen authority missing: {path}")
        digest = sha256(path)
        if digest != expected:
            raise RuntimeError(f"frozen authority mismatch: {path}")
        observed[str(path)] = digest
    return observed


def load_r580() -> ModuleType:
    """Load the hash-pinned scientific implementation without loading a model."""
    verify_authority()
    name = "r586_pinned_r580_scientific_contract"
    spec = importlib.util.spec_from_file_location(name, R580_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned R580 scientific contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    expected = {
        "SPLITS": SPLITS,
        "FORBIDDEN_SPLITS": FORBIDDEN_SPLITS,
        "BATCH": BATCH,
        "EXPECTED_GROUPS": EXPECTED_GROUPS,
        "EXPECTED_ROWS": EXPECTED_ROWS,
        "EXPECTED_SEQUENCES": EXPECTED_SEQUENCES,
        "EXPECTED_FORWARDS": EXPECTED_FORWARDS,
        "BOOTSTRAPS": 2_000,
        "BOOTSTRAP_NAMESPACE": "a8-r580-group-bootstrap-v1",
    }
    for field, value in expected.items():
        if getattr(module, field) != value:
            raise RuntimeError(f"R580 scientific constant changed: {field}")
    return module


def load_result_contract() -> ModuleType:
    """Load the hash-pinned generic result boundary validator."""
    verify_authority()
    name = "r586_pinned_generic_result_contract"
    spec = importlib.util.spec_from_file_location(name, RESULT_CONTRACT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned generic result contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_generic_result_contract(result: Mapping[str, object]) -> dict[str, object]:
    contract_module = load_result_contract()
    authority_rows = json.loads(ROWS.read_text())["rows"]
    contract = contract_module.ResultContract(
        opened_splits=SPLITS,
        allowed_splits=SPLITS,
        forbidden_splits=FORBIDDEN_SPLITS,
        min_model_forwards=EXPECTED_FORWARDS,
        max_model_forwards=EXPECTED_FORWARDS,
        exact_model_forwards=EXPECTED_FORWARDS,
        expected_model_backwards=0,
        expected_weights_updated=False,
        field_types=GENERIC_FIELD_TYPES,
        required_provenance=tuple(str(path) for path in AUTHORITY_HASHES),
        expected_provenance={str(path): digest for path, digest in AUTHORITY_HASHES.items()},
        allow_extra_provenance=False,
        weights_updated_field="model_weights_updated",
    )
    return contract_module.validate_result_contract(
        result,
        result["raw_evidence"]["row_measurements"],
        authority_rows,
        contract,
    )


def load_authority() -> tuple[ModuleType, list[dict], list[dict], list[dict]]:
    r580 = load_r580()
    groups, rows, _ = r580.load_authority()
    specs = r580.collect_sequence_specs(groups, rows)
    if (len(groups), len(rows), len(specs)) != (
        EXPECTED_GROUPS,
        EXPECTED_ROWS,
        EXPECTED_SEQUENCES,
    ):
        raise RuntimeError("R586 authority census changed")
    if {item["split"] for item in groups} != set(SPLITS):
        raise RuntimeError("R586 split envelope changed")
    return r580, groups, rows, specs


def _require_exact_types(value: Mapping[str, object], schema: Mapping[str, type], label: str) -> None:
    if set(value) != set(schema):
        raise TypeError(f"{label} fields changed")
    for field, expected in schema.items():
        if type(value[field]) is not expected:  # bool must not pass as int.
            raise TypeError(
                f"{label}.{field} must be {expected.__name__}, "
                f"got {type(value[field]).__name__}"
            )


def validate_result_envelope(result: Mapping[str, object]) -> dict[str, object]:
    _require_exact_types(result, RESULT_FIELD_TYPES, "result")
    expected_values = {
        "schema": "induction_selector_payload_native_capability_rung586_result_v1",
        "rung": 586,
        "stage": "clean_replication_of_r580_induction_native_capability",
        "instrument_passes": True,
        "model_forwards": EXPECTED_FORWARDS,
        "model_backwards": 0,
        "model_weights_updated": False,
        "unique_sequences": EXPECTED_SEQUENCES,
        "checkpoint_weights_sha256": CHECKPOINT_SHA256,
        "implementation_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST),
        "input_sha256": {str(path): digest for path, digest in AUTHORITY_HASHES.items()},
        "evaluated_splits": list(SPLITS),
        "forbidden_splits_opened": [],
    }
    for field, expected in expected_values.items():
        if result[field] != expected:
            raise ValueError(f"result.{field} changed")
    if result["elapsed_seconds"] < 0 or not math.isfinite(result["elapsed_seconds"]):
        raise ValueError("result.elapsed_seconds must be finite and nonnegative")
    if not all(type(item) is str for item in result["failed_scientific_clauses"]):
        raise TypeError("result.failed_scientific_clauses must contain strings")
    for field in (
        "raw_evidence",
        "factorial_cells",
        "selector_payload_interaction",
        "relation_preserving_controls",
        "selected_match_necessity_and_neutral_selectivity",
        "contrast_source_diagnostics_not_gated",
    ):
        if not result[field]:
            raise ValueError(f"result.{field} must be complete")
    raw = result["raw_evidence"]
    if set(raw) != {
        "sequence_measurements",
        "row_measurements",
        "group_factorial_measurements",
        "group_condition_effect_measurements",
    }:
        raise ValueError("result.raw_evidence fields changed")
    expected_counts = {
        "sequence_measurements": EXPECTED_SEQUENCES,
        "row_measurements": EXPECTED_ROWS,
        "group_factorial_measurements": EXPECTED_GROUPS,
        "group_condition_effect_measurements": EXPECTED_GROUPS * 4,
    }
    if any(type(raw[field]) is not list or len(raw[field]) != count
           for field, count in expected_counts.items()):
        raise ValueError("result.raw_evidence census changed")
    held = result["all_scientific_gates_pass"]
    if held != bool(
        result["pred_a_native_factorial_and_controls"]
        and result["pred_b_selector_payload_interaction"]
        and result["pred_c_selected_match_necessity_and_neutral_selectivity"]
    ):
        raise ValueError("result predicate conjunction changed")
    expected_verdict = "held_capability_screen" if held else "scientific_null"
    expected_next = HELD_NEXT_STEP if held else NULL_NEXT_STEP
    if result["verdict"] != expected_verdict:
        raise ValueError("result verdict disagrees with scientific gates")
    if result["next_step"] != expected_next:
        raise ValueError("result next_step disagrees with scientific gates")
    if held != (result["failed_scientific_clauses"] == []):
        raise ValueError("result failed-clause list disagrees with scientific gates")
    return validate_generic_result_contract(result)


def make_result(scores: Mapping[str, object], raw: dict, *, elapsed_seconds: float) -> dict:
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
        "implementation_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST),
        "input_sha256": {str(path): digest for path, digest in AUTHORITY_HASHES.items()},
        "evaluated_splits": list(SPLITS),
        "forbidden_splits_opened": [],
        "elapsed_seconds": float(elapsed_seconds),
        "next_step": HELD_NEXT_STEP if scores["all_scientific_gates_pass"] else NULL_NEXT_STEP,
    }


def make_fixture_result(*, make_null: bool = False) -> tuple[dict, ModuleType]:
    r580, groups, rows, specs = load_authority()
    measurements = r580.planted_sequence_measurements(specs, rows)
    if make_null:
        measurements = r580.make_planted_scientific_null(measurements, groups)
    raw = r580.build_raw_evidence(groups, rows, measurements)
    result = make_result(r580.score_raw_evidence(raw), raw, elapsed_seconds=0.0)
    validate_result_envelope(result)
    return result, r580


def make_receipt(result: Mapping[str, object], encoded_result: bytes) -> dict:
    receipt = {
        "schema": "induction_selector_payload_native_capability_rung586_receipt_v1",
        "result_path": str(OUT.relative_to(ROOT.parent.parent)),
        "result_sha256": hashlib.sha256(encoded_result).hexdigest(),
        "implementation_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST),
        "preregistration_sha256": sha256(PREREG),
        "input_sha256": {str(path): digest for path, digest in AUTHORITY_HASHES.items()},
        "checkpoint_weights_sha256": result["checkpoint_weights_sha256"],
        "verdict": result["verdict"],
        "model_forwards": result["model_forwards"],
        "model_backwards": result["model_backwards"],
        "model_weights_updated": result["model_weights_updated"],
        "evaluated_splits": result["evaluated_splits"],
        "forbidden_splits_opened": result["forbidden_splits_opened"],
        "next_step": result["next_step"],
    }
    validate_receipt_envelope(receipt, result, encoded_result)
    return receipt


def validate_receipt_envelope(
    receipt: Mapping[str, object], result: Mapping[str, object], encoded_result: bytes
) -> None:
    _require_exact_types(receipt, RECEIPT_FIELD_TYPES, "receipt")
    expected = {
        "schema": "induction_selector_payload_native_capability_rung586_receipt_v1",
        "result_path": "basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_rung586_results.json",
        "result_sha256": hashlib.sha256(encoded_result).hexdigest(),
        "implementation_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST),
        "preregistration_sha256": sha256(PREREG),
        "input_sha256": {str(path): digest for path, digest in AUTHORITY_HASHES.items()},
        "checkpoint_weights_sha256": CHECKPOINT_SHA256,
        "verdict": result["verdict"],
        "model_forwards": EXPECTED_FORWARDS,
        "model_backwards": 0,
        "model_weights_updated": False,
        "evaluated_splits": list(SPLITS),
        "forbidden_splits_opened": [],
        "next_step": result["next_step"],
    }
    for field, value in expected.items():
        if receipt[field] != value:
            raise ValueError(f"receipt.{field} changed")


def write_scientific_result(result: dict) -> None:
    if OUT.exists() or OUT_RECEIPT.exists():
        raise RuntimeError("R586 result namespace already exists; refusing overwrite")
    validate_result_envelope(result)
    encoded = (json.dumps(result, indent=1) + "\n").encode()
    receipt = make_receipt(result, encoded)
    OUT.write_bytes(encoded)
    OUT_RECEIPT.write_text(json.dumps(receipt, indent=1) + "\n")


def run_dryrun() -> dict:
    if OUT.exists() or OUT_RECEIPT.exists():
        raise RuntimeError("future R586 scientific artifact already exists")
    old_before = verify_authority()
    held, r580 = make_fixture_result(make_null=False)
    null, _ = make_fixture_result(make_null=True)
    held_contract = validate_result_envelope(held)
    null_contract = validate_result_envelope(null)
    if held["verdict"] != "held_capability_screen":
        raise RuntimeError("R586 held fixture did not hold")
    if null["verdict"] != "scientific_null" or not null["failed_scientific_clauses"]:
        raise RuntimeError("R586 null fixture did not complete as a scientific null")
    # Explicitly exercise the historical bug in both its Python and serialized forms.
    for malformed in ((HELD_NEXT_STEP,), [HELD_NEXT_STEP]):
        candidate = dict(held)
        candidate["next_step"] = malformed
        try:
            validate_result_envelope(candidate)
        except TypeError:
            pass
        else:
            raise RuntimeError("malformed next_step was accepted")
    # Direct delegation makes these exact R580 scientific scores, not a rewrite.
    rescored_held = r580.score_raw_evidence(held["raw_evidence"])
    rescored_null = r580.score_raw_evidence(null["raw_evidence"])
    if any(held[key] != value for key, value in rescored_held.items()):
        raise RuntimeError("held scientific score differs from R580")
    if any(null[key] != value for key, value in rescored_null.items()):
        raise RuntimeError("null scientific score differs from R580")
    old_after = verify_authority()
    if old_before != old_after:
        raise RuntimeError("a frozen old artifact changed during dry run")
    receipt = {
        "schema": "induction_selector_payload_native_capability_rung586_dryrun_v1",
        "status": "dryrun_passed",
        "implementation_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST),
        "preregistration_sha256": sha256(PREREG),
        "authority_sha256": old_after,
        "old_artifacts_immutable": True,
        "groups": EXPECTED_GROUPS,
        "rows": EXPECTED_ROWS,
        "unique_sequences": EXPECTED_SEQUENCES,
        "batch_size": BATCH,
        "literal_expected_forwards": EXPECTED_FORWARDS,
        "literal_expected_backwards": 0,
        "passing_fixture_verdict": held["verdict"],
        "passing_fixture_next_step": held["next_step"],
        "null_fixture_verdict": null["verdict"],
        "null_fixture_next_step": null["next_step"],
        "null_fixture_failed_clauses": null["failed_scientific_clauses"],
        "r580_scientific_scores_exact": True,
        "generic_result_contract_held": held_contract,
        "generic_result_contract_null": null_contract,
        "tuple_next_step_rejected": True,
        "list_next_step_rejected": True,
        "result_envelope_field_count": len(RESULT_FIELD_TYPES),
        "receipt_envelope_field_count": len(RECEIPT_FIELD_TYPES),
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "model_weights_updated": False,
        "evaluated_splits": list(SPLITS),
        "forbidden_splits_opened": [],
        "future_result_written": False,
        "future_receipt_written": False,
    }
    DRYRUN.write_text(json.dumps(receipt, indent=1) + "\n")
    return receipt


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(run_dryrun(), indent=2))
        return
    if OUT.exists() or OUT_RECEIPT.exists():
        raise RuntimeError("R586 result namespace already exists; refusing overwrite")
    started = time.time()
    r580, groups, rows, specs = load_authority()
    for search_path in (ROOT, OPS, POLY):
        if str(search_path) not in sys.path:
            sys.path.insert(0, str(search_path))
    import torch
    import bilin18_observed_model_facade as facade

    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True
    )
    measurements, calls = r580.evaluate_unique_sequences(model, specs)
    if calls != EXPECTED_FORWARDS:
        raise RuntimeError(f"forward price changed: {calls} != {EXPECTED_FORWARDS}")
    raw = r580.build_raw_evidence(groups, rows, measurements)
    scores = r580.score_raw_evidence(raw)
    result = make_result(scores, raw, elapsed_seconds=time.time() - started)
    if checkpoint.weights_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash changed")
    validate_result_envelope(result)
    write_scientific_result(result)
    print(json.dumps({
        "verdict": result["verdict"],
        "failed_scientific_clauses": result["failed_scientific_clauses"],
        "model_forwards": calls,
        "unique_sequences": len(measurements),
        "next_step": result["next_step"],
    }, indent=2))


if __name__ == "__main__":
    main()
