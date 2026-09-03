#!/usr/bin/env python3
"""R588: independent CPU audit of the R584 downstream-use result."""

# BQLANE: cpu

from __future__ import annotations

import collections
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

import numpy as np


os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
OPS = ROOT / "ops"

ROWS = ROOT / "numbered_list_cached_value_downstream_use_rows_rung582.json"
ROWS_RECEIPT = ROOT / "numbered_list_cached_value_downstream_use_rows_rung582_receipt.json"
R582_PREREG = POLY / "NUMBERED_LIST_CACHED_VALUE_DOWNSTREAM_USE_RUNG582_PREREGISTRATION.md"
R582_HELPER = OPS / "numbered_list_cached_value_downstream_use_rung582.py"
R584_NOTE = POLY / "NUMBERED_LIST_CACHED_VALUE_DOWNSTREAM_USE_RUNG584_IMPLEMENTATION.md"
R584_RUNNER = OPS / "numbered_list_cached_value_downstream_use_rung584.py"
R584_TEST = OPS / "test_numbered_list_cached_value_downstream_use_rung584.py"
R584_ADVERSARIAL_TEST = OPS / "r584_preoutcome_adversarial_tests.py"
RESULT_CONTRACT = OPS / "result_contract.py"
R584_DRYRUN = ROOT / "numbered_list_cached_value_downstream_use_rung584_dryrun.json"
R584_REVIEW = POLY / "NUMBERED_LIST_DOWNSTREAM_USE_R584_REPAIR_PREOUTCOME_REVIEW.md"
R576_RESULT = ROOT / "numbered_list_cached_value_weight_removal_rung576_results.json"
R579_AUDIT = ROOT / "numbered_list_cached_value_weight_removal_rung579_audit.json"
R576_PREREG = POLY / "NUMBERED_LIST_CACHED_VALUE_WEIGHT_REMOVAL_RUNG576_PREREGISTRATION.md"
FACADE = POLY / "bilin18_observed_model_facade.py"
R576_RUNNER = OPS / "numbered_list_cached_value_weight_removal_rung576.py"
R573_RUNNER = OPS / "numbered_list_factor_localization_rung573.py"

# This path is deliberately not touched outside read_stable_source() in the
# non-dry-run main entry point.
SOURCE_RESULT = ROOT / "numbered_list_cached_value_downstream_use_rung584_results.json"
PREREG = POLY / "NUMBERED_LIST_CACHED_VALUE_DOWNSTREAM_USE_AUDIT_RUNG588_PREREGISTRATION.md"
SCRIPT = Path(__file__)
TEST = SCRIPT.with_name("test_audit_numbered_list_cached_value_downstream_use_rung588.py")
OUT = ROOT / "numbered_list_cached_value_downstream_use_rung588_audit.json"
DRYRUN = ROOT / "numbered_list_cached_value_downstream_use_rung588_dryrun.json"

REPAIRED_COMMIT = "55b138ed7d270fa6b103f06006091f761cf54af8"
PREOUTCOME_HASHES = {
    ROWS: "84c6a78882a33c266b3875285f63ceaed746dac7810fce16b591f7b57763cf3b",
    ROWS_RECEIPT: "1511cfd7fcfe729edf4427f9f88f8552c32230e013d01a0661767713fdc29148",
    R582_PREREG: "e7832dc77cabe7a1afba61c759188a0aca73802163cef1abe013ffaff5c987b3",
    R582_HELPER: "b0d99eeeef834091cf9ddfe77b58682f0e9a7e101e143a18570808dacb57bc1c",
    R584_NOTE: "612005760bccda8f1a9f16b540b0734de3241e5da1c40246f514509733539181",
    R584_RUNNER: "50609756d97de2f13f717774f13d72b1c743f38a172375e9b08efc2b055336c7",
    R584_TEST: "37cc8f73ed128ebdb17b5cfcdb1248bc240291e9a10d38c526ac7d4a76ea3cce",
    R584_ADVERSARIAL_TEST: "900883046b648c7c9aa0714fff3d7d0da678b70ab8598623321e4f9d32bb5cd2",
    RESULT_CONTRACT: "af8fb9557dcb77e038319b0fffa919927f3925497a0edafe27fc951125dfb272",
    R584_DRYRUN: "b2ebe65c92ea5170ab13394c1ffee8562ff4241f481a6fce392a00b200149fe8",
    R584_REVIEW: "9294bdf8df18a56cdae8705b69e0129bfe2d6376d642d4c9dc86386c0d898310",
    R576_RESULT: "a6041c28cefc4f695f6e649210884774ed576bae80c14c031473d6b8c8ff2f73",
    R579_AUDIT: "03c03cf9fafe343584f323440d3eab4ab686a70fce44bc36d0fb2ccec945bf2d",
    R576_PREREG: "a776ebc1df29a6f3193d3315e190ec9494c95905596e450461c002378f8f59b6",
    FACADE: "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
    R576_RUNNER: "91db3a2a9210aef915ce2e4f0a62253274e0b5470cbfaa05a95d50a3c0cf985a",
    R573_RUNNER: "5723e42e2a5f72a4ddab7a20b631e18e0b6d28875ff53f3db2d37d1845d6e076",
    PREREG: "1d45b7f4bb111247de813f1e80172564188ed9651dfcd279f3c565426acbfe7d",
}

CHECKPOINT_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SITES = (8, 10, 12, 14)
COMPONENTS = ("background_cross", "contrast_self", "joint_response")
NULLS = ("different_group_same_cell", "same_source_other_action")
SELECTION = tuple((site, component) for site in SITES for component in COMPONENTS)
SELECTION_NAMES = tuple(f"mlp{site}_{component}" for site, component in SELECTION)
ELIGIBLE_CONDITIONS = {
    "factorial_copy", "factorial_successor", "surface_copy", "surface_successor"
}
SPLIT_ROWS = {"FIT": 576, "SELECT": 288, "FINAL_TEST": 288, "OOD": 288}
SPLIT_GROUPS = {"FIT": 16, "SELECT": 8, "FINAL_TEST": 8, "OOD": 8}
NULL_ROWS = {"FIT": 384, "SELECT": 192}
EXACT_BAR = 1e-10
BOOTSTRAPS = 2_000
ABS_TOLERANCE = 1e-12

CAPTURE_KEYS = {
    "row_id", "group_id", "split", "representation", "source_level",
    "source_value", "condition", "action", "token_ids", "query_position",
    "source_position", "source_id", "answer_id", "structural_answer_id",
    "arithmetic_answer_id", "arm", "sites", "native", "source_deleted",
    "source_deleted_logit_difference_squared_sum",
    "source_deleted_logit_vocabulary_count",
    "source_deleted_full_vocabulary_logit_rms", "r576_term_norm",
    "component_norms", "bilinear_response_relative_squared_error",
    "bilinear_response_relative_squared_error_by_site",
    "native_replay_relative_squared_error_by_row",
}
INTERVENTION_BASE_KEYS = {
    "row_id", "group_id", "split", "representation", "source_level",
    "source_value", "condition", "action", "token_ids", "query_position",
    "source_position", "source_id", "answer_id", "structural_answer_id",
    "arithmetic_answer_id", "site", "component", "arm", "native",
    "intervened", "intervention_vector_norm",
    "logit_difference_squared_sum", "logit_vocabulary_count",
    "full_vocabulary_logit_rms", "null_donor_row_id", "source_deleted",
    "source_deleted_logit_difference_squared_sum",
    "source_deleted_logit_vocabulary_count",
    "source_deleted_full_vocabulary_logit_rms",
    "source_deleted_evidence_reason",
}
NORMAL_ENDPOINT_KEYS = {
    "logsumexp", "answer_logit", "max_other_candidate_logit", "margin", "ce",
    "answer_best",
}
CONFLICT_ENDPOINT_KEYS = {
    "logsumexp", "structural_logit", "arithmetic_logit",
    "arithmetic_minus_structural",
}
EXACTNESS_KEYS = {
    "head_source_sum_relative_squared_error",
    "value_split_relative_squared_error",
    "cached_bus_relative_squared_error",
    "projected_term_relative_squared_error",
    "native_end_to_end_smoke_relative_squared_error",
    "native_replay_relative_squared_error",
    "bilinear_response_relative_squared_error",
}
RESULT_KEYS = {
    "rung", "stage", "pred_a_exact_prefix_and_bilinear_decomposition",
    "pred_b_selective_downstream_action_component",
    "pred_c_cross_representation_reuse", "all_required_gates_pass",
    "provisional_fit_selection", "selected_component", "fit_exactness",
    "select_exactness", "fit_capture_raw", "select_capture_raw", "fit_raw",
    "fit_reports", "fit_null_raw", "fit_null_reports", "select_raw",
    "select_reports", "select_null_raw", "select_null_reports",
    "component_interactions", "execution_plan", "model_forwards",
    "model_backwards", "model_weights_updated", "checkpoint_weights_sha256",
    "evaluated_splits", "forbidden_splits_opened", "implementation_sha256",
    "test_sha256", "adversarial_test_sha256", "result_contract_sha256",
    "input_sha256", "elapsed_seconds", "decision", "next_step",
}
PRED_A = "pred_a_exact_prefix_and_bilinear_decomposition"
PRED_B = "pred_b_selective_downstream_action_component"
PRED_C = "pred_c_cross_representation_reuse"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, allow_nan=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    """Write one finite JSON artifact without exposing partial destination bytes."""
    validate_standard_json(value)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"stale temporary audit artifact exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=1, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reject_constant(token: str) -> object:
    raise ValueError(f"non-finite JSON constant: {token}")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_loads(data: bytes, label: str) -> object:
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
        validate_standard_json(value)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as parse_exc:
        raise ValueError(f"{label} is not strict finite JSON: {parse_exc}") from parse_exc
    return value


def validate_standard_json(value: object, path: str = "$") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite number")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            validate_standard_json(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path}: non-string JSON key")
            validate_standard_json(item, f"{path}.{key}")
        return
    raise TypeError(f"{path}: {type(value).__name__} is not a JSON type")


def verify_preoutcome_authority() -> dict[str, str]:
    observed = {}
    for path, expected in PREOUTCOME_HASHES.items():
        if not path.is_file():
            raise RuntimeError(f"frozen pre-outcome authority missing: {path}")
        digest = sha256(path)
        if digest != expected:
            raise RuntimeError(f"frozen pre-outcome authority changed: {path}")
        observed[str(path)] = digest
    return observed


def load_r582_helper() -> ModuleType:
    verify_preoutcome_authority()
    name = "r588_pinned_r582_authority"
    spec = importlib.util.spec_from_file_location(name, R582_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen R582 helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if list(module.SITES) != list(SITES) \
            or list(module.COMPONENT_ARMS) != list(COMPONENTS) \
            or list(module.NULL_ARMS) != list(NULLS):
        raise RuntimeError("R582 frozen component grammar changed")
    return module


def load_authority() -> tuple[list[dict], ModuleType]:
    helper = load_r582_helper()
    document = strict_loads(ROWS.read_bytes(), "R582 rows")
    receipt = strict_loads(ROWS_RECEIPT.read_bytes(), "R582 rows receipt")
    if not isinstance(document, dict) or not isinstance(receipt, dict):
        raise RuntimeError("R582 authority envelopes are not mappings")
    if document.get("model_loaded") is not False or document.get("model_forwards") != 0 \
            or document.get("model_backwards") != 0 or document.get("outcomes_opened") != []:
        raise RuntimeError("R582 row authority is not outcome-blind")
    rows = document.get("rows")
    if type(rows) is not list or rows != helper.build_rows():
        raise RuntimeError("R582 rows do not exactly regenerate")
    if receipt.get("rows_sha256") != PREOUTCOME_HASHES[ROWS] \
            or receipt.get("rows") != 1_440 or receipt.get("groups") != 40:
        raise RuntimeError("R582 receipt does not bind the exact census")
    row_ids = [row["row_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("R582 row IDs are not unique")
    split_counts = collections.Counter(row["split"] for row in rows)
    group_counts = collections.Counter()
    for split in SPLIT_ROWS:
        group_counts[split] = len({row["group_id"] for row in rows if row["split"] == split})
    if dict(split_counts) != SPLIT_ROWS or dict(group_counts) != SPLIT_GROUPS:
        raise RuntimeError("R582 split census changed")
    return rows, helper


def expected_dryrun_provenance(rows: Sequence[dict], helper: ModuleType) -> dict[str, str]:
    paths = (
        ROWS, ROWS_RECEIPT, R582_PREREG, R582_HELPER, R584_NOTE,
        R576_RESULT, R579_AUDIT, R576_PREREG, R584_RUNNER, R584_TEST,
        R584_ADVERSARIAL_TEST, RESULT_CONTRACT, FACADE, R576_RUNNER, R573_RUNNER,
    )
    result = {str(path): PREOUTCOME_HASHES[path] for path in paths}
    for split in ("FIT", "SELECT"):
        maps = helper.deterministic_null_maps(rows, split)
        for null_name, mapping in maps.items():
            result[f"null_map:{split}:{null_name}"] = canonical_sha256(mapping)
    if len(result) != 19:
        raise RuntimeError("R584 dry-run provenance census changed")
    return result


def expected_result_provenance(rows: Sequence[dict], helper: ModuleType) -> dict[str, str]:
    return {
        **expected_dryrun_provenance(rows, helper),
        str(R584_DRYRUN): PREOUTCOME_HASHES[R584_DRYRUN],
        "checkpoint_weights": CHECKPOINT_SHA256,
    }


def load_execution_plan(rows: Sequence[dict], helper: ModuleType) -> dict:
    plan = strict_loads(R584_DRYRUN.read_bytes(), "R584 dry-run receipt")
    if not isinstance(plan, dict):
        raise RuntimeError("R584 dry-run receipt is not a mapping")
    expected = {
        "status": "dryrun_passed", "rung": 584, "rows": 1_440,
        "fit_rows": 576, "select_rows": 288,
        "split_batches": {"FIT": 27, "SELECT": 14},
        "null_eligible_batches": {"FIT": 20, "SELECT": 10},
        "fit_maximum_forwards": 419,
        "conditional_select_maximum_forwards": 91,
        "literal_executable_maximum_forwards": 510,
        "r582_conservative_maximum_forwards": 530,
        "selection_order": list(SELECTION_NAMES),
        "execution_plan_name": "r584_conditional_fit_then_select_v2",
        "model_loaded": False, "model_forwards": 0, "model_backwards": 0,
        "model_weights_updated": False, "opened_splits": [],
        "FINAL_TEST_or_OOD_opened": False,
        "implementation_sha256": PREOUTCOME_HASHES[R584_RUNNER],
        "test_sha256": PREOUTCOME_HASHES[R584_TEST],
        "adversarial_test_sha256": PREOUTCOME_HASHES[R584_ADVERSARIAL_TEST],
        "result_contract_sha256": PREOUTCOME_HASHES[RESULT_CONTRACT],
        "input_sha256": expected_dryrun_provenance(rows, helper),
    }
    if plan != expected:
        raise RuntimeError("R584 dry-run receipt differs from frozen execution plan")
    return plan


def _close(left: object, right: object) -> bool:
    return type(left) in (int, float) and not isinstance(left, bool) \
        and type(right) in (int, float) and not isinstance(right, bool) \
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=ABS_TOLERANCE)


def compare(expected: object, observed: object, path: str, failures: list[str]) -> None:
    if type(expected) is dict:
        if type(observed) is not dict or set(expected) != set(observed):
            failures.append(f"{path}:keys_or_type")
            return
        for key in expected:
            compare(expected[key], observed[key], f"{path}.{key}", failures)
    elif type(expected) is list:
        if type(observed) is not list or len(expected) != len(observed):
            failures.append(f"{path}:length_or_type")
            return
        for index, item in enumerate(expected):
            compare(item, observed[index], f"{path}[{index}]", failures)
    elif type(expected) is float:
        if not _close(expected, observed):
            failures.append(f"{path}:numeric")
    elif type(expected) is not type(observed) or expected != observed:
        failures.append(f"{path}:value_or_type")


def row_coordinates(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "token_ids": [int(value) for value in row["ids"]],
        "query_position": int(row["query_position"]),
        "source_position": int(row["source_position"]),
        "source_value": int(row["source_value"]),
        "source_id": int(row["source_id"]),
        "answer_id": None if row.get("answer_id") is None else int(row["answer_id"]),
        "structural_answer_id": (
            None if row.get("structural_answer_id") is None
            else int(row["structural_answer_id"])
        ),
        "arithmetic_answer_id": (
            None if row.get("arithmetic_answer_id") is None
            else int(row["arithmetic_answer_id"])
        ),
    }


def validate_identity(record: Mapping[str, object], authority: Mapping[str, object]) -> None:
    fields = (
        "row_id", "group_id", "split", "representation", "source_level",
        "source_value", "condition", "action",
    )
    for field in fields:
        if record.get(field) != authority.get(field):
            raise RuntimeError(f"row {authority['row_id']}: {field} changed")
    for field, expected in row_coordinates(authority).items():
        if record.get(field) != expected:
            raise RuntimeError(f"row {authority['row_id']}: {field} changed")


def _finite_number(value: object, label: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or isinstance(value, bool) \
            or not math.isfinite(float(value)):
        raise RuntimeError(f"{label} is not a finite number")
    number = float(value)
    if nonnegative and number < 0:
        raise RuntimeError(f"{label} is negative")
    return number


def validate_endpoint(endpoint: object, condition: str, label: str) -> None:
    if type(endpoint) is not dict:
        raise RuntimeError(f"{label} is not a mapping")
    expected_keys = CONFLICT_ENDPOINT_KEYS if condition == "step_two" else NORMAL_ENDPOINT_KEYS
    if set(endpoint) != expected_keys:
        raise RuntimeError(f"{label} endpoint fields changed")
    for key, value in endpoint.items():
        if key == "answer_best":
            if type(value) is not bool:
                raise RuntimeError(f"{label}.answer_best is not boolean")
        else:
            _finite_number(value, f"{label}.{key}")
    if condition == "step_two":
        expected = float(endpoint["arithmetic_logit"]) - float(endpoint["structural_logit"])
        if not _close(expected, endpoint["arithmetic_minus_structural"]):
            raise RuntimeError(f"{label} conflict preference identity failed")
    else:
        margin = float(endpoint["answer_logit"]) - float(endpoint["max_other_candidate_logit"])
        ce = float(endpoint["logsumexp"]) - float(endpoint["answer_logit"])
        best = bool(float(endpoint["answer_logit"]) >= float(endpoint["max_other_candidate_logit"]))
        if not _close(margin, endpoint["margin"]) or not _close(ce, endpoint["ce"]) \
                or endpoint["answer_best"] is not best:
            raise RuntimeError(f"{label} ordinary endpoint identity failed")


def validate_rms(record: Mapping[str, object], prefix: str) -> None:
    if prefix:
        squared_key = f"{prefix}_logit_difference_squared_sum"
        count_key = f"{prefix}_logit_vocabulary_count"
        rms_key = f"{prefix}_full_vocabulary_logit_rms"
    else:
        squared_key = "logit_difference_squared_sum"
        count_key = "logit_vocabulary_count"
        rms_key = "full_vocabulary_logit_rms"
    squared = _finite_number(record.get(squared_key), squared_key, nonnegative=True)
    count = record.get(count_key)
    rms = _finite_number(record.get(rms_key), rms_key, nonnegative=True)
    if type(count) is not int or isinstance(count, bool) or count <= 0:
        raise RuntimeError(f"{count_key} is not a positive integer")
    expected = math.sqrt(squared / count)
    if not math.isclose(expected, rms, rel_tol=1e-6, abs_tol=1e-8):
        raise RuntimeError(f"{rms_key} disagrees with sufficient statistics")


def exact_membership(records: object, authority: Sequence[dict], split: str,
                     label: str) -> dict[str, Mapping[str, object]]:
    if type(records) is not list:
        raise RuntimeError(f"{label} is not a list")
    expected = {str(row["row_id"]): row for row in authority if row["split"] == split}
    observed = {}
    for record in records:
        if type(record) is not dict or type(record.get("row_id")) is not str:
            raise RuntimeError(f"{label} contains an invalid record")
        if record["row_id"] in observed:
            raise RuntimeError(f"{label} contains a duplicate row")
        observed[record["row_id"]] = record
    if set(observed) != set(expected):
        raise RuntimeError(f"{label} row membership differs from authority")
    for row_id, record in observed.items():
        validate_identity(record, expected[row_id])
    return observed


def validate_capture(records: object, authority: Sequence[dict], split: str) -> dict[str, dict]:
    observed = exact_membership(records, authority, split, f"{split} capture")
    for row_id, record in observed.items():
        if set(record) != CAPTURE_KEYS:
            raise RuntimeError(f"capture row {row_id}: fields changed")
        condition = str(record["condition"])
        validate_endpoint(record["native"], condition, f"capture.{row_id}.native")
        validate_endpoint(record["source_deleted"], condition,
                          f"capture.{row_id}.source_deleted")
        validate_rms(record, "source_deleted")
        if record["arm"] != "source_present_and_deleted_capture" \
                or record["sites"] != list(SITES):
            raise RuntimeError(f"capture row {row_id}: arm/sites changed")
        _finite_number(record["r576_term_norm"], f"capture.{row_id}.term_norm",
                       nonnegative=True)
        errors = record["bilinear_response_relative_squared_error_by_site"]
        norms = record["component_norms"]
        replay = record["native_replay_relative_squared_error_by_row"]
        if type(errors) is not dict or set(errors) != {str(site) for site in SITES}:
            raise RuntimeError(f"capture row {row_id}: site exactness fields changed")
        if type(norms) is not dict or set(norms) != {str(site) for site in SITES}:
            raise RuntimeError(f"capture row {row_id}: component norm sites changed")
        for site in SITES:
            _finite_number(errors[str(site)], f"capture.{row_id}.site{site}.error",
                           nonnegative=True)
            if type(norms[str(site)]) is not dict \
                    or set(norms[str(site)]) != set(COMPONENTS):
                raise RuntimeError(f"capture row {row_id}: component norm fields changed")
            for component in COMPONENTS:
                _finite_number(norms[str(site)][component],
                               f"capture.{row_id}.site{site}.{component}.norm",
                               nonnegative=True)
        if type(replay) is not dict or set(replay) != {
                "source_present", "source_deleted", "maximum"}:
            raise RuntimeError(f"capture row {row_id}: replay fields changed")
        for key, value in replay.items():
            _finite_number(value, f"capture.{row_id}.replay.{key}", nonnegative=True)
        if not _close(max(replay["source_present"], replay["source_deleted"]),
                      replay["maximum"]):
            raise RuntimeError(f"capture row {row_id}: replay maximum identity failed")
        site_max = max(float(value) for value in errors.values())
        if not _close(site_max, record["bilinear_response_relative_squared_error"]):
            raise RuntimeError(f"capture row {row_id}: bilinear maximum identity failed")
    return observed


def validate_interventions(records: object, authority: Sequence[dict], split: str,
                           site: int, component: str, arm: str,
                           helper: ModuleType, *, null_name: str | None = None) -> list[dict]:
    selected_authority = [row for row in authority if row["split"] == split]
    if null_name is not None:
        selected_authority = [row for row in selected_authority
                              if row["condition"] in ELIGIBLE_CONDITIONS]
    observed = exact_membership(records, selected_authority, split, arm)
    donor_map = None
    if null_name is not None:
        donor_map = helper.deterministic_null_maps(authority, split)[null_name]
        if len(donor_map) != NULL_ROWS[split]:
            raise RuntimeError("frozen null donor census changed")
    ordered = []
    for record in records:
        row_id = str(record["row_id"])
        expected_keys = INTERVENTION_BASE_KEYS | (
            {"preference_sign_preserved"} if record["condition"] == "step_two"
            else {"margin_damage", "ce_increase"}
        )
        if set(record) != expected_keys:
            raise RuntimeError(f"intervention row {row_id}: fields changed")
        if record["site"] != site or record["component"] != component \
                or record["arm"] != arm:
            raise RuntimeError(f"intervention row {row_id}: arm/site/component changed")
        validate_endpoint(record["native"], str(record["condition"]),
                          f"intervention.{row_id}.native")
        validate_endpoint(record["intervened"], str(record["condition"]),
                          f"intervention.{row_id}.intervened")
        validate_endpoint(record["source_deleted"], str(record["condition"]),
                          f"intervention.{row_id}.source_deleted")
        validate_rms(record, "")
        validate_rms(record, "source_deleted")
        _finite_number(record["intervention_vector_norm"],
                       f"intervention.{row_id}.vector_norm", nonnegative=True)
        if record["source_deleted_evidence_reason"] is not None:
            raise RuntimeError(f"intervention row {row_id}: source-deleted reason is not null")
        if record["condition"] == "step_two":
            before = float(record["native"]["arithmetic_minus_structural"])
            after = float(record["intervened"]["arithmetic_minus_structural"])
            expected_sign = bool((before >= 0) == (after >= 0))
            if record["preference_sign_preserved"] is not expected_sign:
                raise RuntimeError(f"intervention row {row_id}: conflict sign changed")
        else:
            margin_damage = float(record["native"]["margin"]) \
                - float(record["intervened"]["margin"])
            ce_increase = float(record["intervened"]["ce"]) \
                - float(record["native"]["ce"])
            if not _close(margin_damage, record["margin_damage"]) \
                    or not _close(ce_increase, record["ce_increase"]):
                raise RuntimeError(f"intervention row {row_id}: effect identity failed")
        if donor_map is None:
            if record["null_donor_row_id"] is not None:
                raise RuntimeError(f"real row {row_id}: unexpected null donor")
        elif record["null_donor_row_id"] != donor_map[row_id]:
            raise RuntimeError(f"null row {row_id}: donor differs from frozen map")
        ordered.append(observed[row_id])
    return ordered


class Bootstrapper:
    def __init__(self, replicates: int):
        if type(replicates) is not int or replicates <= 0:
            raise ValueError("bootstrap replicate count must be positive")
        self.replicates = replicates
        self.traces: dict[str, dict] = {}

    def lower(self, cells: Sequence[Mapping[str, object]], key: str,
              cell_id: str) -> float:
        values = {str(cell["group_id"]): float(cell[key]) for cell in cells}
        if len(values) != len(cells) or not values:
            raise RuntimeError(f"bootstrap cell {cell_id} has duplicate/empty groups")
        groups = tuple(sorted(values))
        draws = np.empty((self.replicates, len(groups)), dtype=np.uint16)
        statistics = np.empty(self.replicates, dtype=np.float64)
        for replicate in range(self.replicates):
            sample = []
            for draw in range(len(groups)):
                payload = f"r582-group-bootstrap-v1:{cell_id}:{replicate}:{draw}".encode()
                index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(groups)
                draws[replicate, draw] = index
                sample.append(values[groups[index]])
            statistics[replicate] = float(np.mean(sample))
        if cell_id in self.traces:
            raise RuntimeError(f"bootstrap cell ID reused: {cell_id}")
        self.traces[cell_id] = {
            "ordered_group_ids": list(groups), "replicates": self.replicates,
            "draw_matrix_sha256": hashlib.sha256(
                draws.astype(">u2", copy=False).tobytes(order="C")
            ).hexdigest(),
            "statistic_vector_sha256": hashlib.sha256(
                statistics.astype(">f8", copy=False).tobytes(order="C")
            ).hexdigest(),
        }
        return float(np.quantile(statistics, .025))


def safe_ratio(numerator: float, denominator: float,
               reason: str) -> tuple[float | None, str | None]:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None, "nonfinite_ratio_input"
    if denominator <= 0:
        return None, reason
    value = numerator / denominator
    if not math.isfinite(value):
        return None, "nonfinite_ratio_result"
    return float(value), None


def score_candidate(raw: Sequence[dict], split: str, cell_prefix: str,
                    bootstrapper: Bootstrapper,
                    frozen_scales: Mapping[str, Mapping[str, float]] | None = None) -> dict:
    identity = {field: raw[0][field] for field in ("arm", "site", "component")}
    lookup = {
        (row["group_id"], row["representation"], row["source_level"], row["condition"]): row
        for row in raw
    }
    targets, copies, gaps_report = {}, {}, {}
    scales = {} if frozen_scales is None else {
        key: dict(value) for key, value in frozen_scales.items()
    }
    all_pass = True
    for representation in ("list", "digit", "word"):
        for source in (0, 1):
            for surface in ("factorial", "surface"):
                successor_condition = f"{surface}_successor"
                copy_condition = f"{surface}_copy"
                target = [row for row in raw if row["representation"] == representation
                          and row["source_level"] == source
                          and row["condition"] == successor_condition]
                copy_rows = [row for row in raw if row["representation"] == representation
                             and row["source_level"] == source
                             and row["condition"] == copy_condition]
                if len(target) != SPLIT_GROUPS[split] or len(copy_rows) != SPLIT_GROUPS[split]:
                    raise RuntimeError("candidate action cell census changed")
                key = f"{representation}:source{source}:{surface}"
                positive = float(np.mean([row["margin_damage"] > 0 for row in target]))
                lower_margin = bootstrapper.lower(
                    target, "margin_damage", f"{cell_prefix}:{key}:margin"
                )
                lower_ce = bootstrapper.lower(
                    target, "ce_increase", f"{cell_prefix}:{key}:ce"
                )
                target_norm = float(np.median([
                    row["intervention_vector_norm"] for row in target
                ]))
                copy_norm = float(np.median([
                    row["intervention_vector_norm"] for row in copy_rows
                ]))
                target_pass = bool(
                    positive >= .75 and lower_margin > 0 and lower_ce > 0
                    and target_norm > 0
                )
                targets[key] = {
                    "n": len(target), "positive_margin_damage_fraction": positive,
                    "bootstrap95_lower_mean_margin_damage": lower_margin,
                    "bootstrap95_lower_mean_ce_increase": lower_ce,
                    "median_intervention_norm": target_norm, "passed": target_pass,
                }
                if frozen_scales is None:
                    scales[key] = {
                        "margin_damage": float(np.median([
                            abs(row["margin_damage"]) for row in target
                        ])),
                        "logit_rms": float(np.median([
                            row["full_vocabulary_logit_rms"] for row in target
                        ])),
                    }
                margin_scale = float(scales[key]["margin_damage"])
                rms_scale = float(scales[key]["logit_rms"])
                answer_fraction = float(np.mean([
                    row["intervened"]["answer_best"] for row in copy_rows
                ]))
                mean_ce = float(np.mean([row["ce_increase"] for row in copy_rows]))
                margin_fraction, margin_reason = safe_ratio(
                    float(np.median([abs(row["margin_damage"]) for row in copy_rows])),
                    margin_scale, "nonpositive_successor_margin_scale"
                )
                rms_fraction, rms_reason = safe_ratio(
                    float(np.median([row["full_vocabulary_logit_rms"] for row in copy_rows])),
                    rms_scale, "nonpositive_successor_logit_rms_scale"
                )
                norm_fraction, norm_reason = safe_ratio(
                    copy_norm, target_norm, "nonpositive_successor_intervention_norm"
                )
                copy_pass = bool(
                    norm_fraction is not None and margin_fraction is not None
                    and rms_fraction is not None and norm_fraction >= .10
                    and answer_fraction >= .75 and mean_ce <= .10
                    and margin_fraction <= .25 and rms_fraction <= .25
                )
                copies[key] = {
                    "n": len(copy_rows), "median_norm_fraction": norm_fraction,
                    "median_norm_fraction_reason": norm_reason,
                    "answer_best_fraction": answer_fraction,
                    "mean_ce_increase": mean_ce,
                    "median_absolute_margin_fraction": margin_fraction,
                    "median_absolute_margin_fraction_reason": margin_reason,
                    "median_logit_rms_fraction": rms_fraction,
                    "median_logit_rms_fraction_reason": rms_reason,
                    "passed": copy_pass,
                }
                gap_cells = []
                for target_row in target:
                    copy_row = lookup[(target_row["group_id"], representation, source,
                                       copy_condition)]
                    gap_cells.append({
                        "group_id": target_row["group_id"],
                        "gap": target_row["margin_damage"]
                        - abs(copy_row["margin_damage"]),
                    })
                gap_lower = bootstrapper.lower(
                    gap_cells, "gap", f"{cell_prefix}:{key}:action_gap"
                )
                gap_pass = bool(gap_lower > 0)
                gaps_report[key] = {
                    "n": len(gap_cells),
                    "mean_gap": float(np.mean([item["gap"] for item in gap_cells])),
                    "bootstrap95_lower_mean_gap": gap_lower, "passed": gap_pass,
                }
                all_pass &= target_pass and copy_pass and gap_pass

    conflicts, activity = {}, {}
    for representation in ("list", "digit", "word"):
        for source in (0, 1):
            target_norm = targets[
                f"{representation}:source{source}:factorial"
            ]["median_intervention_norm"]
            for condition in ("relation_break", "step_two"):
                cells = [row for row in raw if row["representation"] == representation
                         and row["source_level"] == source
                         and row["condition"] == condition]
                if len(cells) != SPLIT_GROUPS[split]:
                    raise RuntimeError("control cell census changed")
                median_norm = float(np.median([
                    row["intervention_vector_norm"] for row in cells
                ]))
                fraction, reason = safe_ratio(
                    median_norm, float(target_norm),
                    "nonpositive_successor_intervention_norm"
                )
                passed = bool(fraction is not None and fraction >= .10)
                activity[f"{representation}:source{source}:{condition}"] = {
                    "n": len(cells), "median_norm_fraction_of_successor": fraction,
                    "median_norm_fraction_reason": reason, "passed": passed,
                }
                all_pass &= passed
            step_two = [row for row in raw if row["representation"] == representation
                        and row["source_level"] == source
                        and row["condition"] == "step_two"]
            fraction = float(np.mean([
                row["preference_sign_preserved"] for row in step_two
            ]))
            passed = bool(fraction >= .75)
            conflicts[f"{representation}:source{source}"] = {
                "n": len(step_two),
                "preference_sign_preserved_fraction": fraction,
                "passed": passed,
            }
            all_pass &= passed

    stability = {"source_sign": {}, "surface_recovery": {}}
    for representation in ("list", "digit", "word"):
        for surface in ("factorial", "surface"):
            a = gaps_report[f"{representation}:source0:{surface}"]
            b = gaps_report[f"{representation}:source1:{surface}"]
            groups = sorted({row["group_id"] for row in raw
                             if row["representation"] == representation})
            signs = []
            for group in groups:
                source_gaps = []
                for source in (0, 1):
                    target = lookup[(group, representation, source,
                                     f"{surface}_successor")]
                    copy_row = lookup[(group, representation, source,
                                       f"{surface}_copy")]
                    source_gaps.append(
                        target["margin_damage"] - abs(copy_row["margin_damage"])
                    )
                signs.append(source_gaps[0] * source_gaps[1] > 0)
            fraction = float(np.mean(signs))
            passed = bool(
                fraction >= .75 and a["bootstrap95_lower_mean_gap"] > 0
                and b["bootstrap95_lower_mean_gap"] > 0
            )
            stability["source_sign"][f"{representation}:{surface}"] = {
                "agreement_fraction": fraction, "passed": passed,
            }
            all_pass &= passed
        for source in (0, 1):
            ordinary = gaps_report[
                f"{representation}:source{source}:factorial"
            ]["mean_gap"]
            surface_item = gaps_report[
                f"{representation}:source{source}:surface"
            ]
            ratio, reason = safe_ratio(
                float(surface_item["mean_gap"]), float(ordinary),
                "nonpositive_ordinary_action_gap"
            )
            passed = bool(
                ratio is not None and ratio >= .50
                and surface_item["bootstrap95_lower_mean_gap"] > 0
            )
            stability["surface_recovery"][f"{representation}:source{source}"] = {
                "mean_gap_ratio": ratio, "mean_gap_ratio_reason": reason,
                "passed": passed,
            }
            all_pass &= passed

    relation = {}
    for representation in ("list", "digit", "word"):
        for source in (0, 1):
            coherent = [row["margin_damage"] for row in raw
                        if row["representation"] == representation
                        and row["source_level"] == source
                        and row["condition"] == "factorial_successor"]
            broken = [row["margin_damage"] for row in raw
                      if row["representation"] == representation
                      and row["source_level"] == source
                      and row["condition"] == "relation_break"]
            denominator = float(np.mean(coherent))
            ratio, reason = safe_ratio(
                float(np.mean(broken)), denominator, "nonpositive_coherent_damage"
            )
            relation[f"{representation}:source{source}"] = {
                "mean_broken_damage": float(np.mean(broken)),
                "mean_broken_to_coherent_ratio": ratio,
                "mean_broken_to_coherent_ratio_reason": reason,
                "gate": "characterization_only",
            }
    representation_pass = {
        representation: (
            all(item["passed"] for key, item in targets.items()
                if key.startswith(representation + ":"))
            and all(item["passed"] for key, item in copies.items()
                    if key.startswith(representation + ":"))
            and all(item["passed"] for key, item in gaps_report.items()
                    if key.startswith(representation + ":"))
        )
        for representation in ("list", "digit", "word")
    }
    report = {
        "split": split, **identity, "bootstrap_cell_prefix": cell_prefix,
        "bootstrap_specification": "r582-group-bootstrap-v1:2000_group_resamples",
        "passed_without_nulls": bool(all_pass),
        "all_representations_pass": bool(all(representation_pass.values())),
        "representation_pass": representation_pass, "targets": targets,
        "copies": copies, "action_gaps": gaps_report, "conflicts": conflicts,
        "stability": stability, "active_relation_and_conflict_controls": activity,
        "relation_characterization": relation, "fit_scales": scales,
    }
    validate_standard_json(report)
    return report


def score_null(real_raw: Sequence[dict], null_raw: Sequence[dict], split: str,
               cell_prefix: str, real_report: Mapping[str, object],
               null_name: str, bootstrapper: Bootstrapper) -> dict:
    real_lookup: dict[tuple[str, int, str], list[dict]] = collections.defaultdict(list)
    null_lookup: dict[tuple[str, int, str], list[dict]] = collections.defaultdict(list)
    for row in real_raw:
        if row["condition"] in ELIGIBLE_CONDITIONS:
            real_lookup[(row["representation"], row["source_level"], row["condition"])].append(row)
    for row in null_raw:
        null_lookup[(row["representation"], row["source_level"], row["condition"])].append(row)

    def gap_cells(target: Sequence[dict], copy_rows: Sequence[dict]) -> list[dict]:
        by_group = {row["group_id"]: row for row in copy_rows}
        return [{
            "group_id": row["group_id"],
            "gap": row["margin_damage"]
            - abs(by_group[row["group_id"]]["margin_damage"]),
        } for row in target]

    representation_cells, comparisons = {}, {}
    passed = True
    for source in (0, 1):
        for surface in ("factorial", "surface"):
            real_lowers, null_lowers, activity_passes = [], [], []
            for representation in ("list", "digit", "word"):
                successor = f"{surface}_successor"
                copy_condition = f"{surface}_copy"
                real_cell = real_lookup[(representation, source, successor)] \
                    + real_lookup[(representation, source, copy_condition)]
                null_target = null_lookup[(representation, source, successor)]
                null_copy = null_lookup[(representation, source, copy_condition)]
                key = f"{representation}:source{source}:{surface}"
                real_lower = float(
                    real_report["action_gaps"][key]["bootstrap95_lower_mean_gap"]
                )
                null_lower = bootstrapper.lower(
                    gap_cells(null_target, null_copy), "gap",
                    f"{cell_prefix}:{key}:null"
                )
                real_median = float(np.median([
                    row["intervention_vector_norm"] for row in real_cell
                ]))
                null_median = float(np.median([
                    row["intervention_vector_norm"]
                    for row in null_target + null_copy
                ]))
                ratio, reason = safe_ratio(
                    null_median, real_median, "nonpositive_real_intervention_norm"
                )
                activity_pass = bool(ratio is not None and .8 <= ratio <= 1.25)
                representation_cells[key] = {
                    "real_gap_lower95_reused": real_lower,
                    "null_gap_lower95": null_lower,
                    "real_median_intervention_norm": real_median,
                    "null_median_intervention_norm": null_median,
                    "median_null_norm_over_median_real_norm": ratio,
                    "norm_ratio_reason": reason,
                    "activity_passed": activity_pass,
                }
                real_lowers.append(real_lower)
                null_lowers.append(null_lower)
                activity_passes.append(activity_pass)
            comparison_pass = bool(
                all(activity_passes) and min(real_lowers) > max(null_lowers)
            )
            key = f"source{source}:{surface}"
            comparisons[key] = {
                "minimum_real_gap_lower95_across_representations": min(real_lowers),
                "maximum_null_gap_lower95_across_representations": max(null_lowers),
                "strict_real_exceeds_null": bool(min(real_lowers) > max(null_lowers)),
                "all_representation_activity_cells_passed": bool(all(activity_passes)),
                "passed": comparison_pass,
            }
            passed &= comparison_pass
    report = {
        "null_name": null_name, "split": split,
        "real_arm": real_raw[0]["arm"], "site": real_raw[0]["site"],
        "component": real_raw[0]["component"],
        "bootstrap_cell_prefix": cell_prefix,
        "bootstrap_specification": "r582-group-bootstrap-v1:2000_group_resamples",
        "rule": "min_rep_real_lower95_gt_max_rep_null_lower95_by_source_and_surface",
        "real_bounds_reused_without_redraw": True,
        "activity_rule": "median_null_norm_divided_by_median_real_norm_per_representation_cell",
        "passed": bool(passed), "representation_cells": representation_cells,
        "source_surface_comparisons": comparisons,
        "real_minimum_action_gap_lower95": min(
            float(item["bootstrap95_lower_mean_gap"])
            for item in real_report["action_gaps"].values()
        ),
    }
    validate_standard_json(report)
    return report


def exactness_summary(captures: Sequence[dict], saved: object,
                      label: str) -> tuple[dict, bool]:
    if type(saved) is not dict or set(saved) != EXACTNESS_KEYS:
        raise RuntimeError(f"{label} exactness fields changed")
    for key, value in saved.items():
        _finite_number(value, f"{label}.{key}", nonnegative=True)
    raw_replay = max(
        float(item["native_replay_relative_squared_error_by_row"]["maximum"])
        for item in captures
    )
    raw_bilinear = max(
        float(item["bilinear_response_relative_squared_error"])
        for item in captures
    )
    if not _close(raw_replay, saved["native_replay_relative_squared_error"]) \
            or not _close(raw_bilinear, saved["bilinear_response_relative_squared_error"]):
        raise RuntimeError(f"{label} raw/global exactness maxima disagree")
    passes = bool(
        max(float(value) for value in saved.values()) <= EXACT_BAR
        and all(float(item["r576_term_norm"]) > 0 for item in captures)
    )
    return dict(saved), passes


def interaction_records(raw_by_name: Mapping[str, Sequence[dict]], site: int) -> list[dict]:
    maps = {
        component: {row["row_id"]: row
                    for row in raw_by_name[f"mlp{site}_{component}"]}
        for component in COMPONENTS
    }
    if not (set(maps["background_cross"]) == set(maps["contrast_self"])
            == set(maps["joint_response"])):
        raise RuntimeError("interaction arm memberships differ")
    output = []
    for row_id in maps["joint_response"]:
        c, q, joint = (maps[component][row_id] for component in COMPONENTS)
        if c["condition"] == "step_two":
            continue
        native = float(c["native"]["margin"])
        for other in (q, joint):
            if not _close(native, other["native"]["margin"]):
                raise RuntimeError("interaction arms disagree on native margin")
        remove_c = float(c["intervened"]["margin"])
        remove_q = float(q["intervened"]["margin"])
        remove_joint = float(joint["intervened"]["margin"])
        output.append({
            "row_id": row_id, "group_id": c["group_id"],
            "cross": remove_c - native,
            "self": remove_q - native,
            "cross_x_self": remove_joint - remove_c - remove_q + native,
        })
    return output


def validate_result_schema(result: object, plan: Mapping[str, object],
                           expected_provenance: Mapping[str, str]) -> None:
    validate_standard_json(result)
    if type(result) is not dict or set(result) != RESULT_KEYS:
        raise RuntimeError("R584 result top-level fields changed")
    exact_types = {
        "rung": int, "stage": str,
        PRED_A: bool,
        PRED_B: bool,
        PRED_C: bool,
        "all_required_gates_pass": bool, "fit_exactness": dict,
        "fit_capture_raw": list, "fit_raw": dict, "fit_reports": dict,
        "component_interactions": dict, "execution_plan": dict,
        "model_forwards": int, "model_backwards": int,
        "model_weights_updated": bool, "checkpoint_weights_sha256": str,
        "evaluated_splits": list, "forbidden_splits_opened": list,
        "implementation_sha256": str, "test_sha256": str,
        "adversarial_test_sha256": str, "result_contract_sha256": str,
        "input_sha256": dict, "elapsed_seconds": float,
        "decision": str, "next_step": str,
    }
    for field, expected_type in exact_types.items():
        if type(result[field]) is not expected_type:
            raise RuntimeError(f"result.{field} has wrong type")
    optional_types = {
        "provisional_fit_selection": str,
        "selected_component": str,
        "select_exactness": dict,
        "select_capture_raw": list,
        "fit_null_raw": dict,
        "fit_null_reports": dict,
        "select_raw": dict,
        "select_reports": dict,
        "select_null_raw": dict,
        "select_null_reports": dict,
    }
    for field, expected_type in optional_types.items():
        if result[field] is not None and type(result[field]) is not expected_type:
            raise RuntimeError(f"result.{field} has wrong optional type")
    expected_constants = {
        "rung": 584, "stage": "cached_value_downstream_bilinear_use",
        "execution_plan": plan, "model_backwards": 0,
        "model_weights_updated": False,
        "checkpoint_weights_sha256": CHECKPOINT_SHA256,
        "forbidden_splits_opened": [],
        "implementation_sha256": PREOUTCOME_HASHES[R584_RUNNER],
        "test_sha256": PREOUTCOME_HASHES[R584_TEST],
        "adversarial_test_sha256": PREOUTCOME_HASHES[R584_ADVERSARIAL_TEST],
        "result_contract_sha256": PREOUTCOME_HASHES[RESULT_CONTRACT],
        "input_sha256": dict(expected_provenance),
    }
    failures: list[str] = []
    for key, value in expected_constants.items():
        compare(value, result[key], f"envelope.{key}", failures)
    if failures:
        raise RuntimeError(";".join(failures))
    if result["provisional_fit_selection"] is not None \
            and type(result["provisional_fit_selection"]) is not str:
        raise RuntimeError("provisional selection is not scalar string/null")
    if result["selected_component"] is not None \
            and type(result["selected_component"]) is not str:
        raise RuntimeError("selected component is not scalar string/null")
    if result["elapsed_seconds"] < 0:
        raise RuntimeError("elapsed_seconds is negative")


def expected_forward_count(provisional: str | None, selected: str | None) -> int:
    if provisional is None:
        if selected is not None:
            raise RuntimeError("selected candidate exists without provisional candidate")
        return 379
    if selected is None:
        return 419
    if selected != provisional:
        raise RuntimeError("selected candidate differs from provisional candidate")
    return 510


def audit_payload(result: object, *, replicates: int = BOOTSTRAPS) -> dict:
    failures: list[str] = []
    recomputed_decision = None
    traces: dict[str, dict] = {}
    try:
        rows, helper = load_authority()
        plan = load_execution_plan(rows, helper)
        provenance = expected_result_provenance(rows, helper)
        validate_result_schema(result, plan, provenance)
        assert isinstance(result, dict)
        fit_capture_map = validate_capture(result["fit_capture_raw"], rows, "FIT")
        fit_captures = list(result["fit_capture_raw"])
        fit_exactness, fit_exact_pass = exactness_summary(
            fit_captures, result["fit_exactness"], "FIT"
        )
        bootstrapper = Bootstrapper(replicates)

        if set(result["fit_raw"]) != set(SELECTION_NAMES) \
                or set(result["fit_reports"]) != set(SELECTION_NAMES):
            raise RuntimeError("FIT candidate arm set changed")
        fit_raw, fit_reports = {}, {}
        for site, component in SELECTION:
            name = f"mlp{site}_{component}"
            fit_raw[name] = validate_interventions(
                result["fit_raw"][name], rows, "FIT", site, component, name, helper
            )
            fit_reports[name] = score_candidate(
                fit_raw[name], "FIT", f"FIT:{name}", bootstrapper
            )
            compare(fit_reports[name], result["fit_reports"][name],
                    f"fit_reports.{name}", failures)

        provisional = (
            next((name for name in SELECTION_NAMES
                  if fit_reports[name]["passed_without_nulls"]), None)
            if fit_exact_pass else None
        )
        compare(provisional, result["provisional_fit_selection"],
                "selection.provisional", failures)

        selected = None
        fit_null_raw = fit_null_reports = None
        if provisional is None:
            if result["fit_null_raw"] is not None or result["fit_null_reports"] is not None:
                raise RuntimeError("FIT null evidence exists without provisional candidate")
        else:
            site = int(provisional[3:].split("_", 1)[0])
            component = provisional[3:].split("_", 1)[1]
            expected_null_keys = {
                f"{provisional}:null:{null_name}" for null_name in NULLS
            }
            if type(result["fit_null_raw"]) is not dict \
                    or type(result["fit_null_reports"]) is not dict \
                    or set(result["fit_null_raw"]) != expected_null_keys \
                    or set(result["fit_null_reports"]) != expected_null_keys:
                raise RuntimeError("FIT null arm set changed")
            fit_null_raw, fit_null_reports = {}, {}
            for null_name in NULLS:
                key = f"{provisional}:null:{null_name}"
                fit_null_raw[key] = validate_interventions(
                    result["fit_null_raw"][key], rows, "FIT", site, component,
                    f"null:{null_name}", helper, null_name=null_name
                )
                fit_null_reports[key] = score_null(
                    fit_raw[provisional], fit_null_raw[key], "FIT", f"FIT:{key}",
                    fit_reports[provisional], null_name, bootstrapper
                )
                compare(fit_null_reports[key], result["fit_null_reports"][key],
                        f"fit_null_reports.{key}", failures)
            if all(report["passed"] for report in fit_null_reports.values()):
                selected = provisional
        compare(selected, result["selected_component"], "selection.selected", failures)

        select_raw = select_reports = select_null_raw = select_null_reports = None
        select_exactness = None
        if selected is None:
            for field in (
                "select_capture_raw", "select_exactness", "select_raw", "select_reports",
                "select_null_raw", "select_null_reports",
            ):
                if result[field] is not None:
                    raise RuntimeError(f"{field} exists while SELECT is closed")
            expected_opened = ["FIT"]
        else:
            expected_opened = ["FIT", "SELECT"]
            select_captures = list(result["select_capture_raw"])
            validate_capture(select_captures, rows, "SELECT")
            select_exactness, _ = exactness_summary(
                select_captures, result["select_exactness"], "SELECT"
            )
            site = int(selected[3:].split("_", 1)[0])
            selected_component = selected[3:].split("_", 1)[1]
            select_names = {f"mlp{site}_{component}" for component in COMPONENTS}
            if type(result["select_raw"]) is not dict \
                    or type(result["select_reports"]) is not dict \
                    or set(result["select_raw"]) != select_names \
                    or set(result["select_reports"]) != select_names:
                raise RuntimeError("SELECT component arm set changed")
            select_raw, select_reports = {}, {}
            for component in COMPONENTS:
                name = f"mlp{site}_{component}"
                select_raw[name] = validate_interventions(
                    result["select_raw"][name], rows, "SELECT", site, component,
                    name, helper
                )
                select_reports[name] = score_candidate(
                    select_raw[name], "SELECT", f"SELECT:{name}", bootstrapper,
                    frozen_scales=fit_reports[selected]["fit_scales"]
                )
                compare(select_reports[name], result["select_reports"][name],
                        f"select_reports.{name}", failures)
            expected_null_keys = {f"{selected}:null:{name}" for name in NULLS}
            if type(result["select_null_raw"]) is not dict \
                    or type(result["select_null_reports"]) is not dict \
                    or set(result["select_null_raw"]) != expected_null_keys \
                    or set(result["select_null_reports"]) != expected_null_keys:
                raise RuntimeError("SELECT null arm set changed")
            select_null_raw, select_null_reports = {}, {}
            for null_name in NULLS:
                key = f"{selected}:null:{null_name}"
                select_null_raw[key] = validate_interventions(
                    result["select_null_raw"][key], rows, "SELECT", site,
                    selected_component, f"null:{null_name}", helper,
                    null_name=null_name
                )
                select_null_reports[key] = score_null(
                    select_raw[selected], select_null_raw[key], "SELECT",
                    f"SELECT:{key}", select_reports[selected], null_name,
                    bootstrapper
                )
                compare(select_null_reports[key], result["select_null_reports"][key],
                        f"select_null_reports.{key}", failures)

        compare(expected_opened, result["evaluated_splits"],
                "split.evaluated_splits", failures)
        exact = bool(
            result["checkpoint_weights_sha256"] == CHECKPOINT_SHA256
            and fit_exact_pass
            and (selected is None or (
                max(float(value) for value in select_exactness.values()) <= EXACT_BAR
                and all(float(item["r576_term_norm"]) > 0
                        for item in result["select_capture_raw"])
            ))
        )
        fit_selected_pass = bool(selected is not None)
        select_selected_pass = bool(
            selected is not None
            and select_reports[selected]["passed_without_nulls"]
            and all(report["passed"] for report in select_null_reports.values())
        )
        reuse = bool(
            fit_selected_pass and fit_reports[selected]["all_representations_pass"]
            and select_selected_pass
            and select_reports[selected]["all_representations_pass"]
        )
        all_pass = bool(exact and fit_selected_pass and select_selected_pass and reuse)
        predicates = {
            PRED_A: exact,
            PRED_B: bool(
                fit_selected_pass and select_selected_pass
            ),
            PRED_C: reuse,
            "all_required_gates_pass": all_pass,
        }
        for key, value in predicates.items():
            compare(value, result[key], f"predicate.{key}", failures)

        interactions = {"fit": None, "select": None}
        if selected is not None:
            site = int(selected[3:].split("_", 1)[0])
            interactions["fit"] = interaction_records(fit_raw, site)
            interactions["select"] = interaction_records(select_raw, site)
        compare(interactions, result["component_interactions"],
                "component_interactions", failures)

        price = expected_forward_count(provisional, selected)
        compare(price, result["model_forwards"], "price.model_forwards", failures)
        decision = (
            "downstream_use_component_held" if all_pass
            else "downstream_use_decomposition_null"
        )
        next_step = (
            "independent_cpu_audit_then_FINAL_TEST_remains_separately_preregistered"
            if all_pass else
            "retain_R576_broad_carrier_and_do_not_promote_R582_component"
        )
        compare(decision, result["decision"], "terminal.decision", failures)
        compare(next_step, result["next_step"], "terminal.next_step", failures)
        recomputed_decision = decision
        traces = bootstrapper.traces
        trace_hash = canonical_sha256({key: traces[key] for key in sorted(traces)})
        return {
            "audit_verdict": (
                "held_independent_audit" if not failures
                else "failed_independent_audit"
            ),
            "audit_failures": failures,
            "independently_recomputed_decision": recomputed_decision,
            "independently_recomputed_predicates": predicates,
            "independently_recomputed_provisional": provisional,
            "independently_recomputed_selected": selected,
            "independently_recomputed_opened_splits": expected_opened,
            "independently_recomputed_model_forwards": price,
            "raw_counts": {
                "fit_capture": len(fit_capture_map),
                "fit_real_arms": len(fit_raw),
                "fit_null_arms": 0 if fit_null_raw is None else len(fit_null_raw),
                "select_real_arms": 0 if select_raw is None else len(select_raw),
                "select_null_arms": 0 if select_null_raw is None else len(select_null_raw),
            },
            "bootstrap_replicates_per_cell": replicates,
            "bootstrap_cell_count": len(traces),
            "bootstrap_trace_sha256": trace_hash,
            "bootstrap_traces": {key: traces[key] for key in sorted(traces)},
        }
    except (AssertionError, KeyError, TypeError, ValueError, RuntimeError) as audit_exc:
        return {
            "audit_verdict": "failed_independent_audit",
            "audit_failures": [f"integrity_or_reconstruction:{type(audit_exc).__name__}:{audit_exc}"],
            "independently_recomputed_decision": recomputed_decision,
            "independently_recomputed_predicates": None,
            "independently_recomputed_provisional": None,
            "independently_recomputed_selected": None,
            "independently_recomputed_opened_splits": None,
            "independently_recomputed_model_forwards": None,
            "raw_counts": None,
            "bootstrap_replicates_per_cell": replicates,
            "bootstrap_cell_count": len(traces),
            "bootstrap_trace_sha256": None,
            "bootstrap_traces": {},
        }


def read_stable_source(path: Path = SOURCE_RESULT) -> bytes:
    """Read the source only in the eventual non-dry-run audit entry point."""
    first_stat = path.stat()
    first = path.read_bytes()
    second = path.read_bytes()
    second_stat = path.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(first_stat) != identity(second_stat) or first != second:
        raise RuntimeError("R584 result changed during stable read")
    strict_loads(first, "R584 source result")
    return first


def _native_endpoint(row: Mapping[str, object]) -> dict:
    if row["condition"] == "step_two":
        return {
            "logsumexp": 3.0, "structural_logit": 0.0,
            "arithmetic_logit": 1.0, "arithmetic_minus_structural": 1.0,
        }
    return {
        "logsumexp": 3.0, "answer_logit": 2.0,
        "max_other_candidate_logit": 0.0, "margin": 2.0, "ce": 1.0,
        "answer_best": True,
    }


def _intervened_endpoint(row: Mapping[str, object], damage: float,
                         ce_increase: float) -> dict:
    if row["condition"] == "step_two":
        return {
            "logsumexp": 3.0, "structural_logit": 0.0,
            "arithmetic_logit": .5, "arithmetic_minus_structural": .5,
        }
    answer = 2.0 - damage
    ce = 1.0 + ce_increase
    return {
        "logsumexp": answer + ce, "answer_logit": answer,
        "max_other_candidate_logit": 0.0, "margin": answer, "ce": ce,
        "answer_best": bool(answer >= 0.0),
    }


def fixture_capture(rows: Sequence[dict], split: str) -> list[dict]:
    output = []
    for row in rows:
        if row["split"] != split:
            continue
        output.append({
            "row_id": row["row_id"], "group_id": row["group_id"], "split": split,
            "representation": row["representation"], "source_level": row["source_level"],
            "source_value": row["source_value"], "condition": row["condition"],
            "action": row["action"], **row_coordinates(row),
            "arm": "source_present_and_deleted_capture", "sites": list(SITES),
            "native": _native_endpoint(row), "source_deleted": _native_endpoint(row),
            "source_deleted_logit_difference_squared_sum": 0.0,
            "source_deleted_logit_vocabulary_count": 50_304,
            "source_deleted_full_vocabulary_logit_rms": 0.0,
            "r576_term_norm": 1.0,
            "component_norms": {
                str(site): {component: 1.0 for component in COMPONENTS}
                for site in SITES
            },
            "bilinear_response_relative_squared_error": 0.0,
            "bilinear_response_relative_squared_error_by_site": {
                str(site): 0.0 for site in SITES
            },
            "native_replay_relative_squared_error_by_row": {
                "source_present": 0.0, "source_deleted": 0.0, "maximum": 0.0,
            },
        })
    return output


def fixture_interventions(rows: Sequence[dict], split: str, site: int,
                          component: str, *, target_damage: float,
                          null_name: str | None, helper: ModuleType) -> list[dict]:
    donor_map = None if null_name is None else helper.deterministic_null_maps(rows, split)[null_name]
    output = []
    for row in rows:
        if row["split"] != split or (null_name is not None
                                     and row["condition"] not in ELIGIBLE_CONDITIONS):
            continue
        if null_name is not None:
            damage, ce_increase = 0.0, 0.0
        elif row["condition"] == "step_two":
            damage, ce_increase = 0.0, 0.0
        elif row["action"] == "successor":
            damage, ce_increase = target_damage, (1.0 if target_damage > 0 else 0.0)
        else:
            damage, ce_increase = .05, 0.0
        rms = 0.0 if null_name is not None else (1.0 if row["action"] == "successor" else .1)
        record = {
            "row_id": row["row_id"], "group_id": row["group_id"], "split": split,
            "representation": row["representation"], "source_level": row["source_level"],
            "source_value": row["source_value"], "condition": row["condition"],
            "action": row["action"], **row_coordinates(row),
            "site": site, "component": component,
            "arm": (f"mlp{site}_{component}" if null_name is None
                    else f"null:{null_name}"),
            "native": _native_endpoint(row),
            "intervened": _intervened_endpoint(row, damage, ce_increase),
            "intervention_vector_norm": 1.0,
            "logit_difference_squared_sum": rms * rms * 50_304,
            "logit_vocabulary_count": 50_304,
            "full_vocabulary_logit_rms": rms,
            "null_donor_row_id": None if donor_map is None else donor_map[row["row_id"]],
            "source_deleted": _native_endpoint(row),
            "source_deleted_logit_difference_squared_sum": 0.0,
            "source_deleted_logit_vocabulary_count": 50_304,
            "source_deleted_full_vocabulary_logit_rms": 0.0,
            "source_deleted_evidence_reason": None,
        }
        if row["condition"] == "step_two":
            record["preference_sign_preserved"] = True
        else:
            record["margin_damage"] = damage
            record["ce_increase"] = ce_increase
        output.append(record)
    return output


def fixture_exactness() -> dict[str, float]:
    return {key: 0.0 for key in EXACTNESS_KEYS}


def make_fixture(*, held: bool, replicates: int) -> dict:
    rows, helper = load_authority()
    plan = load_execution_plan(rows, helper)
    target_damage = 1.0 if held else -1.0
    fit_capture = fixture_capture(rows, "FIT")
    fit_raw = {}
    fit_reports = {}
    bootstrapper = Bootstrapper(replicates)
    for site, component in SELECTION:
        name = f"mlp{site}_{component}"
        fit_raw[name] = fixture_interventions(
            rows, "FIT", site, component, target_damage=target_damage,
            null_name=None, helper=helper
        )
        fit_reports[name] = score_candidate(
            fit_raw[name], "FIT", f"FIT:{name}", bootstrapper
        )
    provisional = SELECTION_NAMES[0] if held else None
    selected = provisional
    fit_null_raw = fit_null_reports = None
    select_capture = select_exactness = select_raw = select_reports = None
    select_null_raw = select_null_reports = None
    interactions = {"fit": None, "select": None}
    if held:
        site, component = 8, "background_cross"
        fit_null_raw, fit_null_reports = {}, {}
        for null_name in NULLS:
            key = f"{provisional}:null:{null_name}"
            fit_null_raw[key] = fixture_interventions(
                rows, "FIT", site, component, target_damage=target_damage,
                null_name=null_name, helper=helper
            )
            fit_null_reports[key] = score_null(
                fit_raw[provisional], fit_null_raw[key], "FIT", f"FIT:{key}",
                fit_reports[provisional], null_name, bootstrapper
            )
        select_capture = fixture_capture(rows, "SELECT")
        select_exactness = fixture_exactness()
        select_raw, select_reports = {}, {}
        for selected_component in COMPONENTS:
            name = f"mlp8_{selected_component}"
            select_raw[name] = fixture_interventions(
                rows, "SELECT", 8, selected_component, target_damage=1.0,
                null_name=None, helper=helper
            )
            select_reports[name] = score_candidate(
                select_raw[name], "SELECT", f"SELECT:{name}", bootstrapper,
                frozen_scales=fit_reports[provisional]["fit_scales"]
            )
        select_null_raw, select_null_reports = {}, {}
        for null_name in NULLS:
            key = f"{selected}:null:{null_name}"
            select_null_raw[key] = fixture_interventions(
                rows, "SELECT", 8, component, target_damage=1.0,
                null_name=null_name, helper=helper
            )
            select_null_reports[key] = score_null(
                select_raw[selected], select_null_raw[key], "SELECT",
                f"SELECT:{key}", select_reports[selected], null_name, bootstrapper
            )
        interactions = {
            "fit": interaction_records(fit_raw, 8),
            "select": interaction_records(select_raw, 8),
        }
    exact = True
    pred_b = held
    reuse = held
    all_pass = held
    return {
        "rung": 584, "stage": "cached_value_downstream_bilinear_use",
        PRED_A: exact,
        PRED_B: pred_b,
        PRED_C: reuse,
        "all_required_gates_pass": all_pass,
        "provisional_fit_selection": provisional,
        "selected_component": selected,
        "fit_exactness": fixture_exactness(), "select_exactness": select_exactness,
        "fit_capture_raw": fit_capture, "select_capture_raw": select_capture,
        "fit_raw": fit_raw, "fit_reports": fit_reports,
        "fit_null_raw": fit_null_raw, "fit_null_reports": fit_null_reports,
        "select_raw": select_raw, "select_reports": select_reports,
        "select_null_raw": select_null_raw,
        "select_null_reports": select_null_reports,
        "component_interactions": interactions, "execution_plan": plan,
        "model_forwards": 510 if held else 379, "model_backwards": 0,
        "model_weights_updated": False,
        "checkpoint_weights_sha256": CHECKPOINT_SHA256,
        "evaluated_splits": ["FIT", "SELECT"] if held else ["FIT"],
        "forbidden_splits_opened": [],
        "implementation_sha256": PREOUTCOME_HASHES[R584_RUNNER],
        "test_sha256": PREOUTCOME_HASHES[R584_TEST],
        "adversarial_test_sha256": PREOUTCOME_HASHES[R584_ADVERSARIAL_TEST],
        "result_contract_sha256": PREOUTCOME_HASHES[RESULT_CONTRACT],
        "input_sha256": expected_result_provenance(rows, helper),
        "elapsed_seconds": 0.0,
        "decision": (
            "downstream_use_component_held" if held
            else "downstream_use_decomposition_null"
        ),
        "next_step": (
            "independent_cpu_audit_then_FINAL_TEST_remains_separately_preregistered"
            if held else "retain_R576_broad_carrier_and_do_not_promote_R582_component"
        ),
    }


def make_fit_null_failure_fixture(*, replicates: int) -> dict:
    """Plant the legal 419-forward path: provisional FIT, failed active null."""
    result = make_fixture(held=True, replicates=replicates)
    rows, helper = load_authority()
    selected = str(result["selected_component"])
    first_null = f"{selected}:null:{NULLS[0]}"
    for row in result["fit_null_raw"][first_null]:
        row["intervention_vector_norm"] = 2.0
    bootstrapper = Bootstrapper(replicates)
    for null_name in NULLS:
        key = f"{selected}:null:{null_name}"
        result["fit_null_reports"][key] = score_null(
            result["fit_raw"][selected], result["fit_null_raw"][key], "FIT",
            f"FIT:{key}", result["fit_reports"][selected], null_name,
            bootstrapper,
        )
    result.update({
        "selected_component": None,
        "select_exactness": None,
        "select_capture_raw": None,
        "select_raw": None,
        "select_reports": None,
        "select_null_raw": None,
        "select_null_reports": None,
        "component_interactions": {"fit": None, "select": None},
        PRED_B: False,
        PRED_C: False,
        "all_required_gates_pass": False,
        "model_forwards": 419,
        "evaluated_splits": ["FIT"],
        "decision": "downstream_use_decomposition_null",
        "next_step": "retain_R576_broad_carrier_and_do_not_promote_R582_component",
    })
    return result


def run_dryrun() -> dict:
    # Intentionally no filesystem operation involving SOURCE_RESULT.
    verify_preoutcome_authority()
    fixture_replicates = 7
    held = make_fixture(held=True, replicates=fixture_replicates)
    null = make_fixture(held=False, replicates=fixture_replicates)
    held_audit = audit_payload(held, replicates=fixture_replicates)
    null_audit = audit_payload(null, replicates=fixture_replicates)
    if held_audit["audit_verdict"] != "held_independent_audit" \
            or held_audit["independently_recomputed_decision"] != "downstream_use_component_held":
        raise RuntimeError("planted held fixture failed independent audit")
    if null_audit["audit_verdict"] != "held_independent_audit" \
            or null_audit["independently_recomputed_decision"] != "downstream_use_decomposition_null":
        raise RuntimeError("planted scientific-null fixture failed independent audit")

    null_key = f"{SELECTION_NAMES[0]}:null:{NULLS[0]}"
    mutations = {
        "missing_arm": lambda value: value["fit_raw"].pop(SELECTION_NAMES[-1]),
        "missing_row": lambda value: value["fit_raw"][SELECTION_NAMES[0]].pop(),
        "nonfinite": lambda value: value["fit_raw"][SELECTION_NAMES[0]][0].update(
            {"margin_damage": float("nan")}
        ),
        "wrong_donor": lambda value: value["fit_null_raw"][null_key][0].update({
            "null_donor_row_id": value["fit_null_raw"][null_key][0]["row_id"]
        }),
        "wrong_price": lambda value: value.update({"model_forwards": 509}),
        "wrong_interaction": lambda value: value["component_interactions"]["fit"][0].update({
            "cross_x_self": value["component_interactions"]["fit"][0]["cross_x_self"] + .1
        }),
        "stale_provenance": lambda value: value["input_sha256"].update({
            str(R584_DRYRUN): "0" * 64
        }),
    }
    malformed_verdicts = {}
    for name, mutate in mutations.items():
        malformed = copy.deepcopy(held)
        mutate(malformed)
        malformed_verdicts[name] = audit_payload(
            malformed, replicates=3
        )["audit_verdict"]
        del malformed
    if set(malformed_verdicts.values()) != {"failed_independent_audit"}:
        raise RuntimeError("a malformed planted fixture passed")
    rows, helper = load_authority()
    receipt = {
        "schema": "numbered_list_cached_value_downstream_use_rung588_dryrun_v1",
        "status": "dryrun_passed", "repaired_commit": REPAIRED_COMMIT,
        "preoutcome_authority_sha256": verify_preoutcome_authority(),
        "authority_rows": len(rows),
        "split_rows": dict(SPLIT_ROWS), "split_groups": dict(SPLIT_GROUPS),
        "null_rows": dict(NULL_ROWS), "real_audit_bootstrap_replicates": BOOTSTRAPS,
        "fixture_bootstrap_replicates": fixture_replicates,
        "held_fixture_audit_verdict": held_audit["audit_verdict"],
        "held_fixture_decision": held_audit["independently_recomputed_decision"],
        "held_fixture_forwards": held_audit["independently_recomputed_model_forwards"],
        "null_fixture_audit_verdict": null_audit["audit_verdict"],
        "null_fixture_decision": null_audit["independently_recomputed_decision"],
        "null_fixture_forwards": null_audit["independently_recomputed_model_forwards"],
        "malformed_fixture_verdicts": malformed_verdicts,
        "source_result_touched": False, "model_loaded": False,
        "model_forwards": 0, "model_backwards": 0,
        "model_weights_updated": False, "opened_splits": [],
        "forbidden_splits_opened": [],
        "script_sha256": sha256(SCRIPT),
        "test_sha256": sha256(TEST) if TEST.is_file() else None,
        "preregistration_sha256": sha256(PREREG),
        "execution_plan_sha256": PREOUTCOME_HASHES[R584_DRYRUN],
        "result_provenance_key_count": len(expected_result_provenance(rows, helper)),
    }
    validate_standard_json(receipt)
    atomic_write_json(DRYRUN, receipt)
    return receipt


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(run_dryrun(), indent=2, allow_nan=False))
        return
    if OUT.exists():
        raise RuntimeError("R588 audit namespace already exists")
    source_bytes = read_stable_source()
    result = strict_loads(source_bytes, "R584 source result")
    audit = audit_payload(result, replicates=BOOTSTRAPS)
    audit.update({
        "schema": "numbered_list_cached_value_downstream_use_rung588_audit_v1",
        "rung": 588, "source_result_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_result_bytes_stable": True,
        "preoutcome_authority_sha256": verify_preoutcome_authority(),
        "auditor_implementation_sha256": sha256(SCRIPT),
        "auditor_test_sha256": sha256(TEST),
        "auditor_preregistration_sha256": sha256(PREREG),
        "model_loaded": False, "model_forwards": 0, "model_backwards": 0,
        "model_weights_updated": False, "evaluated_splits": [],
        "forbidden_splits_opened": [],
    })
    atomic_write_json(OUT, audit)
    print(json.dumps({key: audit[key] for key in (
        "audit_verdict", "audit_failures", "independently_recomputed_decision",
        "independently_recomputed_predicates", "independently_recomputed_provisional",
        "independently_recomputed_selected", "independently_recomputed_opened_splits",
        "independently_recomputed_model_forwards", "bootstrap_cell_count",
        "bootstrap_trace_sha256", "source_result_sha256", "model_forwards",
    )}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
