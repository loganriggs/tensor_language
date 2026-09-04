#!/usr/bin/env python3
"""Independent CPU audit of the exact approved R590 evidence package."""

# BQLANE: cpu

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
REPO = ROOT.parents[1]
SCRIPT = Path(__file__).resolve()
TEST = SCRIPT.with_name(
    "test_audit_numbered_list_cached_value_downstream_use_rung590_postexecution.py"
)
APPROVED_COMMIT = "3eb52938b3641f067d8f8eb9e654f461cbd61ad0"
ADAPTER = OPS / "execute_numbered_list_cached_value_downstream_use_rung590.py"
PRODUCER = OPS / "numbered_list_cached_value_downstream_use_rung590.py"
OWNER_TEST = OPS / "test_numbered_list_cached_value_downstream_use_rung590.py"
DRYRUN = ROOT / "numbered_list_cached_value_downstream_use_rung590_dryrun.json"
NOTE = ROOT.parent / "polynomial_causal" / (
    "NUMBERED_LIST_CACHED_VALUE_DOWNSTREAM_USE_RUNG590_"
    "PROSPECTIVE_CONTRACT_REPLICATION.md"
)
RESULT = ROOT / "numbered_list_cached_value_downstream_use_rung590_results.json"
RECEIPT = ROOT / "numbered_list_cached_value_downstream_use_rung590_receipt.json"
EVIDENCE = ROOT / (
    "numbered_list_cached_value_downstream_use_rung590_evidence/primitive_evidence.json"
)
RUNLOG = ROOT / "runlogs/execute_numbered_list_cached_value_downstream_use_rung590.log"
OUT = ROOT / "numbered_list_cached_value_downstream_use_rung590_postexecution_audit.json"

APPROVED_HASHES = {
    PRODUCER: "c38654506f36fcf111f3a34f356893240548c3cfbf4eded58efb04d31fdb2e36",
    OWNER_TEST: "49f6f7a998bfb69331c36391f5d3c16d9b702c1fd60b4da5b09c920f3832e5b0",
    DRYRUN: "3ebada19f74906ba3e7cd1637fc1cd6cdff84936124dee01cb058875432d3b95",
    ADAPTER: "c525cad078935ef0552214fba13c16a5d56483c8e3048bbec4d6ab9ef3f17885",
    NOTE: "dae72b4aee35030f31ce42674d9535d6bff6c857b9beb8633a8ac809edaf031b",
}
OUTCOME_HASHES = {
    RESULT: "e868a0e67aa9e4d3251deccbf625b7708ee1f1ce9070afbdfce2357a9f0bed24",
    RECEIPT: "2789d205c311ffaa8401edd761a787b8061a138ad297ab5c7f4b67b45ba3b20d",
    EVIDENCE: "3025923441b40d7ced3a0d9b8277ade3639d87deb4b8fe2c6a00438c9fcf4815",
}
RUNLOG_SHA256 = "182554a1824fbd7254893d5df28704fb891fc6f78d22d75519f8192c1b657456"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def git_blob(path: Path) -> bytes:
    relative = path.relative_to(REPO).as_posix()
    return subprocess.check_output(
        ["git", "show", f"{APPROVED_COMMIT}:{relative}"], cwd=REPO
    )


def load_approved_modules() -> tuple[ModuleType, ModuleType]:
    adapter_bytes = git_blob(ADAPTER)
    if digest(adapter_bytes) != APPROVED_HASHES[ADAPTER]:
        raise RuntimeError("approved R590 adapter Git blob changed")
    name = "r590_postexecution_approved_adapter"
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(ADAPTER))
    if spec is None:
        raise RuntimeError("cannot construct approved R590 adapter module")
    adapter = importlib.util.module_from_spec(spec)
    adapter.__file__ = str(ADAPTER)
    sys.modules[name] = adapter
    exec(compile(adapter_bytes, str(ADAPTER), "exec"), adapter.__dict__)
    snapshot = adapter.capture_frozen_bytes()
    producer = adapter.load_frozen_producer(snapshot)
    for path, expected in APPROVED_HASHES.items():
        observed = digest(git_blob(path))
        if observed != expected or digest(path.read_bytes()) != expected:
            raise RuntimeError(f"approved R590 byte changed: {path}")
    return adapter, producer


def compare(expected: object, observed: object, label: str, r588: ModuleType) -> None:
    failures: list[str] = []
    r588.compare(expected, observed, label, failures)
    if failures:
        raise RuntimeError(";".join(failures[:20]))


def expected_receipt(
    result: Mapping[str, object], result_bytes: bytes, evidence_bytes: bytes,
    producer: ModuleType,
) -> dict[str, object]:
    result_sha = digest(result_bytes)
    evidence_sha = digest(evidence_bytes)
    return {
        "schema": producer.RECEIPT_SCHEMA,
        "rung": 590,
        "source_rung": 584,
        "package_id": digest(f"r590:{result_sha}:{evidence_sha}".encode()),
        "result_path": str(RESULT.relative_to(REPO)),
        "result_sha256": result_sha,
        "evidence_path": str(EVIDENCE.relative_to(REPO)),
        "evidence_sha256": evidence_sha,
        "implementation_sha256": APPROVED_HASHES[PRODUCER],
        "test_sha256": APPROVED_HASHES[OWNER_TEST],
        "note_sha256": APPROVED_HASHES[NOTE],
        "checkpoint_weights_sha256": result["checkpoint_weights_sha256"],
        "evaluated_splits": result["evaluated_splits"],
        "model_forwards": result["model_forwards"],
        "model_backwards": 0,
        "model_weights_updated": False,
        "decision": result["decision"],
    }


def audit_bytes(
    result_bytes: bytes,
    receipt_bytes: bytes,
    evidence_bytes: bytes,
    producer: ModuleType,
    *,
    bind_observed_hashes: bool,
    replicates: int = 2_000,
) -> dict[str, object]:
    """Rebuild the actual FIT-only R590 terminal from primitive evidence."""
    try:
        if bind_observed_hashes:
            observed = {
                RESULT: digest(result_bytes), RECEIPT: digest(receipt_bytes),
                EVIDENCE: digest(evidence_bytes),
            }
            if observed != OUTCOME_HASHES:
                raise RuntimeError("R590 outcome bytes differ from the landed package")

        r588 = producer.r588
        result = r588.strict_loads(result_bytes, "R590 result")
        receipt = r588.strict_loads(receipt_bytes, "R590 receipt")
        evidence = r588.strict_loads(evidence_bytes, "R590 evidence")
        if not all(type(value) is dict for value in (result, receipt, evidence)):
            raise RuntimeError("R590 package members must be JSON objects")
        if result_bytes != canonical_bytes(result) \
                or receipt_bytes != canonical_bytes(receipt) \
                or evidence_bytes != canonical_bytes(evidence):
            raise RuntimeError("R590 package is not canonical strict JSON")

        if evidence.get("schema") != producer.EVIDENCE_SCHEMA \
                or evidence.get("rung") != 590 or evidence.get("source_rung") != 584:
            raise RuntimeError("R590 evidence identity changed")
        if result.get("schema") != producer.RESULT_SCHEMA \
                or result.get("rung") != 590 or result.get("source_rung") != 584:
            raise RuntimeError("R590 result identity changed")

        rows = producer.load_outcome_blind_authority()
        if len(rows) != 1_440:
            raise RuntimeError("R590 row authority changed")
        opened = evidence.get("evaluated_splits")
        if opened != ["FIT"]:
            raise RuntimeError("landed R590 package is not the registered FIT-only terminal")
        support = producer.frozen_phase_support_census(rows, ["FIT"])
        compare(support, evidence.get("phase_support_census"), "evidence.support", r588)
        support_sha = producer.canonical_sha256(support)
        if evidence.get("phase_support_census_sha256") != support_sha:
            raise RuntimeError("R590 support hash changed")

        captures = list(evidence.get("fit_capture_raw", []))
        capture_by_id = r588.validate_capture(captures, rows, "FIT")
        exactness, exact_pass = r588.exactness_summary(
            captures, evidence.get("fit_exactness"), "FIT"
        )
        if not exact_pass or max(float(value) for value in exactness.values()) > producer.EXACT_BAR:
            raise RuntimeError("R590 instrument exactness is invalid")

        fit_raw_value = evidence.get("fit_raw")
        if type(fit_raw_value) is not dict \
                or set(fit_raw_value) != set(producer.SELECTION_NAMES):
            raise RuntimeError("R590 FIT arm set changed")
        bootstrapper = r588.Bootstrapper(replicates)
        reports: dict[str, dict] = {}
        pass_counts: dict[str, dict[str, int]] = {}
        for site, component in producer.SELECTION:
            name = f"mlp{site}_{component}"
            records = r588.validate_interventions(
                fit_raw_value[name], rows, "FIT", site, component, name,
                producer.r584.r582,
            )
            producer.validate_intervention_capture_join(
                records, capture_by_id, f"audit.FIT.{name}"
            )
            report = r588.score_candidate(
                records, "FIT", f"FIT:{name}", bootstrapper
            )
            reports[name] = report
            pass_counts[name] = {
                "target_cells": sum(item["passed"] for item in report["targets"].values()),
                "copy_cells": sum(item["passed"] for item in report["copies"].values()),
                "action_gap_cells": sum(
                    item["passed"] for item in report["action_gaps"].values()
                ),
                "active_relation_conflict_cells": sum(
                    item["passed"]
                    for item in report["active_relation_and_conflict_controls"].values()
                ),
                "conflict_preservation_cells": sum(
                    item["passed"] for item in report["conflicts"].values()
                ),
            }
        provisional = next(
            (name for name in producer.SELECTION_NAMES if reports[name]["passed_without_nulls"]),
            None,
        )
        if provisional is not None:
            raise RuntimeError("landed FIT evidence unexpectedly has a provisional candidate")
        for field in (
            "fit_null_raw", "select_exactness", "select_capture_raw",
            "select_raw", "select_null_raw",
        ):
            if evidence.get(field) is not None:
                raise RuntimeError(f"{field} exists on the FIT-only terminal")

        compare(reports, result.get("fit_reports"), "result.fit_reports", r588)
        expected_ids = [
            call["call_id"] for call in producer.build_forward_call_manifest(rows)
            if call["guard"] == "fit_always"
        ]
        if len(expected_ids) != 379 \
                or evidence.get("executed_forward_call_ids") != expected_ids \
                or result.get("executed_forward_call_ids") != expected_ids:
            raise RuntimeError("R590 executed call schedule changed")
        trace_sha = producer.canonical_sha256({
            key: bootstrapper.traces[key] for key in sorted(bootstrapper.traces)
        })
        expected_terminal = {
            "provisional_fit_selection": None,
            "selected_component": None,
            "fit_null_reports": None,
            "select_reports": None,
            "select_null_reports": None,
            "component_interactions": {"fit": None, "select": None},
            "fit_exactness": exactness,
            "select_exactness": None,
            "evaluated_splits": ["FIT"],
            "model_forwards": 379,
            "model_backwards": 0,
            "model_weights_updated": False,
            "pred_a_exact_prefix_and_bilinear_decomposition": True,
            "pred_b_selective_downstream_action_component": False,
            "pred_c_cross_representation_reuse": False,
            "pred_d_evidence_derived_terminal": True,
            "all_required_gates_pass": False,
            "decision": "downstream_use_decomposition_null",
            "next_step": "retain_R576_broad_carrier_and_do_not_promote_R582_component",
            "bootstrap_replicates_per_cell": replicates,
            "bootstrap_cell_count": len(bootstrapper.traces),
            "bootstrap_trace_sha256": trace_sha,
            "phase_support_census": support,
            "phase_support_census_sha256": support_sha,
            "checkpoint_weights_sha256": producer.CHECKPOINT_SHA256,
            "forbidden_splits_opened": [],
            "implementation_sha256": APPROVED_HASHES[PRODUCER],
            "test_sha256": APPROVED_HASHES[OWNER_TEST],
            "note_sha256": APPROVED_HASHES[NOTE],
        }
        for key, expected in expected_terminal.items():
            compare(expected, result.get(key), f"result.{key}", r588)
        descriptor = {
            "path": str(EVIDENCE.relative_to(REPO)),
            "sha256": digest(evidence_bytes),
            "schema": producer.EVIDENCE_SCHEMA,
        }
        if result.get("evidence_descriptor") != descriptor:
            raise RuntimeError("R590 result does not bind evidence bytes")
        if receipt != expected_receipt(result, result_bytes, evidence_bytes, producer):
            raise RuntimeError("R590 receipt does not bind result/evidence bytes")

        # Secondary agreement with the exact approved producer validator. The
        # independent reconstruction above determines the audit verdict.
        producer.validate_result_against_evidence(result, evidence, replicates=replicates)

        return {
            "schema": "numbered_list_cached_value_downstream_use_rung590_postexecution_audit_v1",
            "audit_passed": True,
            "audit_failures": [],
            "approved_commit": APPROVED_COMMIT,
            "approved_sha256": {str(path.relative_to(REPO)): value for path, value in APPROVED_HASHES.items()},
            "result_sha256": digest(result_bytes),
            "receipt_sha256": digest(receipt_bytes),
            "evidence_sha256": digest(evidence_bytes),
            "package_id": receipt["package_id"],
            "scientific_terminal_valid": True,
            "instrument_invalid": False,
            "independently_recomputed_decision": "downstream_use_decomposition_null",
            "independently_recomputed_provisional": None,
            "independently_recomputed_selected": None,
            "independently_recomputed_opened_splits": ["FIT"],
            "independently_recomputed_model_forwards": 379,
            "raw_counts": {
                "fit_capture_rows": len(capture_by_id),
                "fit_real_arms": len(reports),
                "rows_per_fit_arm": sorted({len(value) for value in fit_raw_value.values()}),
                "fit_null_arms": 0,
                "select_real_arms": 0,
                "select_null_arms": 0,
            },
            "candidate_cell_pass_counts": pass_counts,
            "fit_exactness": exactness,
            "bootstrap_replicates_per_cell": replicates,
            "bootstrap_cell_count": len(bootstrapper.traces),
            "bootstrap_trace_sha256": trace_sha,
            "phase_support_census_sha256": support_sha,
            "active_scientific_nulls_evaluated": False,
            "active_scientific_nulls_reason": "no_provisional_fit_candidate",
            "FINAL_TEST_or_OOD_opened": False,
            "model_loaded_by_audit": False,
            "model_forwards_by_audit": 0,
            "model_backwards_by_audit": 0,
            "model_weights_updated_by_audit": False,
        }
    except (AssertionError, KeyError, TypeError, ValueError, RuntimeError) as audit_exc:
        return {
            "schema": "numbered_list_cached_value_downstream_use_rung590_postexecution_audit_v1",
            "audit_passed": False,
            "audit_failures": [
                f"integrity_or_reconstruction:{type(audit_exc).__name__}:{audit_exc}"
            ],
            "scientific_terminal_valid": False,
            "instrument_invalid": None,
            "independently_recomputed_decision": None,
            "independently_recomputed_provisional": None,
            "independently_recomputed_selected": None,
            "independently_recomputed_opened_splits": None,
            "independently_recomputed_model_forwards": None,
            "model_loaded_by_audit": False,
            "model_forwards_by_audit": 0,
            "model_backwards_by_audit": 0,
            "model_weights_updated_by_audit": False,
        }


def atomic_write(path: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError("R590 postexecution audit namespace is occupied")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    adapter, producer = load_approved_modules()
    report = audit_bytes(
        RESULT.read_bytes(), RECEIPT.read_bytes(), EVIDENCE.read_bytes(), producer,
        bind_observed_hashes=True,
    )
    if not report["audit_passed"]:
        raise RuntimeError(report["audit_failures"])
    runlog_bytes = RUNLOG.read_bytes()
    if digest(runlog_bytes) != RUNLOG_SHA256:
        raise RuntimeError("R590 managed runlog changed before audit")
    runlog_text = runlog_bytes.decode("utf-8", errors="strict")
    postpublish = "R590 verified scientific entry point unexpectedly returned" in runlog_text
    report.update({
        "audit_script_sha256": digest(SCRIPT.read_bytes()),
        "audit_test_sha256": digest(TEST.read_bytes()),
        "managed_runlog_sha256": RUNLOG_SHA256,
        "managed_runner_exit_code": 1,
        "managed_wrapper_clean": False,
        "postpublication_adapter_error": postpublish,
        "postpublication_error_classification": (
            "wrapper_false_failure_after_valid_receipt_commit"
            if postpublish else "unrecognized_managed_failure"
        ),
        "audit_verdict": (
            "scientific_null_independently_held_with_postpublication_wrapper_bug"
            if postpublish else "failed_independent_audit"
        ),
        "next_step": (
            "accept_scientific_null_do_not_rerun_repair_adapter_before_future_reuse"
            if postpublish else "do_not_accept_result"
        ),
        "approved_adapter_snapshot_file_count": len(adapter.FROZEN_HASHES),
    })
    if not postpublish:
        raise RuntimeError("R590 managed failure was not the known postpublication sentinel")
    atomic_write(OUT, report)
    print(json.dumps({
        "audit_verdict": report["audit_verdict"],
        "decision": report["independently_recomputed_decision"],
        "model_forwards": report["independently_recomputed_model_forwards"],
        "managed_runner_exit_code": report["managed_runner_exit_code"],
        "postpublication_adapter_error": report["postpublication_adapter_error"],
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
