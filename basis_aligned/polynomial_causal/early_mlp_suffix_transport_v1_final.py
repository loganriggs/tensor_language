"""Semantic publication owner for early-MLP suffix transport v1.

This module is deliberately CPU-only and has no model, row-collector, or checkpoint
dependency.  It publishes the already-built canonical program bank, validates the
small semantic envelope returned by a future observed final evaluator, and owns the
create-only result -> manifest -> last-written authority transaction.

It is not a final evaluator and has no command-line entry point.  In particular,
importing it cannot load a role or run a model callback.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

import early_mlp_suffix_transport_v1 as contract
import early_mlp_suffix_transport_v1_final_actions as final_actions
import early_mlp_suffix_transport_v1_lifecycle as lifecycle
import early_mlp_suffix_transport_v1_programs as programs
import early_mlp_suffix_transport_v1_runtime as runtime
import early_mlp_suffix_transport_v1_statistics as statistics


RESULT_KIND = "early_mlp_suffix_transport_v1_final_semantic_result"
CLAIM_BOUNDARY = (
    "fresh-role local objective/transport evidence only; no current-ship, 36-site, "
    "named-behavior, named-causal, semantic, OOD, edit, or whole-model ledger credit"
)
LEDGER_CURRENCIES = (
    "current_ship", "site36", "named_behavior", "named_causal", "semantic",
    "ood", "edit", "whole_model",
)
OBJECTIVE_GATES = (
    "rr_beats_ll_ce",
    "rr_beats_ll_teacher_kl",
    "remaining_kl_ratio_le_half",
    "ce_half_oracle_margin_lcb_positive",
    "rr_beats_r0_l1",
    "rr_beats_l0_r1",
    "suffix_singleton_bonferroni_positive",
    "rr_ce_advantage_n",
    "rr_ce_advantage_e",
    "rr_teacher_kl_advantage_n",
    "rr_beats_shuffled_teacher",
    "rr_beats_new_fit_mean",
    "copy_bound",
    "frequency_bounds",
    "common_integrity",
)
EXECUTION_KEYS = {
    "final_role_loads", "final_evaluation_callbacks", "outer_model_returned",
    "hooks_restored", "hooks_inert", "component_tree_unchanged",
    "student_poison_closed", "programs_reloaded_semantically",
    "common_support_complete", "observational_action_call_ledger_sha256",
    "observational_student_outer_forwards", "gauge_replays",
    "gauge_max_abs_drift", "svd_max_abs_drift",
    "difference_in_differences_max_abs_drift", "row_count",
    "scored_tokens_per_row", "scored_token_count",
}
RESULT_KEYS = {
    "schema_version", "kind", "status", "claim_boundary", "bindings",
    "execution_closure", "objective_route", "transport_route",
    "outcome_class", "ledger_credit", "numerical_payload", "payload_sha256",
}
RESULT_BINDING_KEYS = {
    "final_attempt", "rows_receipt", "programs", "programs_receipt",
    "program_payload_sha256", "source_commit", "source_hashes_sha256",
    "protected_before_sha256",
}
RESPONSE_SUMMARY_KEYS = {
    "unit_identity", "baseline_point", "candidate_point",
    "nre_improvement_point", "nre_improvement_interval95",
    "candidate_nre_interval95", "candidate_r2_interval95",
}
RESPONSE_POINT_KEYS = {
    "error_sum", "teacher_sum", "student_sum", "dot_sum", "nre", "r2",
    "cosine",
}
OUTPUT_KL_SUMMARY_KEYS = {"unit_identity", "point", "interval95"}


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _exact_mapping(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError(f"{name} schema changed")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(
        float(value)
    ):
        raise RuntimeError(f"{name} must be finite")
    return float(value)


def _semantic_identity(value: Any) -> Any:
    """Convert a tensor/scalar tree to a canonical, finite logical identity."""

    if torch.is_tensor(value):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise RuntimeError("semantic payload contains a nonfinite tensor")
        return {"tensor_sha256": runtime.tensor_identity_sha256(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise RuntimeError("semantic payload mapping keys must be nonempty strings")
        return {key: _semantic_identity(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_semantic_identity(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("semantic payload contains a nonfinite scalar")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise RuntimeError(f"unsupported semantic payload type: {type(value).__name__}")


def _snapshot_paths(
    snapshot: Mapping[str, Any], *, required: Sequence[Path] = (),
) -> tuple[Path, ...]:
    if not isinstance(snapshot, Mapping) or not snapshot:
        raise RuntimeError("protected snapshot is absent")
    paths = []
    for key, binding in snapshot.items():
        if not isinstance(key, str) or not Path(key).is_absolute() or not isinstance(
            binding, Mapping
        ) or binding.get("path") != key:
            raise RuntimeError("protected snapshot path binding changed")
        paths.append(Path(key))
    resolved = {path.resolve() for path in paths}
    missing = [str(path) for path in required if path.resolve() not in resolved]
    if missing:
        raise RuntimeError(f"protected snapshot omits required stage inputs: {missing}")
    lifecycle.require_protected_snapshot(paths, snapshot)
    return tuple(paths)


def _required_program_snapshot_paths(paths: lifecycle.ArtifactPaths) -> tuple[Path, ...]:
    return (
        paths.rows_receipt, paths.rows_manifest, paths.fit_ledger,
        paths.fit_manifest, paths.fit_receipt,
    )


def publish_program_bank(
    payload: Mapping[str, Any], *, source_closure: Mapping[str, Any],
    protected_before: Mapping[str, Any], lock_nonce: str,
    paths: lifecycle.ArtifactPaths = lifecycle.PATHS,
    lock_path: Path = lifecycle.RUN_LOCK,
) -> dict[str, Any]:
    """Publish and semantically reload the canonical bank before final is legal."""

    lifecycle.require_run_claim(lock_nonce, lock_path)
    paths.assert_stage_preconditions("programs")
    lifecycle.verify_source_closure(
        source_closure["source_commit"], source_closure["source_hashes"],
    )
    _snapshot_paths(
        protected_before, required=_required_program_snapshot_paths(paths),
    )
    rows_receipt = json.loads(paths.rows_receipt.read_text())
    lifecycle._validate_rows_receipt(rows_receipt, paths)
    validated = programs.validate_canonical_program_bank_payload(payload)
    payload_sha256 = validated["payload_sha256"]

    lifecycle.atomic_create_torch(dict(payload), paths.programs)
    lifecycle.require_run_claim(lock_nonce, lock_path)
    lifecycle.verify_source_closure(
        source_closure["source_commit"], source_closure["source_hashes"],
    )
    _snapshot_paths(
        protected_before, required=_required_program_snapshot_paths(paths),
    )
    reloaded = torch.load(paths.programs, map_location="cpu", weights_only=True)
    replay = programs.validate_canonical_program_bank_payload(reloaded)
    if replay["payload_sha256"] != payload_sha256:
        raise RuntimeError("published canonical program bank changed on reload")

    receipt = {
        "schema_version": 1,
        "status": "frozen_programs_before_final",
        "authority": "early_mlp_suffix_transport_v1_programs_unlock",
        "authorized_for_final_scoring": True,
        "rows_receipt": lifecycle.artifact_binding(paths.rows_receipt),
        "programs": lifecycle.artifact_binding(paths.programs),
        "source_commit": source_closure["source_commit"],
        "source_hashes": dict(source_closure["source_hashes"]),
        "protected_before": dict(protected_before),
    }
    lifecycle.atomic_create_json(receipt, paths.programs_receipt)
    lifecycle.require_run_claim(lock_nonce, lock_path)
    if lifecycle.load_programs_unlock(paths) != receipt:
        raise RuntimeError("canonical programs receipt changed on reload")
    _snapshot_paths(
        protected_before, required=_required_program_snapshot_paths(paths),
    )
    return receipt


def load_program_bank(
    *, paths: lifecycle.ArtifactPaths = lifecycle.PATHS,
) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Reload the unlock and prove that its bytes still denote the frozen bank."""

    unlock = lifecycle.load_programs_unlock(paths)
    _snapshot_paths(
        unlock["protected_before"], required=_required_program_snapshot_paths(paths),
    )
    payload = torch.load(paths.programs, map_location="cpu", weights_only=True)
    validated = programs.validate_canonical_program_bank_payload(payload)
    if not _sha256(validated["payload_sha256"]):
        raise RuntimeError("canonical program bank identity is malformed")
    return unlock, payload, validated


def terminal_bindings(
    *, paths: lifecycle.ArtifactPaths = lifecycle.PATHS,
) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Return the exact context a final semantic result must bind."""

    unlock, payload, validated = load_program_bank(paths=paths)
    attempt = json.loads(paths.final_attempt.read_text())
    rows_receipt = json.loads(paths.rows_receipt.read_text())
    lifecycle._validate_final_attempt(
        paths, rows_receipt, unlock, attempt.get("lock_nonce"),
    )
    bindings = {
        "final_attempt": lifecycle.artifact_binding(paths.final_attempt),
        "rows_receipt": lifecycle.artifact_binding(paths.rows_receipt),
        "programs": lifecycle.artifact_binding(paths.programs),
        "programs_receipt": lifecycle.artifact_binding(paths.programs_receipt),
        "program_payload_sha256": validated["payload_sha256"],
        "source_commit": unlock["source_commit"],
        "source_hashes_sha256": lifecycle.logical_json_sha256(unlock["source_hashes"]),
        "protected_before_sha256": lifecycle.logical_json_sha256(
            unlock["protected_before"]
        ),
    }
    return bindings, validated, attempt


def _validate_execution(value: Any) -> Mapping[str, Any]:
    execution = _exact_mapping(value, EXECUTION_KEYS, "final execution closure")
    literal_true = (
        "outer_model_returned", "hooks_restored", "hooks_inert",
        "component_tree_unchanged", "student_poison_closed",
        "programs_reloaded_semantically", "common_support_complete",
    )
    if any(execution[name] is not True for name in literal_true) or (
        execution["final_role_loads"] != 1
    ) or execution["final_evaluation_callbacks"] != 1 or execution["gauge_replays"] != 8:
        raise RuntimeError("final execution closure is incomplete")
    if not _sha256(execution["observational_action_call_ledger_sha256"]) or (
        type(execution["observational_student_outer_forwards"]) is not int
    ) or execution["observational_student_outer_forwards"] != (
        len(final_actions.CANONICAL_ACTION_KEYS) * final_actions.OBSERVATIONAL_BATCH_COUNT
    ):
        raise RuntimeError("final observational action call ledger is incomplete")
    if execution["row_count"] != 192 or execution["scored_tokens_per_row"] != 192 or (
        execution["scored_token_count"] != 192 * 192
    ):
        raise RuntimeError("final common support changed")
    for name in (
        "gauge_max_abs_drift", "svd_max_abs_drift",
        "difference_in_differences_max_abs_drift",
    ):
        drift = _finite_number(execution[name], name)
        if drift < 0 or drift > 2e-6:
            raise RuntimeError(f"{name} exceeds the frozen replay tolerance")
    return execution


def _validate_route_gates(
    value: Any, *, names: Sequence[str], route: str,
) -> Mapping[str, Any]:
    payload = _exact_mapping(value, {"gates", "passes"}, f"{route} route")
    gates = _exact_mapping(payload["gates"], set(names), f"{route} gates")
    if any(type(gates[name]) is not bool for name in names) or type(
        payload["passes"]
    ) is not bool or payload["passes"] != all(gates.values()):
        raise RuntimeError(f"{route} route decision is inconsistent")
    return payload


def _interval(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise RuntimeError(f"{name} interval changed")
    lower, upper = (_finite_number(value[0], name), _finite_number(value[1], name))
    if lower > upper:
        raise RuntimeError(f"{name} interval is reversed")
    return lower, upper


def _validate_response_summary(value: Any, name: str) -> Mapping[str, Any]:
    summary = _exact_mapping(value, RESPONSE_SUMMARY_KEYS, f"{name} response summary")
    if not _sha256(summary["unit_identity"]):
        raise RuntimeError(f"{name} response unit identity changed")
    points = {}
    for side in ("baseline_point", "candidate_point"):
        point = _exact_mapping(summary[side], RESPONSE_POINT_KEYS, f"{name} {side}")
        points[side] = {key: _finite_number(point[key], f"{name} {side}.{key}") for key in point}
        if any(points[side][key] < 0 for key in (
            "error_sum", "teacher_sum", "student_sum", "nre",
        )) or points[side]["teacher_sum"] <= statistics.DENOMINATOR_FLOOR or (
            points[side]["student_sum"] <= statistics.DENOMINATOR_FLOOR
        ):
            raise RuntimeError(f"{name} response point has an invalid denominator")
        expected_nre = math.sqrt(
            points[side]["error_sum"] / points[side]["teacher_sum"]
        )
        expected_cosine = points[side]["dot_sum"] / math.sqrt(
            points[side]["student_sum"] * points[side]["teacher_sum"]
        )
        if abs(points[side]["nre"] - expected_nre) > 1e-10 or abs(
            points[side]["cosine"] - expected_cosine
        ) > 1e-10 or abs(points[side]["cosine"]) > 1 + 1e-10 or abs(
            points[side]["r2"] - (1 - points[side]["nre"] ** 2)
        ) > 1e-10:
            raise RuntimeError(f"{name} response R2/NRE identity changed")
    improvement = _finite_number(summary["nre_improvement_point"], name)
    expected = points["baseline_point"]["nre"] - points["candidate_point"]["nre"]
    if abs(improvement - expected) > 1e-10:
        raise RuntimeError(f"{name} response improvement identity changed")
    _interval(summary["nre_improvement_interval95"], f"{name} improvement")
    _interval(summary["candidate_nre_interval95"], f"{name} NRE")
    _interval(summary["candidate_r2_interval95"], f"{name} R2")
    return summary


def _validate_output_kl_summary(value: Any, name: str) -> Mapping[str, Any]:
    summary = _exact_mapping(value, OUTPUT_KL_SUMMARY_KEYS, f"{name} output-KL summary")
    if not _sha256(summary["unit_identity"]):
        raise RuntimeError(f"{name} output-KL unit identity changed")
    point = _finite_number(summary["point"], f"{name} output-KL point")
    interval = _interval(summary["interval95"], f"{name} output-KL")
    if point < 0 or interval[0] < 0:
        raise RuntimeError(f"{name} output-KL ratio is negative")
    return summary


def _validate_transport_route(
    value: Any, *, expected_calibration: bool,
) -> Mapping[str, Any]:
    decision = _exact_mapping(value, {
        "unit_identity", "code_response", "logit_response",
        "output_kl_response", "null_logit_nre_improvements",
        "finite_null_rank", "gates", "passes",
    }, "transport route")
    code = _validate_response_summary(decision["code_response"], "code")
    logit = _validate_response_summary(decision["logit_response"], "logit")
    if decision["unit_identity"] != code["unit_identity"] or (
        decision["unit_identity"] != logit["unit_identity"]
    ):
        raise RuntimeError("transport response unit identities differ")
    output_kl = _exact_mapping(
        decision["output_kl_response"], {"baseline", "candidate", "nulls"},
        "transport output-KL response",
    )
    output_kl_summaries = (
        _validate_output_kl_summary(output_kl["baseline"], "baseline"),
        _validate_output_kl_summary(output_kl["candidate"], "candidate"),
    )
    if not isinstance(output_kl["nulls"], (tuple, list)) or len(
        output_kl["nulls"]
    ) != 20:
        raise RuntimeError("transport output-KL null bank changed")
    output_kl_summaries += tuple(
        _validate_output_kl_summary(value, f"null {index}")
        for index, value in enumerate(output_kl["nulls"])
    )
    if any(
        summary["unit_identity"] != decision["unit_identity"]
        for summary in output_kl_summaries
    ):
        raise RuntimeError("output-KL and vector response unit identities differ")
    nulls = decision["null_logit_nre_improvements"]
    if not torch.is_tensor(nulls) or tuple(nulls.shape) != (20,) or not bool(
        torch.isfinite(nulls).all()
    ):
        raise RuntimeError("transport finite-null bank changed")
    rank = contract.finite_null_rank(
        float(logit["nre_improvement_point"]), nulls.detach().cpu().double(),
    )
    if decision["finite_null_rank"] != rank:
        raise RuntimeError("transport finite-null rank was not recomputed")
    names = {
        "calibration", "code_nre_improvement_point_positive",
        "code_nre_improvement_lcb_positive", "logit_nre_improvement_point_positive",
        "logit_nre_improvement_lcb_positive", "logit_nre_point_le_half",
        "logit_nre_ucb_le_half", "logit_r2_point_ge_three_quarters",
        "logit_r2_lcb_ge_three_quarters", "finite_null_rank_one",
        *statistics.TRANSPORT_OBSERVATIONAL_GATES,
    }
    gates = _exact_mapping(decision["gates"], names, "transport gates")
    if any(type(gates[name]) is not bool for name in names):
        raise RuntimeError("transport gate is not a literal boolean")
    expected = {
        "calibration": expected_calibration,
        "code_nre_improvement_point_positive": code["nre_improvement_point"] > 0,
        "code_nre_improvement_lcb_positive": (
            code["nre_improvement_interval95"][0] > 0
        ),
        "logit_nre_improvement_point_positive": logit["nre_improvement_point"] > 0,
        "logit_nre_improvement_lcb_positive": (
            logit["nre_improvement_interval95"][0] > 0
        ),
        "logit_nre_point_le_half": logit["candidate_point"]["nre"] <= 0.5,
        "logit_nre_ucb_le_half": logit["candidate_nre_interval95"][1] <= 0.5,
        "logit_r2_point_ge_three_quarters": logit["candidate_point"]["r2"] >= 0.75,
        "logit_r2_lcb_ge_three_quarters": (
            logit["candidate_r2_interval95"][0] >= 0.75
        ),
        "finite_null_rank_one": rank == 1,
    }
    if any(gates[name] != expected[name] for name in expected) or type(
        decision["passes"]
    ) is not bool or decision["passes"] != all(gates.values()):
        raise RuntimeError("transport route decision is inconsistent")
    return decision


def _outcome_class(objective_passes: bool, transport_passes: bool) -> str:
    if objective_passes and transport_passes:
        return "objective_and_transport_local_positive"
    if objective_passes:
        return "objective_only_local_positive"
    if transport_passes:
        return "transport_only_local_positive"
    return "both_registered_routes_scientific_negative"


def build_final_result(
    *, bindings: Mapping[str, Any], execution_closure: Mapping[str, Any],
    objective_gates: Mapping[str, bool], transport_route: Mapping[str, Any],
    numerical_payload: Mapping[str, Any], expected_calibration: bool,
) -> dict[str, Any]:
    """Build a strict, tensor-aware semantic result without publishing it."""

    _exact_mapping(bindings, RESULT_BINDING_KEYS, "final result bindings")
    _validate_execution(execution_closure)
    objective = {"gates": dict(objective_gates), "passes": all(objective_gates.values())}
    _validate_route_gates(objective, names=OBJECTIVE_GATES, route="objective")
    _validate_transport_route(transport_route, expected_calibration=expected_calibration)
    if not isinstance(numerical_payload, Mapping):
        raise RuntimeError("final numerical payload must be a mapping")
    ledger_credit = {currency: False for currency in LEDGER_CURRENCIES}
    body = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "status": "complete_pending_last_written_outcome_authority",
        "claim_boundary": CLAIM_BOUNDARY,
        "bindings": dict(bindings),
        "execution_closure": dict(execution_closure),
        "objective_route": objective,
        "transport_route": dict(transport_route),
        "outcome_class": _outcome_class(objective["passes"], transport_route["passes"]),
        "ledger_credit": ledger_credit,
        "numerical_payload": dict(numerical_payload),
    }
    return {**body, "payload_sha256": runtime.logical_identity_sha256(
        _semantic_identity(body)
    )}


def validate_final_result_payload(
    value: Any, *, expected_bindings: Mapping[str, Any] | None = None,
    expected_calibration: bool,
) -> Mapping[str, Any]:
    result = _exact_mapping(value, RESULT_KEYS, "final semantic result")
    if result["schema_version"] != 1 or result["kind"] != RESULT_KIND or (
        result["status"] != "complete_pending_last_written_outcome_authority"
    ) or result["claim_boundary"] != CLAIM_BOUNDARY or not _sha256(
        result["payload_sha256"]
    ):
        raise RuntimeError("final semantic result header changed")
    bindings = _exact_mapping(result["bindings"], RESULT_BINDING_KEYS, "final bindings")
    if expected_bindings is not None and dict(bindings) != dict(expected_bindings):
        raise RuntimeError("final result artifact bindings changed")
    for name in (
        "final_attempt", "rows_receipt", "programs", "programs_receipt",
    ):
        _exact_mapping(bindings[name], {"path", "sha256", "bytes"}, f"{name} binding")
        if not _sha256(bindings[name]["sha256"]) or type(bindings[name]["bytes"]) is not int:
            raise RuntimeError(f"{name} binding is malformed")
    if not _sha256(bindings["program_payload_sha256"]) or not _sha256(
        bindings["source_hashes_sha256"]
    ) or not _sha256(bindings["protected_before_sha256"]) or not isinstance(
        bindings["source_commit"], str
    ) or len(bindings["source_commit"]) != 40:
        raise RuntimeError("final source/program binding is malformed")
    _validate_execution(result["execution_closure"])
    objective = _validate_route_gates(
        result["objective_route"], names=OBJECTIVE_GATES, route="objective",
    )
    transport = _validate_transport_route(
        result["transport_route"], expected_calibration=expected_calibration,
    )
    if result["outcome_class"] != _outcome_class(
        objective["passes"], transport["passes"],
    ):
        raise RuntimeError("final outcome class overstates its route decisions")
    credit = _exact_mapping(result["ledger_credit"], set(LEDGER_CURRENCIES), "ledger credit")
    if any(credit[currency] is not False for currency in LEDGER_CURRENCIES):
        raise RuntimeError("local suffix result cannot move a global ledger")
    if not isinstance(result["numerical_payload"], Mapping):
        raise RuntimeError("final numerical payload is malformed")
    body = {key: result[key] for key in result if key != "payload_sha256"}
    if runtime.logical_identity_sha256(_semantic_identity(body)) != result["payload_sha256"]:
        raise RuntimeError("final semantic payload hash changed")
    return MappingProxyType(dict(result))


def _manifest(
    result: Mapping[str, Any], *, paths: lifecycle.ArtifactPaths,
    unlock: Mapping[str, Any], bindings: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "validated_result_before_terminal_authority",
        "authority": "none",
        "authorized_for_scientific_interpretation": False,
        "result": lifecycle.artifact_binding(paths.final_result),
        "result_payload_sha256": result["payload_sha256"],
        "final_attempt": bindings["final_attempt"],
        "programs": bindings["programs"],
        "programs_receipt": bindings["programs_receipt"],
        "source_commit": unlock["source_commit"],
        "source_hashes": dict(unlock["source_hashes"]),
        "protected_before": dict(unlock["protected_before"]),
        "outcome_class": result["outcome_class"],
    }


def _authority(
    result: Mapping[str, Any], *, paths: lifecycle.ArtifactPaths,
    unlock: Mapping[str, Any], bindings: Mapping[str, Any],
) -> dict[str, Any]:
    positive = bool(result["objective_route"]["passes"] or result["transport_route"]["passes"])
    return {
        "schema_version": 1,
        "receipt_kind": "early_mlp_suffix_transport_v1_final_outcome_authority",
        "status": (
            "authoritative_local_route_positive" if positive
            else "authoritative_scientific_negative"
        ),
        "authorized_for_scientific_interpretation": True,
        "authorized_for_global_ledger_credit": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "outcome_class": result["outcome_class"],
        "objective_route_passes": result["objective_route"]["passes"],
        "transport_route_passes": result["transport_route"]["passes"],
        "final_attempt": bindings["final_attempt"],
        "result": lifecycle.artifact_binding(paths.final_result),
        "manifest": lifecycle.artifact_binding(paths.final_manifest),
        "programs": bindings["programs"],
        "programs_receipt": bindings["programs_receipt"],
        "source_commit": unlock["source_commit"],
        "source_hashes": dict(unlock["source_hashes"]),
        "protected_before": dict(unlock["protected_before"]),
    }


def _write_integrity_failure(
    error: BaseException, *, paths: lifecycle.ArtifactPaths,
) -> None:
    if paths.final_authority.exists() or paths.integrity_failure.exists():
        return
    preserved = {
        name: lifecycle.artifact_binding(path)
        for name, path in (
            ("final_attempt", paths.final_attempt),
            ("final_result", paths.final_result),
            ("final_manifest", paths.final_manifest),
        ) if path.is_file()
    }
    if "final_result" not in preserved:
        return
    lifecycle.atomic_create_json({
        "schema_version": 1,
        "status": "integrity_failure_no_outcome_authority",
        "authority": "none",
        "authorized_for_scientific_interpretation": False,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "preserved_outputs": preserved,
    }, paths.integrity_failure)


def publish_terminal_result(
    result: Mapping[str, Any], *, lock_nonce: str,
    paths: lifecycle.ArtifactPaths = lifecycle.PATHS,
    lock_path: Path = lifecycle.RUN_LOCK,
) -> dict[str, Any]:
    """Publish result, then manifest, then the sole terminal authority write."""

    lifecycle.require_run_claim(lock_nonce, lock_path)
    if not paths.final_attempt.is_file() or any(path.exists() for path in (
        paths.final_result, paths.final_manifest, paths.final_authority,
        paths.integrity_failure,
    )):
        raise RuntimeError("terminal namespace is absent, incomplete, or already spent")
    if lifecycle._FINAL_ROLE_LOADS != 1:
        raise RuntimeError("terminal publication requires exactly one final-role load")

    bindings, validated_bank, attempt = terminal_bindings(paths=paths)
    if attempt["lock_nonce"] != lock_nonce:
        raise RuntimeError("terminal publisher does not own the final attempt")
    calibration = validated_bank["teacher_calibration"]["calibration_passed"]
    validate_final_result_payload(
        result, expected_bindings=bindings, expected_calibration=calibration,
    )
    unlock = lifecycle.load_programs_unlock(paths)

    try:
        lifecycle.atomic_create_torch(dict(result), paths.final_result)
        lifecycle.require_run_claim(lock_nonce, lock_path)
        reloaded = torch.load(paths.final_result, map_location="cpu", weights_only=True)
        validated_result = validate_final_result_payload(
            reloaded, expected_bindings=bindings, expected_calibration=calibration,
        )
        manifest = _manifest(
            validated_result, paths=paths, unlock=unlock, bindings=bindings,
        )
        lifecycle.atomic_create_json(manifest, paths.final_manifest)
        if json.loads(paths.final_manifest.read_text()) != manifest:
            raise RuntimeError("final manifest changed on reload")

        # Recheck every mutable boundary immediately before the authority's sole write.
        lifecycle.require_run_claim(lock_nonce, lock_path)
        paths.assert_stage_preconditions("final_authority")
        lifecycle.verify_source_closure(unlock["source_commit"], unlock["source_hashes"])
        _snapshot_paths(
            unlock["protected_before"], required=_required_program_snapshot_paths(paths),
        )
        current_bindings, current_bank, current_attempt = terminal_bindings(paths=paths)
        if current_bindings != bindings or current_attempt != attempt or current_bank[
            "payload_sha256"
        ] != bindings["program_payload_sha256"]:
            raise RuntimeError("terminal context drifted before authority")
        reloaded = torch.load(paths.final_result, map_location="cpu", weights_only=True)
        validate_final_result_payload(
            reloaded, expected_bindings=bindings, expected_calibration=calibration,
        )
        if json.loads(paths.final_manifest.read_text()) != manifest:
            raise RuntimeError("final manifest drifted before authority")
        authority = _authority(
            validated_result, paths=paths, unlock=unlock, bindings=bindings,
        )
        lifecycle.atomic_create_json(authority, paths.final_authority)
        if json.loads(paths.final_authority.read_text()) != authority:
            raise RuntimeError("terminal authority changed on reload")
        return authority
    except BaseException as error:
        _write_integrity_failure(error, paths=paths)
        raise
