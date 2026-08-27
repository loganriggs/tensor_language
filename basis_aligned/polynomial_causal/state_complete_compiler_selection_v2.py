"""Validation-only direct-response metrics and frozen selection for compiler v2."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

import early_mlp_state_complete_compiler_v2 as compiler


FAMILY_ORDER = {
    "B_state_complete_affine_euclidean": 0,
    "C_state_complete_affine_causal": 1,
    "D_state_complete_native_euclidean": 2,
    "E_state_complete_native_causal": 3,
}
ALL_FAMILIES = ("A_v1_like_z_only_affine_euclidean", *FAMILY_ORDER)


def token_weighted_teacher_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    valid: torch.Tensor,
) -> float:
    teacher = teacher_logits.float()
    student = student_logits.float()
    if teacher.shape != student.shape or teacher.ndim != 3:
        raise ValueError("teacher/student logits must be aligned rank-3 tensors")
    if valid.shape != teacher.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("KL validity mask does not align")
    if not bool(valid.any()):
        raise ValueError("KL validity mask is empty")
    teacher_logp = F.log_softmax(teacher, dim=-1)
    student_logp = F.log_softmax(student, dim=-1)
    teacher_p = teacher_logp.exp()
    per_token = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1)
    return float(per_token[valid].double().mean())


def direct_recovery(candidate_kl: float, oracle_denominator_kl: float) -> dict[str, float]:
    if not (oracle_denominator_kl > 0.0):
        raise ValueError("teacher-KL oracle denominator must be positive")
    ratio = candidate_kl / oracle_denominator_kl
    return {"candidate_teacher_kl": float(candidate_kl),
            "oracle_denominator_kl": float(oracle_denominator_kl),
            "remaining_kl_ratio": float(ratio),
            "recovery": float(1.0 - ratio)}


def state_price(state: Mapping[str, Any]) -> dict[str, Any]:
    grammar = state.get("grammar")
    if grammar == "affine":
        return compiler.corrected_affine_price(int(state["rank"]), include_basis=True)
    if grammar == "native":
        return compiler.native_program_price(int(state["k"]), include_basis=True)
    raise ValueError(f"candidate grammar has no registered selectable price: {grammar}")


def candidate_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    state = candidate["state"]
    family = state["family"]
    price = state_price(state)
    grammar_count = 0 if state["grammar"] == "affine" else int(state["k"])
    size = int(state.get("rank", state.get("k")))
    ridge = state.get("lambda")
    ridge_tie = -float(ridge) if ridge is not None else 0.0
    return (price["total_reals"], grammar_count, FAMILY_ORDER.get(family, -1),
            size, ridge_tie, candidate["name"])


def freeze_validation_selection(
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    recovery_slack: float = 0.99,
    max_copy_worsening: float = 0.01,
) -> dict[str, Any]:
    """Freeze B--E winner and one A--E representative without outcome reuse."""

    if not 0.0 < recovery_slack <= 1.0:
        raise ValueError("recovery slack must lie in (0,1]")
    if not candidates:
        raise ValueError("validation candidate bank is empty")
    rows = []
    for name, candidate in candidates.items():
        state = candidate.get("state")
        metrics = candidate.get("metrics")
        if not isinstance(state, Mapping) or not isinstance(metrics, Mapping):
            raise ValueError(f"candidate {name} lacks state/metrics")
        family = state.get("family")
        if family not in ALL_FAMILIES:
            raise ValueError(f"candidate {name} has unknown family {family}")
        recovery = float(metrics["recovery"])
        copy = float(metrics["copy_worsening"])
        rows.append({"name": name, "state": state, "metrics": metrics,
                     "eligible": family in FAMILY_ORDER and recovery > 0.0
                     and copy <= max_copy_worsening})
    selectable = [row for row in rows if row["eligible"]]
    if not selectable:
        raise RuntimeError("no B-E validation candidate has positive constrained recovery")
    best_recovery = max(float(row["metrics"]["recovery"]) for row in selectable)
    eligible = [row for row in selectable if float(row["metrics"]["recovery"])
                >= recovery_slack * best_recovery]
    selected = min(eligible, key=candidate_key)

    representatives = {}
    for family in ALL_FAMILIES:
        family_rows = [row for row in rows if row["state"]["family"] == family]
        if not family_rows:
            raise RuntimeError(f"validation bank lacks family {family}")
        best = max(float(row["metrics"]["recovery"]) for row in family_rows)
        near = [row for row in family_rows if float(row["metrics"]["recovery"])
                >= recovery_slack * best]
        representatives[family] = min(near, key=candidate_key)["name"]
    return {
        "selected": selected["name"],
        "selected_family": selected["state"]["family"],
        "best_constrained_recovery": best_recovery,
        "recovery_slack": recovery_slack,
        "max_copy_worsening": max_copy_worsening,
        "eligible": sorted(row["name"] for row in eligible),
        "family_representatives": representatives,
    }
