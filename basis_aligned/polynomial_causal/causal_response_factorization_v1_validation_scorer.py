"""Pure validation scoring of the complete frozen candidate library.

This module has no filesystem, model, corpus, training-response, or EVAL surface. It
receives the 114-document validation role from the reducer and the 27 frozen programs
from the lifecycle, and scores every program at every registered calibration budget and
design. It forms no Pareto frontier and selects nothing (Amendment 16).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from causal_response_factorization_v1 import (
    ResponseProgram, block_d_optimal_anchor_mask, prospective_anchor_arm_mask,
    score_program_on_validation,
)
from causal_response_factorization_v1_validation_input import (
    FitValidationInput, PRODUCTION_RANK_PAIRS, PRODUCTION_SEEDS,
)


CALIBRATION_ARM_BUDGETS = (2, 4, 8, 16)
DESIGNS = ("sha256_outcome_blind_blocks", "training_only_block_d_optimal")
SUPPORT_GATE = 0.90
TABLE_SCHEMA = "causal_response_factorization_v1_validation_table"
TABLE_STATUS = "complete_all_candidates_all_panels_no_selection"


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class FrozenCandidate:
    """One frozen program plus the training-only facts its scoring is allowed to use."""

    global_rank: int
    private_rank_each_owner: int
    seed: int
    artifact: str
    artifact_sha256: str
    bytes: int
    persistent_values: int
    per_document_values: int
    training_response_rms: float
    program: ResponseProgram
    training_codes: torch.Tensor

    def __post_init__(self) -> None:
        program = self.program
        if not isinstance(program, ResponseProgram):
            raise TypeError("frozen candidate program must be a ResponseProgram")
        if program.global_phase.shape[1] != self.global_rank or any(
            block.shape[1] != self.private_rank_each_owner for block in program.private_phase
        ):
            raise ValueError("frozen candidate registered ranks do not match its program")
        if program.persistent_values != self.persistent_values or (
            program.code_dimension != self.per_document_values
        ):
            raise ValueError("frozen candidate literal price does not match its program")
        codes = self.training_codes
        if type(codes) is not torch.Tensor or codes.dtype != torch.float64 or (
            codes.device.type != "cpu" or not codes.is_contiguous() or codes.ndim != 2
            or codes.shape[0] < 1 or codes.shape[1] != program.code_dimension
            or not bool(torch.isfinite(codes).all())
        ):
            raise ValueError("frozen candidate training codes are malformed")
        rms = self.training_response_rms
        if not isinstance(rms, float) or not math.isfinite(rms) or rms <= 0:
            raise ValueError("frozen candidate training RMS must be finite and positive")
        if not _is_sha256(self.artifact_sha256) or not isinstance(self.bytes, int) or (
            self.bytes <= 0 or not isinstance(self.artifact, str) or not self.artifact
        ):
            raise ValueError("frozen candidate artifact identity is malformed")

    @property
    def identity(self) -> tuple[int, int, int]:
        return self.global_rank, self.private_rank_each_owner, self.seed


def validate_candidate_library(
    candidates: Sequence[FrozenCandidate],
    freeze: Mapping[str, Any],
    *,
    source_groups: torch.Tensor,
    require_production: bool = True,
) -> float:
    """Bind the loaded library to the freeze census; return the shared training RMS."""

    programs = freeze.get("candidate_programs")
    if not isinstance(programs, list) or len(programs) != len(candidates):
        raise RuntimeError("candidate library census differs from the freeze")
    if require_production:
        expected = [
            (*pair, seed) for pair in PRODUCTION_RANK_PAIRS for seed in PRODUCTION_SEEDS
        ]
        if [candidate.identity for candidate in candidates] != expected:
            raise RuntimeError("candidate library identities differ from production")
    for candidate, record in zip(candidates, programs):
        if not isinstance(candidate, FrozenCandidate):
            raise TypeError("candidate library must contain FrozenCandidate values")
        if (
            record.get("global_rank") != candidate.global_rank
            or record.get("private_rank_each_owner") != candidate.private_rank_each_owner
            or record.get("seed") != candidate.seed
            or record.get("artifact") != candidate.artifact
            or record.get("artifact_sha256") != candidate.artifact_sha256
            or record.get("bytes") != candidate.bytes
            or record.get("persistent_values") != candidate.persistent_values
            or record.get("per_document_values") != candidate.per_document_values
        ):
            raise RuntimeError("candidate library record differs from the freeze")
        if not torch.equal(candidate.program.source_groups, source_groups):
            raise RuntimeError("candidate program owner topology differs from validation")
    rms_values = {candidate.training_response_rms for candidate in candidates}
    if len(rms_values) != 1:
        raise RuntimeError("frozen candidates disagree on the training RMS currency")
    return rms_values.pop()


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return _finite_or_none(value)
    if isinstance(value, torch.Tensor):
        raise TypeError("validation table must not contain tensors")
    raise TypeError(f"validation table cannot serialize {type(value).__name__}")


def _masked_nrmse(
    truth: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor, training_rms: float,
) -> dict[str, float | int | None]:
    cells = int(mask.sum())
    if cells == 0:
        return {"cells": 0, "mse": None, "nrmse_by_training_rms": None}
    residual = prediction[mask] - truth[mask]
    mse = float((residual * residual).mean())
    return {
        "cells": cells, "mse": _finite_or_none(mse),
        "nrmse_by_training_rms": _finite_or_none(math.sqrt(mse) / training_rms),
    }


def _phase_and_owner_breakdown(
    truth: torch.Tensor, prediction: torch.Tensor, score_mask: torch.Tensor, *,
    shape: tuple[int, int, int, int], source_groups: torch.Tensor, training_rms: float,
) -> dict[str, Any]:
    """Registered error slices: each phase, each source owner, each target owner."""

    p, s, t, d = shape
    truth4 = truth.reshape(p, s, t, d)
    prediction4 = prediction.reshape(p, s, t, d)
    mask4 = score_mask.reshape(p, s, t, d)
    group_count = int(source_groups.max()) + 1
    phases = []
    for phase in range(p):
        local = torch.zeros_like(mask4)
        local[phase] = mask4[phase]
        phases.append(_masked_nrmse(truth4, prediction4, local, training_rms))
    source_owners = []
    target_owners = []
    for group in range(group_count):
        members = source_groups == group
        local = torch.zeros_like(mask4)
        local[:, members] = mask4[:, members]
        source_owners.append(_masked_nrmse(truth4, prediction4, local, training_rms))
        local = torch.zeros_like(mask4)
        local[:, :, members] = mask4[:, :, members]
        target_owners.append(_masked_nrmse(truth4, prediction4, local, training_rms))
    return {"phase": phases, "source_owner": source_owners, "target_owner": target_owners}


def _design_conditioning(
    basis: torch.Tensor, valid_matrix: torch.Tensor, anchors: torch.Tensor,
    supported: Sequence[int],
) -> dict[str, Any]:
    """Smallest singular value of the valid selected design, over supported documents."""

    smallest: list[float] = []
    selected_cells: list[int] = []
    for document in supported:
        mask = anchors & valid_matrix[:, document]
        design = basis[mask]
        selected_cells.append(int(mask.sum()))
        if design.shape[1] == 0:
            continue
        smallest.append(float(torch.linalg.svdvals(design)[-1]))
    if not smallest:
        return {
            "supported_documents": len(supported),
            "smallest_singular_value_min": None,
            "smallest_singular_value_median": None,
            "valid_selected_cells_min": min(selected_cells) if selected_cells else 0,
        }
    ordered = sorted(smallest)
    return {
        "supported_documents": len(supported),
        "smallest_singular_value_min": _finite_or_none(ordered[0]),
        "smallest_singular_value_median": _finite_or_none(ordered[len(ordered) // 2]),
        "valid_selected_cells_min": min(selected_cells),
    }


def _replay_predictions(
    program: ResponseProgram, training_codes: torch.Tensor,
    validation: FitValidationInput, anchors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute the two arms' predictions for the registered slice breakdown."""

    from causal_response_factorization_v1 import infer_document_codes, predict_from_codes

    p, s, t, d = validation.response.shape
    response_matrix = validation.response.reshape(p * s * t, d)
    valid_matrix = validation.valid.reshape(p * s * t, d)
    basis = program.basis()
    mean_code = training_codes.mean(dim=0, keepdim=True).expand(d, -1).contiguous()
    unconditional = predict_from_codes(basis, mean_code)
    inferred, supported = infer_document_codes(basis, response_matrix, valid_matrix, anchors)
    calibrated = predict_from_codes(basis, inferred)
    return basis, response_matrix, valid_matrix, unconditional, (calibrated, supported)


def _panel_costs(arms: int, shape: tuple[int, int, int], code_dimension: int) -> dict[str, int]:
    cells = arms * shape[2]
    return {
        "physical_source_arms": arms,
        "calibration_cells": cells,
        "code_solve_multiply_add_upper_bound": (
            cells * code_dimension * (code_dimension + 1) + code_dimension ** 3
        ),
        "prediction_multiply_adds_per_document": shape[0] * shape[1] * shape[2] * code_dimension,
    }


def score_candidate(
    candidate: FrozenCandidate, validation: FitValidationInput, *,
    training_rms: float, arm_budgets: Sequence[int] = CALIBRATION_ARM_BUDGETS,
    designs: Sequence[str] = DESIGNS,
) -> dict[str, Any]:
    """Score one program on every panel. Panels that fail numerically are recorded."""

    if candidate.training_response_rms != training_rms:
        raise ValueError("candidate training RMS differs from the library currency")
    program = candidate.program
    p, s, t, d = validation.response.shape
    if program.shape != (p, s, t):
        raise ValueError("candidate program shape differs from the validation role")
    unknown = [design for design in designs if design not in DESIGNS]
    if unknown or len(set(designs)) != len(designs) or not designs:
        raise ValueError("validation designs are not the registered designs")
    if not arm_budgets or len(set(arm_budgets)) != len(arm_budgets) or any(
        not isinstance(arms, int) or arms < 1 for arms in arm_budgets
    ):
        raise ValueError("calibration arm budgets are malformed")

    unconditional: dict[str, Any] | None = None
    calibrated: dict[str, dict[str, Any]] = {design: {} for design in designs}
    basis_rank: int | None = None
    for design in designs:
        for arms in arm_budgets:
            panel: dict[str, Any]
            try:
                if design == "sha256_outcome_blind_blocks":
                    anchors, selected = prospective_anchor_arm_mask(p, s, t, arms=arms)
                    design_diagnostics: dict[str, Any] = {"selected_arms": list(selected)}
                else:
                    basis = program.basis()
                    basis_rank = int(torch.linalg.matrix_rank(basis))
                    anchors, selected, logdet = block_d_optimal_anchor_mask(
                        basis, shape=(p, s, t), arms=arms,
                    )
                    design_diagnostics = {
                        "selected_arms": list(selected),
                        "logdet_path": [_finite_or_none(value) for value in logdet],
                    }
                report = score_program_on_validation(
                    program, candidate.training_codes, validation.response,
                    validation.valid, anchors, training_rms=training_rms,
                )
                _, response_matrix, valid_matrix, unconditional_prediction, (
                    calibrated_prediction, supported,
                ) = _replay_predictions(program, candidate.training_codes, validation, anchors)
                non_anchor = (~anchors)[:, None].expand_as(valid_matrix)
                score_mask = valid_matrix & non_anchor & supported[None, :]
                slices = _phase_and_owner_breakdown(
                    response_matrix, calibrated_prediction, score_mask,
                    shape=(p, s, t, d), source_groups=program.source_groups,
                    training_rms=training_rms,
                ) if bool(score_mask.any()) else None
                if unconditional is None:
                    unconditional = {
                        **report["unconditional"],
                        "slices": _phase_and_owner_breakdown(
                            response_matrix, unconditional_prediction, valid_matrix,
                            shape=(p, s, t, d), source_groups=program.source_groups,
                            training_rms=training_rms,
                        ),
                        "uses_validation_responses": False,
                    }
                elif report["unconditional"] != unconditional_reference(unconditional):
                    raise RuntimeError("unconditional arm changed between panels")
                panel = {
                    "status": "scored",
                    "design": design_diagnostics,
                    "costs": _panel_costs(arms, (p, s, t), program.code_dimension),
                    "anchor_cells": report["anchor_cells"],
                    "anchor_source_arms": report["anchor_source_arms"],
                    "supported_documents": report["supported_documents"],
                    "supported_document_fraction": report["supported_document_fraction"],
                    "support_gate": SUPPORT_GATE,
                    "support_gate_passes": report["support_gate_passes"],
                    "eligible_for_frontier": bool(report["support_gate_passes"]),
                    "calibrated": report["calibrated"],
                    "slices": slices,
                    "conditioning": _design_conditioning(
                        program.basis(), valid_matrix, anchors,
                        report["supported_document_indices"],
                    ),
                    "claim_boundary": report["claim_boundary"],
                }
            except (ValueError, RuntimeError, torch.linalg.LinAlgError) as error:
                panel = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "costs": _panel_costs(arms, (p, s, t), program.code_dimension),
                    "eligible_for_frontier": False,
                }
            calibrated[design][str(arms)] = panel
    if unconditional is None:
        raise RuntimeError("no panel produced the unconditional arm")
    return _jsonable({
        "global_rank": candidate.global_rank,
        "private_rank_each_owner": candidate.private_rank_each_owner,
        "seed": candidate.seed,
        "artifact": candidate.artifact,
        "artifact_sha256": candidate.artifact_sha256,
        "bytes": candidate.bytes,
        "persistent_values": candidate.persistent_values,
        "per_document_values": candidate.per_document_values,
        "code_dimension": program.code_dimension,
        "basis_rank": basis_rank,
        "training_documents_in_codes": int(candidate.training_codes.shape[0]),
        "unconditional": unconditional,
        "calibrated": calibrated,
    })


def unconditional_reference(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item for key, item in value.items()
        if key not in ("slices", "uses_validation_responses")
    }


def score_library(
    candidates: Sequence[FrozenCandidate], validation: FitValidationInput,
    freeze: Mapping[str, Any], *, arm_budgets: Sequence[int] = CALIBRATION_ARM_BUDGETS,
    designs: Sequence[str] = DESIGNS, require_production: bool = True,
) -> dict[str, Any]:
    """Score every frozen program; publish the complete table with no winner."""

    if require_production and (
        tuple(arm_budgets) != CALIBRATION_ARM_BUDGETS or tuple(designs) != DESIGNS
        or validation.response.shape != (2, 49, 49, 114)
    ):
        raise RuntimeError("production validation panels or role changed")
    training_rms = validate_candidate_library(
        candidates, freeze, source_groups=validation.source_groups,
        require_production=require_production,
    )
    rows = [
        score_candidate(
            candidate, validation, training_rms=training_rms,
            arm_budgets=arm_budgets, designs=designs,
        )
        for candidate in candidates
    ]
    p, s, t, d = validation.response.shape
    return _jsonable({
        "schema": TABLE_SCHEMA,
        "status": TABLE_STATUS,
        "role": "FIT_INTERNAL_VALIDATION",
        "validation_documents": d,
        "response_shape": [p, s, t, d],
        "valid_cells": int(validation.valid.sum()),
        "owner_components": list(validation.owner_components),
        "phases": list(validation.phases),
        "training_response_rms": training_rms,
        "normalization": "every NRMSE divides by the training-role response RMS frozen before validation",
        "calibration_arm_budgets": list(arm_budgets),
        "designs": list(designs),
        "support_gate": SUPPORT_GATE,
        "candidate_count": len(rows),
        "candidates": rows,
        "candidates_dropped_after_scoring": 0,
        "candidate_selected": False,
        "pareto_frontier_formed": False,
        "validation_values_read": True,
        "eval_values_read": False,
        "claim_boundary": {
            "response_tomography_not_eval": True,
            "ood_transport": False,
            "semantic_extraction": False,
            "selective_removal": False,
            "terminal_circuit": False,
            "whole_model_ledger_credit": False,
        },
    })
