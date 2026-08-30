"""Pure validation-role reduction after the exact candidate library is frozen.

This module deliberately has no filesystem, model, corpus, training-output, or EVAL
capability. A later source-closed lifecycle owns the receipt-bound full FIT payload,
validates the freeze/audit mappings, calls this reducer once, destroys its private
payload alias, and receives only cloned 114-document validation tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import torch

import causal_response_tensor_v1_fit_bundle as fit_bundle
from causal_response_factorization_v1 import (
    FIT_TRAIN_DOCUMENTS, prospective_document_split, signed_response_from_sums,
)
from causal_response_factorization_v1_fit_adapter import (
    FitArtifactBinding,
)


PRODUCTION_FREEZE_ARTIFACT_SHA256 = "53f8264228e905ad1a459f32204d1acb07fa044e7753026dbb0bcfb91ac77b98"
PRODUCTION_FREEZE_MANIFEST_SHA256 = "3b386a38e9bf79f90e01c87ef6471770dfd7bae73ffb89ef3749a182968b5500"
PRODUCTION_RANK_PAIRS = (
    (1, 0), (2, 0), (4, 0), (4, 1), (8, 0), (8, 2),
    (16, 0), (16, 4), (32, 0),
)
PRODUCTION_SEEDS = (2026083001, 2026083002, 2026083003)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _logical_sha256(value: object) -> str:
    return _sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode())


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class CandidateFreezeBinding:
    artifact_sha256: str
    manifest_sha256: str
    audit_artifact_sha256: str
    candidate_rank_pairs: int
    candidate_programs: int

    def __post_init__(self) -> None:
        if not all(_is_sha256(value) for value in (
            self.artifact_sha256, self.manifest_sha256, self.audit_artifact_sha256,
        )):
            raise ValueError("candidate freeze identities must be SHA-256 values")
        if (self.candidate_rank_pairs, self.candidate_programs) != (9, 27):
            raise ValueError("candidate freeze census changed")


def validate_candidate_freeze(
    freeze: Mapping[str, Any], audit: Mapping[str, Any], *,
    freeze_artifact_sha256: str, audit_artifact_sha256: str,
    require_production: bool,
) -> CandidateFreezeBinding:
    if type(freeze) is not dict or set(freeze) != {
        "schema", "status", "source_closure", "training_analysis_sha256",
        "grid_terminal_sha256", "grid_manifest_sha256", "selection_rule",
        "candidate_rank_pairs", "candidate_rank_pair_count", "candidate_programs",
        "candidate_program_count", "candidate_selected", "validation_values_read",
        "eval_values_read", "manifest_sha256",
    } or freeze.get("schema") != "causal_response_factorization_v1_candidate_freeze_v2" or (
        freeze.get("status") != "complete_training_frontier_freeze_no_scores"
    ):
        raise RuntimeError("candidate freeze schema or status changed")
    body = {key: value for key, value in freeze.items() if key != "manifest_sha256"}
    if freeze.get("manifest_sha256") != _logical_sha256(body):
        raise RuntimeError("candidate freeze logical identity does not replay")
    pairs = freeze.get("candidate_rank_pairs")
    programs = freeze.get("candidate_programs")
    if pairs != [list(value) for value in PRODUCTION_RANK_PAIRS] or (
        freeze.get("candidate_rank_pair_count") != 9
        or freeze.get("candidate_program_count") != 27
        or not isinstance(programs, list) or len(programs) != 27
    ):
        raise RuntimeError("candidate freeze census or rank pairs changed")
    program_keys = {
        "global_rank", "private_rank_each_owner", "seed", "artifact",
        "artifact_sha256", "bytes", "persistent_values", "per_document_values",
    }
    identities = []
    for program in programs:
        if type(program) is not dict or set(program) != program_keys:
            raise RuntimeError("candidate program schema changed or contains a score")
        identities.append((
            program["global_rank"], program["private_rank_each_owner"], program["seed"],
        ))
    expected = [(*pair, seed) for pair in PRODUCTION_RANK_PAIRS for seed in PRODUCTION_SEEDS]
    if identities != expected:
        raise RuntimeError("candidate rank/seed identities changed")
    if freeze.get("candidate_selected") is not False or (
        freeze.get("validation_values_read") is not False
        or freeze.get("eval_values_read") is not False
    ):
        raise RuntimeError("candidate freeze role boundary changed")

    audit_keys = {
        "schema", "status", "approved", "reviewer", "outcome_access",
        "candidate_tensor_deserialized", "validation_values_read", "eval_values_read",
        "source_hashes", "focused_tests_passed", "additional_attacks_passed",
        "candidate_freeze_artifact_sha256", "candidate_freeze_manifest_sha256",
        "candidate_rank_pairs", "candidate_programs", "seeds_per_rank_pair",
        "score_fields_present", "candidate_selected", "remaining_execution_blockers",
    }
    if type(audit) is not dict or set(audit) != audit_keys or (
        audit.get("schema") != "causal_response_factorization_v1_candidate_freeze_v2_independent_audit"
        or audit.get("status") != "GO" or audit.get("approved") is not True
        or not isinstance(audit.get("reviewer"), str) or not audit["reviewer"]
        or audit.get("outcome_access") is not False
        or audit.get("candidate_tensor_deserialized") is not False
        or audit.get("validation_values_read") is not False
        or audit.get("eval_values_read") is not False
        or audit.get("focused_tests_passed") != 6
        or audit.get("additional_attacks_passed") != [
            "analysis_mutation_rejected", "terminal_mutation_rejected",
        ]
        or audit.get("candidate_freeze_artifact_sha256") != freeze_artifact_sha256
        or audit.get("candidate_freeze_manifest_sha256") != freeze["manifest_sha256"]
        or audit.get("candidate_rank_pairs") != 9
        or audit.get("candidate_programs") != 27
        or audit.get("seeds_per_rank_pair") != 3
        or audit.get("score_fields_present") is not False
        or audit.get("candidate_selected") is not False
        or audit.get("remaining_execution_blockers") != []
    ):
        raise RuntimeError("candidate freeze independent audit changed")
    if require_production and (
        freeze_artifact_sha256 != PRODUCTION_FREEZE_ARTIFACT_SHA256
        or freeze["manifest_sha256"] != PRODUCTION_FREEZE_MANIFEST_SHA256
    ):
        raise RuntimeError("production candidate freeze identity changed")
    return CandidateFreezeBinding(
        artifact_sha256=freeze_artifact_sha256,
        manifest_sha256=freeze["manifest_sha256"],
        audit_artifact_sha256=audit_artifact_sha256,
        candidate_rank_pairs=9,
        candidate_programs=27,
    )


@dataclass(frozen=True)
class FitValidationInput:
    response: torch.Tensor
    valid: torch.Tensor
    document_ids: torch.Tensor
    original_document_indices: torch.Tensor
    source_groups: torch.Tensor
    phases: tuple[str, ...]
    source_tags: tuple[str, ...]
    target_tags: tuple[str, ...]
    source_components: tuple[str, ...]
    owner_components: tuple[str, ...]
    artifacts: FitArtifactBinding
    candidate_freeze: CandidateFreezeBinding

    def __post_init__(self) -> None:
        p, s, t, d = self.response.shape
        if self.response.dtype != torch.float64 or self.response.device.type != "cpu" or (
            not self.response.is_contiguous()
        ):
            raise TypeError("validation response must be contiguous CPU float64")
        if self.valid.dtype != torch.bool or self.valid.shape != self.response.shape or (
            self.valid.device.type != "cpu" or not self.valid.is_contiguous()
        ):
            raise TypeError("validation validity mask does not align")
        if self.document_ids.dtype != torch.int64 or self.document_ids.shape != (d,) or (
            self.document_ids.device.type != "cpu" or not self.document_ids.is_contiguous()
            or torch.unique(self.document_ids).numel() != d
        ):
            raise ValueError("validation document IDs are malformed")
        if self.original_document_indices.dtype != torch.int64 or (
            self.original_document_indices.shape != (d,)
            or self.original_document_indices.device.type != "cpu"
            or not self.original_document_indices.is_contiguous()
            or torch.unique(self.original_document_indices).numel() != d
        ):
            raise ValueError("validation source-document indices are malformed")
        expected_owners = tuple(dict.fromkeys(self.source_components))
        owner_index = {owner: index for index, owner in enumerate(expected_owners)}
        expected_groups = torch.tensor(
            [owner_index[owner] for owner in self.source_components], dtype=torch.int64,
        )
        if self.source_groups.dtype != torch.int64 or self.source_groups.shape != (s,) or (
            self.source_groups.device.type != "cpu" or not self.source_groups.is_contiguous()
            or not torch.equal(self.source_groups, expected_groups)
            or self.owner_components != expected_owners
        ):
            raise ValueError("validation owner topology changed")
        if p != len(self.phases) or s != len(self.source_tags) or t != len(self.target_tags) or (
            self.source_tags != self.target_tags or len(self.source_components) != s
        ):
            raise ValueError("validation tensor topology changed")
        if not isinstance(self.artifacts, FitArtifactBinding) or not isinstance(
            self.candidate_freeze, CandidateFreezeBinding
        ):
            raise TypeError("validation input lacks immutable parent bindings")


def validation_input_from_fit_payload(
    payload: Mapping[str, Any], *, parent_binding: Mapping[str, Any],
    candidate_freeze: Mapping[str, Any], candidate_freeze_audit: Mapping[str, Any],
    candidate_freeze_artifact_sha256: str, candidate_freeze_audit_artifact_sha256: str,
    require_production: bool = True, train_documents: int = FIT_TRAIN_DOCUMENTS,
) -> FitValidationInput:
    if require_production and train_documents != FIT_TRAIN_DOCUMENTS:
        raise RuntimeError("production validation split differs from preregistration")
    freeze_binding = validate_candidate_freeze(
        candidate_freeze, candidate_freeze_audit,
        freeze_artifact_sha256=candidate_freeze_artifact_sha256,
        audit_artifact_sha256=candidate_freeze_audit_artifact_sha256,
        require_production=require_production,
    )
    artifacts = FitArtifactBinding.from_parent_binding(parent_binding)
    fit_bundle.validate_fit_bundle_payload(payload, require_production=require_production)
    if payload["binding"]["authority_sha256"] != artifacts.authority_logical_sha256:
        raise RuntimeError("FIT validation authority differs from the receipt binding")
    if payload["binding"]["source_closure_sha256"] != artifacts.source_closure_sha256:
        raise RuntimeError("FIT validation source closure differs from the receipt binding")
    raw = payload["fit_response"]
    response, valid = signed_response_from_sums(
        raw["statistics"], raw["member_count"], raw["off_count"],
    )
    _, validation_indices = prospective_document_split(
        raw["document_ids"], train_documents=train_documents,
    )
    source_components = tuple(payload["source_components"])
    owner_components = tuple(dict.fromkeys(source_components))
    owner_index = {owner: index for index, owner in enumerate(owner_components)}
    source_groups = torch.tensor(
        [owner_index[owner] for owner in source_components], dtype=torch.int64,
    )
    result = FitValidationInput(
        response=response[..., validation_indices].clone().contiguous(),
        valid=valid[..., validation_indices].clone().contiguous(),
        document_ids=raw["document_ids"][validation_indices].clone().contiguous(),
        original_document_indices=validation_indices.clone().contiguous(),
        source_groups=source_groups.contiguous(),
        phases=tuple(payload["phases"]), source_tags=tuple(payload["source_tags"]),
        target_tags=tuple(payload["target_tags"]), source_components=source_components,
        owner_components=owner_components, artifacts=artifacts,
        candidate_freeze=freeze_binding,
    )
    if require_production and result.response.shape != (2, 49, 49, 114):
        raise RuntimeError("production validation shape changed")
    return result
