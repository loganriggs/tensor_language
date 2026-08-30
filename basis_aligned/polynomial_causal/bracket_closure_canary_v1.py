"""Pure source-closure contract for the L13H8 bracket-closure canary.

This module has no loader, CLI, row publisher, parser callback, or result writer.  It
freezes the four-arm topology and pre-forward bindings needed by a later adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bracket_closure_masks_v1 as masks_module
import bracket_closure_tensor_v1 as tensor_module
import circuit_campaign_runtime as campaign
from tensor_preserving_attention import TensorPreservingSquaredAttention


SCHEMA = "bracket_closure_canary_v1"
SITE_COUNT = 18
ROW_LENGTH = 257
SCORED_POSITIONS = 256
LOGIT_VOCAB = 50_304
SOURCE_CLOSURE = (
    "basis_aligned/polynomial_causal/BRACKET_CLOSURE_CANARY_V1_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
    "basis_aligned/polynomial_causal/bracket_closure_canary_v1.py",
    "basis_aligned/polynomial_causal/bracket_closure_masks_v1.py",
    "basis_aligned/polynomial_causal/bracket_closure_tensor_v1.py",
    "basis_aligned/polynomial_causal/circuit_campaign_runtime.py",
    "basis_aligned/polynomial_causal/tensor_preserving_attention.py",
)


class BracketRole(str, Enum):
    SELECT_PROSE = "select_prose"
    SELECT_CODE = "select_code"
    SYNTHETIC_CANARY = "synthetic_canary"


ARM_NAMES = (
    "native",
    tensor_module.BracketTensorArm.STORED_ALL_HEADS.value,
    tensor_module.BracketTensorArm.DELETE_H8.value,
    tensor_module.BracketTensorArm.DERANGED_H8.value,
)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_git_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def tensor_sha256(value: torch.Tensor) -> str:
    if not torch.is_tensor(value) or value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("hashed tensors must be contiguous CPU tensors")
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def support_sha256(rows: torch.Tensor, masks: masks_module.BracketMasks) -> str:
    """Bind exact ordered rows and all score-only parser/control metadata."""

    if rows.device.type != "cpu" or rows.dtype != torch.long or rows.ndim != 2 or (
        rows.shape[1] != ROW_LENGTH or not rows.is_contiguous()
    ):
        raise ValueError("support rows must be contiguous CPU int64 [N,257]")
    masks.validate()
    expected = (rows.shape[0], SCORED_POSITIONS)
    if masks.compatible.shape != expected:
        raise ValueError("mask support does not match exact row predictions")
    digest = hashlib.sha256(tensor_sha256(rows).encode("ascii"))
    ordered = (
        *masks.named_cells().items(),
        ("family_index", masks.family_index), ("depth", masks.depth),
        ("distance", masks.distance), ("domain_index", masks.domain_index),
    )
    for name, value in ordered:
        digest.update(name.encode("ascii"))
        digest.update(tensor_sha256(value.contiguous()).encode("ascii"))
    return digest.hexdigest()


def build_circuit_plan() -> campaign.CircuitPlan:
    arms = [campaign.ArmPlan.build(
        "native", campaign.ArmKind.NATIVE, site_count=SITE_COUNT,
    )]
    for arm in ARM_NAMES[1:]:
        arms.append(campaign.ArmPlan.build(
            arm, campaign.ArmKind.CANDIDATE, site_count=SITE_COUNT,
            attention_replacements={tensor_module.TARGET_SITE: arm},
        ))
    return campaign.CircuitPlan(SCHEMA, SITE_COUNT, tuple(arms))


@dataclass(frozen=True)
class ProgramBinding:
    arm: str
    sha256: str
    stored_values: int
    native_calls_per_forward: int
    token_table_values: int
    total_input_support: bool

    def __post_init__(self) -> None:
        if self.arm not in ARM_NAMES[1:] or not _is_sha256(self.sha256):
            raise ValueError("program identity is outside the bracket arm registry")
        if type(self.stored_values) is not int or (
            self.stored_values != tensor_module.PRODUCTION_STORED_VALUES
        ):
            raise ValueError("program price differs from dense stored-all-head price")
        if (type(self.native_calls_per_forward) is not int
                or self.native_calls_per_forward != 0):
            raise ValueError("bracket replacement must make zero native attention calls")
        if type(self.token_table_values) is not int or self.token_table_values != 0 or (
            type(self.total_input_support) is not bool or not self.total_input_support
        ):
            raise ValueError("bracket replacement cannot use a token table or partial support")


@dataclass(frozen=True)
class RoleBinding:
    role: BracketRole
    rows_sha256: str
    support_sha256: str
    document_ids_sha256: str
    source_files_sha256: str
    document_count: int

    def __post_init__(self) -> None:
        if type(self.role) is not BracketRole or any(not _is_sha256(value) for value in (
            self.rows_sha256, self.support_sha256, self.document_ids_sha256,
            self.source_files_sha256,
        )):
            raise ValueError("role identity/hash binding is malformed")
        if type(self.document_count) is not int or self.document_count <= 0:
            raise ValueError("role document count must be positive")


@dataclass(frozen=True)
class PairwiseDisjointness:
    left: BracketRole
    right: BracketRole
    row_collisions: int
    document_collisions: int
    source_file_collisions: int

    def __post_init__(self) -> None:
        roles = tuple(BracketRole)
        if type(self.left) is not BracketRole or type(self.right) is not BracketRole or (
            roles.index(self.left) >= roles.index(self.right)
        ):
            raise ValueError("disjointness pair must follow frozen role order")
        counts = (self.row_collisions, self.document_collisions, self.source_file_collisions)
        if any(type(value) is not int or value != 0 for value in counts):
            raise ValueError("row/document/source roles must be exactly disjoint")


@dataclass(frozen=True)
class BracketCanaryAuthority:
    source_commit: str
    source_files: tuple[tuple[str, str], ...]
    model_config_sha256: str
    model_weights_sha256: str
    delimiter_registry_sha256: str
    derangement_sha256: str
    roles: tuple[RoleBinding, ...]
    disjointness: tuple[PairwiseDisjointness, ...]
    programs: tuple[ProgramBinding, ...]

    def __post_init__(self) -> None:
        if not _is_git_hash(self.source_commit):
            raise ValueError("authority source commit is malformed")
        if type(self.source_files) is not tuple or tuple(
            item[0] for item in self.source_files
        ) != SOURCE_CLOSURE or any(
            type(item) is not tuple or len(item) != 2 or not _is_sha256(item[1])
            for item in self.source_files
        ):
            raise ValueError("authority source closure differs from the frozen path set")
        if any(not _is_sha256(value) for value in (
            self.model_config_sha256, self.model_weights_sha256,
            self.delimiter_registry_sha256, self.derangement_sha256,
        )):
            raise ValueError("authority model/registry/control hash is malformed")
        if type(self.roles) is not tuple or tuple(binding.role for binding in self.roles) != tuple(
            BracketRole
        ):
            raise ValueError("authority must bind all roles in frozen order")
        role_fields = (
            "rows_sha256", "support_sha256", "document_ids_sha256", "source_files_sha256",
        )
        for field in role_fields:
            values = tuple(getattr(binding, field) for binding in self.roles)
            if len(set(values)) != len(values):
                raise ValueError(f"authority role {field} values must be distinct")
        expected_pairs = tuple(
            (left, right) for index, left in enumerate(BracketRole)
            for right in tuple(BracketRole)[index + 1:]
        )
        if type(self.disjointness) is not tuple or tuple(
            (item.left, item.right) for item in self.disjointness
        ) != expected_pairs:
            raise ValueError("authority needs every pairwise disjointness proof")
        if type(self.programs) is not tuple or tuple(
            binding.arm for binding in self.programs
        ) != ARM_NAMES[1:]:
            raise ValueError("authority program bindings differ from exact arm order")
        hashes = tuple(binding.sha256 for binding in self.programs)
        if len(set(hashes)) != len(hashes):
            raise ValueError("stored replay, deletion, and null must be distinct programs")


def make_attention_replacement(program: TensorPreservingSquaredAttention):
    """Create a mask-blind replacement callback from an owned stored program."""

    if not isinstance(program, TensorPreservingSquaredAttention):
        raise ValueError("replacement requires the exact stored-attention program type")
    receipt = program.cost_receipt()
    if receipt.total_stored_values <= 0 or receipt.native_calls_per_forward != 0 or (
        receipt.token_table_values != 0 or not receipt.total_input_support
    ):
        raise ValueError("replacement program price/support contract changed")

    def replacement(event: campaign.AttentionReplacementEvent):
        if type(event) is not campaign.AttentionReplacementEvent or (
            event.site != tensor_module.TARGET_SITE
        ):
            raise ValueError("bracket callback received the wrong physical site")
        # Tokens and all parser-derived masks are intentionally ignored.
        return program(event.state, event.first_value)

    return replacement


def validate_forward_closure(
    closure: campaign.ForwardClosure, arm: str, *, document_count: int,
) -> None:
    if type(closure) is not campaign.ForwardClosure or arm not in ARM_NAMES or (
        type(document_count) is not int or document_count <= 0
    ):
        raise ValueError("forward closure identity is malformed")
    if closure.circuit != SCHEMA or closure.arm != arm or (
        closure.document_count != document_count or not closure.closed
    ) or (closure.attempted_outer_forwards, closure.completed_outer_forwards,
          closure.outer_returns) != (1, 1, 1):
        raise ValueError("forward closure differs from one exact bracket batch")
    if len(closure.sites) != SITE_COUNT:
        raise ValueError("forward closure site topology changed")
    for ledger in closure.sites:
        replaced = arm != "native" and ledger.site == tensor_module.TARGET_SITE
        if (ledger.native_attention_calls, ledger.replacement_attention_calls) != (
            (0, 1) if replaced else (1, 0)
        ) or (ledger.native_mlp_calls, ledger.replacement_mlp_calls) != (1, 0):
            raise ValueError("component-call ledger differs from frozen topology")
    if arm != "native" and not closure.candidate_native_call_prohibition_passed:
        raise ValueError("candidate called native layer-13 attention")


def validate_logits(logits: torch.Tensor, *, documents: int) -> None:
    if not torch.is_tensor(logits) or logits.dtype != torch.float32 or logits.ndim != 3 or (
        logits.shape != (documents, SCORED_POSITIONS, LOGIT_VOCAB)
    ) or not bool(torch.isfinite(logits.detach()).all()):
        raise ValueError("bracket logits must use exact float32 [documents,256,50304] currency")


__all__ = (
    "ARM_NAMES", "BracketCanaryAuthority", "BracketRole", "PairwiseDisjointness",
    "ProgramBinding", "RoleBinding", "SOURCE_CLOSURE", "build_circuit_plan",
    "make_attention_replacement", "support_sha256", "tensor_sha256",
    "validate_forward_closure", "validate_logits",
)
