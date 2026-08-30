"""Pure, outcome-blind fresh-role allocation for the newline L12H6 canary.

No filesystem, tokenizer, model, checkpoint, or publication access occurs here.
Candidates are pre-tokenized one-row-per-document records from a future separately
audited builder.  Allocation uses only frozen metadata, token rows, and SHA ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Mapping

import torch

from circuit_newline_fixed_crew_v1 import NewlineMaskSpec, NewlineMasks, build_newline_masks


ROW_LENGTH = 257
PREFIX_LENGTH = 32
ROLE_ORDER = ("CANARY_SELECT", "FINAL", "OOD")
DOMAIN_ORDER = ("prose", "code", "list")
ROLE_DOMAIN_QUOTAS = {
    "CANARY_SELECT": {domain: 48 for domain in DOMAIN_ORDER},
    "FINAL": {domain: 64 for domain in DOMAIN_ORDER},
    "OOD": {domain: 64 for domain in DOMAIN_ORDER},
}
MIN_TARGET_DOCUMENTS = 128
MIN_TARGET_POSITIONS = 256


class NewlineDomain(str, Enum):
    PROSE = "prose"
    CODE = "code"
    LIST = "list"


def _sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def tensor_sha256(value: torch.Tensor) -> str:
    if value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("newline row hash requires a contiguous CPU tensor")
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def logical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


@dataclass(frozen=True)
class CandidateRecord:
    document_id: str
    source_document_index: int
    source_file: str
    source_revision: str
    source_blob_sha256: str
    domain: NewlineDomain
    license_id: str
    role_license: str
    structural_partition: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in (
            self.document_id, self.source_file, self.source_revision,
            self.license_id, self.role_license, self.structural_partition,
        )) or type(self.source_document_index) is not int or self.source_document_index < 0:
            raise ValueError("newline candidate provenance is malformed")
        if not _sha(self.source_blob_sha256) or type(self.domain) is not NewlineDomain:
            raise ValueError("newline candidate source/domain identity is malformed")
        if self.role_license not in ROLE_ORDER:
            raise ValueError("newline candidate role license is outside the frozen registry")


@dataclass(frozen=True)
class HistoricalExclusions:
    document_ids: frozenset[str]
    source_files: frozenset[str]
    source_blobs: frozenset[str]
    row_sha256s: frozenset[str]
    prefix_sha256s: frozenset[str]

    @classmethod
    def empty(cls) -> "HistoricalExclusions":
        return cls(*(frozenset() for _ in range(5)))


@dataclass(frozen=True)
class FrozenRole:
    role: str
    rows: torch.Tensor
    records: tuple[CandidateRecord, ...]
    masks: NewlineMasks
    support: Mapping[str, object]


def _eligible(
    row: torch.Tensor, record: CandidateRecord, spec: NewlineMaskSpec,
    exclusions: HistoricalExclusions,
) -> bool:
    row_hash = tensor_sha256(row.contiguous())
    prefix_hash = tensor_sha256(row[:PREFIX_LENGTH].contiguous())
    if record.document_id in exclusions.document_ids or record.source_file in (
        exclusions.source_files
    ) or record.source_blob_sha256 in exclusions.source_blobs or row_hash in (
        exclusions.row_sha256s
    ) or prefix_hash in exclusions.prefix_sha256s:
        return False
    try:
        masks = build_newline_masks(row.unsqueeze(0).contiguous(), spec)
    except (ValueError, RuntimeError):
        return False
    return int(masks.newline_target.sum()) > 0


def _order_key(seed: str, role: str, record: CandidateRecord, row: torch.Tensor) -> str:
    return hashlib.sha256("\0".join((
        seed, role, record.domain.value, record.document_id,
        str(record.source_document_index), tensor_sha256(row.contiguous()),
    )).encode()).hexdigest()


def support_census(masks: NewlineMasks, records: tuple[CandidateRecord, ...]) -> dict[str, object]:
    masks.validate()
    if len(records) != masks.newline_target.shape[0]:
        raise ValueError("newline support provenance count changed")
    named = {
        "newline_target": masks.newline_target,
        "position_jitter": masks.position_jitter,
        "matched_random": masks.matched_random,
        "punctuation": masks.punctuation,
        "capitalized": masks.capitalized,
        "quote_bracket": masks.quote_bracket,
        "global_off_target": masks.global_off_target,
    }
    return {
        "documents": len(records),
        "domains": {
            domain: sum(record.domain.value == domain for record in records)
            for domain in DOMAIN_ORDER
        },
        "cells": {name: int(value.sum()) for name, value in named.items()},
        "target_documents": int(masks.newline_target.any(1).sum()),
        "structural_partitions": sorted(set(record.structural_partition for record in records)),
    }


def allocate_roles(
    rows: torch.Tensor,
    records: tuple[CandidateRecord, ...],
    spec: NewlineMaskSpec,
    exclusions: HistoricalExclusions,
    *,
    seed: str,
) -> tuple[FrozenRole, ...]:
    """Allocate exact fresh roles without replacement using frozen SHA order."""

    if rows.device.type != "cpu" or rows.dtype != torch.long or rows.ndim != 2 or (
        rows.shape[1] != ROW_LENGTH or not rows.is_contiguous() or len(records) != rows.shape[0]
    ):
        raise ValueError("newline candidates must be contiguous CPU int64 [N,257]")
    if not isinstance(seed, str) or not seed or type(exclusions) is not HistoricalExclusions:
        raise ValueError("newline allocation seed/exclusions are malformed")
    identities = {
        "document": [record.document_id for record in records],
        "source": [record.source_file for record in records],
        "row": [tensor_sha256(row.contiguous()) for row in rows],
        "prefix": [tensor_sha256(row[:PREFIX_LENGTH].contiguous()) for row in rows],
    }
    if any(len(values) != len(set(values)) for values in identities.values()):
        raise ValueError("newline candidate pool repeats document/source/row/prefix identity")
    available = {
        index for index, (row, record) in enumerate(zip(rows, records, strict=True))
        if _eligible(row, record, spec, exclusions)
    }
    output: list[FrozenRole] = []
    for role in ROLE_ORDER:
        chosen: list[int] = []
        for domain in DOMAIN_ORDER:
            candidates = sorted(
                (index for index in available if records[index].domain.value == domain
                 and records[index].role_license == role),
                key=lambda index: _order_key(seed, role, records[index], rows[index]),
            )
            quota = ROLE_DOMAIN_QUOTAS[role][domain]
            if len(candidates) < quota:
                raise RuntimeError(f"newline {role}/{domain} has {len(candidates)} < {quota} candidates")
            selected = candidates[:quota]; chosen.extend(selected); available.difference_update(selected)
        role_rows = rows[chosen].contiguous()
        role_records = tuple(records[index] for index in chosen)
        role_masks = build_newline_masks(role_rows, spec)
        support = support_census(role_masks, role_records)
        if support["domains"] != ROLE_DOMAIN_QUOTAS[role] or (
            support["target_documents"] < MIN_TARGET_DOCUMENTS
            or support["cells"]["newline_target"] < MIN_TARGET_POSITIONS
        ):
            raise RuntimeError(f"newline {role} support is underpowered")
        output.append(FrozenRole(role, role_rows, role_records, role_masks, support))
    validate_role_disjointness(tuple(output))
    return tuple(output)


def validate_role_disjointness(roles: tuple[FrozenRole, ...]) -> None:
    if type(roles) is not tuple or tuple(role.role for role in roles) != ROLE_ORDER:
        raise ValueError("newline roles differ from frozen order")
    identity_sets = []
    for role in roles:
        if role.rows.shape[0] != sum(ROLE_DOMAIN_QUOTAS[role.role].values()):
            raise RuntimeError("newline role document count changed")
        identity_sets.append({
            "document": {record.document_id for record in role.records},
            "source": {record.source_file for record in role.records},
            "blob": {record.source_blob_sha256 for record in role.records},
            "partition": {record.structural_partition for record in role.records},
            "row": {tensor_sha256(row.contiguous()) for row in role.rows},
            "prefix": {tensor_sha256(row[:PREFIX_LENGTH].contiguous()) for row in role.rows},
        })
    for left in range(len(roles)):
        for right in range(left + 1, len(roles)):
            if any(identity_sets[left][key] & identity_sets[right][key] for key in identity_sets[left]):
                raise RuntimeError("newline role identities overlap")


def role_summary(role: FrozenRole) -> dict[str, object]:
    if role.role not in ROLE_ORDER or role.rows.shape[0] != sum(
        ROLE_DOMAIN_QUOTAS[role.role].values()
    ) or len(role.records) != role.rows.shape[0]:
        raise ValueError("newline role summary received a malformed role")
    role.masks.validate()
    records = [{
        "document_id": record.document_id,
        "source_document_index": record.source_document_index,
        "source_file": record.source_file,
        "source_revision": record.source_revision,
        "source_blob_sha256": record.source_blob_sha256,
        "domain": record.domain.value,
        "license_id": record.license_id,
        "role_license": record.role_license,
        "structural_partition": record.structural_partition,
    } for record in role.records]
    return {
        "role": role.role,
        "rows_sha256": tensor_sha256(role.rows),
        "records_sha256": logical_sha256(records),
        "document_ids_sha256": logical_sha256([record.document_id for record in role.records]),
        "support_sha256": logical_sha256(role.support),
        "support": role.support,
    }


__all__ = (
    "CandidateRecord", "DOMAIN_ORDER", "FrozenRole", "HistoricalExclusions",
    "MIN_TARGET_DOCUMENTS", "MIN_TARGET_POSITIONS", "NewlineDomain", "ROLE_DOMAIN_QUOTAS",
    "ROLE_ORDER", "ROW_LENGTH", "allocate_roles", "logical_sha256", "role_summary",
    "support_census", "tensor_sha256", "validate_role_disjointness",
)
