"""Pure, model-free allocation contract for bracket closure FIT/SELECT/OOD rows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from bracket_closure_masks_v1 import (
    BracketDomain, BracketMasks, DelimiterRegistry, build_bracket_masks,
)


ROW_LENGTH = 257
SCORE_START = 64
SCORE_STOP = 256
ROWS_PER_CELL = 32
MIN_CELL_DOCUMENTS = 30
MIN_CELL_POSITIONS = 30
CELL_ORDER = (
    "compatible_closer", "incompatible_closer", "no_opener",
    "quote_control", "punctuation_control",
)


class RowRole(str, Enum):
    FIT = "fit"
    SELECT = "select"
    OOD = "ood"


@dataclass(frozen=True)
class CandidateRecord:
    document_id: str
    source_document_index: int
    source_file: str
    source_revision: str
    source_blob_sha256: str
    domain: BracketDomain
    license_id: str
    normalized_python_sha256: str | None = None

    def __post_init__(self) -> None:
        strings = (
            self.document_id, self.source_file, self.source_revision,
            self.source_blob_sha256, self.license_id,
        )
        if any(not isinstance(value, str) or not value for value in strings):
            raise ValueError("candidate provenance strings must be nonempty")
        if type(self.source_document_index) is not int or self.source_document_index < 0:
            raise ValueError("candidate source index must be a nonnegative integer")
        if type(self.domain) is not BracketDomain:
            raise ValueError("candidate domain must be typed code/prose")
        if len(self.source_blob_sha256) != 64:
            raise ValueError("candidate source blob hash is malformed")
        if self.domain is BracketDomain.CODE:
            if not isinstance(self.normalized_python_sha256, str) or len(
                self.normalized_python_sha256
            ) != 64:
                raise ValueError("code candidate needs a normalized Python hash")
        elif self.normalized_python_sha256 is not None:
            raise ValueError("prose candidate cannot carry a normalized Python hash")


@dataclass(frozen=True)
class PriorExclusions:
    documents: frozenset[str]
    source_files: frozenset[str]
    source_blobs: frozenset[str]
    normalized_python: frozenset[str]
    row_sha256: frozenset[str]
    prefix32_sha256: frozenset[str]

    @classmethod
    def empty(cls) -> "PriorExclusions":
        return cls(*(frozenset() for _ in range(6)))


@dataclass(frozen=True)
class FrozenRole:
    role: RowRole
    rows: torch.Tensor
    records: tuple[CandidateRecord, ...]
    masks: BracketMasks
    support: Mapping[str, Mapping[str, int]]


def tensor_sha256(value: torch.Tensor) -> str:
    if not torch.is_tensor(value) or value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("hashed tensor must be contiguous CPU")
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def row_sha256(row: torch.Tensor) -> str:
    if row.device.type != "cpu" or row.dtype != torch.long or row.shape != (ROW_LENGTH,):
        raise ValueError("row hash needs one CPU int64 row")
    return tensor_sha256(row.contiguous())


def prefix32_sha256(row: torch.Tensor) -> str:
    return tensor_sha256(row[:32].contiguous())


def registry_sha256(registry: DelimiterRegistry) -> str:
    payload = {
        "families": [{"name": family.name, "open": list(family.opener_ids),
                      "close": list(family.closer_ids)} for family in registry.families],
        "quote": list(registry.quote_control_ids),
        "punctuation": list(registry.punctuation_control_ids),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _walk(value: Any):
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def historical_exclusions(
    registry_payloads: Sequence[Mapping[str, Any]],
) -> PriorExclusions:
    """Metadata-only historical census; tensor paths are never deserialized."""

    sets = {name: set() for name in (
        "documents", "source_files", "source_blobs", "normalized_python",
        "row_sha256", "prefix32_sha256",
    )}
    for payload in registry_payloads:
        if not isinstance(payload, Mapping):
            raise ValueError("historical registry payload must be a mapping")
        for value in _walk(payload):
            if not isinstance(value, Mapping):
                continue
            mapping = {
                "document_id": "documents", "source_file": "source_files",
                "path": "source_files", "source_blob_sha256": "source_blobs",
                "blob_sha256": "source_blobs",
                "normalized_python_sha256": "normalized_python",
                "row_sha256": "row_sha256",
                "prefix32_sha256": "prefix32_sha256",
            }
            for key, target in mapping.items():
                item = value.get(key)
                if isinstance(item, str) and item:
                    sets[target].add(item)
    return PriorExclusions(*(frozenset(sets[name]) for name in (
        "documents", "source_files", "source_blobs", "normalized_python",
        "row_sha256", "prefix32_sha256",
    )))


def _primary_cell(mask: BracketMasks) -> str | None:
    named = mask.named_cells()
    candidates = []
    for order, name in enumerate(CELL_ORDER):
        positions = torch.nonzero(named[name][0, SCORE_START:SCORE_STOP], as_tuple=False)
        if positions.numel():
            candidates.append((int(positions[0, 0]), order, name))
    return min(candidates)[2] if candidates else None


def support_census(masks: BracketMasks, domain: BracketDomain) -> dict[str, dict[str, int]]:
    masks.validate()
    domain_mask = masks.domain_index.eq(tuple(BracketDomain).index(domain))
    scored = torch.zeros_like(masks.compatible)
    scored[:, SCORE_START:SCORE_STOP] = True
    cells = dict(masks.named_cells()); cells["all"] = scored
    output = {}
    for name, mask in cells.items():
        selected = mask & domain_mask & scored
        output[name] = {
            "positions": int(selected.sum()),
            "documents": int(selected.any(1).sum()),
        }
    if any(
        output[name]["positions"] < MIN_CELL_POSITIONS
        or output[name]["documents"] < MIN_CELL_DOCUMENTS
        for name in (*CELL_ORDER, "all")
    ):
        raise RuntimeError(f"{domain.value} bracket score support is underpowered")
    return output


def allocate_roles(
    rows: torch.Tensor,
    records: Sequence[CandidateRecord],
    registry: DelimiterRegistry,
    prior: PriorExclusions,
    *, seed: str,
) -> tuple[FrozenRole, ...]:
    """Allocate exact role/domain/cell quotas in a fixed content-hash order."""

    if rows.device.type != "cpu" or rows.dtype != torch.long or rows.ndim != 2 or (
        rows.shape[1] != ROW_LENGTH or not rows.is_contiguous()
    ) or len(records) != rows.shape[0] or not isinstance(seed, str) or not seed:
        raise ValueError("candidate pool currency is malformed")
    if any(type(record) is not CandidateRecord for record in records):
        raise ValueError("candidate provenance needs exact CandidateRecord objects")
    buckets: dict[tuple[BracketDomain, str], list[tuple[str, int]]] = {
        (domain, cell): [] for domain in BracketDomain for cell in CELL_ORDER
    }
    seen = {"documents": set(), "source_files": set(), "rows": set(), "prefixes": set()}
    for index, record in enumerate(records):
        row = rows[index]
        row_hash, prefix_hash = row_sha256(row), prefix32_sha256(row)
        blocked = (
            record.document_id in prior.documents or row_hash in prior.row_sha256
            or prefix_hash in prior.prefix32_sha256
            or (record.domain is BracketDomain.CODE and (
                record.source_file in prior.source_files
                or record.source_blob_sha256 in prior.source_blobs
            ))
            or (record.normalized_python_sha256 is not None
                and record.normalized_python_sha256 in prior.normalized_python)
        )
        if blocked:
            continue
        file_identity = (
            record.source_file if record.domain is BracketDomain.CODE
            else f"prose-document:{record.document_id}"
        )
        identities = (record.document_id, file_identity, row_hash, prefix_hash)
        if any(identity in seen[name] for identity, name in zip(
            identities, ("documents", "source_files", "rows", "prefixes"), strict=True,
        )):
            continue
        one = build_bracket_masks(
            row.unsqueeze(0).contiguous(), registry, (record.domain,),
            first_prediction=SCORE_START,
        )
        cell = _primary_cell(one)
        if cell is None:
            continue
        order = hashlib.sha256(
            (seed + record.document_id + record.source_file + row_hash).encode()
        ).hexdigest()
        buckets[(record.domain, cell)].append((order, index))
        for identity, name in zip(
            identities, ("documents", "source_files", "rows", "prefixes"), strict=True,
        ):
            seen[name].add(identity)

    selected: dict[RowRole, list[int]] = {role: [] for role in RowRole}
    needed = ROWS_PER_CELL * len(tuple(RowRole))
    for domain in BracketDomain:
        for cell in CELL_ORDER:
            candidates = [index for _, index in sorted(buckets[(domain, cell)])]
            if len(candidates) < needed:
                raise RuntimeError(f"candidate pool lacks {domain.value}/{cell} quota")
            for role_index, role in enumerate(RowRole):
                start = role_index * ROWS_PER_CELL
                selected[role].extend(candidates[start:start + ROWS_PER_CELL])

    output = []
    global_docs: set[str] = set(); global_files: set[str] = set()
    for role in RowRole:
        indices = selected[role]
        role_rows = rows[indices].contiguous()
        role_records = tuple(records[index] for index in indices)
        docs = {record.document_id for record in role_records}
        files = {record.source_file for record in role_records
                 if record.domain is BracketDomain.CODE}
        code_count = sum(record.domain is BracketDomain.CODE for record in role_records)
        if len(docs) != len(role_records) or len(files) != code_count or (
            docs & global_docs or files & global_files
        ):
            raise RuntimeError("allocated roles are not document/source-file disjoint")
        global_docs |= docs; global_files |= files
        domains = tuple(record.domain for record in role_records)
        masks = build_bracket_masks(
            role_rows, registry, domains, first_prediction=SCORE_START,
        )
        support = {domain.value: support_census(masks, domain) for domain in BracketDomain}
        output.append(FrozenRole(role, role_rows, role_records, masks, support))
    return tuple(output)


__all__ = (
    "CELL_ORDER", "CandidateRecord", "FrozenRole", "PriorExclusions", "RowRole",
    "allocate_roles", "historical_exclusions", "prefix32_sha256", "registry_sha256",
    "row_sha256", "support_census", "tensor_sha256",
)
