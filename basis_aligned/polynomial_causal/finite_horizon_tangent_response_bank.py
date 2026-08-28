"""Source-closed CPU contract for a finite-horizon tangent response bank.

The model-side collector is deliberately absent.  It must emit one detached response
matrix per registered source site and row; this module validates, copies, hashes, and
seals those matrices into document-disjoint split operators.  Partial banks, duplicate
rows, graph-bearing tensors, and missing directions are never interpreted as zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch

import finite_horizon_tangent_realization as realization


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def allocate_whole_document_splits(
    document_ids: Sequence[str], split_names: tuple[str, str] = ("primary", "replication"),
) -> tuple[str, ...]:
    """Greedily balance row counts while keeping every document in one split."""
    if len(split_names) != 2 or len(set(split_names)) != 2 or any(not x for x in split_names):
        raise ValueError("exactly two distinct nonempty split names are required")
    if not document_ids or any(not isinstance(value, str) or not value for value in document_ids):
        raise ValueError("document ids must be nonempty strings")
    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    for index, document in enumerate(document_ids):
        counts[document] = counts.get(document, 0) + 1
        first.setdefault(document, index)
    if len(counts) < 2:
        raise ValueError("at least two source documents are required")
    assignment: dict[str, str] = {}
    loads = {name: 0 for name in split_names}
    for document in sorted(counts, key=lambda d: (-counts[d], first[d], d)):
        destination = min(split_names, key=lambda name: (loads[name], split_names.index(name)))
        assignment[document] = destination
        loads[destination] += counts[document]
    if min(loads.values()) == 0:
        raise ValueError("whole-document allocation produced an empty split")
    return tuple(assignment[document] for document in document_ids)


@dataclass(frozen=True)
class TangentResponsePlan:
    experiment_id: str
    row_artifact_sha256: str
    row_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    splits: tuple[str, ...]
    scored_positions: tuple[int, ...]
    input_dims: tuple[tuple[int, int], ...]
    target_site: int
    probes_per_row: int
    direction_seed: int
    probe_seed: int
    position_seed: int

    def __post_init__(self) -> None:
        n_rows = len(self.row_ids)
        dimensions = dict(self.input_dims)
        if not self.experiment_id or len(self.row_artifact_sha256) != 64:
            raise ValueError("experiment identity or row hash is malformed")
        if n_rows == 0 or len(set(self.row_ids)) != n_rows:
            raise ValueError("row ids must be nonempty and unique")
        if len(self.document_ids) != n_rows or len(self.splits) != n_rows or len(
            self.scored_positions
        ) != n_rows:
            raise ValueError("row, document, split, and position ledgers must align")
        if any(type(position) is not int or not 64 <= position < 256
               for position in self.scored_positions):
            raise ValueError("scored positions must lie in the frozen 64:256 window")
        if any(not value for value in self.document_ids) or set(self.splits) != {
            "primary", "replication"
        }:
            raise ValueError("both frozen splits must be populated")
        document_splits: dict[str, str] = {}
        for document, split in zip(self.document_ids, self.splits, strict=True):
            previous = document_splits.setdefault(document, split)
            if previous != split:
                raise ValueError("a source document crosses tangent-bank splits")
        if not dimensions or len(dimensions) != len(self.input_dims) or any(
            type(site) is not int or type(width) is not int or width <= 0
            for site, width in self.input_dims
        ):
            raise ValueError("input dimensions are malformed or duplicated")
        if type(self.target_site) is not int or any(site >= self.target_site for site in dimensions):
            raise ValueError("the final behavioral target must follow every source site")
        if type(self.probes_per_row) is not int or self.probes_per_row <= 0:
            raise ValueError("probe count must be positive")
        if any(type(seed) is not int or seed < 0 for seed in (
            self.direction_seed, self.probe_seed, self.position_seed,
        )):
            raise ValueError("seeds must be nonnegative integers")

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256({
            "experiment_id": self.experiment_id,
            "row_artifact_sha256": self.row_artifact_sha256,
            "row_ids": self.row_ids,
            "document_ids": self.document_ids,
            "splits": self.splits,
            "scored_positions": self.scored_positions,
            "input_dims": self.input_dims,
            "target_site": self.target_site,
            "probes_per_row": self.probes_per_row,
            "direction_seed": self.direction_seed,
            "probe_seed": self.probe_seed,
            "position_seed": self.position_seed,
        })


@dataclass(frozen=True)
class SealedTangentResponseBank:
    plan_fingerprint: str
    split_blocks: Mapping[str, Mapping[tuple[int, int], torch.Tensor]]
    input_dims: Mapping[int, int]
    output_dims_by_split: Mapping[str, Mapping[int, int]]
    receipt: Mapping[str, Any]


class TangentResponseBankTransaction:
    """One-use assembler that cannot emit a partial response operator."""

    def __init__(self, plan: TangentResponsePlan) -> None:
        if not isinstance(plan, TangentResponsePlan):
            raise TypeError("a validated tangent response plan is required")
        self.__plan: TangentResponsePlan | None = plan
        self.__rows: dict[str, dict[int, torch.Tensor]] | None = {}
        self.__hashes: dict[str, dict[int, str]] | None = {}
        self.__closed = False

    @property
    def closed(self) -> bool:
        return self.__closed

    @property
    def aliases_revoked(self) -> bool:
        return self.__closed and self.__plan is None and self.__rows is None and self.__hashes is None

    def _revoke(self) -> None:
        self.__plan = None
        self.__rows = None
        self.__hashes = None
        self.__closed = True

    def add_row(self, row_id: str, responses: Mapping[int, torch.Tensor]) -> None:
        if self.__closed:
            raise RuntimeError("tangent response bank transaction is spent")
        plan, rows, hashes = self.__plan, self.__rows, self.__hashes
        assert plan is not None and rows is not None and hashes is not None
        if row_id not in plan.row_ids:
            raise ValueError("response row is not registered")
        if row_id in rows:
            raise ValueError("duplicate response row")
        dimensions = dict(plan.input_dims)
        if set(responses) != set(dimensions):
            raise ValueError("every and only registered source sites must be supplied")
        copied: dict[int, torch.Tensor] = {}
        row_hashes: dict[int, str] = {}
        for site in sorted(dimensions):
            value = responses[site]
            expected = (plan.probes_per_row, dimensions[site])
            if not torch.is_tensor(value) or tuple(value.shape) != expected:
                raise ValueError(f"source {site} response has the wrong shape")
            if value.device.type != "cpu" or value.dtype != torch.float64 or value.requires_grad:
                raise ValueError("responses must be detached CPU float64 tensors")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("responses must be finite")
            copied[site] = value.contiguous().clone()
            row_hashes[site] = _tensor_sha256(value)
        rows[row_id] = copied
        hashes[row_id] = row_hashes

    def seal(self) -> SealedTangentResponseBank:
        if self.__closed:
            raise RuntimeError("tangent response bank transaction is spent")
        plan, rows, hashes = self.__plan, self.__rows, self.__hashes
        assert plan is not None and rows is not None and hashes is not None
        try:
            missing = tuple(row_id for row_id in plan.row_ids if row_id not in rows)
            if missing:
                raise RuntimeError(f"tangent response bank is incomplete: {len(missing)} rows missing")
            dimensions = dict(plan.input_dims)
            split_blocks: dict[str, dict[tuple[int, int], torch.Tensor]] = {}
            output_dims: dict[str, dict[int, int]] = {}
            split_receipts: dict[str, Any] = {}
            for split in ("primary", "replication"):
                selected = [row_id for row_id, role in zip(
                    plan.row_ids, plan.splits, strict=True,
                ) if role == split]
                blocks = {
                    (plan.target_site, site): torch.cat(
                        [rows[row_id][site] for row_id in selected], dim=0,
                    )
                    for site in sorted(dimensions)
                }
                split_blocks[split] = blocks
                output_dims[split] = {plan.target_site: len(selected) * plan.probes_per_row}
                split_receipts[split] = {
                    "rows": len(selected),
                    "source_documents": len({
                        document for document, role in zip(
                            plan.document_ids, plan.splits, strict=True,
                        ) if role == split
                    }),
                    "output_dimension": len(selected) * plan.probes_per_row,
                }
            ordered_hashes = [
                [row_id, [[site, hashes[row_id][site]] for site, _ in plan.input_dims]]
                for row_id in plan.row_ids
            ]
            receipt = {
                "status": "complete",
                "plan_fingerprint": plan.fingerprint,
                "row_artifact_sha256": plan.row_artifact_sha256,
                "response_bank_sha256": _canonical_sha256(ordered_hashes),
                "registered_rows": len(plan.row_ids),
                "registered_source_sites": list(dimensions),
                "directions_per_source": dimensions,
                "probes_per_row": plan.probes_per_row,
                "scored_position_sha256": _canonical_sha256(plan.scored_positions),
                "every_direction_evaluated_on_every_row": True,
                "whole_document_splits": True,
                "splits": split_receipts,
            }
            bank = SealedTangentResponseBank(
                plan_fingerprint=plan.fingerprint,
                split_blocks=split_blocks,
                input_dims=dimensions,
                output_dims_by_split=output_dims,
                receipt=receipt,
            )
        finally:
            self._revoke()
        return bank


def analyze_bank(
    bank: SealedTangentResponseBank, cuts: tuple[int, ...], **analysis_kwargs: Any,
) -> dict[str, Any]:
    if not isinstance(bank, SealedTangentResponseBank):
        raise TypeError("a sealed tangent response bank is required")
    return {
        split: realization.analyze_all_cuts(
            blocks, bank.input_dims, bank.output_dims_by_split[split], cuts,
            **analysis_kwargs,
        )
        for split, blocks in bank.split_blocks.items()
    }
