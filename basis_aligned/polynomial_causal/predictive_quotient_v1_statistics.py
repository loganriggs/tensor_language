"""Tensor-only sufficient statistics for predictive quotient v1.

This module has no model, row loader, artifact publisher, or role authority. A future
source-closed observed consumer may pass its ephemeral VJP sketches here and receive
only mergeable quadratic summaries; logits, codes, graphs, and raw VJPs are not stored.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Sequence

import torch


BATCH_ROWS = 4
SCORED_POSITIONS = 192
SCORE_START = 64
SCORE_STOP = 256
CODE_DIM = 64
ROLE_ROWS = 192
ROLE_BATCHES = ROLE_ROWS // BATCH_ROWS
PRIMARY_PROBE_SEEDS = tuple(range(2026082860, 2026082868))
REPLICATION_PROBE_SEEDS = tuple(range(2026082868, 2026082876))
PROBE_SEEDS = {
    "primary": PRIMARY_PROBE_SEEDS,
    "replication": REPLICATION_PROBE_SEEDS,
}
PROBES_PER_BANK = len(PRIMARY_PROBE_SEEDS)
SKETCHES_PER_BATCH = PROBES_PER_BANK * BATCH_ROWS * SCORED_POSITIONS


def _sha256_text(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _logical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def _finite_float64(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if not torch.is_tensor(value) or value.dtype != torch.float64 or value.device.type != (
        "cpu"
    ) or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite CPU float64 with shape {shape}")
    return value.detach().clone().contiguous()


def _symmetric_rows(name: str, value: torch.Tensor) -> None:
    error = float(torch.max(torch.abs(value - value.transpose(-1, -2))))
    scale = max(1.0, float(torch.max(torch.linalg.matrix_norm(value, ord=2))))
    if error > 1e-10 * scale:
        raise ValueError(f"{name} is not symmetric")
    minimum = float(torch.min(torch.linalg.eigvalsh(value)))
    if minimum < -1e-10 * scale:
        raise ValueError(f"{name} is not positive semidefinite")


@dataclass(frozen=True)
class FisherBatchSummary:
    """One bank/batch summary containing no graph-bearing or vocabulary tensor."""

    bank: str
    probe_seeds: tuple[int, ...]
    batch_ordinal: int
    ordered_row_indices: tuple[int, ...]
    source_identity_sha256: str
    target_ids_sha256: str
    sketch_count: int
    row_sketch_count: int
    outer_product_sum: torch.Tensor
    row_outer_product_sums: torch.Tensor
    assigned_position_outer_sums: torch.Tensor
    assigned_positions: torch.Tensor
    assigned_direction_indices: torch.Tensor

    def __post_init__(self) -> None:
        expected_rows = tuple(range(
            self.batch_ordinal * BATCH_ROWS,
            (self.batch_ordinal + 1) * BATCH_ROWS,
        )) if type(self.batch_ordinal) is int and 0 <= self.batch_ordinal < (
            ROLE_BATCHES
        ) else ()
        if self.bank not in PROBE_SEEDS or self.probe_seeds != PROBE_SEEDS.get(
            self.bank
        ) or expected_rows != self.ordered_row_indices or not _sha256_text(
            self.source_identity_sha256
        ) or not _sha256_text(self.target_ids_sha256) or self.sketch_count != (
            SKETCHES_PER_BATCH
        ) or self.row_sketch_count != PROBES_PER_BANK * SCORED_POSITIONS:
            raise ValueError("Fisher batch scalar/identity contract changed")
        outer = _finite_float64(
            "outer product sum", self.outer_product_sum, (CODE_DIM, CODE_DIM),
        )
        rows = _finite_float64(
            "row outer product sums", self.row_outer_product_sums,
            (BATCH_ROWS, CODE_DIM, CODE_DIM),
        )
        assigned = _finite_float64(
            "assigned-position outer sums", self.assigned_position_outer_sums,
            (BATCH_ROWS, CODE_DIM, CODE_DIM),
        )
        positions = self.assigned_positions
        directions = self.assigned_direction_indices
        if not torch.is_tensor(positions) or positions.dtype != torch.long or (
            positions.device.type != "cpu"
        ) or tuple(positions.shape) != (BATCH_ROWS,) or bool((positions < SCORE_START).any()) or (
            bool((positions >= SCORE_STOP).any())
        ) or not torch.is_tensor(directions) or directions.dtype != torch.long or (
            directions.device.type != "cpu"
        ) or tuple(directions.shape) != (BATCH_ROWS,) or bool((directions < 0).any()) or (
            bool((directions >= 32).any())
        ):
            raise ValueError("Fisher batch intervention assignment changed")
        _symmetric_rows("row outer sums", rows)
        _symmetric_rows("assigned-position outer sums", assigned)
        if not torch.allclose(outer, rows.sum(dim=0), rtol=1e-12, atol=1e-12):
            raise ValueError("Fisher batch global and row outer sums differ")
        object.__setattr__(self, "outer_product_sum", outer)
        object.__setattr__(self, "row_outer_product_sums", rows)
        object.__setattr__(self, "assigned_position_outer_sums", assigned)
        object.__setattr__(self, "assigned_positions", positions.detach().clone().contiguous())
        object.__setattr__(
            self, "assigned_direction_indices", directions.detach().clone().contiguous(),
        )

    @property
    def sha256(self) -> str:
        return _logical_sha256({
            "bank": self.bank, "probe_seeds": list(self.probe_seeds),
            "batch_ordinal": self.batch_ordinal,
            "ordered_row_indices": list(self.ordered_row_indices),
            "source_identity_sha256": self.source_identity_sha256,
            "target_ids_sha256": self.target_ids_sha256,
            "sketch_count": self.sketch_count,
            "row_sketch_count": self.row_sketch_count,
            "outer_product_sum": _tensor_sha256(self.outer_product_sum),
            "row_outer_product_sums": _tensor_sha256(self.row_outer_product_sums),
            "assigned_position_outer_sums": _tensor_sha256(
                self.assigned_position_outer_sums
            ),
            "assigned_positions": _tensor_sha256(self.assigned_positions),
            "assigned_direction_indices": _tensor_sha256(
                self.assigned_direction_indices
            ),
        })


def summarize_fisher_batch(
    sketches: torch.Tensor,
    probe_token_ids: torch.Tensor,
    *,
    bank: str,
    batch_ordinal: int,
    source_identity_sha256: str,
    assigned_positions: torch.Tensor,
    assigned_direction_indices: torch.Tensor,
) -> FisherBatchSummary:
    """Reduce one ephemeral VJP bank before any caller-visible return."""

    expected = (PROBES_PER_BANK, BATCH_ROWS, SCORED_POSITIONS, CODE_DIM)
    if not torch.is_tensor(sketches) or tuple(sketches.shape) != expected or not bool(
        torch.isfinite(sketches).all()
    ):
        raise ValueError(f"Fisher sketches must be finite with shape {expected}")
    target_shape = (PROBES_PER_BANK, BATCH_ROWS, SCORED_POSITIONS)
    if not torch.is_tensor(probe_token_ids) or probe_token_ids.dtype != torch.long or (
        tuple(probe_token_ids.shape) != target_shape
    ) or probe_token_ids.device.type != "cpu" or bool((probe_token_ids < 0).any()) or (
        bool((probe_token_ids >= 50304).any())
    ):
        raise ValueError("Fisher target IDs changed shape, dtype, device, or vocabulary")
    values = sketches.detach().cpu().double()
    row_outer = torch.einsum("pbti,pbtj->bij", values, values).contiguous()
    positions = assigned_positions.detach().cpu().long().contiguous()
    if tuple(positions.shape) != (BATCH_ROWS,) or bool((positions < SCORE_START).any()) or (
        bool((positions >= SCORE_STOP).any())
    ):
        raise ValueError("assigned positions are outside scored support")
    relative = positions - SCORE_START
    selected = values[:, torch.arange(BATCH_ROWS), relative]
    assigned_outer = torch.einsum("pbi,pbj->bij", selected, selected).contiguous()
    return FisherBatchSummary(
        bank=bank, probe_seeds=PROBE_SEEDS.get(bank, ()),
        batch_ordinal=batch_ordinal,
        ordered_row_indices=tuple(range(
            batch_ordinal * BATCH_ROWS, (batch_ordinal + 1) * BATCH_ROWS,
        )) if type(batch_ordinal) is int else (),
        source_identity_sha256=source_identity_sha256,
        target_ids_sha256=_tensor_sha256(probe_token_ids),
        sketch_count=SKETCHES_PER_BATCH,
        row_sketch_count=PROBES_PER_BANK * SCORED_POSITIONS,
        outer_product_sum=row_outer.sum(dim=0),
        row_outer_product_sums=row_outer,
        assigned_position_outer_sums=assigned_outer,
        assigned_positions=positions,
        assigned_direction_indices=assigned_direction_indices,
    )


@dataclass(frozen=True)
class FisherRoleStatistics:
    """Complete ordered validation-role sufficient statistics for both probe banks."""

    common_support_sha256: str
    source_identity_sha256: Mapping[str, tuple[str, ...]]
    target_ids_sha256: Mapping[str, tuple[str, ...]]
    row_outer_product_sums: Mapping[str, torch.Tensor]
    assigned_position_outer_sums: Mapping[str, torch.Tensor]
    assigned_positions: torch.Tensor
    assigned_direction_indices: torch.Tensor
    count_per_bank: int

    def __post_init__(self) -> None:
        if not _sha256_text(self.common_support_sha256) or set(
            self.source_identity_sha256
        ) != set(PROBE_SEEDS) or set(self.target_ids_sha256) != set(PROBE_SEEDS) or set(
            self.row_outer_product_sums
        ) != set(PROBE_SEEDS) or set(self.assigned_position_outer_sums) != set(PROBE_SEEDS) or (
            self.count_per_bank != PROBES_PER_BANK * ROLE_ROWS * SCORED_POSITIONS
        ):
            raise ValueError("Fisher role statistics schema changed")
        row_outer = {}
        assigned_outer = {}
        for bank in PROBE_SEEDS:
            if len(self.source_identity_sha256[bank]) != ROLE_BATCHES or len(
                self.target_ids_sha256[bank]
            ) != ROLE_BATCHES or any(not _sha256_text(value) for value in (
                *self.source_identity_sha256[bank], *self.target_ids_sha256[bank],
            )):
                raise ValueError("Fisher role hash ledger is incomplete")
            row_outer[bank] = _finite_float64(
                f"{bank} row outer sums", self.row_outer_product_sums[bank],
                (ROLE_ROWS, CODE_DIM, CODE_DIM),
            )
            assigned_outer[bank] = _finite_float64(
                f"{bank} assigned outer sums", self.assigned_position_outer_sums[bank],
                (ROLE_ROWS, CODE_DIM, CODE_DIM),
            )
            _symmetric_rows(f"{bank} row outer sums", row_outer[bank])
            _symmetric_rows(f"{bank} assigned outer sums", assigned_outer[bank])
        positions = self.assigned_positions.detach().cpu().long().contiguous()
        directions = self.assigned_direction_indices.detach().cpu().long().contiguous()
        if tuple(positions.shape) != (ROLE_ROWS,) or tuple(directions.shape) != (
            ROLE_ROWS,
        ) or bool((positions < SCORE_START).any()) or bool((positions >= SCORE_STOP).any()) or (
            bool((directions < 0).any()) or bool((directions >= 32).any())
        ):
            raise ValueError("Fisher role intervention assignment changed")
        object.__setattr__(self, "source_identity_sha256", MappingProxyType({
            bank: tuple(self.source_identity_sha256[bank]) for bank in PROBE_SEEDS
        }))
        object.__setattr__(self, "target_ids_sha256", MappingProxyType({
            bank: tuple(self.target_ids_sha256[bank]) for bank in PROBE_SEEDS
        }))
        object.__setattr__(self, "row_outer_product_sums", MappingProxyType(row_outer))
        object.__setattr__(
            self, "assigned_position_outer_sums", MappingProxyType(assigned_outer),
        )
        object.__setattr__(self, "assigned_positions", positions)
        object.__setattr__(self, "assigned_direction_indices", directions)

    def observability(self, bank: str) -> torch.Tensor:
        if bank not in PROBE_SEEDS:
            raise ValueError("unknown Fisher probe bank")
        return (self.row_outer_product_sums[bank].sum(dim=0) / self.count_per_bank).contiguous()

    def assigned_quadratic_response(
        self, bank: str, directions: torch.Tensor,
    ) -> torch.Tensor:
        """Return each row's mean-probe local response for one assigned direction."""

        if bank not in PROBE_SEEDS or not torch.is_tensor(directions) or tuple(
            directions.shape
        ) != (ROLE_ROWS, CODE_DIM) or not bool(torch.isfinite(directions).all()):
            raise ValueError("assigned response directions are malformed")
        values = directions.detach().cpu().double()
        return torch.einsum(
            "bi,bij,bj->b", values, self.assigned_position_outer_sums[bank], values,
        ) / PROBES_PER_BANK


class FisherStatisticsCollector:
    """One-use exact 48-batch collector with prospective identity commitments."""

    def __init__(
        self, *, common_support_sha256: str,
        expected_source_identity_sha256: Mapping[str, Sequence[str]],
    ) -> None:
        if not _sha256_text(common_support_sha256) or set(
            expected_source_identity_sha256
        ) != set(PROBE_SEEDS):
            raise ValueError("Fisher collector support/identity schema changed")
        expected = {}
        for bank in PROBE_SEEDS:
            values = tuple(expected_source_identity_sha256[bank])
            if len(values) != ROLE_BATCHES or len(set(values)) != ROLE_BATCHES or any(
                not _sha256_text(value) for value in values
            ):
                raise ValueError("Fisher collector identity plan is malformed")
            expected[bank] = values
        if expected["primary"] != expected["replication"]:
            raise ValueError("Fisher probe banks must share one source identity plan")
        self._common_support_sha256 = common_support_sha256
        self._expected = expected
        self._summaries: dict[tuple[str, int], FisherBatchSummary] = {}
        self._summary_sha256: dict[tuple[str, int], str] = {}
        self._closed = False

    def add(self, summary: FisherBatchSummary) -> None:
        if self._closed or not isinstance(summary, FisherBatchSummary):
            raise RuntimeError("Fisher collector is closed or received the wrong type")
        key = (summary.bank, summary.batch_ordinal)
        if key in self._summaries or summary.source_identity_sha256 != self._expected[
            summary.bank
        ][summary.batch_ordinal]:
            raise RuntimeError("Fisher summary is duplicated or differs from its plan")
        self._summaries[key] = summary
        self._summary_sha256[key] = summary.sha256

    def finalize(self) -> FisherRoleStatistics:
        if self._closed:
            raise RuntimeError("Fisher collector was already finalized")
        self._closed = True
        required = {
            (bank, ordinal) for bank in PROBE_SEEDS for ordinal in range(ROLE_BATCHES)
        }
        if set(self._summaries) != required:
            raise RuntimeError("Fisher collector is missing an exact bank/batch summary")
        if any(
            self._summaries[key].sha256 != self._summary_sha256[key] for key in required
        ):
            raise RuntimeError("Fisher summary mutated after collector admission")
        row_outer = {}
        assigned_outer = {}
        source_hashes = {}
        target_hashes = {}
        positions = None
        directions = None
        for bank in PROBE_SEEDS:
            ordered = [self._summaries[(bank, ordinal)] for ordinal in range(ROLE_BATCHES)]
            row_outer[bank] = torch.cat([
                summary.row_outer_product_sums for summary in ordered
            ]).contiguous()
            assigned_outer[bank] = torch.cat([
                summary.assigned_position_outer_sums for summary in ordered
            ]).contiguous()
            source_hashes[bank] = tuple(
                summary.source_identity_sha256 for summary in ordered
            )
            target_hashes[bank] = tuple(summary.target_ids_sha256 for summary in ordered)
            bank_positions = torch.cat([summary.assigned_positions for summary in ordered])
            bank_directions = torch.cat([
                summary.assigned_direction_indices for summary in ordered
            ])
            if positions is None:
                positions, directions = bank_positions, bank_directions
            elif not torch.equal(positions, bank_positions) or not torch.equal(
                directions, bank_directions
            ):
                raise RuntimeError("Fisher banks use different intervention assignments")
        if source_hashes["primary"] != source_hashes["replication"]:
            raise RuntimeError("Fisher banks came from different graph-bearing sources")
        assert positions is not None and directions is not None
        return FisherRoleStatistics(
            common_support_sha256=self._common_support_sha256,
            source_identity_sha256=source_hashes, target_ids_sha256=target_hashes,
            row_outer_product_sums=row_outer,
            assigned_position_outer_sums=assigned_outer,
            assigned_positions=positions, assigned_direction_indices=directions,
            count_per_bank=PROBES_PER_BANK * ROLE_ROWS * SCORED_POSITIONS,
        )
