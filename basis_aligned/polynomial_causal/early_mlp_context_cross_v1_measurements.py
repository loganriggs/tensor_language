"""Pure ordered two-role measurement boundary for early-MLP/context cross v1.

The eventual GPU adapter supplies one row-level correct-count/CE column at a time.
This module binds exact requests, common support, cell receipts, literal document
aggregation, and sealed discovery/validation/heldout capabilities.  It performs no
row, checkpoint, model, CUDA, artifact, or git I/O.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import torch

import early_mlp_context_cross_v1 as cross
import early_mlp_context_cross_v1_statistics as statistics


SCHEMA_VERSION = 1
PREREGISTRATION_SHA256 = (
    "b1fa36f022ccd7e403a1c82dc77c5d888a26f8e615120dd577a57ff2cd9a6970"
)
PROGRAM_FAMILY = "section1786_contextfree_rank64_table_learned_rank64_map"
SHARED_PROGRAM_SHA256 = (
    "cad513c942cccaf01e747cb600428b427c03d98dd0dddc710a4028ff1ba9d0bb"
)
MODEL_REALIZATION_SHA256 = (
    "cf3ca3f55028979ef6f87ac4afa08a7d90fc01dfa4fc2ce037343ac3c69688eb"
)
COMPONENT_TREE_SHA256 = (
    "94cbebb35ca3f8c6923f5040b76d243c3f3fa192496604bd40abeb2e4077da0c"
)
ROLE_DOCUMENT_COUNTS = {"skip7000": 79, "skip11000": 105}
ROW_COUNT = 192
SCORED_TOKENS_PER_ROW = 192
BATCH_COUNT = 24


def _sha256_text(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _commit_text(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _stage_for_cell(cell: cross.Cell) -> str:
    matches = [
        stage for stage, cells in statistics.STAGE_CELLS.items() if cell in cells
    ]
    if len(matches) != 1:
        raise RuntimeError("cell does not have exactly one evidence stage")
    return matches[0]


@dataclass(frozen=True, slots=True)
class MeasurementRequest:
    ordinal: int
    cell: cross.Cell
    stage: str
    sites: tuple[cross.Site, ...]
    schema_version: int = SCHEMA_VERSION
    preregistration_sha256: str = PREREGISTRATION_SHA256
    program_family: str = PROGRAM_FAMILY

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or not 0 <= self.ordinal < 64 or self.cell != (
            self.ordinal // 8, self.ordinal % 8
        ) or self.stage != _stage_for_cell(self.cell) or self.sites != (
            cross.mask_for_cell(self.cell)
        ) or len(self.sites) != len(set(self.sites)) or self.schema_version != (
            SCHEMA_VERSION
        ) or self.preregistration_sha256 != PREREGISTRATION_SHA256 or (
            self.program_family != PROGRAM_FAMILY
        ):
            raise ValueError("measurement request changed")

    @property
    def sha256(self) -> str:
        return _logical_sha256(asdict(self))


REQUESTS = tuple(
    MeasurementRequest(
        ordinal=ordinal, cell=cell, stage=_stage_for_cell(cell),
        sites=cross.mask_for_cell(cell),
    )
    for ordinal, cell in enumerate(cross.ALL_CELLS)
)
REQUEST_PLAN_SHA256 = _logical_sha256({
    "schema_version": SCHEMA_VERSION,
    "preregistration_sha256": PREREGISTRATION_SHA256,
    "request_sha256s": [request.sha256 for request in REQUESTS],
})


@dataclass(frozen=True, slots=True)
class RoleAuthority:
    """Outcome-blind authority for exactly one role in one two-role transaction."""

    role: str
    source_commit: str
    source_closure_sha256: str
    source_receipt_file_sha256: str
    row_file_sha256: str
    row_raw_sha256: str
    row_provenance_sha256: str
    ordered_row_identity_sha256: str
    ordered_row_to_document_sha256: str
    ordered_document_ids_sha256: str
    row_token_count_sha256: str
    document_identity_set_sha256: str
    cross_role_disjointness_sha256: str
    wave_nonce_sha256: str
    row_count: int
    document_count: int
    total_scored_token_count: int
    batch_count: int
    request_plan_sha256: str = REQUEST_PLAN_SHA256
    model_realization_sha256: str = MODEL_REALIZATION_SHA256
    component_tree_sha256: str = COMPONENT_TREE_SHA256
    shared_program_sha256: str = SHARED_PROGRAM_SHA256
    authorized_for_final_role: bool = False

    def __post_init__(self) -> None:
        hashes = tuple(
            getattr(self, name) for name in self.__dataclass_fields__
            if name.endswith("sha256") and name != "request_plan_sha256"
        )
        if self.role not in statistics.ROLE_NAMES or not _commit_text(
            self.source_commit
        ) or any(not _sha256_text(value) for value in hashes) or (
            self.request_plan_sha256 != REQUEST_PLAN_SHA256
        ) or self.model_realization_sha256 != MODEL_REALIZATION_SHA256 or (
            self.component_tree_sha256 != COMPONENT_TREE_SHA256
        ) or self.shared_program_sha256 != SHARED_PROGRAM_SHA256 or (
            self.authorized_for_final_role is not False
        ) or self.row_count != ROW_COUNT or self.document_count != (
            ROLE_DOCUMENT_COUNTS[self.role]
        ) or self.total_scored_token_count != ROW_COUNT * SCORED_TOKENS_PER_ROW or (
            self.batch_count != BATCH_COUNT
        ):
            raise ValueError("role measurement authority changed")

    @property
    def sha256(self) -> str:
        return _logical_sha256(asdict(self))


class RowCellStatistics:
    """Sealed row-level values for one cell on the role's common support."""

    __slots__ = ("_ce_sum", "_expected_sha256", "_row_token_count", "_sealed", "_top1")

    def __init__(
        self, *, top1_correct: torch.Tensor, ce_sum: torch.Tensor,
        row_token_count: torch.Tensor,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        if not torch.is_tensor(top1_correct) or not torch.is_tensor(ce_sum) or not (
            torch.is_tensor(row_token_count)
        ) or top1_correct.ndim != 1 or len(top1_correct) == 0 or ce_sum.shape != (
            top1_correct.shape
        ) or (
            row_token_count.shape != top1_correct.shape
        ) or top1_correct.dtype != torch.long or ce_sum.dtype != torch.float64 or (
            row_token_count.dtype != torch.long
        ):
            raise ValueError("row-cell statistic schema changed")
        values = (top1_correct, ce_sum, row_token_count)
        if any(
            value.device.type != "cpu" or not value.is_contiguous() or value.requires_grad
            for value in values
        ) or not torch.equal(row_token_count, torch.full(
            (len(row_token_count),), SCORED_TOKENS_PER_ROW, dtype=torch.long,
        )) or bool((top1_correct < 0).any()) or bool(
            (top1_correct > row_token_count).any()
        ) or bool((ce_sum < 0).any()) or not bool(torch.isfinite(ce_sum).all()):
            raise ValueError("row-cell statistics violate support/count bounds")
        self._top1 = top1_correct.detach().clone().contiguous()
        self._ce_sum = ce_sum.detach().clone().contiguous()
        self._row_token_count = row_token_count.detach().clone().contiguous()
        self._expected_sha256 = self._compute_sha256()
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("row-cell statistics are sealed")
        object.__setattr__(self, name, value)

    def _compute_sha256(self) -> str:
        return _logical_sha256({
            "top1_correct_sha256": statistics.tensor_sha256(self._top1),
            "ce_sum_sha256": statistics.tensor_sha256(self._ce_sum),
            "row_token_count_sha256": statistics.tensor_sha256(self._row_token_count),
        })

    def _require_pristine(self) -> None:
        if self._compute_sha256() != self._expected_sha256:
            raise RuntimeError("row-cell statistics mutated")

    @property
    def sha256(self) -> str:
        self._require_pristine()
        return self._expected_sha256

    @property
    def top1_correct_sha256(self) -> str:
        self._require_pristine()
        return statistics.tensor_sha256(self._top1)

    @property
    def ce_sum_sha256(self) -> str:
        self._require_pristine()
        return statistics.tensor_sha256(self._ce_sum)

    def _values(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._require_pristine()
        return self._top1.clone(), self._ce_sum.clone(), self._row_token_count.clone()


@dataclass(frozen=True, slots=True)
class CellReceipt:
    authority_sha256: str
    request_sha256: str
    ordinal: int
    cell: cross.Cell
    statistics_sha256: str
    top1_correct_sha256: str
    ce_sum_sha256: str
    row_token_count_sha256: str
    call_ledger_sha256: str
    source_closure_sha256: str
    model_tree_before_sha256: str
    model_tree_after_sha256: str
    shared_program_before_sha256: str
    shared_program_after_sha256: str
    outer_forward_count: int
    batch_count: int

    def __post_init__(self) -> None:
        hashes = tuple(
            getattr(self, name) for name in self.__dataclass_fields__
            if name.endswith("sha256")
        )
        if any(not _sha256_text(value) for value in hashes) or type(
            self.ordinal
        ) is not int or not 0 <= self.ordinal < 64 or self.cell != (
            self.ordinal // 8, self.ordinal % 8
        ) or self.outer_forward_count != BATCH_COUNT or self.batch_count != BATCH_COUNT:
            raise ValueError("cell receipt changed")

    @property
    def sha256(self) -> str:
        return _logical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class RoleReceipt:
    role: str
    authority_sha256: str
    request_plan_sha256: str
    source_commit: str
    source_closure_sha256: str
    row_file_sha256: str
    row_raw_sha256: str
    ordered_document_ids_sha256: str
    shared_program_sha256: str
    cell_receipt_sha256s: tuple[str, ...]
    stage_payload_sha256s: tuple[str, str, str]
    row_count: int
    document_count: int
    total_scored_token_count: int
    cell_count: int = 64

    def __post_init__(self) -> None:
        hashes = tuple(
            getattr(self, name) for name in self.__dataclass_fields__
            if name.endswith("sha256") and name != "request_plan_sha256"
        )
        if self.role not in statistics.ROLE_NAMES or any(
            not _sha256_text(value) for value in hashes
        ) or self.request_plan_sha256 != REQUEST_PLAN_SHA256 or not _commit_text(
            self.source_commit
        ) or len(self.cell_receipt_sha256s) != 64 or any(
            not _sha256_text(value) for value in self.cell_receipt_sha256s
        ) or len(self.stage_payload_sha256s) != 3 or any(
            not _sha256_text(value) for value in self.stage_payload_sha256s
        ) or self.row_count != ROW_COUNT or self.document_count != (
            ROLE_DOCUMENT_COUNTS[self.role]
        ) or self.total_scored_token_count != ROW_COUNT * SCORED_TOKENS_PER_ROW or (
            self.cell_count != 64
        ):
            raise ValueError("role receipt changed")

    @property
    def sha256(self) -> str:
        return _logical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class StagedRoleBundle:
    discovery: statistics.StageStatistics
    validation: statistics.StageStatistics
    heldout: statistics.StageStatistics
    receipt: RoleReceipt

    def __post_init__(self) -> None:
        stages = (self.discovery, self.validation, self.heldout)
        if not isinstance(self.receipt, RoleReceipt) or any(
            not isinstance(stage, statistics.StageStatistics) for stage in stages
        ) or tuple(stage.stage for stage in stages) != (
            "discovery", "validation", "heldout"
        ) or any(stage.role != self.receipt.role for stage in stages) or tuple(
            stage.sha256 for stage in stages
        ) != self.receipt.stage_payload_sha256s:
            raise ValueError("staged role bundle differs from its receipt")


def _first_seen_document_order(row_to_document: torch.Tensor, count: int) -> bool:
    seen: set[int] = set()
    expected = 0
    for raw in row_to_document.tolist():
        value = int(raw)
        if value not in seen:
            if value != expected:
                return False
            seen.add(value)
            expected += 1
    return expected == count


class RoleCollector:
    """One-use 64-cell role collector; partial outcome access is impossible."""

    __slots__ = (
        "_authority", "_ce", "_finalized", "_next", "_receipts",
        "_row_to_document", "_row_token_count", "_top1",
    )

    def __init__(
        self, *, authority: RoleAuthority, row_to_document: torch.Tensor,
        row_token_count: torch.Tensor,
    ) -> None:
        if not isinstance(authority, RoleAuthority) or not torch.is_tensor(
            row_to_document
        ) or not torch.is_tensor(row_token_count) or row_to_document.shape != (
            ROW_COUNT,
        ) or row_token_count.shape != (ROW_COUNT,) or row_to_document.dtype != (
            torch.long
        ) or row_token_count.dtype != torch.long or any(
            value.device.type != "cpu" or not value.is_contiguous() or value.requires_grad
            for value in (row_to_document, row_token_count)
        ) or bool((row_to_document < 0).any()) or bool(
            (row_to_document >= authority.document_count).any()
        ) or not _first_seen_document_order(row_to_document, authority.document_count) or (
            statistics.tensor_sha256(row_to_document)
            != authority.ordered_row_to_document_sha256
        ) or statistics.tensor_sha256(row_token_count) != authority.row_token_count_sha256 or (
            not torch.equal(row_token_count, torch.full(
                (ROW_COUNT,), SCORED_TOKENS_PER_ROW, dtype=torch.long,
            ))
        ):
            raise ValueError("role collector support differs from authority")
        self._authority = authority
        self._row_to_document = row_to_document.detach().clone().contiguous()
        self._row_token_count = row_token_count.detach().clone().contiguous()
        self._top1: list[torch.Tensor] = []
        self._ce: list[torch.Tensor] = []
        self._receipts: list[CellReceipt] = []
        self._next = 0
        self._finalized = False

    @property
    def next_ordinal(self) -> int:
        return self._next

    def add_cell(
        self, *, request: MeasurementRequest, values: RowCellStatistics,
        receipt: CellReceipt,
    ) -> None:
        if self._finalized or self._next >= 64:
            raise RuntimeError("role collector is closed")
        expected = REQUESTS[self._next]
        if request != expected or not isinstance(values, RowCellStatistics) or not isinstance(
            receipt, CellReceipt
        ):
            raise RuntimeError("cell type/order differs from frozen request plan")
        if receipt.authority_sha256 != self._authority.sha256 or receipt.request_sha256 != (
            expected.sha256
        ) or receipt.ordinal != expected.ordinal or receipt.cell != expected.cell or (
            receipt.statistics_sha256 != values.sha256
        ) or receipt.top1_correct_sha256 != values.top1_correct_sha256 or (
            receipt.ce_sum_sha256 != values.ce_sum_sha256
        ) or receipt.row_token_count_sha256 != self._authority.row_token_count_sha256 or (
            receipt.source_closure_sha256 != self._authority.source_closure_sha256
        ) or receipt.model_tree_before_sha256 != COMPONENT_TREE_SHA256 or (
            receipt.model_tree_after_sha256 != COMPONENT_TREE_SHA256
        ) or receipt.shared_program_before_sha256 != SHARED_PROGRAM_SHA256 or (
            receipt.shared_program_after_sha256 != SHARED_PROGRAM_SHA256
        ):
            raise RuntimeError("cell receipt differs from authority/statistics")
        top1, ce, tokens = values._values()
        if not torch.equal(tokens, self._row_token_count):
            raise RuntimeError("cell token support differs by value")
        self._top1.append(top1)
        self._ce.append(ce)
        self._receipts.append(receipt)
        self._next += 1

    def finalize(self) -> StagedRoleBundle:
        if self._finalized:
            raise RuntimeError("role collector finalization already attempted")
        self._finalized = True
        if self._next != 64 or len(self._top1) != 64 or len(self._ce) != 64 or len(
            self._receipts
        ) != 64:
            raise RuntimeError("role collector cannot expose an incomplete grid")
        row_top1 = torch.stack(self._top1, dim=1).contiguous()
        row_ce = torch.stack(self._ce, dim=1).contiguous()
        document_tokens = torch.zeros(self._authority.document_count, dtype=torch.long)
        document_top1 = torch.zeros(
            (self._authority.document_count, 64), dtype=torch.long,
        )
        document_ce = torch.zeros(
            (self._authority.document_count, 64), dtype=torch.float64,
        )
        for row, raw_document in enumerate(self._row_to_document.tolist()):
            document = int(raw_document)
            document_tokens[document] += self._row_token_count[row]
            document_top1[document] += row_top1[row]
            document_ce[document] += row_ce[row]
        if int(document_tokens.sum()) != self._authority.total_scored_token_count or (
            not torch.equal(document_top1.sum(0), row_top1.sum(0))
        ) or not torch.allclose(document_ce.sum(0), row_ce.sum(0), atol=1e-10, rtol=1e-14):
            raise RuntimeError("literal document aggregation changed totals")
        payloads = []
        for stage, cells in statistics.STAGE_CELLS.items():
            columns = [8 * i + j for i, j in cells]
            payloads.append(statistics.StageStatistics(
                role=self._authority.role, stage=stage,
                authority_sha256=self._authority.sha256,
                ordered_document_ids_sha256=self._authority.ordered_document_ids_sha256,
                document_token_count=document_tokens.contiguous(),
                top1_correct=document_top1[:, columns].contiguous(),
                ce_sum=document_ce[:, columns].contiguous(),
            ))
        discovery, validation, heldout = payloads
        receipt = RoleReceipt(
            role=self._authority.role,
            authority_sha256=self._authority.sha256,
            request_plan_sha256=REQUEST_PLAN_SHA256,
            source_commit=self._authority.source_commit,
            source_closure_sha256=self._authority.source_closure_sha256,
            row_file_sha256=self._authority.row_file_sha256,
            row_raw_sha256=self._authority.row_raw_sha256,
            ordered_document_ids_sha256=self._authority.ordered_document_ids_sha256,
            shared_program_sha256=self._authority.shared_program_sha256,
            cell_receipt_sha256s=tuple(value.sha256 for value in self._receipts),
            stage_payload_sha256s=(
                discovery.sha256, validation.sha256, heldout.sha256,
            ),
            row_count=self._authority.row_count,
            document_count=self._authority.document_count,
            total_scored_token_count=self._authority.total_scored_token_count,
        )
        self._top1.clear()
        self._ce.clear()
        self._receipts.clear()
        return StagedRoleBundle(
            discovery=discovery, validation=validation, heldout=heldout,
            receipt=receipt,
        )


def validate_request_plan() -> None:
    if len(REQUESTS) != 64 or tuple(request.ordinal for request in REQUESTS) != tuple(
        range(64)
    ) or tuple(request.cell for request in REQUESTS) != cross.ALL_CELLS or (
        REQUESTS[0].sites
    ) or tuple(request.stage for request in REQUESTS).count("discovery") != 48 or (
        tuple(request.stage for request in REQUESTS).count("validation") != 7
    ) or tuple(request.stage for request in REQUESTS).count("heldout") != 9:
        raise RuntimeError("measurement request plan changed")


validate_request_plan()
