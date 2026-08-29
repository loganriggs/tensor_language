"""Pure ordered measurement boundary for the eight new broad-MLP suffix cells."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

import broad_mlp_suffix_dealias_v1 as assay
import early_mlp_context_cross_v1_measurements as parent


SCHEMA_VERSION = 1
PREREGISTRATION_SHA256 = "065f1458a93685432596abf43c6628935e93c4e84d8210cb19d7f346d1e9b24f"
PROGRAM_FAMILY = parent.PROGRAM_FAMILY
SHARED_PROGRAM_SHA256 = parent.SHARED_PROGRAM_SHA256
MODEL_REALIZATION_SHA256 = parent.MODEL_REALIZATION_SHA256
COMPONENT_TREE_SHA256 = parent.COMPONENT_TREE_SHA256
ROLE_DOCUMENT_COUNTS = parent.ROLE_DOCUMENT_COUNTS
ROW_COUNT = parent.ROW_COUNT
SCORED_TOKENS_PER_ROW = parent.SCORED_TOKENS_PER_ROW
BATCH_COUNT = parent.BATCH_COUNT
RowCellStatistics = parent.RowCellStatistics


def _logical_sha256(value: Any) -> str:
    return assay.logical_sha256(value)


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True, slots=True)
class MeasurementRequest:
    ordinal: int
    prefix_index: int
    sites: tuple[tuple[str, int], ...]
    schema_version: int = SCHEMA_VERSION
    preregistration_sha256: str = PREREGISTRATION_SHA256
    program_family: str = PROGRAM_FAMILY

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or not 0 <= self.ordinal < assay.CELL_COUNT or (
            self.prefix_index != self.ordinal
        ) or self.sites != assay.REQUEST_MASKS[self.ordinal] or self.schema_version != (
            SCHEMA_VERSION
        ) or self.preregistration_sha256 != PREREGISTRATION_SHA256 or (
            self.program_family != PROGRAM_FAMILY
        ):
            raise ValueError("broad-MLP measurement request changed")

    @property
    def sha256(self) -> str:
        return _logical_sha256(asdict(self))


REQUESTS = tuple(
    MeasurementRequest(ordinal=i, prefix_index=i, sites=assay.REQUEST_MASKS[i])
    for i in range(assay.CELL_COUNT)
)
REQUEST_PLAN_SHA256 = _logical_sha256({
    "schema_version": SCHEMA_VERSION,
    "preregistration_sha256": PREREGISTRATION_SHA256,
    "request_sha256s": [request.sha256 for request in REQUESTS],
})


@dataclass(frozen=True, slots=True)
class RoleAuthority:
    role: str
    source_commit: str
    source_closure_sha256: str
    row_file_sha256: str
    row_raw_sha256: str
    ordered_row_identity_sha256: str
    ordered_row_to_document_sha256: str
    ordered_document_ids_sha256: str
    row_token_count_sha256: str
    common_support_sha256: str
    parent_role_authority_sha256: str
    parent_measurement_receipt_sha256: str
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
        hashes = [
            getattr(self, name) for name in self.__dataclass_fields__
            if name.endswith("sha256") and name != "request_plan_sha256"
        ]
        if self.role not in assay.ROLE_NAMES or not _commit(self.source_commit) or any(
            not _sha(value) for value in hashes
        ) or self.request_plan_sha256 != REQUEST_PLAN_SHA256 or self.model_realization_sha256 != (
            MODEL_REALIZATION_SHA256
        ) or self.component_tree_sha256 != COMPONENT_TREE_SHA256 or self.shared_program_sha256 != (
            SHARED_PROGRAM_SHA256
        ) or self.authorized_for_final_role is not False or self.row_count != ROW_COUNT or (
            self.document_count != ROLE_DOCUMENT_COUNTS[self.role]
        ) or self.total_scored_token_count != ROW_COUNT * SCORED_TOKENS_PER_ROW or (
            self.batch_count != BATCH_COUNT
        ):
            raise ValueError("broad-MLP role authority changed")

    @property
    def sha256(self) -> str:
        return _logical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class CellReceipt:
    authority_sha256: str
    request_sha256: str
    ordinal: int
    prefix_index: int
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
        hashes = [getattr(self, name) for name in self.__dataclass_fields__ if name.endswith("sha256")]
        if any(not _sha(value) for value in hashes) or type(self.ordinal) is not int or not (
            0 <= self.ordinal < assay.CELL_COUNT
        ) or self.prefix_index != self.ordinal or self.outer_forward_count != BATCH_COUNT or (
            self.batch_count != BATCH_COUNT
        ):
            raise ValueError("broad-MLP cell receipt changed")

    @property
    def sha256(self) -> str:
        return _logical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class RoleStatistics:
    role: str
    authority_sha256: str
    ordered_document_ids_sha256: str
    document_token_count: torch.Tensor
    top1_correct: torch.Tensor
    ce_sum: torch.Tensor

    def __post_init__(self) -> None:
        expected_docs = ROLE_DOCUMENT_COUNTS.get(self.role)
        if not _sha(self.authority_sha256) or not _sha(self.ordered_document_ids_sha256) or (
            expected_docs is None
        ) or self.document_token_count.shape != (expected_docs,) or self.top1_correct.shape != (
            expected_docs, assay.CELL_COUNT
        ) or self.ce_sum.shape != self.top1_correct.shape or self.document_token_count.dtype != (
            torch.long
        ) or self.top1_correct.dtype != torch.long or self.ce_sum.dtype != torch.float64 or any(
            tensor.device.type != "cpu" or not tensor.is_contiguous() or tensor.requires_grad
            for tensor in (self.document_token_count, self.top1_correct, self.ce_sum)
        ) or int(self.document_token_count.sum()) != ROW_COUNT * SCORED_TOKENS_PER_ROW or (
            bool((self.document_token_count <= 0).any())
        ) or bool((self.top1_correct < 0).any()) or bool((self.ce_sum < 0).any()) or not bool(
            torch.isfinite(self.ce_sum).all()
        ):
            raise ValueError("broad-MLP role statistics changed")

    @property
    def sha256(self) -> str:
        return _logical_sha256({
            "role": self.role,
            "authority_sha256": self.authority_sha256,
            "ordered_document_ids_sha256": self.ordered_document_ids_sha256,
            "document_token_count_sha256": parent.statistics.tensor_sha256(self.document_token_count),
            "top1_correct_sha256": parent.statistics.tensor_sha256(self.top1_correct),
            "ce_sum_sha256": parent.statistics.tensor_sha256(self.ce_sum),
        })


@dataclass(frozen=True, slots=True)
class RoleReceipt:
    role: str
    authority_sha256: str
    source_commit: str
    source_closure_sha256: str
    row_file_sha256: str
    row_raw_sha256: str
    ordered_document_ids_sha256: str
    shared_program_sha256: str
    cell_receipt_sha256s: tuple[str, ...]
    statistics_sha256: str
    row_count: int
    document_count: int
    total_scored_token_count: int
    request_plan_sha256: str = REQUEST_PLAN_SHA256
    cell_count: int = assay.CELL_COUNT

    def __post_init__(self) -> None:
        hashes = [
            getattr(self, name) for name in self.__dataclass_fields__
            if name.endswith("sha256") and name != "request_plan_sha256"
        ]
        if self.role not in assay.ROLE_NAMES or not _commit(self.source_commit) or any(
            not _sha(value) for value in hashes
        ) or self.request_plan_sha256 != REQUEST_PLAN_SHA256 or len(self.cell_receipt_sha256s) != (
            assay.CELL_COUNT
        ) or any(not _sha(value) for value in self.cell_receipt_sha256s) or self.row_count != (
            ROW_COUNT
        ) or self.document_count != ROLE_DOCUMENT_COUNTS[self.role] or self.total_scored_token_count != (
            ROW_COUNT * SCORED_TOKENS_PER_ROW
        ) or self.cell_count != assay.CELL_COUNT:
            raise ValueError("broad-MLP role receipt changed")

    @property
    def sha256(self) -> str:
        return _logical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class RoleBundle:
    statistics: RoleStatistics
    receipt: RoleReceipt

    def __post_init__(self) -> None:
        if self.statistics.role != self.receipt.role or self.statistics.sha256 != (
            self.receipt.statistics_sha256
        ) or self.statistics.authority_sha256 != self.receipt.authority_sha256:
            raise ValueError("broad-MLP role bundle changed")


class RoleCollector:
    def __init__(
        self, *, authority: RoleAuthority, row_to_document: torch.Tensor,
        row_token_count: torch.Tensor,
    ) -> None:
        if row_to_document.shape != (ROW_COUNT,) or row_to_document.dtype != torch.long or (
            row_token_count.shape != (ROW_COUNT,)
        ) or row_token_count.dtype != torch.long or parent.statistics.tensor_sha256(
            row_to_document
        ) != authority.ordered_row_to_document_sha256 or parent.statistics.tensor_sha256(
            row_token_count
        ) != authority.row_token_count_sha256 or bool((row_to_document < 0).any()) or bool(
            (row_to_document >= authority.document_count).any()
        ):
            raise ValueError("collector support differs from authority")
        self.authority = authority
        self.row_to_document = row_to_document.clone().contiguous()
        self.row_token_count = row_token_count.clone().contiguous()
        self.values: list[RowCellStatistics] = []
        self.receipts: list[CellReceipt] = []
        self.finalized = False

    @property
    def next_ordinal(self) -> int:
        return len(self.values)

    def add(self, statistics: RowCellStatistics, receipt: CellReceipt) -> None:
        ordinal = self.next_ordinal
        if self.finalized or ordinal >= assay.CELL_COUNT or receipt.ordinal != ordinal or (
            receipt.authority_sha256 != self.authority.sha256
        ) or receipt.request_sha256 != REQUESTS[ordinal].sha256 or receipt.statistics_sha256 != (
            statistics.sha256
        ):
            raise RuntimeError("cell is missing, duplicated, reordered, or rebound")
        _top1, _ce, tokens = statistics._values()
        if not torch.equal(tokens, self.row_token_count):
            raise RuntimeError("cell token support changed")
        self.values.append(statistics)
        self.receipts.append(receipt)

    def finalize(self) -> RoleBundle:
        if self.finalized or len(self.values) != assay.CELL_COUNT:
            raise RuntimeError("partial or repeated role finalization")
        self.finalized = True
        document_count = self.authority.document_count
        document_tokens = torch.zeros(document_count, dtype=torch.long)
        document_tokens.index_add_(0, self.row_to_document, self.row_token_count)
        top1 = torch.empty((document_count, assay.CELL_COUNT), dtype=torch.long)
        ce = torch.empty((document_count, assay.CELL_COUNT), dtype=torch.float64)
        for ordinal, value in enumerate(self.values):
            row_top1, row_ce, _tokens = value._values()
            top1[:, ordinal] = torch.zeros(document_count, dtype=torch.long).index_add_(
                0, self.row_to_document, row_top1,
            )
            ce[:, ordinal] = torch.zeros(document_count, dtype=torch.float64).index_add_(
                0, self.row_to_document, row_ce,
            )
        statistics = RoleStatistics(
            role=self.authority.role, authority_sha256=self.authority.sha256,
            ordered_document_ids_sha256=self.authority.ordered_document_ids_sha256,
            document_token_count=document_tokens.contiguous(),
            top1_correct=top1.contiguous(), ce_sum=ce.contiguous(),
        )
        receipt = RoleReceipt(
            role=self.authority.role, authority_sha256=self.authority.sha256,
            source_commit=self.authority.source_commit,
            source_closure_sha256=self.authority.source_closure_sha256,
            row_file_sha256=self.authority.row_file_sha256,
            row_raw_sha256=self.authority.row_raw_sha256,
            ordered_document_ids_sha256=self.authority.ordered_document_ids_sha256,
            shared_program_sha256=self.authority.shared_program_sha256,
            cell_receipt_sha256s=tuple(value.sha256 for value in self.receipts),
            statistics_sha256=statistics.sha256, row_count=ROW_COUNT,
            document_count=document_count,
            total_scored_token_count=ROW_COUNT * SCORED_TOKENS_PER_ROW,
        )
        return RoleBundle(statistics=statistics, receipt=receipt)
