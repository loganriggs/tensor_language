"""Pure source/currency boundary for the compilation-mask cut-rank assay.

This module performs no row, model, checkpoint, or artifact I/O.  It freezes the
64 requested mask measurements and accepts only row-level top-1/CE sufficient
statistics from a future observed adapter.  No outcome tensor is released until
all cells close in canonical order; the sole output is a bootstrap-ready,
per-document sufficient-statistic payload plus a tensor-free receipt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Sequence

import torch

import compilation_mask_cut_rank_v1 as cut


SCHEMA_VERSION = 1
PREREGISTRATION_SHA256 = (
    "85e0cb8a66f0a1db3ab146d3ae9f147b9b685d6fc55441c41c4d3ec0e46c54e0"
)
AUTHORITY_SCOPE = "discovery_only_compilation_mask_cut_rank_v1_measurement"
SOURCE_ROLE = "compilation_mask_cut_rank_v1_measurement"
METRIC_CONTRACT = (
    "top1_correct_count_on_common_scored_targets",
    "cross_entropy_sum_nats_on_identical_common_scored_targets",
)
ALWAYS_COMPILED_SITES: tuple[cut.Site, ...] = (("attn", 0), ("mlp", 0))
FLOAT64_AGGREGATION_EPSILON_MULTIPLIER = 16


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


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash exact CPU tensor shape, dtype, and contiguous bytes."""

    if not torch.is_tensor(value) or value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("measurement tensor hashing requires a contiguous CPU tensor")
    header = json.dumps({
        "shape": list(value.shape), "dtype": str(value.dtype),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(header)
    digest.update(value.detach().numpy().tobytes(order="C"))
    return digest.hexdigest()


def _split_for_cell(cell: cut.Cell) -> str:
    if cell in cut.ANCHOR_CELLS:
        return "anchor"
    if cell in cut.TRAIN_CELLS:
        return "train"
    if cell in cut.VALIDATION_CELLS:
        return "validation"
    if cell in cut.HELDOUT_CELLS:
        return "heldout"
    raise RuntimeError("measurement cell is outside the frozen split")


def _symbols(sites: Sequence[cut.Site]) -> tuple[str, ...]:
    site_set = set(sites)
    symbols = []
    for layer in range(1, 18):
        attention = ("attn", layer) in site_set
        mlp = ("mlp", layer) in site_set
        symbols.append(
            "both" if attention and mlp else "attn" if attention else "mlp" if mlp else "none"
        )
    return tuple(symbols)


@dataclass(frozen=True, slots=True)
class MeasurementRequest:
    """One exact physical mask request, with no outcome or runtime authority."""

    ordinal: int
    cell: cut.Cell
    split: str
    prefix_sites: tuple[cut.Site, ...]
    suffix_sites: tuple[cut.Site, ...]
    additional_sites: tuple[cut.Site, ...]
    layer_symbols: tuple[str, ...]
    always_compiled_sites: tuple[cut.Site, ...] = ALWAYS_COMPILED_SITES
    metric_contract: tuple[str, str] = METRIC_CONTRACT
    schema_version: int = SCHEMA_VERSION
    preregistration_sha256: str = PREREGISTRATION_SHA256

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or not 0 <= self.ordinal < 64 or self.cell != (
            self.ordinal // 8, self.ordinal % 8
        ) or self.schema_version != SCHEMA_VERSION or self.preregistration_sha256 != (
            PREREGISTRATION_SHA256
        ) or self.always_compiled_sites != ALWAYS_COMPILED_SITES or (
            self.metric_contract != METRIC_CONTRACT
        ):
            raise ValueError("measurement request header changed")
        i, j = self.cell
        expected_prefix = cut.PREFIX_MASKS[i]
        expected_suffix = cut.SUFFIX_MASKS[j]
        expected_sites = (*expected_prefix, *expected_suffix)
        if self.split != _split_for_cell(self.cell) or self.prefix_sites != (
            expected_prefix
        ) or self.suffix_sites != expected_suffix or self.additional_sites != (
            expected_sites
        ) or len(expected_sites) != len(set(expected_sites)) or self.layer_symbols != (
            _symbols(expected_sites)
        ):
            raise ValueError("measurement request mask/split changed")

    @property
    def sha256(self) -> str:
        return _logical_sha256(asdict(self))


def measurement_requests() -> tuple[MeasurementRequest, ...]:
    requests = tuple(
        MeasurementRequest(
            ordinal=ordinal, cell=cell, split=_split_for_cell(cell),
            prefix_sites=cut.PREFIX_MASKS[cell[0]],
            suffix_sites=cut.SUFFIX_MASKS[cell[1]],
            additional_sites=(
                *cut.PREFIX_MASKS[cell[0]], *cut.SUFFIX_MASKS[cell[1]],
            ),
            layer_symbols=_symbols((
                *cut.PREFIX_MASKS[cell[0]], *cut.SUFFIX_MASKS[cell[1]],
            )),
        )
        for ordinal, cell in enumerate(cut.ALL_CELLS)
    )
    if len(requests) != 64 or tuple(request.cell for request in requests) != (
        cut.ALL_CELLS
    ) or requests[0].additional_sites or requests[0].split != "anchor":
        raise RuntimeError("canonical 64-cell measurement request plan changed")
    return requests


REQUESTS = measurement_requests()
REQUEST_PLAN_SHA256 = _logical_sha256({
    "schema_version": SCHEMA_VERSION,
    "preregistration_sha256": PREREGISTRATION_SHA256,
    "request_sha256s": [request.sha256 for request in REQUESTS],
})


def program_bank_sha256(program_realization_sha256s: tuple[str, ...]) -> str:
    """Hash the ordered realization corresponding exactly to ``REQUESTS``."""

    if not isinstance(program_realization_sha256s, tuple) or len(
        program_realization_sha256s
    ) != 64 or any(not _sha256_text(value) for value in program_realization_sha256s):
        raise ValueError("program realization set is not the exact 64-cell tuple")
    return _logical_sha256({
        "request_plan_sha256": REQUEST_PLAN_SHA256,
        "program_realization_sha256s": list(program_realization_sha256s),
    })


@dataclass(frozen=True, slots=True)
class MeasurementWaveAuthority:
    """Outcome-blind binding of one complete same-wave measurement realization."""

    source_commit: str
    source_receipt_sha256: str
    row_tensor_sha256: str
    row_provenance_sha256: str
    ordered_row_identity_sha256: str
    ordered_row_to_document_sha256: str
    ordered_document_ids_sha256: str
    row_token_count_sha256: str
    common_support_sha256: str
    model_realization_sha256: str
    component_tree_sha256: str
    program_bank_sha256: str
    source_closure_sha256: str
    wave_nonce_sha256: str
    program_realization_sha256s: tuple[str, ...]
    row_count: int
    document_count: int
    total_scored_token_count: int
    batch_count: int
    request_plan_sha256: str = REQUEST_PLAN_SHA256
    authority_scope: str = AUTHORITY_SCOPE
    source_role: str = SOURCE_ROLE
    authorized_for_final_role: bool = False

    def __post_init__(self) -> None:
        if not _commit_text(self.source_commit) or any(not _sha256_text(value) for value in (
            self.source_receipt_sha256, self.row_tensor_sha256,
            self.row_provenance_sha256, self.ordered_row_identity_sha256,
            self.ordered_row_to_document_sha256, self.ordered_document_ids_sha256,
            self.row_token_count_sha256, self.common_support_sha256,
            self.model_realization_sha256, self.component_tree_sha256,
            self.program_bank_sha256,
            self.source_closure_sha256, self.wave_nonce_sha256,
        )):
            raise ValueError("measurement wave source/currency identity is malformed")
        if self.request_plan_sha256 != REQUEST_PLAN_SHA256 or self.authority_scope != (
            AUTHORITY_SCOPE
        ) or self.source_role != SOURCE_ROLE or self.authorized_for_final_role is not False:
            raise ValueError("measurement wave acquired an unauthorized scope")
        if not isinstance(self.program_realization_sha256s, tuple) or len(
            self.program_realization_sha256s
        ) != 64 or any(not _sha256_text(value) for value in self.program_realization_sha256s):
            raise ValueError("measurement wave program realization set is incomplete")
        if self.program_bank_sha256 != program_bank_sha256(
            self.program_realization_sha256s
        ):
            raise ValueError("measurement wave program bank does not bind its ordered cells")
        integers = (
            self.row_count, self.document_count, self.total_scored_token_count,
            self.batch_count,
        )
        if any(type(value) is not int or value <= 0 for value in integers) or (
            self.document_count > self.row_count
        ):
            raise ValueError("measurement wave row/document/count contract is malformed")

    @property
    def sha256(self) -> str:
        return _logical_sha256(asdict(self))


class RowCellSufficientStatistics:
    """One cell's row-level top-1/CE sums on the common scored support."""

    __slots__ = (
        "_ce_sum", "_expected_sha256", "_row_token_count", "_sealed",
        "_top1_correct",
    )

    def __init__(
        self, *, top1_correct: torch.Tensor, ce_sum: torch.Tensor,
        row_token_count: torch.Tensor,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        if not torch.is_tensor(top1_correct) or not torch.is_tensor(
            ce_sum
        ) or not torch.is_tensor(row_token_count) or top1_correct.ndim != 1 or (
            ce_sum.shape != top1_correct.shape or row_token_count.shape != top1_correct.shape
        ) or top1_correct.dtype != torch.long or row_token_count.dtype != torch.long or (
            ce_sum.dtype != torch.float64
        ) or any(value.device.type != "cpu" for value in (
            top1_correct, ce_sum, row_token_count,
        )) or any(not value.is_contiguous() for value in (
            top1_correct, ce_sum, row_token_count,
        )) or top1_correct.requires_grad or ce_sum.requires_grad or (
            row_token_count.requires_grad
        ):
            raise ValueError("row sufficient statistics have the wrong schema")
        if len(top1_correct) == 0 or bool((row_token_count <= 0).any()) or bool(
            (top1_correct < 0).any()
        ) or bool((top1_correct > row_token_count).any()) or bool((ce_sum < 0).any()) or (
            not bool(torch.isfinite(ce_sum).all())
        ):
            raise ValueError("row sufficient statistics violate count/finite bounds")
        self._top1_correct = top1_correct.detach().clone().contiguous()
        self._ce_sum = ce_sum.detach().clone().contiguous()
        self._row_token_count = row_token_count.detach().clone().contiguous()
        self._expected_sha256 = self._compute_sha256()
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("row sufficient statistics are sealed")
        object.__setattr__(self, name, value)

    def _compute_sha256(self) -> str:
        return _logical_sha256({
            "top1_correct_sha256": tensor_sha256(self._top1_correct),
            "ce_sum_sha256": tensor_sha256(self._ce_sum),
            "row_token_count_sha256": tensor_sha256(self._row_token_count),
        })

    def _require_pristine(self) -> None:
        if self._compute_sha256() != self._expected_sha256:
            raise RuntimeError("row sufficient statistics mutated after construction")

    @property
    def sha256(self) -> str:
        self._require_pristine()
        return self._expected_sha256

    @property
    def row_count(self) -> int:
        return len(self._top1_correct)

    @property
    def top1_correct_sha256(self) -> str:
        self._require_pristine()
        return tensor_sha256(self._top1_correct)

    @property
    def ce_sum_sha256(self) -> str:
        self._require_pristine()
        return tensor_sha256(self._ce_sum)

    @property
    def row_token_count_sha256(self) -> str:
        self._require_pristine()
        return tensor_sha256(self._row_token_count)

    def _clone_values(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._require_pristine()
        return (
            self._top1_correct.clone(), self._ce_sum.clone(),
            self._row_token_count.clone(),
        )


@dataclass(frozen=True, slots=True)
class CellMeasurementReceipt:
    """Tensor-free adapter proof for one complete cell measurement."""

    authority_sha256: str
    request_sha256: str
    ordinal: int
    cell: cut.Cell
    program_realization_sha256: str
    common_support_sha256: str
    ordered_row_identity_sha256: str
    top1_correct_sha256: str
    ce_sum_sha256: str
    row_token_count_sha256: str
    statistics_sha256: str
    call_ledger_sha256: str
    source_closure_sha256: str
    model_tree_before_sha256: str
    model_tree_after_sha256: str
    outer_forward_count: int
    batch_count: int

    def __post_init__(self) -> None:
        hashes = (
            self.authority_sha256, self.request_sha256,
            self.program_realization_sha256, self.common_support_sha256,
            self.ordered_row_identity_sha256, self.top1_correct_sha256,
            self.ce_sum_sha256, self.row_token_count_sha256,
            self.statistics_sha256, self.call_ledger_sha256,
            self.source_closure_sha256, self.model_tree_before_sha256,
            self.model_tree_after_sha256,
        )
        if any(not _sha256_text(value) for value in hashes) or type(
            self.ordinal
        ) is not int or not 0 <= self.ordinal < 64 or self.cell != (
            self.ordinal // 8, self.ordinal % 8
        ) or type(self.outer_forward_count) is not int or self.outer_forward_count <= 0 or (
            type(self.batch_count) is not int or self.batch_count <= 0
        ):
            raise ValueError("cell measurement receipt is malformed")

    @property
    def sha256(self) -> str:
        return _logical_sha256(asdict(self))


class PerDocumentSufficientStatistics:
    """Sealed bootstrap boundary; accessors return clones, never internal aliases."""

    __slots__ = (
        "_authority_sha256", "_ce_sum", "_document_row_count",
        "_document_token_count", "_expected_sha256", "_ordered_document_ids_sha256",
        "_sealed", "_top1_correct",
    )

    def __init__(
        self, *, authority_sha256: str, ordered_document_ids_sha256: str,
        document_row_count: torch.Tensor, document_token_count: torch.Tensor,
        top1_correct: torch.Tensor, ce_sum: torch.Tensor,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        if not _sha256_text(authority_sha256) or not _sha256_text(
            ordered_document_ids_sha256
        ) or not torch.is_tensor(document_row_count) or not torch.is_tensor(
            document_token_count
        ) or not torch.is_tensor(top1_correct) or not torch.is_tensor(ce_sum) or (
            document_row_count.ndim != 1 or document_token_count.shape != (
                document_row_count.shape
            ) or top1_correct.shape != (len(document_row_count), 64) or ce_sum.shape != (
                len(document_row_count), 64
            ) or document_row_count.dtype != torch.long or document_token_count.dtype != (
                torch.long
            ) or top1_correct.dtype != torch.long or ce_sum.dtype != torch.float64
        ):
            raise ValueError("per-document sufficient-statistic schema changed")
        values = (document_row_count, document_token_count, top1_correct, ce_sum)
        if any(value.device.type != "cpu" or not value.is_contiguous() for value in values) or (
            len(document_row_count) == 0 or bool((document_row_count <= 0).any()) or bool(
                (document_token_count <= 0).any()
            ) or bool((top1_correct < 0).any()) or bool((
                top1_correct > document_token_count[:, None]
            ).any()) or bool((ce_sum < 0).any()) or not bool(torch.isfinite(ce_sum).all())
        ):
            raise ValueError("per-document sufficient statistics violate bounds")
        self._authority_sha256 = authority_sha256
        self._ordered_document_ids_sha256 = ordered_document_ids_sha256
        self._document_row_count = document_row_count.detach().clone().contiguous()
        self._document_token_count = document_token_count.detach().clone().contiguous()
        self._top1_correct = top1_correct.detach().clone().contiguous()
        self._ce_sum = ce_sum.detach().clone().contiguous()
        self._expected_sha256 = self._compute_sha256()
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("per-document sufficient statistics are sealed")
        object.__setattr__(self, name, value)

    def _compute_sha256(self) -> str:
        return _logical_sha256({
            "authority_sha256": self._authority_sha256,
            "ordered_document_ids_sha256": self._ordered_document_ids_sha256,
            "document_row_count_sha256": tensor_sha256(self._document_row_count),
            "document_token_count_sha256": tensor_sha256(self._document_token_count),
            "top1_correct_sha256": tensor_sha256(self._top1_correct),
            "ce_sum_sha256": tensor_sha256(self._ce_sum),
        })

    def _require_pristine(self) -> None:
        if self._compute_sha256() != self._expected_sha256:
            raise RuntimeError("per-document sufficient statistics mutated")

    @property
    def sha256(self) -> str:
        self._require_pristine()
        return self._expected_sha256

    @property
    def document_count(self) -> int:
        return len(self._document_row_count)

    @property
    def row_count(self) -> int:
        self._require_pristine()
        return int(self._document_row_count.sum())

    @property
    def total_scored_token_count(self) -> int:
        self._require_pristine()
        return int(self._document_token_count.sum())

    @property
    def authority_sha256(self) -> str:
        self._require_pristine()
        return self._authority_sha256

    @property
    def ordered_document_ids_sha256(self) -> str:
        self._require_pristine()
        return self._ordered_document_ids_sha256

    @property
    def document_row_count(self) -> torch.Tensor:
        self._require_pristine()
        return self._document_row_count.clone()

    @property
    def document_token_count(self) -> torch.Tensor:
        self._require_pristine()
        return self._document_token_count.clone()

    @property
    def top1_correct(self) -> torch.Tensor:
        self._require_pristine()
        return self._top1_correct.clone()

    @property
    def ce_sum(self) -> torch.Tensor:
        self._require_pristine()
        return self._ce_sum.clone()


@dataclass(frozen=True, slots=True)
class MeasurementReceipt:
    schema_version: int
    preregistration_sha256: str
    request_plan_sha256: str
    authority_sha256: str
    source_commit: str
    source_receipt_sha256: str
    row_tensor_sha256: str
    row_provenance_sha256: str
    ordered_row_identity_sha256: str
    ordered_row_to_document_sha256: str
    ordered_document_ids_sha256: str
    row_token_count_sha256: str
    common_support_sha256: str
    model_realization_sha256: str
    component_tree_sha256: str
    program_bank_sha256: str
    source_closure_sha256: str
    wave_nonce_sha256: str
    b0_request_sha256: str
    b0_cell_receipt_sha256: str
    cell_receipt_sha256s: tuple[str, ...]
    top1_correct_row_sha256s: tuple[str, ...]
    ce_sum_row_sha256s: tuple[str, ...]
    statistics_sha256s: tuple[str, ...]
    per_document_payload_sha256: str
    row_count: int
    document_count: int
    total_scored_token_count: int
    batch_count: int
    cell_count: int
    authority_scope: str = AUTHORITY_SCOPE
    source_role: str = SOURCE_ROLE
    authorized_for_final_role: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.preregistration_sha256 != (
            PREREGISTRATION_SHA256
        ) or self.request_plan_sha256 != REQUEST_PLAN_SHA256 or not _commit_text(
            self.source_commit
        ) or self.authority_scope != AUTHORITY_SCOPE or self.source_role != (
            SOURCE_ROLE
        ) or self.authorized_for_final_role is not False:
            raise ValueError("measurement receipt header/scope changed")
        hashes = tuple(
            getattr(self, name) for name in self.__dataclass_fields__
            if name.endswith("sha256") and name != "request_plan_sha256"
        )
        hash_vectors = (
            self.cell_receipt_sha256s, self.top1_correct_row_sha256s,
            self.ce_sum_row_sha256s, self.statistics_sha256s,
        )
        if any(not _sha256_text(value) for value in hashes) or any(
            not isinstance(vector, tuple) or len(vector) != 64 or any(
                not _sha256_text(value) for value in vector
            ) for vector in hash_vectors
        ) or self.b0_request_sha256 != REQUESTS[0].sha256 or self.b0_cell_receipt_sha256 != (
            self.cell_receipt_sha256s[0]
        ):
            raise ValueError("measurement receipt hash/B0 binding changed")
        if self.cell_count != 64 or any(type(value) is not int or value <= 0 for value in (
            self.row_count, self.document_count, self.total_scored_token_count,
            self.batch_count,
        )) or self.document_count > self.row_count or self.total_scored_token_count < (
            self.row_count
        ):
            raise ValueError("measurement receipt count contract changed")

    @property
    def sha256(self) -> str:
        return _logical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class FinalizedMeasurementBundle:
    payload: PerDocumentSufficientStatistics
    receipt: MeasurementReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.payload, PerDocumentSufficientStatistics) or not isinstance(
            self.receipt, MeasurementReceipt
        ) or self.payload.sha256 != self.receipt.per_document_payload_sha256 or (
            self.payload.authority_sha256 != self.receipt.authority_sha256
        ) or self.payload.ordered_document_ids_sha256 != (
            self.receipt.ordered_document_ids_sha256
        ) or self.payload.document_count != self.receipt.document_count or (
            self.payload.row_count != self.receipt.row_count
        ) or self.payload.total_scored_token_count != (
            self.receipt.total_scored_token_count
        ):
            raise ValueError("measurement bundle payload/receipt identity differs")


def _first_seen_document_order(row_to_document: torch.Tensor, document_count: int) -> bool:
    seen: set[int] = set()
    next_document = 0
    for raw in row_to_document.tolist():
        value = int(raw)
        if value not in seen:
            if value != next_document:
                return False
            seen.add(value)
            next_document += 1
    return next_document == document_count


class MeasurementCollector:
    """Sequential 64-cell transaction which reveals nothing before finalization."""

    __slots__ = (
        "_authority", "_ce_columns", "_finalized", "_next_ordinal",
        "_receipts", "_row_to_document", "_row_token_count", "_top1_columns",
    )

    def __init__(
        self, *, authority: MeasurementWaveAuthority,
        row_to_document: torch.Tensor, row_token_count: torch.Tensor,
    ) -> None:
        if not isinstance(authority, MeasurementWaveAuthority) or not torch.is_tensor(
            row_to_document
        ) or not torch.is_tensor(row_token_count) or row_to_document.dtype != (
            torch.long
        ) or row_token_count.dtype != torch.long or tuple(row_to_document.shape) != (
            authority.row_count,
        ) or row_token_count.shape != row_to_document.shape or any(
            value.device.type != "cpu" or not value.is_contiguous()
            for value in (row_to_document, row_token_count)
        ) or bool((row_to_document < 0).any()) or bool((
            row_to_document >= authority.document_count
        ).any()) or bool((row_token_count <= 0).any()):
            raise ValueError("measurement collector row/document mapping is malformed")
        if tensor_sha256(row_to_document) != authority.ordered_row_to_document_sha256 or (
            tensor_sha256(row_token_count) != authority.row_token_count_sha256
        ) or int(row_token_count.sum()) != authority.total_scored_token_count or not (
            _first_seen_document_order(row_to_document, authority.document_count)
        ):
            raise RuntimeError("measurement collector differs from sealed row/support authority")
        self._authority = authority
        self._row_to_document = row_to_document.detach().clone().contiguous()
        self._row_token_count = row_token_count.detach().clone().contiguous()
        self._top1_columns: list[torch.Tensor] = []
        self._ce_columns: list[torch.Tensor] = []
        self._receipts: list[CellMeasurementReceipt] = []
        self._next_ordinal = 0
        self._finalized = False

    @property
    def next_ordinal(self) -> int:
        return self._next_ordinal

    def add_cell(
        self, *, request: MeasurementRequest,
        statistics: RowCellSufficientStatistics,
        receipt: CellMeasurementReceipt,
    ) -> None:
        if self._finalized:
            raise RuntimeError("measurement collector was already finalized")
        if self._next_ordinal >= 64:
            raise RuntimeError("measurement collector already contains every cell")
        expected = REQUESTS[self._next_ordinal]
        if not isinstance(request, MeasurementRequest) or request != expected or not isinstance(
            statistics, RowCellSufficientStatistics
        ) or not isinstance(receipt, CellMeasurementReceipt):
            raise RuntimeError("measurement cell type/order differs from the frozen request")
        if statistics.row_count != self._authority.row_count or (
            statistics.row_token_count_sha256 != self._authority.row_token_count_sha256
        ):
            raise RuntimeError("measurement cell row support differs from the common support")
        expected_program = self._authority.program_realization_sha256s[expected.ordinal]
        if receipt.authority_sha256 != self._authority.sha256 or receipt.request_sha256 != (
            expected.sha256
        ) or receipt.ordinal != expected.ordinal or receipt.cell != expected.cell or (
            receipt.program_realization_sha256 != expected_program
        ) or receipt.common_support_sha256 != self._authority.common_support_sha256 or (
            receipt.ordered_row_identity_sha256 != self._authority.ordered_row_identity_sha256
        ) or receipt.top1_correct_sha256 != statistics.top1_correct_sha256 or (
            receipt.ce_sum_sha256 != statistics.ce_sum_sha256
        ) or receipt.row_token_count_sha256 != statistics.row_token_count_sha256 or (
            receipt.statistics_sha256 != statistics.sha256
        ) or receipt.source_closure_sha256 != self._authority.source_closure_sha256 or (
            receipt.model_tree_before_sha256 != self._authority.component_tree_sha256
        ) or (
            receipt.model_tree_after_sha256 != self._authority.component_tree_sha256
        ) or receipt.outer_forward_count != self._authority.batch_count or (
            receipt.batch_count != self._authority.batch_count
        ):
            raise RuntimeError("measurement cell receipt differs from authority/statistics")
        top1, ce, tokens = statistics._clone_values()
        if not torch.equal(tokens, self._row_token_count):
            raise RuntimeError("measurement cell token denominators differ by value")
        self._top1_columns.append(top1)
        self._ce_columns.append(ce)
        self._receipts.append(receipt)
        self._next_ordinal += 1

    def finalize(self) -> FinalizedMeasurementBundle:
        if self._finalized:
            raise RuntimeError("measurement collector finalization was already attempted")
        self._finalized = True
        if self._next_ordinal != 64 or len(self._top1_columns) != 64 or len(
            self._ce_columns
        ) != 64 or len(self._receipts) != 64:
            raise RuntimeError("measurement collector cannot finalize an incomplete grid")
        row_top1 = torch.stack(self._top1_columns, dim=1).contiguous()
        row_ce = torch.stack(self._ce_columns, dim=1).contiguous()
        document_rows = torch.bincount(
            self._row_to_document, minlength=self._authority.document_count,
        ).to(torch.long).contiguous()
        document_tokens = torch.zeros(
            self._authority.document_count, dtype=torch.long,
        )
        document_top1 = torch.zeros(
            (self._authority.document_count, 64), dtype=torch.long,
        )
        document_ce = torch.zeros(
            (self._authority.document_count, 64), dtype=torch.float64,
        )
        # Literal ordered accumulation makes the payload independent of parallel
        # reduction scheduling and preserves the exact row-to-document provenance.
        for row, raw_document in enumerate(self._row_to_document.tolist()):
            document = int(raw_document)
            document_tokens[document] += self._row_token_count[row]
            document_top1[document] += row_top1[row]
            document_ce[document] += row_ce[row]
        ce_row_total = row_ce.sum(dim=0)
        ce_document_total = document_ce.sum(dim=0)
        ce_absolute_mass = row_ce.abs().sum(dim=0)
        ce_error_bound = (
            FLOAT64_AGGREGATION_EPSILON_MULTIPLIER
            * (self._authority.row_count + self._authority.document_count)
            * torch.finfo(torch.float64).eps
            * torch.maximum(ce_absolute_mass, torch.ones_like(ce_absolute_mass))
        )
        if int(document_rows.sum()) != self._authority.row_count or int(
            document_tokens.sum()
        ) != self._authority.total_scored_token_count or not torch.equal(
            document_top1.sum(dim=0), row_top1.sum(dim=0)
        ) or bool((torch.abs(ce_document_total - ce_row_total) > ce_error_bound).any()):
            raise RuntimeError("row-to-document sufficient-statistic aggregation changed")
        payload = PerDocumentSufficientStatistics(
            authority_sha256=self._authority.sha256,
            ordered_document_ids_sha256=self._authority.ordered_document_ids_sha256,
            document_row_count=document_rows, document_token_count=document_tokens,
            top1_correct=document_top1, ce_sum=document_ce,
        )
        receipt = MeasurementReceipt(
            schema_version=SCHEMA_VERSION,
            preregistration_sha256=PREREGISTRATION_SHA256,
            request_plan_sha256=REQUEST_PLAN_SHA256,
            authority_sha256=self._authority.sha256,
            source_commit=self._authority.source_commit,
            source_receipt_sha256=self._authority.source_receipt_sha256,
            row_tensor_sha256=self._authority.row_tensor_sha256,
            row_provenance_sha256=self._authority.row_provenance_sha256,
            ordered_row_identity_sha256=self._authority.ordered_row_identity_sha256,
            ordered_row_to_document_sha256=(
                self._authority.ordered_row_to_document_sha256
            ),
            ordered_document_ids_sha256=self._authority.ordered_document_ids_sha256,
            row_token_count_sha256=self._authority.row_token_count_sha256,
            common_support_sha256=self._authority.common_support_sha256,
            model_realization_sha256=self._authority.model_realization_sha256,
            component_tree_sha256=self._authority.component_tree_sha256,
            program_bank_sha256=self._authority.program_bank_sha256,
            source_closure_sha256=self._authority.source_closure_sha256,
            wave_nonce_sha256=self._authority.wave_nonce_sha256,
            b0_request_sha256=REQUESTS[0].sha256,
            b0_cell_receipt_sha256=self._receipts[0].sha256,
            cell_receipt_sha256s=tuple(value.sha256 for value in self._receipts),
            top1_correct_row_sha256s=tuple(
                value.top1_correct_sha256 for value in self._receipts
            ),
            ce_sum_row_sha256s=tuple(value.ce_sum_sha256 for value in self._receipts),
            statistics_sha256s=tuple(value.statistics_sha256 for value in self._receipts),
            per_document_payload_sha256=payload.sha256,
            row_count=self._authority.row_count,
            document_count=self._authority.document_count,
            total_scored_token_count=self._authority.total_scored_token_count,
            batch_count=self._authority.batch_count,
            cell_count=64,
        )
        self._top1_columns.clear()
        self._ce_columns.clear()
        self._receipts.clear()
        return FinalizedMeasurementBundle(payload=payload, receipt=receipt)


def validate_request_plan() -> None:
    if measurement_requests() != REQUESTS or REQUEST_PLAN_SHA256 != _logical_sha256({
        "schema_version": SCHEMA_VERSION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "request_sha256s": [request.sha256 for request in REQUESTS],
    }):
        raise RuntimeError("measurement request plan failed deterministic replay")
    if tuple(request.ordinal for request in REQUESTS) != tuple(range(64)) or tuple(
        request.split for request in REQUESTS
    ).count("heldout") != len(cut.HELDOUT_CELLS):
        raise RuntimeError("measurement request ordering/split census changed")


validate_request_plan()
