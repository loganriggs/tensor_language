"""Pure prospective row-budget contract for ordered-successor SELECT v3.

This module has no row publication, authority, checkpoint, model, GPU, or outcome
capability.  It preserves the v2 mask and deterministic selection rule while exposing
the larger v3 budget as a synthetic-testable contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

import ordered_successor_tensor_select_registry_v2 as protocol
import prepare_ordered_successor_tensor_select_v2_rows as v2
from ordered_successor_masks_v1 import OrderedLexicon, SuccessorMasks


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = ROOT / "basis_aligned/bilinear_quotient"
AMENDMENT = HERE / "ORDERED_SUCCESSOR_TENSOR_SELECT_V3_ROWS_AMENDMENT.md"
V2_AUDIT = HERE / "ordered_successor_tensor_select_v2_rows_independent_audit.json"
V2_FAILURE = BQ / "ordered_successor_tensor_select_v2_rows_failure.json"

STATUS = "PROSPECTIVE_NO_GO_REQUIRES_V3_FREEZER_AND_INDEPENDENT_AUDIT"
V2_SOURCE_COMMIT = "320dc5537d3fe99b14c29d54f74073714edb21af"
V2_FAILURE_COMMIT = "5f40025895ba0887ad00cc7b3200fc51c8ab823b"
V2_AUDIT_SHA256 = "d5747b99a2ab224fad569c460b5bcc59a695e93ad4595bd300f99e7930d14b1d"
V2_FAILURE_SHA256 = "ba852204a585592c699b8df3554e1dcbe951f964d6da13bbc6d39dbf507278d9"
V2_SELECT_DOCUMENTS = 192
V3_SELECT_DOCUMENTS = 384
REGISTERED_GREEDY_MINIMUM = 335
V3_MARGIN_DOCUMENTS = V3_SELECT_DOCUMENTS - REGISTERED_GREEDY_MINIMUM
V3_SCORED_POSITIONS = V3_SELECT_DOCUMENTS * (256 - 64)

EXPECTED_V2_FAILURE = {
    "cache_exists": False,
    "error": "powered successor support requires more than 192 documents",
    "error_type": "RuntimeError",
    "outcome_access": False,
    "schema": "ordered_successor_tensor_select_v2_rows_failure",
    "status": "terminal_failure_no_receipt",
}


def _stable_bytes(path: Path) -> tuple[bytes, str]:
    before = v2.file_sha256(path)
    raw = path.read_bytes()
    after = v2.file_sha256(path)
    digest = hashlib.sha256(raw).hexdigest()
    if before != digest or after != before:
        raise RuntimeError(f"successor lineage changed during stable read: {path}")
    return raw, digest


def validate_v2_lineage(
    *, audit_path: Path = V2_AUDIT, failure_path: Path = V2_FAILURE,
) -> dict[str, str]:
    """Bind the exact independent GO and spent terminal failure as immutable bytes."""

    audit_raw, audit_sha = _stable_bytes(audit_path)
    failure_raw, failure_sha = _stable_bytes(failure_path)
    if audit_sha != V2_AUDIT_SHA256 or failure_sha != V2_FAILURE_SHA256:
        raise RuntimeError("successor v2 lineage byte hash changed")
    audit = json.loads(audit_raw)
    failure = json.loads(failure_raw)
    if (
        not isinstance(audit, dict)
        or audit.get("schema") != "ordered_successor_tensor_select_v2_rows_independent_audit"
        or audit.get("status") != "GO"
        or audit.get("outcome_access") is not False
        or audit.get("audited_source_commit") != V2_SOURCE_COMMIT
        or failure != EXPECTED_V2_FAILURE
    ):
        raise RuntimeError("successor v2 lineage semantics changed")
    return {
        "v2_source_commit": V2_SOURCE_COMMIT,
        "v2_failure_commit": V2_FAILURE_COMMIT,
        "v2_independent_audit_sha256": audit_sha,
        "v2_terminal_failure_sha256": failure_sha,
    }


@dataclass(frozen=True)
class BudgetAllocation:
    selected_rows: torch.Tensor
    selected_records: tuple[Mapping[str, Any], ...]
    support_first_count: int
    support_first_last_candidate: int
    census: Mapping[str, Mapping[str, int | bool]]
    pair_occupancy: Mapping[str, Mapping[str, int]]


def _validate_contract() -> None:
    protocol.validate_registry()
    if (
        v2.N_SELECT != V2_SELECT_DOCUMENTS
        or v2.START_DOCUMENT_INDEX != 200_000
        or v2.CANDIDATE_DOCUMENTS != 4_096
        or v2.ROW_LENGTH != 257
        or v2.PREFIX_LENGTH != 32
        or v2.MIN_POSITIONS != 200
        or v2.MIN_DOCUMENTS != 30
        or V3_SELECT_DOCUMENTS != 2 * V2_SELECT_DOCUMENTS
        or V3_MARGIN_DOCUMENTS != 49
        or V3_SCORED_POSITIONS != 73_728
        or tuple(protocol.ARM_NAMES) != tuple(v2.V2_ARM_NAMES)
        or len(protocol.ARM_NAMES) != 15
    ):
        raise RuntimeError("successor v3 prospective budget contract changed")


def allocate_v3_budget(
    candidate_rows: torch.Tensor,
    candidate_records: list[dict[str, Any]],
    lexicon: OrderedLexicon,
    *,
    budget: int = V3_SELECT_DOCUMENTS,
    mask_builder: Callable[[torch.Tensor, OrderedLexicon], SuccessorMasks] = v2.build_masks,
    require_registered_stopping_count: bool = False,
) -> BudgetAllocation:
    """Replay v2 support-first selection and fill to a fixed prospective budget."""

    _validate_contract()
    if type(budget) is not int or budget <= 0 or budget > len(candidate_rows):
        raise ValueError("successor v3 document budget is malformed")
    if (
        not torch.is_tensor(candidate_rows)
        or candidate_rows.device.type != "cpu"
        or candidate_rows.dtype != torch.long
        or candidate_rows.ndim != 2
        or candidate_rows.shape[1] != v2.ROW_LENGTH
        or len(candidate_rows) != len(candidate_records)
    ):
        raise ValueError("successor v3 candidate rows are malformed")
    masks = mask_builder(candidate_rows, lexicon)
    masks.validate_partition()
    named = masks.named_cells()
    positions = {name: 0 for name in protocol.POWERED_CELLS}
    documents = {name: 0 for name in protocol.POWERED_CELLS}
    chosen: list[int] = []
    for index in range(len(candidate_rows)):
        amounts = {name: int(named[name][index].sum()) for name in protocol.POWERED_CELLS}
        contributes = any(
            amounts[name]
            and (positions[name] < v2.MIN_POSITIONS or documents[name] < v2.MIN_DOCUMENTS)
            for name in protocol.POWERED_CELLS
        )
        if not contributes:
            continue
        chosen.append(index)
        for name, amount in amounts.items():
            positions[name] += amount
            documents[name] += int(amount > 0)
        if all(
            positions[name] >= v2.MIN_POSITIONS and documents[name] >= v2.MIN_DOCUMENTS
            for name in protocol.POWERED_CELLS
        ):
            break
    else:
        raise RuntimeError("candidate scan cannot power every successor cell")
    support_first_count = len(chosen)
    if require_registered_stopping_count and support_first_count != REGISTERED_GREEDY_MINIMUM:
        raise RuntimeError("frozen successor support stopping count changed")
    if support_first_count > budget:
        raise RuntimeError(
            f"powered successor support requires {support_first_count} documents, budget is {budget}"
        )
    selected = set(chosen)
    for index in range(len(candidate_rows)):
        if len(selected) == budget:
            break
        selected.add(index)
    if len(selected) != budget:
        raise RuntimeError("candidate scan cannot fill the fixed successor v3 budget")
    ordered = sorted(selected)
    index_tensor = torch.tensor(ordered, dtype=torch.long)
    rows = candidate_rows.index_select(0, index_tensor).contiguous()
    records = []
    for ordinal, index in enumerate(ordered):
        record = dict(candidate_records[index])
        record["candidate_scan_ordinal"] = index
        record["source_document_ordinal"] = ordinal
        record["row_index"] = ordinal
        records.append(record)
    selected_masks = mask_builder(rows, lexicon)
    census = v2.powered_census(selected_masks)
    if any(census[name]["passed"] is not True for name in protocol.POWERED_CELLS):
        raise RuntimeError("successor v3 selected role failed its powered census")
    return BudgetAllocation(
        selected_rows=rows,
        selected_records=tuple(records),
        support_first_count=support_first_count,
        support_first_last_candidate=chosen[-1],
        census=census,
        pair_occupancy=v2.pair_occupancy(selected_masks),
    )


def prospective_status() -> dict[str, Any]:
    _validate_contract()
    return {
        "status": STATUS,
        "row_materialization_authorized": False,
        "model_forward_authorized": False,
        "v2_select_documents": V2_SELECT_DOCUMENTS,
        "v3_select_documents": V3_SELECT_DOCUMENTS,
        "registered_greedy_minimum": REGISTERED_GREEDY_MINIMUM,
        "margin_documents": V3_MARGIN_DOCUMENTS,
        "scored_positions": V3_SCORED_POSITIONS,
    }


__all__ = (
    "AMENDMENT", "BudgetAllocation", "EXPECTED_V2_FAILURE",
    "REGISTERED_GREEDY_MINIMUM", "STATUS", "V2_AUDIT_SHA256",
    "V2_FAILURE_SHA256", "V2_SELECT_DOCUMENTS", "V3_MARGIN_DOCUMENTS",
    "V3_SCORED_POSITIONS", "V3_SELECT_DOCUMENTS", "allocate_v3_budget",
    "prospective_status", "validate_v2_lineage",
)

