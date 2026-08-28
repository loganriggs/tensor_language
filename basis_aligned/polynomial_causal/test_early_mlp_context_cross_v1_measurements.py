from __future__ import annotations

import pytest
import torch

import early_mlp_context_cross_v1 as cross
import early_mlp_context_cross_v1_measurements as measurement
import early_mlp_context_cross_v1_statistics as statistics


H = "d" * 64


def _support(role: str = "skip7000") -> tuple[torch.Tensor, torch.Tensor]:
    documents = measurement.ROLE_DOCUMENT_COUNTS[role]
    row_to_document = torch.tensor(
        [*range(documents), *([0] * (measurement.ROW_COUNT - documents))],
        dtype=torch.long,
    ).contiguous()
    tokens = torch.full(
        (measurement.ROW_COUNT,), measurement.SCORED_TOKENS_PER_ROW,
        dtype=torch.long,
    ).contiguous()
    return row_to_document, tokens


def _authority(role: str = "skip7000") -> measurement.RoleAuthority:
    row_to_document, tokens = _support(role)
    return measurement.RoleAuthority(
        role=role, source_commit="c" * 40, source_closure_sha256=H,
        source_receipt_file_sha256=H, row_file_sha256=H, row_raw_sha256=H,
        row_provenance_sha256=H, ordered_row_identity_sha256=H,
        ordered_row_to_document_sha256=statistics.tensor_sha256(row_to_document),
        ordered_document_ids_sha256=H,
        row_token_count_sha256=statistics.tensor_sha256(tokens),
        document_identity_set_sha256=H, cross_role_disjointness_sha256=H,
        wave_nonce_sha256=H, row_count=measurement.ROW_COUNT,
        document_count=measurement.ROLE_DOCUMENT_COUNTS[role],
        total_scored_token_count=(
            measurement.ROW_COUNT * measurement.SCORED_TOKENS_PER_ROW
        ), batch_count=measurement.BATCH_COUNT,
    )


def _values(ordinal: int) -> measurement.RowCellStatistics:
    tokens = torch.full(
        (measurement.ROW_COUNT,), measurement.SCORED_TOKENS_PER_ROW,
        dtype=torch.long,
    ).contiguous()
    return measurement.RowCellStatistics(
        top1_correct=torch.full(
            (measurement.ROW_COUNT,), 50 + ordinal, dtype=torch.long,
        ).contiguous(),
        ce_sum=torch.full(
            (measurement.ROW_COUNT,), 200.0 + ordinal, dtype=torch.float64,
        ).contiguous(),
        row_token_count=tokens,
    )


def _receipt(
    authority: measurement.RoleAuthority, request: measurement.MeasurementRequest,
    values: measurement.RowCellStatistics,
) -> measurement.CellReceipt:
    return measurement.CellReceipt(
        authority_sha256=authority.sha256, request_sha256=request.sha256,
        ordinal=request.ordinal, cell=request.cell, statistics_sha256=values.sha256,
        top1_correct_sha256=values.top1_correct_sha256,
        ce_sum_sha256=values.ce_sum_sha256,
        row_token_count_sha256=authority.row_token_count_sha256,
        call_ledger_sha256=H, source_closure_sha256=authority.source_closure_sha256,
        model_tree_before_sha256=measurement.COMPONENT_TREE_SHA256,
        model_tree_after_sha256=measurement.COMPONENT_TREE_SHA256,
        shared_program_before_sha256=measurement.SHARED_PROGRAM_SHA256,
        shared_program_after_sha256=measurement.SHARED_PROGRAM_SHA256,
        outer_forward_count=measurement.BATCH_COUNT,
        batch_count=measurement.BATCH_COUNT,
    )


def test_request_plan_has_a_truly_live_origin_and_real_mlp0_factor() -> None:
    measurement.validate_request_plan()
    assert measurement.REQUESTS[0].cell == (0, 0)
    assert measurement.REQUESTS[0].sites == ()
    assert measurement.REQUESTS[8].cell == (1, 0)
    assert measurement.REQUESTS[8].sites == (("mlp", 0),)
    assert [request.stage for request in measurement.REQUESTS].count("discovery") == 48
    assert [request.stage for request in measurement.REQUESTS].count("validation") == 7
    assert [request.stage for request in measurement.REQUESTS].count("heldout") == 9


def test_collector_rejects_skips_duplicates_and_mixed_support() -> None:
    authority = _authority()
    row_to_document, tokens = _support()
    collector = measurement.RoleCollector(
        authority=authority, row_to_document=row_to_document, row_token_count=tokens,
    )
    values = _values(0)
    with pytest.raises(RuntimeError, match="order"):
        collector.add_cell(
            request=measurement.REQUESTS[1], values=values,
            receipt=_receipt(authority, measurement.REQUESTS[1], values),
        )
    collector.add_cell(
        request=measurement.REQUESTS[0], values=values,
        receipt=_receipt(authority, measurement.REQUESTS[0], values),
    )
    with pytest.raises(RuntimeError, match="order"):
        collector.add_cell(
            request=measurement.REQUESTS[0], values=values,
            receipt=_receipt(authority, measurement.REQUESTS[0], values),
        )
    with pytest.raises(ValueError, match="support"):
        measurement.RoleCollector(
            authority=authority, row_to_document=row_to_document.flip(0).contiguous(),
            row_token_count=tokens,
        )


def test_complete_collector_exposes_only_sealed_stage_payloads() -> None:
    authority = _authority()
    row_to_document, tokens = _support()
    collector = measurement.RoleCollector(
        authority=authority, row_to_document=row_to_document, row_token_count=tokens,
    )
    for request in measurement.REQUESTS:
        values = _values(request.ordinal)
        collector.add_cell(
            request=request, values=values,
            receipt=_receipt(authority, request, values),
        )
    bundle = collector.finalize()
    assert bundle.receipt.role == "skip7000"
    assert bundle.discovery.top1_correct.shape == (79, 48)
    assert bundle.validation.top1_correct.shape == (79, 7)
    assert bundle.heldout.top1_correct.shape == (79, 9)
    assert bundle.receipt.stage_payload_sha256s == (
        bundle.discovery.sha256, bundle.validation.sha256, bundle.heldout.sha256,
    )
    with pytest.raises(RuntimeError, match="already attempted"):
        collector.finalize()
