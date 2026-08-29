from dataclasses import replace

import pytest
import torch

import broad_mlp_suffix_dealias_v1 as assay
import broad_mlp_suffix_dealias_v1_measurements as measurement
import early_mlp_context_cross_v1_measurements as parent


def _authority():
    value = "a" * 64
    return measurement.RoleAuthority(
        role="skip7000", source_commit="b" * 40, source_closure_sha256=value,
        row_file_sha256=value, row_raw_sha256=value,
        ordered_row_identity_sha256=value,
        ordered_row_to_document_sha256=parent.statistics.tensor_sha256(
            torch.arange(measurement.ROW_COUNT) % measurement.ROLE_DOCUMENT_COUNTS["skip7000"]
        ),
        ordered_document_ids_sha256=value,
        row_token_count_sha256=parent.statistics.tensor_sha256(torch.full(
            (measurement.ROW_COUNT,), measurement.SCORED_TOKENS_PER_ROW, dtype=torch.long,
        )),
        common_support_sha256=value, parent_role_authority_sha256=value,
        parent_measurement_receipt_sha256=value, wave_nonce_sha256=value,
        row_count=measurement.ROW_COUNT,
        document_count=measurement.ROLE_DOCUMENT_COUNTS["skip7000"],
        total_scored_token_count=measurement.ROW_COUNT * measurement.SCORED_TOKENS_PER_ROW,
        batch_count=measurement.BATCH_COUNT,
    )


def _cell(authority, ordinal):
    tokens = torch.full(
        (measurement.ROW_COUNT,), measurement.SCORED_TOKENS_PER_ROW, dtype=torch.long,
    )
    statistics = measurement.RowCellStatistics(
        top1_correct=torch.full((measurement.ROW_COUNT,), ordinal, dtype=torch.long),
        ce_sum=torch.full((measurement.ROW_COUNT,), float(ordinal + 1), dtype=torch.float64),
        row_token_count=tokens,
    )
    h = "c" * 64
    receipt = measurement.CellReceipt(
        authority_sha256=authority.sha256,
        request_sha256=measurement.REQUESTS[ordinal].sha256,
        ordinal=ordinal, prefix_index=ordinal, statistics_sha256=statistics.sha256,
        top1_correct_sha256=statistics.top1_correct_sha256,
        ce_sum_sha256=statistics.ce_sum_sha256,
        row_token_count_sha256=parent.statistics.tensor_sha256(tokens),
        call_ledger_sha256=h, source_closure_sha256=h,
        model_tree_before_sha256=h, model_tree_after_sha256=h,
        shared_program_before_sha256=h, shared_program_after_sha256=h,
        outer_forward_count=measurement.BATCH_COUNT, batch_count=measurement.BATCH_COUNT,
    )
    return statistics, receipt


def test_requests_are_exact_and_ordered():
    assert tuple(request.ordinal for request in measurement.REQUESTS) == tuple(range(8))
    assert tuple(request.sites for request in measurement.REQUESTS) == assay.REQUEST_MASKS


def test_collector_rejects_reorder_and_finalizes_all_eight():
    authority = _authority()
    mapping = torch.arange(measurement.ROW_COUNT) % authority.document_count
    tokens = torch.full((measurement.ROW_COUNT,), measurement.SCORED_TOKENS_PER_ROW, dtype=torch.long)
    collector = measurement.RoleCollector(
        authority=authority, row_to_document=mapping, row_token_count=tokens,
    )
    statistics, receipt = _cell(authority, 1)
    with pytest.raises(RuntimeError, match="reordered"):
        collector.add(statistics, receipt)
    for ordinal in range(8):
        collector.add(*_cell(authority, ordinal))
    bundle = collector.finalize()
    assert bundle.receipt.cell_count == 8
    assert bundle.statistics.top1_correct.shape == (authority.document_count, 8)
    assert int(bundle.statistics.document_token_count.sum()) == 192 * 192
    with pytest.raises(RuntimeError, match="repeated"):
        collector.finalize()


def test_authority_rejects_wrong_component_tree():
    authority = _authority()
    with pytest.raises(ValueError, match="authority changed"):
        replace(authority, component_tree_sha256="d" * 64)
