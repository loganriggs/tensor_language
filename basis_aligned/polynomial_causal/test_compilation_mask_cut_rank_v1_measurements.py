from dataclasses import replace

import pytest
import torch

import compilation_mask_cut_rank_v1 as cut
import compilation_mask_cut_rank_v1_measurements as measurement


def _hash(label: str) -> str:
    return measurement._logical_sha256(label)


def _rows():
    row_to_document = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long)
    row_token_count = torch.tensor([4, 4, 3, 2, 2, 2], dtype=torch.long)
    return row_to_document, row_token_count


def _authority(**changes):
    row_to_document, row_token_count = _rows()
    program_realizations = tuple(
        _hash(f"program realization {ordinal}") for ordinal in range(64)
    )
    values = dict(
        source_commit="1" * 40,
        source_receipt_sha256=_hash("source receipt"),
        row_tensor_sha256=_hash("row tensor"),
        row_provenance_sha256=_hash("row provenance"),
        ordered_row_identity_sha256=_hash("ordered row identity"),
        ordered_row_to_document_sha256=measurement.tensor_sha256(row_to_document),
        ordered_document_ids_sha256=_hash("ordered document ids"),
        row_token_count_sha256=measurement.tensor_sha256(row_token_count),
        common_support_sha256=_hash("common scored targets"),
        model_realization_sha256=_hash("checkpoint/config realization"),
        component_tree_sha256=_hash("immutable loaded component tree"),
        program_bank_sha256=measurement.program_bank_sha256(program_realizations),
        source_closure_sha256=_hash("source closure"),
        wave_nonce_sha256=_hash("wave nonce"),
        program_realization_sha256s=program_realizations,
        row_count=len(row_to_document),
        document_count=3,
        total_scored_token_count=int(row_token_count.sum()),
        batch_count=2,
    )
    values.update(changes)
    return measurement.MeasurementWaveAuthority(**values)


def _statistics(ordinal: int):
    _, token_count = _rows()
    correct = token_count - (ordinal % 2)
    ce = token_count.to(torch.float64) * (0.1 + ordinal / 1000.0)
    return measurement.RowCellSufficientStatistics(
        top1_correct=correct.contiguous(), ce_sum=ce.contiguous(),
        row_token_count=token_count,
    )


def _cell_receipt(authority, request, statistics, **changes):
    values = dict(
        authority_sha256=authority.sha256,
        request_sha256=request.sha256,
        ordinal=request.ordinal,
        cell=request.cell,
        program_realization_sha256=(
            authority.program_realization_sha256s[request.ordinal]
        ),
        common_support_sha256=authority.common_support_sha256,
        ordered_row_identity_sha256=authority.ordered_row_identity_sha256,
        top1_correct_sha256=statistics.top1_correct_sha256,
        ce_sum_sha256=statistics.ce_sum_sha256,
        row_token_count_sha256=statistics.row_token_count_sha256,
        statistics_sha256=statistics.sha256,
        call_ledger_sha256=_hash(f"call ledger {request.ordinal}"),
        source_closure_sha256=authority.source_closure_sha256,
        model_tree_before_sha256=authority.component_tree_sha256,
        model_tree_after_sha256=authority.component_tree_sha256,
        outer_forward_count=authority.batch_count,
        batch_count=authority.batch_count,
    )
    values.update(changes)
    return measurement.CellMeasurementReceipt(**values)


def _collector(authority=None):
    if authority is None:
        authority = _authority()
    row_to_document, row_token_count = _rows()
    return measurement.MeasurementCollector(
        authority=authority, row_to_document=row_to_document,
        row_token_count=row_token_count,
    )


def _complete_collector():
    authority = _authority()
    collector = _collector(authority)
    statistics = []
    receipts = []
    for request in measurement.REQUESTS:
        statistic = _statistics(request.ordinal)
        receipt = _cell_receipt(authority, request, statistic)
        collector.add_cell(request=request, statistics=statistic, receipt=receipt)
        statistics.append(statistic)
        receipts.append(receipt)
    return authority, collector, statistics, receipts


def test_request_plan_is_exact_row_major_frozen_grid():
    measurement.validate_request_plan()
    assert len(measurement.REQUESTS) == 64
    assert tuple(request.cell for request in measurement.REQUESTS) == cut.ALL_CELLS
    assert tuple(request.ordinal for request in measurement.REQUESTS) == tuple(range(64))
    assert measurement.REQUESTS[0].additional_sites == ()
    assert measurement.REQUESTS[0].layer_symbols == ("none",) * 17
    assert measurement.REQUESTS[0].always_compiled_sites == (
        ("attn", 0), ("mlp", 0),
    )
    assert sum(request.split == "heldout" for request in measurement.REQUESTS) == 11
    with pytest.raises(ValueError, match="mask/split"):
        replace(measurement.REQUESTS[9], split="heldout")


def test_authority_is_discovery_only_and_binds_exact_program_set():
    authority = _authority()
    assert authority.authorized_for_final_role is False
    assert authority.authority_scope == measurement.AUTHORITY_SCOPE
    with pytest.raises(ValueError, match="unauthorized"):
        replace(authority, authorized_for_final_role=True)
    with pytest.raises(ValueError, match="program realization"):
        _authority(program_realization_sha256s=authority.program_realization_sha256s[:-1])
    with pytest.raises(ValueError, match="program bank"):
        replace(authority, program_bank_sha256=_hash("wrong bank"))
    with pytest.raises(ValueError, match="identity"):
        _authority(source_commit="not-a-commit")


def test_row_statistics_are_exact_typed_sealed_and_fail_closed():
    statistic = _statistics(3)
    original_sha = statistic.sha256
    external = statistic.top1_correct_sha256
    assert len(external) == 64 and original_sha == statistic.sha256
    _, tokens = _rows()
    with pytest.raises(ValueError, match="wrong schema"):
        measurement.RowCellSufficientStatistics(
            top1_correct=tokens.to(torch.float64),
            ce_sum=tokens.to(torch.float64), row_token_count=tokens,
        )
    with pytest.raises(ValueError, match="bounds"):
        measurement.RowCellSufficientStatistics(
            top1_correct=tokens + 1, ce_sum=tokens.to(torch.float64),
            row_token_count=tokens,
        )
    with pytest.raises(AttributeError, match="sealed"):
        statistic._ce_sum = torch.zeros(6, dtype=torch.float64)
    statistic._ce_sum[0] += 1.0
    with pytest.raises(RuntimeError, match="mutated"):
        _ = statistic.sha256


def test_collector_rejects_order_program_model_and_common_support_forgery():
    authority = _authority()
    collector = _collector(authority)
    statistic = _statistics(0)
    receipt = _cell_receipt(authority, measurement.REQUESTS[0], statistic)
    with pytest.raises(RuntimeError, match="type/order"):
        collector.add_cell(
            request=measurement.REQUESTS[1], statistics=_statistics(1),
            receipt=_cell_receipt(authority, measurement.REQUESTS[1], _statistics(1)),
        )
    with pytest.raises(RuntimeError, match="receipt differs"):
        collector.add_cell(
            request=measurement.REQUESTS[0], statistics=statistic,
            receipt=replace(receipt, program_realization_sha256=_hash("wrong program")),
        )
    with pytest.raises(RuntimeError, match="receipt differs"):
        collector.add_cell(
            request=measurement.REQUESTS[0], statistics=statistic,
            receipt=replace(receipt, model_tree_after_sha256=_hash("mutated model")),
        )
    with pytest.raises(RuntimeError, match="receipt differs"):
        collector.add_cell(
            request=measurement.REQUESTS[0], statistics=statistic,
            receipt=replace(receipt, common_support_sha256=_hash("different support")),
        )
    with pytest.raises(RuntimeError, match="receipt differs"):
        collector.add_cell(
            request=measurement.REQUESTS[0], statistics=statistic,
            receipt=replace(receipt, source_closure_sha256=_hash("different source")),
        )
    collector.add_cell(
        request=measurement.REQUESTS[0], statistics=statistic, receipt=receipt,
    )
    assert collector.next_ordinal == 1


def test_collector_constructor_rejects_mapping_order_and_token_currency():
    authority = _authority()
    mapping, tokens = _rows()
    with pytest.raises(RuntimeError, match="sealed row/support"):
        measurement.MeasurementCollector(
            authority=authority, row_to_document=mapping[[2, 0, 1, 3, 4, 5]].contiguous(),
            row_token_count=tokens,
        )
    bad_authority = replace(
        authority, ordered_row_to_document_sha256=measurement.tensor_sha256(
            torch.tensor([0, 2, 1, 2, 0, 2], dtype=torch.long)
        ),
    )
    with pytest.raises(RuntimeError, match="sealed row/support"):
        measurement.MeasurementCollector(
            authority=bad_authority,
            row_to_document=torch.tensor([0, 2, 1, 2, 0, 2], dtype=torch.long),
            row_token_count=tokens,
        )
    with pytest.raises(RuntimeError, match="sealed row/support"):
        measurement.MeasurementCollector(
            authority=replace(authority, total_scored_token_count=999),
            row_to_document=mapping, row_token_count=tokens,
        )


def test_incomplete_finalize_is_terminal_and_reveals_no_payload():
    authority = _authority()
    collector = _collector(authority)
    statistic = _statistics(0)
    collector.add_cell(
        request=measurement.REQUESTS[0], statistics=statistic,
        receipt=_cell_receipt(authority, measurement.REQUESTS[0], statistic),
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        collector.finalize()
    with pytest.raises(RuntimeError, match="already finalized"):
        collector.add_cell(
            request=measurement.REQUESTS[1], statistics=_statistics(1),
            receipt=_cell_receipt(authority, measurement.REQUESTS[1], _statistics(1)),
        )
    with pytest.raises(RuntimeError, match="already attempted"):
        collector.finalize()


def test_complete_grid_emits_bootstrap_ready_document_payload_and_bound_receipt():
    authority, collector, statistics, receipts = _complete_collector()
    bundle = collector.finalize()
    assert bundle.receipt.authority_sha256 == authority.sha256
    assert bundle.receipt.b0_request_sha256 == measurement.REQUESTS[0].sha256
    assert bundle.receipt.b0_cell_receipt_sha256 == receipts[0].sha256
    assert bundle.receipt.cell_receipt_sha256s == tuple(x.sha256 for x in receipts)
    assert bundle.receipt.top1_correct_row_sha256s == tuple(
        x.top1_correct_sha256 for x in receipts
    )
    assert bundle.receipt.ce_sum_row_sha256s == tuple(x.ce_sum_sha256 for x in receipts)
    assert bundle.receipt.statistics_sha256s == tuple(
        x.statistics_sha256 for x in receipts
    )
    assert bundle.receipt.source_closure_sha256 == authority.source_closure_sha256
    assert bundle.receipt.component_tree_sha256 == authority.component_tree_sha256
    assert bundle.receipt.batch_count == authority.batch_count
    assert bundle.receipt.authorized_for_final_role is False
    expected_top1 = torch.stack([
        statistic._clone_values()[0] for statistic in statistics
    ], dim=1)
    expected_ce = torch.stack([
        statistic._clone_values()[1] for statistic in statistics
    ], dim=1)
    mapping, tokens = _rows()
    assert torch.equal(bundle.payload.document_row_count, torch.tensor([2, 1, 3]))
    assert torch.equal(bundle.payload.document_token_count, torch.tensor([8, 3, 6]))
    for document in range(3):
        selected = mapping == document
        assert torch.equal(
            bundle.payload.top1_correct[document], expected_top1[selected].sum(dim=0),
        )
        assert torch.allclose(
            bundle.payload.ce_sum[document], expected_ce[selected].sum(dim=0),
            rtol=0.0, atol=0.0,
        )
        assert int(tokens[selected].sum()) == int(bundle.payload.document_token_count[document])
    escaped = bundle.payload.ce_sum
    escaped.zero_()
    assert bool((bundle.payload.ce_sum > 0).all())
    with pytest.raises(ValueError, match="payload/receipt"):
        measurement.FinalizedMeasurementBundle(
            payload=bundle.payload,
            receipt=replace(bundle.receipt, row_count=bundle.receipt.row_count + 1),
        )
    with pytest.raises(RuntimeError, match="already attempted"):
        collector.finalize()


def test_inputs_are_cloned_and_late_external_mutation_cannot_change_payload():
    authority = _authority()
    mapping, tokens = _rows()
    collector = measurement.MeasurementCollector(
        authority=authority, row_to_document=mapping, row_token_count=tokens,
    )
    mapping.fill_(2)
    tokens.fill_(99)
    for request in measurement.REQUESTS:
        statistic = _statistics(request.ordinal)
        receipt = _cell_receipt(authority, request, statistic)
        collector.add_cell(request=request, statistics=statistic, receipt=receipt)
    bundle = collector.finalize()
    assert torch.equal(bundle.payload.document_row_count, torch.tensor([2, 1, 3]))
    assert torch.equal(bundle.payload.document_token_count, torch.tensor([8, 3, 6]))
