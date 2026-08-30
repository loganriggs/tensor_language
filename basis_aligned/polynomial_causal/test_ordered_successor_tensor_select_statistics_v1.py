from __future__ import annotations

import dataclasses

import pytest
import torch

import ordered_successor_tensor_discovery_v1 as discovery
import ordered_successor_tensor_select_registry_v2 as v2_registry
import ordered_successor_tensor_select_statistics_v1 as stats


def _ledger(*, documents: int = 32, zero_first: bool = False) -> stats.SelectDocumentLedger:
    arms = len(discovery.ARM_NAMES)
    cells = len(stats.CELL_NAMES)
    count = torch.full((documents, cells), 8, dtype=torch.int64)
    if zero_first:
        count[0] = 0
    ce = torch.zeros(documents, arms, cells, dtype=torch.float64)
    margin = torch.zeros_like(ce)
    native = discovery.ARM_NAMES.index("native")
    deleted = discovery.ARM_NAMES.index(discovery.HEAD_DELETED)
    ce[:, native] = count.double() * 1.0
    ce[:, deleted] = count.double() * 2.0
    margin[:, native] = count.double() * 2.0
    margin[:, deleted] = count.double() * 1.0
    for candidate in discovery.CANDIDATES:
        index = discovery.ARM_NAMES.index(candidate.arm)
        if candidate.kind is discovery.CandidateKind.TRUE:
            fraction = 0.1 if candidate.rank >= 64 else 0.5
        else:
            fraction = 0.8
        ce[:, index] = count.double() * (1.0 + fraction)
        margin[:, index] = count.double() * (2.0 - fraction)
        if candidate.kind is discovery.CandidateKind.TRUE and candidate.rank >= 64:
            ce[:, index, stats.CELL_NAMES.index("all_positions")] = (
                count[:, stats.CELL_NAMES.index("all_positions")].double() * 1.005
            )
            for control in ("wrong_source_clean", "no_source_clean"):
                cell = stats.CELL_NAMES.index(control)
                ce[:, index, cell] = count[:, cell].double()
    # Make deletion specificity positive and true-arm control drift zero.
    for cell in ("wrong_source_clean", "no_source_clean"):
        position = stats.CELL_NAMES.index(cell)
        ce[:, deleted, position] = count[:, position].double() * 1.2
    eligible = count[:, 1:].sum(dim=1)
    pair_count = torch.stack(
        (eligible, torch.zeros_like(eligible), torch.zeros_like(eligible)), dim=1,
    )
    return stats.SelectDocumentLedger(
        document_ids=tuple(f"doc-{index}" for index in range(documents)),
        pair_names=("0->1", "1->2", "2->3"),
        count=count.contiguous(),
        pair_count=pair_count.contiguous(),
        ce_sum=ce.contiguous(),
        native_kl_sum=torch.zeros_like(ce).contiguous(),
        top1_change_sum=torch.zeros_like(ce).contiguous(),
        successor_margin_sum=margin.contiguous(),
    )


def test_readiness_is_exact_three_blocker_pre_authority_no_go() -> None:
    report = stats.evaluate_v1_readiness()
    assert report.status == "PROSPECTIVE_NO_GO"
    assert not report.authority_allowed
    assert not report.row_freeze_allowed
    assert not report.model_forward_allowed
    assert report.frozen_arm_count == 17
    assert report.promotive_arm_count == 6
    assert tuple(blocker.code for blocker in report.blockers) == (
        "nonmaterializable_registered_diagnostics",
        "digit_lexicon_not_frozen",
        "fresh_select_rule_not_frozen",
    )
    assert "drop the two nonpromotive diagnostics" in report.blockers[0].cheapest_prospective_repair
    with pytest.raises(RuntimeError, match="prospectively NO-GO"):
        stats.require_v1_launch_ready()


def test_source_paths_bind_frozen_science_backend_and_new_assurance_tests_once() -> None:
    assert len(stats.SOURCE_PATHS) == len(set(stats.SOURCE_PATHS))
    assert tuple(stats.SOURCE_PATHS[:len(discovery.SOURCE_CLOSURE)]) == discovery.SOURCE_CLOSURE
    assert stats.SOURCE_PATHS[-3:] == (
        "basis_aligned/polynomial_causal/ordered_successor_tensor_select_statistics_v1.py",
        "basis_aligned/polynomial_causal/test_ordered_successor_tensor_select_statistics_v1.py",
        "basis_aligned/polynomial_causal/ordered_successor_tensor_select_registry_v2.py",
    )


def test_ledger_is_sufficient_statistics_only_and_binds_exact_frozen_arm_order() -> None:
    ledger = _ledger()
    assert ledger.ce_sum.shape == (32, 17, 7)
    assert not any("logit" in field.name or "token" in field.name or "row" in field.name
                   for field in dataclasses.fields(ledger))
    with pytest.raises(ValueError, match="metric ledger"):
        stats.SelectDocumentLedger(
            ledger.document_ids, ledger.pair_names, ledger.count, ledger.pair_count,
            ledger.ce_sum.float(), ledger.native_kl_sum, ledger.top1_change_sum,
            ledger.successor_margin_sum,
        )


def test_versioned_v2_currency_scores_exactly_15_arms_and_rejects_other_shapes() -> None:
    v1_ledger = _ledger()
    v2_ledger = dataclasses.replace(
        v1_ledger,
        ce_sum=v1_ledger.ce_sum[:, :15].contiguous(),
        native_kl_sum=v1_ledger.native_kl_sum[:, :15].contiguous(),
        top1_change_sum=v1_ledger.top1_change_sum[:, :15].contiguous(),
        successor_margin_sum=v1_ledger.successor_margin_sum[:, :15].contiguous(),
        arm_names=v2_registry.ARM_NAMES,
    )
    v1_score = stats.score_select_ledger(v1_ledger)
    v2_score = stats.score_select_ledger(v2_ledger)
    assert v2_ledger.ce_sum.shape[1] == 15
    assert v2_score.coordinate_names == v1_score.coordinate_names
    assert torch.equal(v2_score.point, v1_score.point)
    assert torch.equal(v2_score.lower, v1_score.lower)
    assert torch.equal(v2_score.upper, v1_score.upper)
    assert len(stats.point_metric_table(v2_ledger)) == 15 * len(stats.CELL_NAMES)
    for width in (14, 16):
        names = tuple(f"arm-{index}" for index in range(width))
        with pytest.raises(ValueError, match="registry"):
            dataclasses.replace(
                v2_ledger,
                arm_names=names,
                ce_sum=torch.zeros(32, width, 7, dtype=torch.float64),
                native_kl_sum=torch.zeros(32, width, 7, dtype=torch.float64),
                top1_change_sum=torch.zeros(32, width, 7, dtype=torch.float64),
                successor_margin_sum=torch.zeros(32, width, 7, dtype=torch.float64),
            )
    with pytest.raises(ValueError, match="metric ledger"):
        dataclasses.replace(
            v2_ledger,
            ce_sum=v1_ledger.ce_sum,
            native_kl_sum=v1_ledger.native_kl_sum,
            top1_change_sum=v1_ledger.top1_change_sum,
            successor_margin_sum=v1_ledger.successor_margin_sum,
        )


def test_frozen_20k_bootstrap_is_deterministic_shared_and_uses_exact_order_index() -> None:
    ledger = _ledger()
    first = stats.score_select_ledger(ledger)
    second = stats.score_select_ledger(ledger)
    assert first.bootstrap_draws == second.bootstrap_draws == 20_000
    assert first.bootstrap_seed == second.bootstrap_seed == 2_026_083_013
    assert first.order_index == second.order_index == 18_999
    assert first.coordinate_names == second.coordinate_names
    assert torch.equal(first.point, second.point)
    assert torch.equal(first.lower, second.lower)
    assert torch.equal(first.upper, second.upper)
    assert first.critical_value == second.critical_value
    assert first.critical_value < 2e-15
    assert first.support["positive_clean"] == {"positions": 256, "documents": 32}
    assert first.pair_support["0->1"] == {"positions": 1536, "documents": 32}
    table = stats.point_metric_table(ledger)
    assert len(table) == 17 * 7
    assert table[0].arm == "native" and table[0].cell == "all_positions"
    assert table[0].ce == pytest.approx(1.0)


def test_promotion_gate_uses_only_true_arms_and_lowest_price_then_rank() -> None:
    score = stats.score_select_ledger(_ledger())
    integrity = stats.IntegrityEvidence(
        0.0, 0.0, 0.0, 0.0, True, True, True,
    )
    decision = stats.decide_promotions(score, integrity)
    assert decision.integrity_passed and decision.support_passed
    assert decision.selected_arm == "head8_7_both_r64_true"
    assert decision.selected_rank == 64
    assert discovery.CURRENT_ONLY not in decision.passing_arms
    assert discovery.V1_ONLY not in decision.passing_arms
    failed = stats.decide_promotions(
        score, dataclasses.replace(integrity, call_ledgers_passed=False),
    )
    assert failed.passing_arms == () and failed.selected_arm is None


def test_nonpositive_bootstrap_recovery_denominator_fails_without_redraw() -> None:
    ledger = _ledger(documents=2, zero_first=True)
    deleted = discovery.ARM_NAMES.index(discovery.HEAD_DELETED)
    native = discovery.ARM_NAMES.index("native")
    ce = ledger.ce_sum.clone()
    margin = ledger.successor_margin_sum.clone()
    # One document has positive support but a reversed deletion effect. Some shared
    # bootstrap draws are singular/nonpositive and must not be repaired or redrawn.
    count = ledger.count.clone()
    count[0] = 8
    ce[0, native] = 8.0
    ce[0, deleted] = 4.0
    margin[0, native] = 8.0
    margin[0, deleted] = 12.0
    pair_count = ledger.pair_count.clone()
    pair_count[0, 0] = count[0, 1:].sum()
    bad = dataclasses.replace(
        ledger, count=count.contiguous(), pair_count=pair_count.contiguous(),
        ce_sum=ce.contiguous(),
        successor_margin_sum=margin.contiguous(),
    )
    with pytest.raises(ZeroDivisionError, match="nonpositive deletion denominator"):
        stats.score_select_ledger(bad)


def test_support_and_integrity_are_conjoined_not_laundered_by_metric_passes() -> None:
    score = stats.score_select_ledger(_ledger(documents=20))
    integrity = stats.IntegrityEvidence(0.0, 0.0, 0.0, 0.0, True, True, True)
    decision = stats.decide_promotions(score, integrity)
    assert not decision.support_passed
    assert decision.passing_arms == ()
    with pytest.raises(ValueError, match="integrity numerics"):
        stats.IntegrityEvidence(float("nan"), 0.0, 0.0, 0.0, True, True, True)
