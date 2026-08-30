import dataclasses

import pytest
import torch

import circuit_campaign_statistics as stats


def _logits(rows: torch.Tensor, *, target_logit: float) -> torch.Tensor:
    result = torch.zeros(rows.shape[0], rows.shape[1] - 1, 7, dtype=torch.float32)
    result.scatter_(2, rows[:, 1:].unsqueeze(-1), target_logit)
    return result


def _reduced_ledger():
    rows = torch.tensor([
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [1, 1, 1, 1],
    ], dtype=torch.long)
    masks = {
        "target": torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.bool),
        "matched": torch.tensor([[0, 0, 1], [0, 1, 1], [1, 0, 0]], dtype=torch.bool),
        "background": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.bool),
    }
    native = _logits(rows, target_logit=3.0)
    removal = _logits(rows, target_logit=0.0)
    extracted = _logits(rows, target_logit=2.0)
    collateral = native.clone()
    collateral[:, :, 0] += 0.1
    return stats.reduce_document_batch(
        {
            "native": native,
            "removal": removal,
            "extracted": extracted,
            "collateral": collateral,
        },
        rows,
        masks,
        ("doc-a", "doc-a", "doc-b"),
        kl_pairs=(("native", "extracted"), ("native", "collateral")),
    )


def test_reducer_aggregates_repeated_rows_and_returns_only_sufficient_statistics():
    ledger = _reduced_ledger()
    assert tuple(ledger) == ("doc-a", "doc-b")
    assert ledger["doc-a"]["target"].n == 3
    assert ledger["doc-b"]["target"].n == 3
    cell = ledger["doc-a"]["target"]
    assert tuple(item.arm for item in cell.arms) == (
        "collateral", "extracted", "native", "removal",
    )
    assert tuple((item.source_arm, item.target_arm) for item in cell.directed_kls) == (
        ("native", "collateral"), ("native", "extracted"),
    )
    assert len(cell.support_sha256) == 64
    assert not any(torch.is_tensor(value) for value in dataclasses.astuple(cell))


def test_support_digest_binds_ordered_row_bytes_not_only_mask_shape_or_count():
    rows = torch.tensor([[0, 1, 2], [0, 3, 4]], dtype=torch.long)
    masks = {"cell": torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)}
    logits = torch.zeros(2, 2, 5)
    first = stats.reduce_document_batch({"native": logits}, rows, masks, ("d", "d"))
    changed_rows = rows.clone()
    changed_rows[0, 0] = 4  # input-only byte; target support/count remain unchanged.
    second = stats.reduce_document_batch(
        {"native": logits}, changed_rows, masks, ("d", "d"),
    )
    assert first["d"]["cell"].n == second["d"]["cell"].n == 2
    assert first["d"]["cell"].support_sha256 != second["d"]["cell"].support_sha256


def test_all_coordinate_kinds_have_frozen_higher_is_better_formulas():
    ledger = _reduced_ledger()
    roles = {"fit": ledger, "ood": ledger}
    specs = (
        stats.CoordinateSpec(
            "damage", stats.CoordinateKind.TARGET_DAMAGE, "fit", "target",
            candidate_arm="removal",
        ),
        stats.CoordinateSpec(
            "specific", stats.CoordinateKind.SPECIFICITY, "fit", "target",
            candidate_arm="removal", comparison_cell="matched",
        ),
        stats.CoordinateSpec(
            "collateral", stats.CoordinateKind.COLLATERAL, "fit", "background",
            candidate_arm="collateral", limit=0.02,
        ),
        stats.CoordinateSpec(
            "recovery", stats.CoordinateKind.EXTRACTION_RECOVERY, "fit", "target",
            candidate_arm="extracted", stake_arm="removal",
        ),
        stats.CoordinateSpec(
            "ood", stats.CoordinateKind.OOD_RETENTION, "ood", "target",
            candidate_arm="extracted", stake_arm="removal",
        ),
        stats.CoordinateSpec(
            "kl", stats.CoordinateKind.KL, "fit", "target",
            candidate_arm="extracted", source_arm="native", limit=1.0,
        ),
        stats.CoordinateSpec(
            "top1", stats.CoordinateKind.TOP1, "fit", "target",
            candidate_arm="extracted", limit=0.05,
        ),
    )
    values = stats.evaluate_coordinates(roles, specs)
    assert values.shape == (7,)
    assert values.dtype == torch.float64
    damage = stats.evaluate_coordinate(ledger, specs[0])
    matched = stats.CoordinateSpec(
        "matched-damage", stats.CoordinateKind.TARGET_DAMAGE, "fit", "matched",
        candidate_arm="removal",
    )
    assert values[1].item() == pytest.approx(
        damage - stats.evaluate_coordinate(ledger, matched),
    )
    assert 0 < values[3].item() < 1
    assert values[3].item() == pytest.approx(values[4].item())
    assert values[5].item() <= 1.0


def test_coordinate_schema_is_typed_and_rejects_ambiguous_formulas():
    with pytest.raises(ValueError, match="CoordinateKind"):
        stats.CoordinateSpec("x", "kl", "fit", "target")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires comparison_cell"):
        stats.CoordinateSpec("x", stats.CoordinateKind.SPECIFICITY, "fit", "target")
    with pytest.raises(ValueError, match="requires stake_arm"):
        stats.CoordinateSpec("x", stats.CoordinateKind.OOD_RETENTION, "fit", "target")
    with pytest.raises(ValueError, match="required exactly"):
        stats.CoordinateSpec("x", stats.CoordinateKind.KL, "fit", "target", source_arm="native")
    with pytest.raises(ValueError, match="finite nonnegative"):
        stats.CoordinateSpec(
            "x", stats.CoordinateKind.TOP1, "fit", "target", limit=float("nan"),
        )


def test_coordinate_known_answers_include_recovery_kl_and_top1_currencies():
    def cell(n, native, removal, extracted, native_correct, extracted_correct, kl):
        return stats.DocumentCellSums(
            n=n,
            support_sha256="a" * 64,
            arms=(
                stats.ArmCellSums("extracted", float(extracted), extracted_correct),
                stats.ArmCellSums("native", float(native), native_correct),
                stats.ArmCellSums("removal", float(removal), 0),
            ),
            directed_kls=(stats.DirectedKLSums("native", "extracted", float(kl)),),
        )

    ledger = {"doc": {
        "target": cell(10, 10, 30, 15, 8, 7, 2.5),
        "matched": cell(5, 5, 10, 6, 4, 4, 0.5),
    }}
    specs = (
        stats.CoordinateSpec(
            "damage", stats.CoordinateKind.TARGET_DAMAGE, "fit", "target",
            candidate_arm="removal",
        ),
        stats.CoordinateSpec(
            "specificity", stats.CoordinateKind.SPECIFICITY, "fit", "target",
            candidate_arm="removal", comparison_cell="matched",
        ),
        stats.CoordinateSpec(
            "collateral", stats.CoordinateKind.COLLATERAL, "fit", "target",
            candidate_arm="extracted", limit=1.0,
        ),
        stats.CoordinateSpec(
            "recovery", stats.CoordinateKind.EXTRACTION_RECOVERY, "fit", "target",
            candidate_arm="extracted", stake_arm="removal",
        ),
        stats.CoordinateSpec(
            "ood", stats.CoordinateKind.OOD_RETENTION, "fit", "target",
            candidate_arm="extracted", stake_arm="removal",
        ),
        stats.CoordinateSpec(
            "kl", stats.CoordinateKind.KL, "fit", "target",
            candidate_arm="extracted", source_arm="native", limit=0.5,
        ),
        stats.CoordinateSpec(
            "top1", stats.CoordinateKind.TOP1, "fit", "target",
            candidate_arm="extracted", limit=0.15,
        ),
    )
    assert stats.evaluate_coordinates({"fit": ledger}, specs).tolist() == pytest.approx(
        [2.0, 1.0, 0.5, 0.75, 0.75, 0.25, 0.05],
    )


def _manual_cell(n: int, native: float, candidate: float, support: str):
    return stats.DocumentCellSums(
        n=n,
        support_sha256=support,
        arms=(
            stats.ArmCellSums("candidate", float(candidate), 0),
            stats.ArmCellSums("native", float(native), 0),
        ),
        directed_kls=(),
    )


def _manual_role(values):
    return {
        document: {"target": _manual_cell(n, 2.0 * n, 2.0 * n + delta, hex_digit * 64)}
        for document, n, delta, hex_digit in values
    }


def test_point_estimate_is_token_weighted_not_equal_document_weighted():
    ledger = _manual_role((
        ("a", 1, 1.0, "a"),
        ("b", 9, 18.0, "b"),
    ))
    spec = stats.CoordinateSpec(
        "damage", stats.CoordinateKind.TARGET_DAMAGE, "fit", "target",
    )
    assert stats.evaluate_coordinate(ledger, spec) == pytest.approx(1.9)
    assert stats.evaluate_coordinate(ledger, spec) != pytest.approx(1.5)


def test_bootstrap_is_deterministic_and_shares_draws_across_paired_roles():
    left = _manual_role((("a", 1, 1.0, "a"), ("b", 1, 3.0, "b")))
    right = _manual_role((("a", 1, 2.0, "c"), ("b", 1, 6.0, "d")))
    specs = (
        stats.CoordinateSpec(
            "left", stats.CoordinateKind.TARGET_DAMAGE, "fit", "target",
            draw_group="paired",
        ),
        stats.CoordinateSpec(
            "right", stats.CoordinateKind.TARGET_DAMAGE, "ood", "target",
            draw_group="paired",
        ),
    )
    first = stats.simultaneous_document_bootstrap(
        {"fit": left, "ood": right}, specs, repetitions=200, seed="fixed",
    )
    second = stats.simultaneous_document_bootstrap(
        {"fit": left, "ood": right}, specs, repetitions=200, seed="fixed",
    )
    assert first.coordinate_names == ("left", "right")
    assert torch.equal(first.point_estimates, second.point_estimates)
    assert torch.equal(first.simultaneous_lower_bounds, second.simultaneous_lower_bounds)
    assert first.critical_value == second.critical_value
    # Every right-role deviation is exactly twice the paired left-role deviation;
    # its point and confidence bound therefore preserve that relation.
    assert first.point_estimates[1] == 2 * first.point_estimates[0]
    assert first.simultaneous_lower_bounds[1] <= first.point_estimates[1]


def test_bootstrap_rejects_mismatched_documents_in_one_draw_group():
    left = _manual_role((("a", 1, 1.0, "a"), ("b", 1, 2.0, "b")))
    right = _manual_role((("a", 1, 1.0, "c"), ("c", 1, 2.0, "d")))
    specs = (
        stats.CoordinateSpec(
            "left", stats.CoordinateKind.TARGET_DAMAGE, "fit", "target",
            draw_group="paired",
        ),
        stats.CoordinateSpec(
            "right", stats.CoordinateKind.TARGET_DAMAGE, "ood", "target",
            draw_group="paired",
        ),
    )
    with pytest.raises(ValueError, match="share exact document IDs"):
        stats.simultaneous_document_bootstrap(
            {"fit": left, "ood": right}, specs, repetitions=10, seed="fixed",
        )


def test_zero_denominator_draw_fails_without_redraw_or_repair():
    ledger = _manual_role((("a", 0, 0.0, "a"), ("b", 1, 1.0, "b")))
    spec = stats.CoordinateSpec(
        "damage", stats.CoordinateKind.TARGET_DAMAGE, "fit", "target",
    )
    with pytest.raises(ZeroDivisionError, match="zero token denominator"):
        stats.simultaneous_document_bootstrap(
            {"fit": ledger}, (spec,), repetitions=100, seed="find-zero",
        )


def test_reducer_rejects_nonfinite_logits_and_unregistered_kl_pairs():
    rows = torch.tensor([[0, 1]], dtype=torch.long)
    masks = {"target": torch.ones(1, 1, dtype=torch.bool)}
    bad = torch.zeros(1, 1, 2)
    bad[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="logits"):
        stats.reduce_document_batch({"native": bad}, rows, masks, ("d",))
    with pytest.raises(ValueError, match="KL pairs"):
        stats.reduce_document_batch(
            {"native": torch.zeros(1, 1, 2)}, rows, masks, ("d",),
            kl_pairs=(("native", "missing"),),
        )
