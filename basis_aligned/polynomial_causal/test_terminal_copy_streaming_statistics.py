import pytest
import torch

import terminal_copy_streaming_statistics as stats


def _cell(n, delta, support):
    return stats.DocumentCellSums(
        n=n, native_nll_sum=2.0 * n, ablated_nll_sum=2.0 * n + delta,
        native_correct_count=0, ablated_correct_count=0,
        native_to_ablated_kl_sum=max(delta, 0.0), support_sha256=support,
    )


def _ledger(positive=(1.0, 9.0), negative=(0.0, 0.0), off=(0.0, 0.0), counts=(1, 9)):
    output = {}
    for index, document in enumerate(("d0", "d1")):
        support = (str(index + 1) * 64)[:64]
        output[document] = {
            "positive": _cell(counts[index], positive[index], support),
            "matched_negative": _cell(counts[index], negative[index], support),
            "off_target": _cell(counts[index], off[index], support),
        }
    return output


def test_pooled_effect_is_token_weighted_not_equal_document_weighted():
    effect = stats.pooled_effects(_ledger(positive=(1.0, 18.0)))
    assert effect.tau_positive == pytest.approx(1.9)
    assert effect.tau_positive != pytest.approx((1.0 + 2.0) / 2)
    assert effect.specificity == pytest.approx(1.9)
    assert effect.collateral_margin == pytest.approx(0.01)


def test_streaming_reducer_is_shift_invariant_and_uses_native_to_ablated_kl():
    rows = torch.arange(514, dtype=torch.long).reshape(2, 257) % 7
    masks = {name: torch.zeros(2, 256, dtype=torch.bool) for name in stats.CELL_NAMES}
    masks["positive"][:, 64] = True
    masks["matched_negative"][:, 65] = True
    masks["off_target"][:, 66:] = True
    native = torch.randn(2, 256, 8, generator=torch.Generator().manual_seed(4))
    ablated = native.clone()
    ablated[:, :, 0] += 0.5
    first = stats.reduce_document_batch(native, ablated, rows, masks, ("a", "b"))
    second = stats.reduce_document_batch(native + 13, ablated + 13, rows, masks, ("a", "b"))
    for document in ("a", "b"):
        for cell in stats.CELL_NAMES:
            left, right = first[document][cell], second[document][cell]
            assert left.n == right.n
            assert left.support_sha256 == right.support_sha256
            assert left.native_nll_sum == pytest.approx(right.native_nll_sum, abs=1e-5)
            assert left.ablated_nll_sum == pytest.approx(right.ablated_nll_sum, abs=1e-5)
            assert left.native_to_ablated_kl_sum == pytest.approx(
                right.native_to_ablated_kl_sum, abs=1e-5,
            )
    assert first["a"]["positive"].native_to_ablated_kl_sum >= 0


def test_candidate_support_mismatch_is_rejected_even_at_equal_counts():
    first = _ledger()
    second = _ledger()
    changed = second["d0"]["positive"]
    second["d0"]["positive"] = stats.DocumentCellSums(
        **{**changed.__dict__, "support_sha256": "f" * 64}
    )
    with pytest.raises(ValueError, match="exact input support"):
        stats.simultaneous_selection_bootstrap(
            {"a": first, "b": second}, repetitions=20, expected_candidates=("a", "b"),
        )


def test_bootstrap_is_deterministic_shared_and_uses_three_coordinates_per_candidate():
    ledgers = {"b": _ledger(), "a": _ledger(positive=(2.0, 18.0))}
    first = stats.simultaneous_selection_bootstrap(
        ledgers, repetitions=200, seed="fixed", expected_candidates=("a", "b"),
    )
    second = stats.simultaneous_selection_bootstrap(
        ledgers, repetitions=200, seed="fixed", expected_candidates=("a", "b"),
    )
    assert first.coordinate_names == (
        "a:tau_positive", "a:specificity", "a:collateral_margin",
        "b:tau_positive", "b:specificity", "b:collateral_margin",
    )
    assert torch.equal(first.point_estimates, second.point_estimates)
    assert torch.equal(first.simultaneous_lower_bounds, second.simultaneous_lower_bounds)
    assert first.critical_value == second.critical_value


def test_bootstrap_zero_denominator_draw_fails_without_redraw():
    ledger = _ledger(counts=(0, 1))
    with pytest.raises(ZeroDivisionError, match="zero cell denominator"):
        stats.simultaneous_selection_bootstrap(
            {"a": ledger}, repetitions=100, seed="find-zero", expected_candidates=("a",),
        )


def test_selection_boundaries_and_lexicographic_tie_are_frozen():
    passing = _ledger(positive=(1.0, 9.0), negative=(-1.0, -9.0), off=(-1.0, -9.0))
    result = stats.simultaneous_selection_bootstrap(
        {"z": passing, "a": passing}, repetitions=200, seed="tie",
        expected_candidates=("a", "z"),
    )
    assert result.selected_candidate == "a"


def test_default_selection_rejects_any_nonfrozen_candidate_family():
    with pytest.raises(ValueError, match="frozen bank"):
        stats.simultaneous_selection_bootstrap({"a": _ledger()}, repetitions=20)


def test_final_and_ood_use_one_six_coordinate_gate_with_independent_role_draws():
    passing = _ledger(positive=(1.0, 9.0), negative=(-1.0, -9.0), off=(-1.0, -9.0))
    result = stats.simultaneous_final_ood_bootstrap(
        {"final_natural": passing, "ood_code": passing},
        repetitions=200, seed="replicate",
    )
    assert result.coordinate_names == (
        "final_natural:tau_positive", "final_natural:specificity",
        "final_natural:collateral_margin", "ood_code:tau_positive",
        "ood_code:specificity", "ood_code:collateral_margin",
    )
    assert result.passed
