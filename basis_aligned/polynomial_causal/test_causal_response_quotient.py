import numpy as np
import pytest

from causal_response_quotient import (
    ConsumerGate,
    pointwise_dominates,
    rank_passing_candidates,
    score_partition,
    score_worst_cell_equivalence,
)


def _gate(q95=0.30, maximum=0.50, ratio=2.0):
    return ConsumerGate(q95, maximum, ratio)


def test_valid_coarse_quotient_passes_every_consumer():
    # Two tight cells, each repeated over two live backgrounds.
    base = np.array(
        [
            [[0.00], [0.05]],
            [[0.04], [0.09]],
            [[2.00], [2.10]],
            [[2.05], [2.15]],
        ]
    )
    responses = {"attn1": base, "mlp1": 0.5 * base}
    report = score_partition(
        responses,
        ["a", "a", "b", "b"],
        scales={"attn1": 1.0, "mlp1": 0.5},
        gates={"attn1": _gate(), "mlp1": _gate()},
        declared_price_bits=64,
    )
    assert report["causal_quotient_passes"] is True
    assert report["passes_all_consumers"] is True
    assert report["acceptance_uses_mean_response"] is False


def test_one_bad_background_cannot_hide_behind_small_mean():
    # The first cell looks interchangeable in nine backgrounds and fails badly in
    # one.  An average-only criterion would accept it; the max-background gate must
    # reject it.
    response = np.zeros((4, 10, 1))
    response[1, -1, 0] = 1.0
    response[2:, :, 0] = 3.0
    report = score_partition(
        {"suffix": response},
        [0, 0, 1, 1],
        scales={"suffix": 1.0},
        gates={"suffix": _gate(q95=0.20, maximum=0.50)},
        declared_price_bits=1,
    )
    consumer = report["consumers"]["suffix"]
    assert consumer["within_mean_diagnostic_only"] < 0.20
    assert consumer["within_worst"] == pytest.approx(1.0)
    assert consumer["passes"] is False
    assert report["causal_quotient_passes"] is False


def test_consumer_failure_cannot_free_ride_on_other_consumer():
    good = np.array([[[0.0]], [[0.1]], [[2.0]], [[2.1]]])
    bad = np.array([[[0.0]], [[1.0]], [[2.0]], [[3.0]]])
    report = score_partition(
        {"attn1": good, "mlp1": bad},
        [0, 0, 1, 1],
        scales={"attn1": 1.0, "mlp1": 1.0},
        gates={"attn1": _gate(), "mlp1": _gate()},
        declared_price_bits=12,
    )
    assert report["consumers"]["attn1"]["passes"] is True
    assert report["consumers"]["mlp1"]["passes"] is False
    assert report["passes_all_consumers"] is False


def test_singletons_cannot_game_within_cell_gate():
    response = np.arange(6.0).reshape(6, 1, 1)
    report = score_partition(
        {"suffix": response},
        [0, 1, 2, 3, 4, 4],
        scales={"suffix": 1.0},
        gates={"suffix": _gate(q95=2.0, maximum=2.0, ratio=1.0)},
        declared_price_bits=100,
    )
    assert report["non_singleton_state_coverage"] == pytest.approx(2 / 6)
    assert report["coverage_passes"] is False
    assert report["causal_quotient_passes"] is False


def test_missing_consumer_or_invalid_scale_is_rejected():
    response = np.zeros((2, 1, 1))
    with pytest.raises(ValueError, match="match exactly"):
        score_partition(
            {"a": response},
            [0, 0],
            scales={"b": 1.0},
            gates={"a": _gate()},
            declared_price_bits=1,
        )
    with pytest.raises(ValueError, match="positive"):
        score_partition(
            {"a": response},
            [0, 0],
            scales={"a": 0.0},
            gates={"a": _gate()},
            declared_price_bits=1,
        )


def test_rank_only_passing_candidates_by_declared_price():
    reports = {
        "expensive": {"causal_quotient_passes": True, "declared_price_bits": 20},
        "failed": {"causal_quotient_passes": False, "declared_price_bits": 1},
        "cheap": {"causal_quotient_passes": True, "declared_price_bits": 10},
    }
    assert rank_passing_candidates(reports) == ["cheap", "expensive"]


def test_worst_cell_bootstrap_passes_small_effects():
    sums = np.full((40, 2), 0.002)
    counts = np.ones((40, 2))
    report = score_worst_cell_equivalence(
        {"kl": sums, "attn1": sums},
        {"kl": counts, "attn1": counts},
        margins={"kl": 0.01, "attn1": 0.05},
        cell_names=["low", "high"],
        n_bootstrap=200,
        seed=3,
    )
    assert report["equivalence_passes"] is True
    assert report["acceptance_uses_pooled_average"] is False
    assert report["simultaneous_95pct_ucb_max_standardized_effect"] < 1


def test_single_bad_cell_falsifies_even_when_pooled_average_is_small():
    sums = np.zeros((40, 16))
    counts = np.ones((40, 16))
    sums[:, 7] = 0.02  # one KL cell is 2x its margin; pooled mean is tiny
    report = score_worst_cell_equivalence(
        {"kl": sums},
        {"kl": counts},
        margins={"kl": 0.01},
        cell_names=[f"c{i}" for i in range(16)],
        n_bootstrap=200,
    )
    assert sums.mean() < 0.01
    assert report["point_max_standardized_effect"] == pytest.approx(2.0)
    assert report["equivalence_passes"] is False


def test_sparse_cell_fails_closed():
    sums = np.full((40, 2), 0.001)
    counts = np.ones((40, 2))
    sums[20:, 1] = 0
    counts[20:, 1] = 0
    report = score_worst_cell_equivalence(
        {"ce": sums},
        {"ce": counts},
        margins={"ce": 0.0075},
        cell_names=["dense", "sparse"],
        minimum_documents_per_cell=30,
        n_bootstrap=100,
    )
    assert report["support_passes"] is False
    assert report["equivalence_passes"] is False


def test_pointwise_dominance_requires_every_cell_and_strict_max():
    counts = np.ones((40, 2))
    candidate = score_worst_cell_equivalence(
        {"kl": np.tile([0.002, 0.003], (40, 1))},
        {"kl": counts},
        margins={"kl": 0.01},
        cell_names=["a", "b"],
        n_bootstrap=50,
    )
    baseline = score_worst_cell_equivalence(
        {"kl": np.tile([0.004, 0.005], (40, 1))},
        {"kl": counts},
        margins={"kl": 0.01},
        cell_names=["a", "b"],
        n_bootstrap=50,
    )
    assert pointwise_dominates(candidate, baseline) is True

    worse_one_cell = score_worst_cell_equivalence(
        {"kl": np.tile([0.006, 0.003], (40, 1))},
        {"kl": counts},
        margins={"kl": 0.01},
        cell_names=["a", "b"],
        n_bootstrap=50,
    )
    assert pointwise_dominates(worse_one_cell, baseline) is False
