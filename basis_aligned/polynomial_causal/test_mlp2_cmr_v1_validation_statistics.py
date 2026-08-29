from __future__ import annotations

import pytest
import torch

import mlp2_cmr_v1_validation_statistics as stats


def fixture(batch: int = 3, vocab: int = 17):
    generator = torch.Generator().manual_seed(301)
    rows = torch.randint(0, vocab, (batch, 257), generator=generator)
    eligible = torch.zeros(batch, 256, dtype=torch.bool)
    eligible[:, 64:] = True
    fit_counts = torch.arange(50_304, dtype=torch.long).remainder(200)
    boundaries = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128])
    cells = stats.validation_cells(rows, eligible, fit_counts, boundaries)
    native = torch.randn(batch, 256, vocab, generator=generator)
    candidate = native + 0.03 * torch.randn(
        batch, 256, vocab, generator=generator,
    )
    return rows, eligible, cells, native, candidate


def test_validation_cells_have_two_exact_partitions_and_all_scored() -> None:
    rows, eligible, cells, _, _ = fixture()
    assert set(cells) == set(stats.CELL_NAMES)
    assert torch.equal(cells["all_scored"], eligible)
    assert torch.equal(
        torch.stack([cells[name] for name in stats.FREQUENCY_CELL_NAMES]).sum(0),
        eligible.long(),
    )
    assert torch.equal(
        torch.stack([cells[name] for name in stats.COPY_CELL_NAMES]).sum(0),
        eligible.long(),
    )


def test_reducer_is_logit_shift_invariant_and_self_arm_is_exact() -> None:
    rows, _, cells, native, candidate = fixture()
    first = stats.reduce_arm_batch(native, candidate, rows, cells, (0, 1, 2))
    shifted = stats.reduce_arm_batch(
        native + 11, candidate + 11, rows, cells, (0, 1, 2),
    )
    for document in range(3):
        for cell in stats.CELL_NAMES:
            left, right = first[document][cell], shifted[document][cell]
            assert left.count == right.count
            assert left.support_sha256 == right.support_sha256
            assert left.teacher_kl_sum == pytest.approx(right.teacher_kl_sum, abs=2e-4)
            assert left.centered_logit_sse == pytest.approx(
                right.centered_logit_sse, abs=2e-4,
            )
    identity = stats.reduce_arm_batch(native, native, rows, cells, (0, 1, 2))
    assert identity[0]["all_scored"].teacher_kl_sum == 0
    assert identity[0]["all_scored"].centered_logit_sse == 0
    assert identity[0]["all_scored"].raw_logit_sse == 0
    assert identity[0]["all_scored"].native_top1_agreement_count == 192


def test_arm_summary_is_token_weighted_and_reports_worst_document() -> None:
    rows, _, cells, native, candidate = fixture()
    ledger = stats.reduce_arm_batch(native, candidate, rows, cells, (0, 1, 2))
    summary = stats.summarize_arm(ledger, prefix_documents=3)
    all_scored = summary["cells"]["all_scored"]
    expected_count = sum(ledger[d]["all_scored"].count for d in range(3))
    expected_delta = sum(
        ledger[d]["all_scored"].candidate_nll_sum
        - ledger[d]["all_scored"].native_nll_sum for d in range(3)
    ) / expected_count
    assert all_scored["count"] == expected_count
    assert all_scored["candidate_minus_native_ce"] == pytest.approx(expected_delta)
    assert len(summary["raw_sufficient_statistics"]) == 3


def test_margin_certificate_uses_vocabulary_sum_D2_and_frozen_grid() -> None:
    rows, eligible, cells, native, candidate = fixture()
    ledger = stats.reduce_arm_batch(native, candidate, rows, cells, (0, 1, 2))
    grid = torch.tensor([0.1, 0.5, 2.0], dtype=torch.float64)
    margin, support = stats.native_margin_counts(native, eligible, grid)
    curve = stats.margin_certificate_curve(
        ledger, margin, support, grid, prefix_documents=3,
    )
    assert curve["epsilon_grid"] == grid.tolist()
    assert len(curve["bounds"]) == 3
    assert 0 <= curve["maximum_bound"] <= 1
    expected_d2 = sum(
        ledger[d]["all_scored"].raw_logit_sse for d in range(3)
    ) / int(support.sum())
    assert curve["raw_logit_D2"] == pytest.approx(expected_d2)


def _bootstrap_ledger(scale: float):
    rows, _, cells, native, candidate = fixture()
    base = stats.reduce_arm_batch(native, candidate, rows, cells, (0, 1, 2))
    output = {}
    for document, cell_map in base.items():
        output[document] = {}
        for cell, value in cell_map.items():
            fields = value.__dict__.copy()
            fields["teacher_kl_sum"] = scale * (document + 1) * max(value.count, 1)
            output[document][cell] = stats.CellSums(**fields)
    return output


def test_relative_kl_bootstrap_is_shared_deterministic_and_pooled() -> None:
    ledgers = {
        "SUFFIX": _bootstrap_ledger(0.5),
        "LOCAL": _bootstrap_ledger(1.0),
        "RMS": _bootstrap_ledger(1.1),
        "MASS": _bootstrap_ledger(1.2),
        "DERANGED": _bootstrap_ledger(1.3),
        "HASH_RANDOM": _bootstrap_ledger(1.4),
    }
    first = stats._relative_kl_bootstrap(
        ledgers, repetitions=500, seed="fixed",
    )
    second = stats._relative_kl_bootstrap(
        ledgers, repetitions=500, seed="fixed",
    )
    assert first == second
    assert first["minimum_point_relative_kl_improvement"] == pytest.approx(0.5)
    assert first["simultaneous_lower_bound"] > 0


def test_canonical_bootstrap_rejects_small_or_tunable_protocol() -> None:
    ledgers = {
        "SUFFIX": _bootstrap_ledger(0.5), "LOCAL": _bootstrap_ledger(1.0),
        "RMS": _bootstrap_ledger(1.1), "MASS": _bootstrap_ledger(1.2),
        "DERANGED": _bootstrap_ledger(1.3), "HASH_RANDOM": _bootstrap_ledger(1.4),
    }
    with pytest.raises(ValueError, match="canonical"):
        stats.simultaneous_relative_kl_bootstrap(ledgers)
    with pytest.raises(ValueError, match="canonical"):
        stats.simultaneous_relative_kl_bootstrap(
            ledgers, primary="LOCAL", controls=(
                "SUFFIX", "RMS", "MASS", "DERANGED", "HASH_RANDOM",
            ),
        )
    assert int.from_bytes(
        __import__("hashlib").sha256(stats.BOOTSTRAP_SEED.encode()).digest()[:8], "little",
    ) == 13_376_816_517_823_017_776


def test_signed_geometry_recovers_one_direction_and_rejects_zero_norm() -> None:
    rows, _, cells, native, _ = fixture()
    direction = torch.randn_like(native, generator=torch.Generator().manual_seed(302))
    signed = {
        "minus_0p25": native - 0.25 * direction,
        "minus_0p10": native - 0.10 * direction,
        "plus_0p10": native + 0.10 * direction,
        "plus_0p25": native + 0.25 * direction,
    }
    batch = stats.reduce_signed_geometry_batch(
        native, native + direction, signed, cells,
    )
    summary = stats.summarize_signed_geometry([batch])
    for pair in stats.GEOMETRY_PAIRS:
        assert summary["all_scored"][pair]["cosine"] == pytest.approx(1, abs=1e-6)
    zero = {key: native for key in stats.SIGNED_KEYS}
    summary = stats.summarize_signed_geometry([
        stats.reduce_signed_geometry_batch(native, native, zero, cells)
    ])
    assert summary["all_scored"]["g0p10_vs_full"]["cosine"] is None
    assert not summary["all_scored"]["g0p10_vs_full"]["nonzero"]


def test_float32_precision_audit_passes_small_fixture_and_is_fail_closed() -> None:
    rows, eligible, _, native, candidate = fixture()
    audit = stats.enforce_float32_precision_audit(native, candidate, rows, eligible)
    assert audit["passed"]
    assert audit["maximum_native_nll_absolute_error"] <= 1e-4
    assert audit["maximum_candidate_nll_absolute_error"] <= 1e-4
    with pytest.raises(ValueError, match="support"):
        stats.float32_precision_audit(native, candidate, rows, eligible[:, :-1])
    old = stats.NLL_KL_PRECISION_TOLERANCE
    try:
        stats.NLL_KL_PRECISION_TOLERANCE = -1
        with pytest.raises(RuntimeError, match="frozen"):
            stats.enforce_float32_precision_audit(native, candidate, rows, eligible)
    finally:
        stats.NLL_KL_PRECISION_TOLERANCE = old
