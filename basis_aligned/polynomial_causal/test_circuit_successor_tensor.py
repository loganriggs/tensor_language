from __future__ import annotations

import pytest
import torch

import circuit_successor_tensor as successor


def _known_factors() -> tuple[torch.Tensor, ...]:
    scores = torch.tensor(
        [[[1.0, 0.0, 0.0], [2.0, -1.0, 0.0], [0.5, 3.0, 1.0]]],
        dtype=torch.float64,
    )
    current = torch.tensor(
        [[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]], dtype=torch.float64,
    )
    saved = torch.tensor(
        [[[2.0, -1.0], [4.0, -3.0], [6.0, -5.0]]], dtype=torch.float64,
    )
    value_current = torch.tensor([[1.0, 2.0], [-1.0, 1.0]], dtype=torch.float64)
    value_saved = torch.tensor([[2.0, 1.0], [3.0, -2.0]], dtype=torch.float64)
    output = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]], dtype=torch.float64)
    return scores, current, saved, value_current, value_saved, output


def test_two_source_contraction_recovers_literal_current_and_v1_algebra() -> None:
    scores, current, saved, value_current, value_saved, output = _known_factors()
    mix = 0.25
    actual = successor.two_source_successor_write(
        scores, current, saved, value_current, value_saved, output, mix,
    )
    mixed = (1 - mix) * (current @ value_current.T) + mix * (saved @ value_saved.T)
    expected = (scores @ mixed) @ output.T
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    folded = successor.folded_two_source_map(
        value_current, value_saved, output, mix,
    )
    concatenated = torch.cat([current, saved], dim=-1)
    torch.testing.assert_close(
        actual, (scores @ concatenated) @ folded.T, rtol=0, atol=0,
    )


def test_shared_head_gauge_leaves_physical_write_and_folded_map_invariant() -> None:
    scores, current, saved, value_current, value_saved, output = _known_factors()
    gauge = torch.tensor([[2.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    inverse = torch.linalg.inv(gauge)
    moved_current = gauge @ value_current
    moved_saved = gauge @ value_saved
    moved_output = output @ inverse
    original = successor.two_source_successor_write(
        scores, current, saved, value_current, value_saved, output, 0.25,
    )
    moved = successor.two_source_successor_write(
        scores, current, saved, moved_current, moved_saved, moved_output, 0.25,
    )
    torch.testing.assert_close(moved, original, rtol=0, atol=2e-14)
    torch.testing.assert_close(
        successor.folded_two_source_map(
            moved_current, moved_saved, moved_output, 0.25,
        ),
        successor.folded_two_source_map(
            value_current, value_saved, output, 0.25,
        ),
        rtol=0,
        atol=2e-14,
    )


def test_folded_rank_certificate_recovers_known_rank_and_factor_price() -> None:
    _, _, _, value_current, value_saved, output = _known_factors()
    folded = successor.folded_two_source_map(
        value_current, value_saved, output, 0.25,
    )
    assert successor.tolerance_rank(
        folded, absolute_tolerance=1e-12, relative_tolerance=1e-12,
    ) == 2
    rank_one_output = output[:, :1]
    rank_one = successor.folded_two_source_map(
        value_current[:1], value_saved[:1], rank_one_output, 0.25,
    )
    assert successor.tolerance_rank(
        rank_one, absolute_tolerance=1e-12, relative_tolerance=1e-12,
    ) == 1
    assert successor.factor_complete_parameter_count(1152, 1152, 128, 1152) == 442_368


def test_spectral_derangement_is_nontrivial_same_rank_same_spectrum_null() -> None:
    physical = torch.tensor(
        [[9.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    null = successor.spectral_deranged_control(
        physical, torch.tensor([1, 2, 0], dtype=torch.long),
    )
    assert not torch.equal(null, physical)
    torch.testing.assert_close(
        torch.linalg.svdvals(null), torch.linalg.svdvals(physical), rtol=0, atol=0,
    )
    assert successor.tolerance_rank(
        null, absolute_tolerance=1e-12, relative_tolerance=1e-12,
    ) == 3
    with pytest.raises(ValueError, match="no fixed"):
        successor.spectral_deranged_control(
            physical, torch.tensor([0, 2, 1], dtype=torch.long),
        )


def test_intervention_arms_use_one_explicit_head_free_parent() -> None:
    base = torch.tensor([[10.0, 20.0]])
    native = torch.tensor([[1.0, 2.0]])
    extracted = torch.tensor([[3.0, 4.0]])
    null = torch.tensor([[5.0, 6.0]])
    expected = {
        successor.SuccessorArm.NATIVE: base + native,
        successor.SuccessorArm.REMOVE: base,
        successor.SuccessorArm.EXTRACT: base + extracted,
        successor.SuccessorArm.DERANGED: base + null,
    }
    for arm, target in expected.items():
        torch.testing.assert_close(
            successor.compose_successor_arm(base, native, extracted, null, arm),
            target,
            rtol=0,
            atol=0,
        )
    with pytest.raises(ValueError, match="SuccessorArm"):
        successor.compose_successor_arm(base, native, extracted, null, "native")  # type: ignore[arg-type]


def test_shape_currency_and_rank_tolerances_fail_closed() -> None:
    scores, current, saved, value_current, value_saved, output = _known_factors()
    with pytest.raises(ValueError, match="key axes"):
        successor.two_source_successor_write(
            scores, current[:, :-1], saved, value_current, value_saved, output, 0.25,
        )
    with pytest.raises(ValueError, match="one dtype"):
        successor.two_source_successor_write(
            scores.float(), current, saved, value_current, value_saved, output, 0.25,
        )
    with pytest.raises(ValueError, match="finite Python float"):
        successor.two_source_successor_write(
            scores, current, saved, value_current, value_saved, output, float("nan"),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        successor.tolerance_rank(
            output, absolute_tolerance=-1.0, relative_tolerance=0.0,
        )
