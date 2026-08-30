from __future__ import annotations

import pytest
import torch

import circuit_previous_token_tensor as previous


def _toy_head() -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.tensor(
        [
            [2.0, 3.0, 5.0, 7.0, 11.0],
            [13.0, 17.0, 19.0, 23.0, 29.0],
            [31.0, 37.0, 41.0, 43.0, 47.0],
            [53.0, 59.0, 61.0, 67.0, 71.0],
        ],
        dtype=torch.float64,
    )
    values = torch.tensor(
        [
            [1.0, -1.0],
            [2.0, -2.0],
            [4.0, -4.0],
            [8.0, -8.0],
            [16.0, -16.0],
        ],
        dtype=torch.float64,
    )
    return scores, values


def test_fixed_shift_mask_has_literal_rectangular_offset_semantics() -> None:
    expected_previous = torch.tensor(
        [
            [False, False, False, False, False],
            [True, False, False, False, False],
            [False, True, False, False, False],
            [False, False, True, False, False],
        ],
    )
    expected_plus_two = torch.tensor(
        [
            [False, False, True, False, False],
            [False, False, False, True, False],
            [False, False, False, False, True],
            [False, False, False, False, False],
        ],
    )
    assert torch.equal(previous.fixed_shift_mask(4, 5, -1), expected_previous)
    assert torch.equal(previous.fixed_shift_mask(4, 5, 2), expected_plus_two)
    assert not bool(previous.fixed_shift_mask(4, 5, 99).any())


def test_native_remove_and_extract_obey_exact_additive_decomposition() -> None:
    scores, values = _toy_head()
    native = previous.run_previous_token_arm(
        scores, values, previous.PreviousTokenArm.NATIVE,
    )
    removed = previous.run_previous_token_arm(
        scores, values, previous.PreviousTokenArm.REMOVE_PREVIOUS,
    )
    extracted = previous.run_previous_token_arm(
        scores, values, previous.PreviousTokenArm.EXTRACT_PREVIOUS,
    )
    torch.testing.assert_close(native, scores @ values, rtol=0, atol=0)
    torch.testing.assert_close(native, removed + extracted, rtol=0, atol=0)
    expected_extract = torch.tensor(
        [[0.0, 0.0], [13.0, -13.0], [74.0, -74.0], [244.0, -244.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(extracted, expected_extract, rtol=0, atol=0)


def test_deranged_arms_select_only_the_registered_minus_two_and_plus_two_edges() -> None:
    scores, values = _toy_head()
    minus_two = previous.run_previous_token_arm(
        scores, values, previous.PreviousTokenArm.DERANGED_MINUS_2,
    )
    plus_two = previous.run_previous_token_arm(
        scores, values, previous.PreviousTokenArm.DERANGED_PLUS_2,
    )
    expected_minus_two = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [31.0, -31.0], [118.0, -118.0]],
        dtype=torch.float64,
    )
    expected_plus_two = torch.tensor(
        [[20.0, -20.0], [184.0, -184.0], [752.0, -752.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(minus_two, expected_minus_two, rtol=0, atol=0)
    torch.testing.assert_close(plus_two, expected_plus_two, rtol=0, atol=0)


def test_batched_multihead_axes_are_independent_and_gradients_flow() -> None:
    generator = torch.Generator().manual_seed(20260830)
    scores = torch.randn(
        2, 3, 5, 5, generator=generator, dtype=torch.float64, requires_grad=True,
    )
    values = torch.randn(
        2, 3, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True,
    )
    result = previous.contract_fixed_shift(scores, values, -1)
    assert result.shape == (2, 3, 5, 4)
    expected = torch.stack([
        torch.stack([
            previous.contract_fixed_shift(scores[b, h], values[b, h], -1)
            for h in range(3)
        ])
        for b in range(2)
    ])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result.square().sum().backward()
    assert scores.grad is not None and values.grad is not None
    support = previous.fixed_shift_mask(5, 5, -1).expand(2, 3, -1, -1)
    assert torch.equal(scores.grad == 0, ~support)


@pytest.mark.parametrize(
    ("scores", "values", "message"),
    [
        (torch.ones(2, 3), torch.ones(4, 1), "key axis"),
        (torch.ones(2, 3), torch.ones(1, 3, 1), "shape"),
        (torch.ones(2, 3), torch.ones(3, 1, dtype=torch.float64), "dtype"),
        (torch.ones(2, 3, dtype=torch.int64), torch.ones(3, 1, dtype=torch.int64), "floating"),
    ],
)
def test_malformed_inputs_fail_closed(
    scores: torch.Tensor,
    values: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        previous.contract_full_head(scores, values)


def test_arm_rejects_untyped_strings_and_shift_arguments_reject_bools() -> None:
    scores, values = _toy_head()
    with pytest.raises(ValueError, match="PreviousTokenArm"):
        previous.run_previous_token_arm(scores, values, "native_full_head")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Python integer"):
        previous.contract_fixed_shift(scores, values, True)
    with pytest.raises(ValueError, match="positive Python integer"):
        previous.fixed_shift_mask(0, 5, -1)
