import pytest
import torch

import circuit_induction_tensor as subject


def test_induction_mask_fetches_token_after_previous_equal_query():
    # At q=4 token A repeats token 0, so standard induction fetches key 1 (B).
    tokens = torch.tensor([[10, 20, 30, 40, 10, 99]])
    mask = subject.induction_fetch_mask(tokens)
    assert mask.shape == (1, 6, 6)
    assert bool(mask[0, 4, 1])
    assert not bool(mask[0, 4, 0])
    assert not bool(mask[0, 4, 5])
    assert int(mask.sum()) == 1


def test_fetch_plus_removed_reconstructs_full_head_exactly():
    generator = torch.Generator().manual_seed(18)
    tokens = torch.tensor([[1, 2, 7, 1, 2], [4, 5, 4, 6, 4]])
    scores = torch.randn(2, 5, 5, generator=generator, dtype=torch.float64)
    scores = scores * torch.tril(torch.ones(5, 5))
    values = torch.randn(2, 5, 3, generator=generator, dtype=torch.float64)
    fetched = subject.contract_induction_fetch(scores, values, tokens)
    removed = subject.contract_without_induction_fetch(scores, values, tokens)
    torch.testing.assert_close(
        fetched + removed, torch.bmm(scores, values), rtol=1e-14, atol=1e-14,
    )


def test_multiple_matches_sum_without_nearest_or_argmax_router():
    tokens = torch.tensor([[1, 5, 1, 6, 1]])
    scores = torch.ones(1, 5, 5)
    values = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    output = subject.contract_induction_fetch(scores, values, tokens)
    # q=4 matches predecessor positions 0 and 2, fetching keys 1 and 3.
    assert float(output[0, 4, 0]) == 1 + 3


def test_fixed_vocabulary_permutation_is_a_same_shape_deranged_null():
    tokens = torch.tensor([[4, 9, 8, 3, 1]])
    native = subject.induction_fetch_mask(tokens)
    deranged = subject.induction_fetch_mask(
        tokens, vocabulary_offset=1, vocabulary_size=10,
    )
    assert native.shape == deranged.shape
    # The q=0 derangement cannot select a future key because causality is explicit.
    assert not bool(deranged[0, 0].any())
    # query token 3 at q=3 deranges to 4, matching predecessor position 0 and
    # therefore fetching key 1.
    assert bool(deranged[0, 3, 1])
    assert not torch.equal(native, deranged)


def test_bad_shapes_fail_closed():
    tokens = torch.tensor([[1, 2, 1]])
    with pytest.raises(ValueError):
        subject.contract_induction_fetch(torch.ones(3, 3), torch.ones(1, 3, 2), tokens)
    with pytest.raises(ValueError):
        subject.induction_fetch_mask(tokens.float())
    with pytest.raises(ValueError):
        subject.induction_fetch_mask(tokens, vocabulary_offset=1)
