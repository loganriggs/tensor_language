"""Answer-preserving complete-record permutations for the fixed cold-query domain."""
import torch


def permutations():
    swap = torch.arange(24)
    swap[[0, 1]] = torch.tensor([1, 0])
    return {'swap_first_two': swap, 'cyclic_shift_one': torch.arange(24).roll(-1)}


def function(tokens):
    assert tokens.shape[-1] == 51
    pairs = tokens[..., :48].reshape(*tokens.shape[:-1], 24, 2)
    expected = torch.arange(24, device=tokens.device).expand_as(pairs[..., 0])
    assert torch.equal(pairs[..., 0].sort(-1).values, expected)
    assert torch.equal(pairs[..., 1].sort(-1).values, expected)
    return torch.zeros_like(pairs[..., 0]).scatter(-1, pairs[..., 0], pairs[..., 1])


def reorder(tokens, permutation):
    assert permutation.shape == (24,)
    assert torch.equal(permutation.sort().values.cpu(), torch.arange(24))
    result = tokens.clone()
    records = tokens[..., :48].reshape(*tokens.shape[:-1], 24, 2)
    result[..., :48] = records[..., permutation.to(tokens.device), :].flatten(-2)
    assert torch.equal(function(result), function(tokens))
    assert torch.equal(result[..., -3:], tokens[..., -3:])
    return result
