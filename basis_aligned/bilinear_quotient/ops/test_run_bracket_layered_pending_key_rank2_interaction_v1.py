#!/usr/bin/env python3
from collections import defaultdict

import torch

import run_bracket_layered_pending_key_rank2_interaction_v1 as candidate


def test_three_centered_type_prototypes_have_exact_rank_at_most_two():
    values = defaultdict(list)
    values["parenthesis"] = [torch.tensor([1., 2., 4.]), torch.tensor([3., 4., 6.])]
    values["square"] = [torch.tensor([4., 1., 2.]), torch.tensor([6., 3., 4.])]
    values["quote"] = [torch.tensor([2., 5., 1.]), torch.tensor([4., 7., 3.])]
    table = candidate.low_rank_type_table(values, torch)
    prototypes = torch.stack([torch.stack(values[name]).double().mean(0) for name in candidate.TYPES])
    rebuilt = table["mean"] + table["coefficients"] @ table["basis"]
    assert table["rank"] <= 2
    assert torch.allclose(rebuilt, prototypes, atol=1e-12, rtol=1e-12)


def test_score_is_product_of_both_normalized_query_key_contractions():
    torch.manual_seed(15092026)
    tensors = [torch.randn(5, candidate.HEAD_D) for _ in range(4)]
    q1, q2, k1, k2 = tensors
    expected = torch.einsum("bd,bd->b", q1, k1)/candidate.HEAD_D
    expected *= torch.einsum("bd,bd->b", q2, k2)/candidate.HEAD_D
    actual = candidate.score_from_query_keys(q1, q2, k1, k2)
    assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-6)
