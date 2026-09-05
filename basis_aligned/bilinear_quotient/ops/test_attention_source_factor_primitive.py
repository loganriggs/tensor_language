#!/usr/bin/env python3

import torch
import torch.nn.functional as F

import attention_source_factor_primitive as primitive


def apply_rotary_emb(value, _cos, _sin):
    return value


class FakeAttention:
    def __init__(self, width):
        self.c_q = torch.nn.Linear(width, width, bias=False)
        self.c_k = torch.nn.Linear(width, width, bias=False)
        self.c_q2 = torch.nn.Linear(width, width, bias=False)
        self.c_k2 = torch.nn.Linear(width, width, bias=False)
        self.c_v = torch.nn.Linear(width, width, bias=False)
        self.c_proj = torch.nn.Linear(width, width, bias=False)
        self.lamb = .25

    def rotary(self, value):
        return torch.ones_like(value), torch.zeros_like(value)


def test_source_term_is_exact_score_times_projected_value():
    factors = {
        "p": torch.tensor([[2.0, 3.0], [5.0, 7.0]]),
        "u": torch.tensor([[[11.0, 13.0], [17.0, 19.0]],
                           [[23.0, 29.0], [31.0, 37.0]]]),
    }
    got = primitive.source_terms(factors, torch.tensor([1, 0]), torch)
    assert torch.equal(got, torch.tensor([[51.0, 57.0], [115.0, 145.0]]))


def test_install_changes_only_selected_final_query_term():
    write = torch.zeros(2, 3, 2)
    original = write.clone()
    factors = {
        "p": torch.tensor([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]]),
        "u": torch.ones(2, 3, 2),
    }
    replacement = torch.tensor([[20.0, 30.0], [40.0, 50.0]])
    got = primitive.install_source_terms(
        write, factors, torch.tensor([2, 1]), torch.tensor([0, 2]), replacement, torch,
    )
    expected = torch.zeros_like(write)
    expected[0, 2] = torch.tensor([18.0, 28.0])
    expected[1, 1] = torch.tensor([27.0, 37.0])
    assert torch.equal(got, expected)
    assert torch.equal(write, original)


def test_generic_replay_equals_direct_formula_and_source_sum():
    generator = torch.Generator().manual_seed(13)
    batch, length, width, heads, head_width = 2, 4, 18, 9, 2
    attention = FakeAttention(width)
    for layer in (attention.c_q, attention.c_k, attention.c_q2,
                  attention.c_k2, attention.c_v, attention.c_proj):
        layer.weight.data.copy_(torch.randn(layer.weight.shape, generator=generator))
    state = torch.randn(batch, length, width, generator=generator)
    first = torch.randn(batch, length, heads, head_width, generator=generator)
    finals = torch.tensor([2, 3])
    head = 3
    write, factors = primitive.replay_attention_with_source_factors(
        state, first, attention, finals, head, torch, F,
    )

    def project(layer):
        return F.linear(state, layer.weight).view(batch, length, heads, head_width)
    q, k, q2, k2 = (F.rms_norm(project(layer), (head_width,)) for layer in
                    (attention.c_q, attention.c_k, attention.c_q2, attention.c_k2))
    value = (1 - attention.lamb) * project(attention.c_v) + attention.lamb * first
    pattern = torch.einsum("bqhd,bkhd->bhqk", q, k) / head_width
    pattern *= torch.einsum("bqhd,bkhd->bhqk", q2, k2) / head_width
    pattern = pattern.masked_fill(~torch.tril(torch.ones(length, length, dtype=torch.bool)), 0)
    all_heads = torch.einsum("bhqk,bkhd->bhqd", pattern, value)
    direct = F.linear(all_heads.transpose(1, 2).contiguous().view(batch, length, width),
                      attention.c_proj.weight)
    assert torch.allclose(write, direct, atol=1e-5, rtol=1e-5)
    assert torch.allclose(torch.einsum("bk,bkd->bd", factors["p"], factors["u"]),
                          factors["head"], atol=1e-5, rtol=1e-5)
