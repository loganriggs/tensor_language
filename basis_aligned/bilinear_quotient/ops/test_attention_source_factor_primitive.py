#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import pytest

import attention_source_factor_primitive as primitive
import attention_source_destination_eval as attention_eval


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
        half = value[..., :value.shape[-1] // 2]
        return torch.ones_like(half), torch.zeros_like(half)


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


def test_row_varying_source_subset_replacement_is_exact():
    native = {
        "p": torch.tensor([[.2, .8], [.3, .7]]),
        "u": torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]),
    }
    donor = {
        "p": torch.tensor([[.6, .4], [.9, .1]]),
        "u": torch.tensor([[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]]),
    }
    for factors in (native, donor):
        factors["head"] = torch.einsum("bk,bkd->bd", factors["p"], factors["u"])
    empty = torch.zeros(2, 2, dtype=torch.bool)
    full = torch.ones(2, 2, dtype=torch.bool)
    mixed = torch.tensor([[True, False], [False, True]])
    assert torch.equal(
        primitive.replace_head_source_subset(native, donor, empty, "joint", torch),
        native["head"],
    )
    assert torch.allclose(
        primitive.replace_head_source_subset(native, donor, full, "joint", torch),
        donor["head"],
    )
    for mode in ("score", "value", "joint"):
        got = primitive.replace_head_source_subset(native, donor, mixed, mode, torch)
        chosen_p = donor["p"] if mode in {"score", "joint"} else native["p"]
        chosen_u = donor["u"] if mode in {"value", "joint"} else native["u"]
        expected = native["head"].clone()
        for row, source in ((0, 0), (1, 1)):
            expected[row] += (chosen_p[row, source] * chosen_u[row, source]
                              - native["p"][row, source] * native["u"][row, source])
        assert torch.allclose(got, expected)
    left = torch.tensor([[True, False], [True, False]])
    right = ~left
    left_head = primitive.replace_head_source_subset(native, donor, left, "joint", torch)
    right_head = primitive.replace_head_source_subset(native, donor, right, "joint", torch)
    assert torch.allclose(left_head + right_head - native["head"], donor["head"])


def test_source_subset_replacement_rejects_ambiguous_masks_and_shapes():
    native = {"p": torch.ones(1, 2), "u": torch.ones(1, 2, 3),
              "head": torch.ones(1, 3)}
    donor = {key: value.clone() for key, value in native.items()}
    with pytest.raises(ValueError, match="boolean"):
        primitive.replace_head_source_subset(
            native, donor, torch.ones(1, 2), "joint", torch)
    with pytest.raises(ValueError, match="shapes"):
        bad = {**donor, "u": torch.ones(1, 3, 3)}
        primitive.replace_head_source_subset(
            native, bad, torch.ones(1, 2, dtype=torch.bool), "joint", torch)
    with pytest.raises(ValueError, match="mode"):
        primitive.replace_head_source_subset(
            native, donor, torch.ones(1, 2, dtype=torch.bool), "blend", torch)
    with pytest.raises(ValueError, match="contain p, u, and head"):
        missing_with_extra = {"p": donor["p"], "u": donor["u"], "extra": donor["u"]}
        primitive.replace_head_source_subset(
            native, missing_with_extra, torch.ones(1, 2, dtype=torch.bool), "joint", torch)


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
    assert set(factors) == {"p", "u", "head"}


def test_optional_qk_factors_expose_selected_normalized_rotary_vectors():
    generator = torch.Generator().manual_seed(17)
    batch, length, width, heads, head_width = 2, 4, 18, 9, 2
    attention = FakeAttention(width)
    for layer in (attention.c_q, attention.c_k, attention.c_q2,
                  attention.c_k2, attention.c_v, attention.c_proj):
        layer.weight.data.copy_(torch.randn(layer.weight.shape, generator=generator))
    state = torch.randn(batch, length, width, generator=generator)
    first = torch.randn(batch, length, heads, head_width, generator=generator)
    finals = torch.tensor([2, 3])
    _, factors = primitive.replay_attention_with_source_factors(
        state, first, attention, finals, 3, torch, F, include_qk_factors=True,
    )
    assert factors["q"].shape == factors["q2"].shape == (batch, head_width)
    assert factors["k"].shape == factors["k2"].shape == (batch, length, head_width)
    self_score = ((factors["q"] * factors["k"][torch.arange(batch), finals]).sum(-1) / head_width
                  * (factors["q2"] * factors["k2"][torch.arange(batch), finals]).sum(-1)
                  / head_width)
    assert torch.allclose(self_score, factors["p"][torch.arange(batch), finals])


def _five_factor_fixture():
    native = {
        "q": torch.tensor([[1.0, 2.0], [2.0, -1.0]]),
        "k": torch.tensor([[[1.0, 0.0], [0.0, 1.0]],
                           [[2.0, 1.0], [1.0, -2.0]]]),
        "q2": torch.tensor([[2.0, -1.0], [1.0, 3.0]]),
        "k2": torch.tensor([[[1.0, 1.0], [-1.0, 2.0]],
                            [[0.0, 2.0], [3.0, 1.0]]]),
        "u": torch.tensor([[[1.0, 3.0, 2.0], [4.0, -1.0, 2.0]],
                           [[2.0, 2.0, -3.0], [1.0, 5.0, 2.0]]]),
    }
    donor = {name: value + (index + 1) * .25
             for index, (name, value) in enumerate(native.items())}
    return native, donor


def test_five_factor_source_mixture_matches_direct_attention_term():
    native, donor = _five_factor_fixture()
    for selected in ((), ("q",), ("k", "u"), primitive.SOURCE_FACTORS):
        got = primitive.mixed_source_terms(native, donor, selected, torch)
        chosen = {name: donor[name] if name in selected else native[name]
                  for name in primitive.SOURCE_FACTORS}
        score1 = torch.einsum("bd,btd->bt", chosen["q"], chosen["k"]) / 2
        score2 = torch.einsum("bd,btd->bt", chosen["q2"], chosen["k2"]) / 2
        expected = (score1 * score2).unsqueeze(-1) * chosen["u"]
        assert torch.equal(got, expected)


def test_five_factor_mobius_closes_sourcewise_and_after_source_sum():
    native, donor = _five_factor_fixture()
    dividends = primitive.source_factor_mobius(native, donor, torch)
    assert len(dividends) == 32
    native_terms = primitive.mixed_source_terms(native, donor, (), torch)
    donor_terms = primitive.mixed_source_terms(
        native, donor, primitive.SOURCE_FACTORS, torch)
    reconstructed = sum((dividends[mask] for mask in range(1, 32)),
                        torch.zeros_like(native_terms))
    assert torch.allclose(reconstructed, donor_terms - native_terms, atol=2e-5, rtol=2e-5)
    assert torch.allclose(reconstructed.sum(1),
                          donor_terms.sum(1) - native_terms.sum(1),
                          atol=2e-5, rtol=2e-5)
    assert primitive.factor_names(0b10101) == ("q", "q2", "u")


def test_five_factor_source_game_rejects_unknown_factors_and_bad_shapes():
    native, donor = _five_factor_fixture()
    with pytest.raises(ValueError, match="unknown"):
        primitive.mixed_source_terms(native, donor, ("score3",), torch)
    with pytest.raises(ValueError, match="shapes"):
        primitive.mixed_source_terms(
            {**native, "u": native["u"][:, :1]}, donor, (), torch)
    with pytest.raises(ValueError, match="must contain"):
        primitive.mixed_source_terms(
            {name: value for name, value in native.items() if name != "q2"}, donor, (), torch)


def _all_query_fixture():
    native, donor = _five_factor_fixture()
    native = {**native, "q": native["q"][:, None].repeat(1, 2, 1),
              "q2": native["q2"][:, None].repeat(1, 2, 1)}
    donor = {**donor, "q": donor["q"][:, None].repeat(1, 2, 1),
             "q2": donor["q2"][:, None].repeat(1, 2, 1)}
    groups = torch.tensor([[[True, False], [False, True], [False, False]],
                           [[False, True], [True, False], [False, False]]])
    return native, donor, groups


def test_grouped_all_query_game_is_causal_and_closes_exactly():
    native, donor, groups = _all_query_fixture()
    base = primitive.mixed_grouped_query_source_writes(native, donor, (), groups, torch)
    changed = primitive.mixed_grouped_query_source_writes(
        native, donor, primitive.SOURCE_FACTORS, groups, torch)
    # Query zero cannot receive source one, whichever group contains it.
    assert torch.equal(base[0, 0, 1], torch.zeros_like(base[0, 0, 1]))
    dividends = primitive.grouped_query_source_factor_mobius(native, donor, groups, torch)
    reconstructed = sum((dividends[mask] for mask in range(1, 32)), torch.zeros_like(base))
    assert torch.allclose(reconstructed, changed - base, atol=2e-5, rtol=2e-5)
    assert torch.allclose(reconstructed.sum(2),
                          (changed - base).sum(2), atol=2e-5, rtol=2e-5)


def test_token_role_partition_covers_actual_v23_rows_without_padding_leakage():
    import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23 as v23

    rows = v23.build_rows()
    masks = primitive.batch_token_role_partitions(rows, 9, torch)
    assert masks.shape == (64, 3, 9)
    assert not (masks.sum(1) > 1).any()
    for index, row in enumerate(rows):
        endpoint = row["base_semantic_position"]
        assert masks[index, :, :endpoint + 1].sum() == endpoint + 1
        assert masks[index, :, endpoint + 1:].sum() == 0
        changed = masks[index, primitive.SOURCE_GROUPS.index("changed")].nonzero().flatten()
        suffix = masks[index, primitive.SOURCE_GROUPS.index("matched_suffix")].nonzero().flatten()
        assert all(row["base_ids"][position] != row["donor_ids"][position]
                   for position in changed.tolist())
        assert all(row["base_ids"][position] == row["donor_ids"][position]
                   for position in suffix.tolist())


def test_grouped_game_rejects_overlapping_source_roles():
    native, donor, groups = _all_query_fixture()
    groups[:, 2] = groups[:, 0]
    with pytest.raises(ValueError, match="overlap"):
        primitive.mixed_grouped_query_source_writes(native, donor, (), groups, torch)


def test_public_attention_factor_replay_exposes_exact_full_query_factors():
    generator = torch.Generator().manual_seed(23)
    batch, length, width, heads, head_width = 2, 4, 18, 9, 2
    attention = FakeAttention(width)
    attention.squared_attn = True
    for layer in (attention.c_q, attention.c_k, attention.c_q2,
                  attention.c_k2, attention.c_v, attention.c_proj):
        layer.weight.data.copy_(torch.randn(layer.weight.shape, generator=generator))
    state = torch.randn(batch, length, width, generator=generator)
    first = torch.randn(batch, length, heads, head_width, generator=generator)
    config = type("Config", (), {"n_head": heads, "n_embd": width})()
    model = type("Model", (), {"config": config})()
    backend = type("Backend", (), {"model": model})()
    factors = attention_eval.attention_factor_terms(backend, attention, state, first)
    assert set(factors) == {"q", "k", "q2", "k2", "value", "pattern", "head_output"}
    expected = torch.einsum("bhqs,bshd->bqhd", factors["pattern"], factors["value"])
    assert torch.allclose(factors["head_output"], expected, atol=1e-5, rtol=1e-5)
    for name in ("q", "k", "q2", "k2", "value"):
        assert factors[name].shape == (batch, length, heads, head_width)
