import torch

import mlp0_coupled_block1_bigram_response_rung486 as subject


def test_mobius_reconstructs_three_carrier_game():
    performance = torch.zeros(8, dtype=torch.float64)
    for mask in range(8):
        performance[mask] = (
            (1.0 if mask & 1 else 0.0)
            + (2.0 if mask & 2 else 0.0)
            + (4.0 if mask & 4 else 0.0)
            + (3.0 if mask & 1 and mask & 4 else 0.0))
    effects = subject._mobius(performance)
    assert torch.allclose(effects[1:], torch.tensor(
        [1.0, 2.0, 0.0, 4.0, 3.0, 0.0, 0.0], dtype=torch.float64))
    assert torch.allclose(effects[1:].sum(), performance[7] - performance[0])


def test_group_means_and_sparse_lookup_are_exact():
    ids = torch.tensor([[20, 10, 20], [10, 20, 10]])
    values = torch.tensor([[2.0, 10.0, 4.0], [12.0, 6.0, 14.0]])
    groups = torch.tensor([10, 20])
    means, counts, supported, index = subject._group_means(values, ids, groups)
    assert counts.tolist() == [3.0, 3.0]
    assert means[:, 0].tolist() == [12.0, 4.0]
    assert subject._lookup(means, supported, index).squeeze(-1).tolist() == \
        [4.0, 12.0, 4.0, 12.0, 4.0, 12.0]


def test_carrier_analysis_preserves_all_terms_and_detects_stability():
    generator = torch.Generator().manual_seed(4)
    base = (torch.rand(3, 4, subject.TOKENS, generator=generator) + .5).double()
    term_scale = torch.tensor([.1, .2, .3, .4, .5, .6, .7], dtype=torch.float64)
    effects = base[..., None] * term_scale
    performance = torch.zeros(3, 4, subject.TOKENS, 8, dtype=torch.float64)
    for mask in range(8):
        for child in range(1, 8):
            if child & ~mask == 0:
                performance[..., mask] += effects[..., child - 1]
    report = subject.analyze_carriers(-performance, split_index=2)
    assert report["pred_b_carrier_profiles_stable"] is True
    assert report["all_routes_and_terms_live"] is True
    assert report["mobius_closure_holds"] is True


def test_named_bigram_predictor_beats_current_token_and_finds_shared_law():
    generator = torch.Generator().manual_seed(19)
    tokens = torch.randint(1, 5, (500, subject.TOKENS), generator=generator)
    pair_ids = subject._pair_ids(tokens)
    groups = torch.unique(pair_ids).sort().values
    # The effect depends on the ordered pair, while current token alone averages
    # over several predecessor-specific values.
    pair_value = ((groups % subject.VOCAB).double()
                  - (groups // subject.VOCAB).double())
    _, pair_index = subject._membership(pair_ids, groups)
    scalar = pair_value[pair_index]
    coefficients = torch.tensor(
        [.08, .12, .16, .20, .14, .18, .12], dtype=torch.float64)
    route = torch.zeros(3, 500, subject.TOKENS, dtype=torch.float64)
    effects = torch.zeros(3, 500, subject.TOKENS, 7, dtype=torch.float64)
    for branch, scale in enumerate((1.0, -.5, 2.0)):
        route[branch, :, 1:] = scale * scalar
        effects[branch, :, 1:] = scale * scalar[..., None] * coefficients
    report = subject.analyze_named_context(
        {"complete_routes": route, "effects": torch.nn.functional.pad(
            effects, (1, 0))}, tokens, pair_ids, groups)
    assert report["pred_c_bigram_predicts_T_response"] is True
    assert report["shared_holds"] is True
    assert report["split_holds"] is False
    assert report["relation"] == "shared"
