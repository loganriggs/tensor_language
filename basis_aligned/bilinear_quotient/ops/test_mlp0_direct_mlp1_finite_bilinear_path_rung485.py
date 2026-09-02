import torch

import mlp0_direct_mlp1_finite_bilinear_path_rung485 as subject


def test_mobius_reconstructs_two_side_game():
    performance = torch.tensor([0.0, 2.0, 3.0, 12.0], dtype=torch.float64)
    effects = subject._mobius(performance)
    assert torch.allclose(
        effects, torch.tensor([0.0, 2.0, 3.0, 7.0], dtype=torch.float64))
    assert torch.allclose(effects[1:].sum(), performance[3] - performance[0])


def test_side_selection_excludes_the_trivial_full_arm():
    benefits = torch.zeros(3, 5, 4, dtype=torch.float64)
    benefits[..., 3] = 1.0
    selected, candidates = subject._select_side(benefits)
    assert selected is None
    assert {row["mask"] for row in candidates} == {1, 2}


def test_analysis_detects_stable_split_and_reports_equality_positions():
    generator = torch.Generator().manual_seed(11)
    effect = torch.randn(4, subject.TOKENS, generator=generator).double()
    ce = torch.ones(2, 4, subject.TOKENS, 4, dtype=torch.float64)
    for mask in range(4):
        if mask & 1:
            ce[0, ..., mask] -= effect
        if mask & 2:
            ce[1, ..., mask] -= effect
    positive = torch.zeros(4, subject.TOKENS, dtype=torch.bool)
    positive[:, ::7] = True
    token_ids = torch.zeros(4, subject.TOKENS, dtype=torch.long)
    report = subject.analyze_phase(
        ce, token_ids, positive, split_index=2)
    assert report["selected_sides"] == {"T": 1, "I": 2}
    assert report["pred_b_physical_side_predicts_route"] is True
    assert report["pred_c_path_profile_stable"] is True
    assert report["split_holds"] is True
    assert report["shared_holds"] is False
    assert report["relation"] == "split"
    assert report["mobius_closure_holds"] is True
    assert report["branch_reports"]["T"]["halves"][0][
        "equality_positive_positions"] > 0
    assert len(report["branch_reports"]["T"]["halves"][0][
        "equality_candidate_reports"]) == 2


def test_token_means_are_exact_for_repeated_ids():
    token_ids = torch.tensor([[1, 2, 1], [2, 1, 2]])
    values = torch.tensor([[2.0, 10.0, 4.0], [12.0, 6.0, 14.0]])
    means, counts = subject._token_means(values, token_ids)
    assert counts[1] == 3 and counts[2] == 3
    assert means[1] == 4.0
    assert means[2] == 12.0


def test_error_aggregation_does_not_double_suffix_max_abs():
    assert subject._aggregate_error_key("mlp0_state_max_abs") == \
        "mlp0_state_max_abs"
    assert subject._aggregate_error_key("all_native_relative_squared") == \
        "all_native_relative_squared_max"
