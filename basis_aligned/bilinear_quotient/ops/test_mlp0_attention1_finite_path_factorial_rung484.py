import torch

import mlp0_attention1_finite_path_factorial_rung484 as subject


def test_mobius_reconstructs_three_component_game():
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


def test_selection_excludes_trivial_full_arm():
    benefits = torch.zeros(2, 3, 8, dtype=torch.float64)
    benefits[..., 7] = 1.0
    positive = torch.ones(2, 3, dtype=torch.bool)
    selected, candidates = subject._select_mask(benefits, positive)
    assert selected is None
    assert {row["mask"] for row in candidates} == set(range(1, 7))


def test_analysis_detects_stable_split_physical_paths():
    ce = torch.ones(2, 4, subject.TOKENS, 8, dtype=torch.float64)
    positive = torch.zeros(4, subject.TOKENS, dtype=torch.bool)
    positive[:, 0] = True
    for mask in range(8):
        if mask & 1:
            ce[0, :, 0, mask] -= 1.0
        if mask & 4:
            ce[1, :, 0, mask] -= 1.0
    report = subject.analyze_phase(ce, positive, split_index=2)
    assert report["selected_masks"] == {"T": 1, "I": 4}
    assert report["pred_b_physical_path_predicts_route"] is True
    assert report["pred_c_path_decomposition_stable"] is True
    assert report["split_holds"] is True
    assert report["shared_holds"] is False
    assert report["relation"] == "split"
    assert report["mobius_closure_holds"] is True


def test_scaled_error_allows_a_frozen_scalar_not_only_unit_slope():
    predictor = torch.tensor([1.0, -2.0, 3.0])
    alpha, error = subject._scaled_error(predictor, 2.5 * predictor)
    assert abs(alpha - 2.5) < 1e-12
    assert error < 1e-12
