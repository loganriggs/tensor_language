import torch

import equality_query_circuit_fingerprint_rung475 as subject


def test_mask_from_indices():
    mask = subject._mask_from_indices([1, 3], 5)
    assert torch.equal(mask, torch.tensor([0, 1, 0, 1, 0]).bool())


def test_difficulty_residualization_removes_affine_relation():
    base = torch.arange(8, dtype=torch.float32)
    effect = 2 + 3 * base
    positive = torch.ones(8, dtype=torch.bool)
    residual, coefficients = subject.residualize_difficulty(effect, base, positive)
    assert float(residual.abs().max()) < 1e-5
    assert abs(coefficients[0] - 2) < 1e-5
    assert abs(coefficients[1] - 3) < 1e-5


def test_expected_call_formulas():
    assert subject.FORWARDS_PER_BATCH == 15
    assert subject.EXPECTED_FORWARDS == 3750
    assert subject.EXPECTED_PATCH_CALLS_PER_BATCH == 18
    assert subject.EXPECTED_PATCH_CALLS == 4500
