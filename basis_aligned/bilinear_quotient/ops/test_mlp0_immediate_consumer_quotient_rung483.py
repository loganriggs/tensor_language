import torch

import mlp0_immediate_consumer_quotient_rung483 as subject


def test_held_scale_error_transfers_exact_scalar_relation():
    fit = torch.tensor([[4.0, 0.0, 6.0], [0.0, 1.0, 0.0], [6.0, 0.0, 9.0]])
    held = torch.tensor([[16.0, 0.0, 24.0], [0.0, 1.0, 0.0], [24.0, 0.0, 36.0]])
    alpha, error = subject._held_scale_error(fit, held, 0, 2)
    assert alpha == 1.5
    assert error == 0.0


def test_scaled_error_ignores_gain_but_not_rotation():
    predictor = torch.tensor([1.0, 2.0, 3.0])
    target = 7.0 * predictor
    dot = float(torch.dot(predictor, target))
    assert subject._scaled_error_from_sums(
        dot, float(predictor.square().sum()), float(target.square().sum())) < 1e-7
    orthogonal = torch.tensor([2.0, -1.0, 0.0])
    assert subject._scaled_error_from_sums(
        float(torch.dot(predictor, orthogonal)),
        float(predictor.square().sum()), float(orthogonal.square().sum())) > .99


def test_gram_cosines_and_relation_categories_are_mutually_exclusive():
    gram = torch.tensor([[4.0, 6.0], [6.0, 9.0]])
    cosine = subject._gram_cosines(gram)
    assert torch.allclose(cosine, torch.ones_like(cosine))
    assert .85 > .65 and .90 > .55


def test_position_shuffle_offsets_are_nonzero_unique_and_in_range():
    assert len(subject.POSITION_SHIFTS) == 16
    assert len(set(subject.POSITION_SHIFTS)) == 16
    assert min(subject.POSITION_SHIFTS) >= 1
    assert max(subject.POSITION_SHIFTS) <= 255


def test_subset_nested_respects_batch_split():
    values = [(
        torch.arange(4 * 3).view(4, 3),
        torch.arange(4 * 3).view(4, 3) + 100,
        torch.arange(4 * 3).view(4, 3) + 200,
    )]
    selected = subject._subset_nested(values, torch.tensor([False, True, True, False]))
    assert selected[0][0].shape == (2, 3)
    assert selected[0][0][:, 0].tolist() == [3, 6]
