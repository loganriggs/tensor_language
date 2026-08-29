from __future__ import annotations

import torch

from run_copy_edge_simple_gates import (
    _distance_prediction,
    _fit_affine,
    _fit_distance_table,
    _predict_affine,
    _r2,
)


def test_affine_fit_recovers_two_exact_outputs():
    score = torch.linspace(-2, 2, 17)
    target = torch.stack((3 * score + 2, -0.5 * score + 0.25), 1)
    coefficients = _fit_affine(score, target)
    prediction = _predict_affine(score, coefficients)
    assert torch.allclose(prediction, target, atol=1e-6, rtol=1e-6)
    assert all(value > 0.999999 for value in _r2(target, prediction))


def test_distance_table_uses_frozen_half_open_bins():
    distance = torch.tensor([1, 8, 9, 32, 33, 64, 65, 128])
    target = torch.stack((distance.float(), -distance.float()), 1)
    table, counts = _fit_distance_table(distance, target)
    prediction = _distance_prediction(distance, table)
    expected = torch.tensor([4.5, 20.5, 48.5, 96.5])
    assert counts == [2, 2, 2, 2]
    assert torch.equal(table[:, 0], expected)
    assert torch.equal(table[:, 1], -expected)
    assert torch.equal(prediction[:2], table[0].expand(2, -1))
    assert torch.equal(prediction[2:4], table[1].expand(2, -1))
    assert torch.equal(prediction[4:6], table[2].expand(2, -1))
    assert torch.equal(prediction[6:], table[3].expand(2, -1))

