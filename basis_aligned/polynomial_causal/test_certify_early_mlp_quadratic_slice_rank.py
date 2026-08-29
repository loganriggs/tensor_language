import pytest
import torch

import certify_early_mlp_quadratic_slice_rank as certificate


def test_one_product_symmetric_slice_has_rank_at_most_one(monkeypatch):
    monkeypatch.setattr(certificate, "NATIVE_PRODUCTS", 1)
    monkeypatch.setattr(certificate, "DIMENSION", 4)
    left = torch.tensor([[1.0, 2.0, 0.0, 1.0]])
    right = torch.tensor([[0.0, 1.0, 3.0, 2.0]])
    down = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    matrix = certificate.symmetric_slice(down, left, right)
    assert int(torch.linalg.matrix_rank(matrix)) <= 1


def test_shape_and_coordinate_fail_closed(monkeypatch):
    monkeypatch.setattr(certificate, "NATIVE_PRODUCTS", 1)
    monkeypatch.setattr(certificate, "DIMENSION", 2)
    with pytest.raises(ValueError):
        certificate.symmetric_slice(torch.ones(2, 1), torch.ones(1, 2),
                                    torch.ones(1, 3))
    with pytest.raises(ValueError):
        certificate.symmetric_slice(torch.ones(2, 1), torch.ones(1, 2),
                                    torch.ones(1, 2), coordinate=2)
