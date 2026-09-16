import pytest
import torch

from .node import SCORE_SCALE, execute


def test_scale_and_composition():
    left = torch.tensor([[[1.0, 2.0]]])
    right = torch.tensor([[[3.0, -1.0]]])
    assert torch.equal(execute(left), left * SCORE_SCALE)
    assert torch.allclose(execute(left + right), execute(left) + execute(right))


def test_shape_failure():
    with pytest.raises(ValueError):
        execute(torch.zeros(2, 3))
