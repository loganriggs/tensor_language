import pytest
import torch

import normalized_weight_reader_contract as contract


def test_rms_jvp_matches_autograd_directional_derivative():
    state = torch.tensor([[1.0, -2.0, 3.0]], requires_grad=True)
    delta = torch.tensor([[.2, .1, -.3]])
    expected = torch.func.jvp(
        lambda value: torch.nn.functional.rms_norm(value, (3,), eps=torch.finfo(value.dtype).eps),
        (state,), (delta,))[1]
    actual = contract.rms_jvp(torch, state.detach(), delta)
    assert torch.allclose(actual, expected.detach(), atol=1e-6, rtol=1e-6)


def test_reader_response_distinguishes_raw_tangent_and_exact():
    matrix = torch.eye(3)
    state = torch.tensor([[1.0, 2.0, 4.0]])
    delta = torch.tensor([[.4, -.3, .2]])
    report = contract.reader_response(torch, matrix, state, delta)
    assert report["raw_strength"] == pytest.approx(1 / 3 ** .5)
    assert report["tangent_exact_cosine"] > .9
    assert report["tangent_exact_relative_l2"] > 0


def test_rank_and_spearman_are_label_deterministic():
    left = [{"label": "b", "exact_strength": 1.0}, {"label": "a", "exact_strength": 2.0}]
    right = list(reversed(left))
    ranked = contract.ranked(left, "exact_strength")
    assert [row["label"] for row in ranked] == ["a", "b"]
    assert contract.spearman(left, right, "exact_strength") == pytest.approx(1.0)


def test_shape_errors_fail_closed():
    with pytest.raises(contract.NormalizedReaderError):
        contract.rms_jvp(torch, torch.ones(2, 3), torch.ones(2, 4))
    with pytest.raises(contract.NormalizedReaderError):
        contract.reader_response(torch, torch.ones(2, 4), torch.ones(2, 3), torch.ones(2, 3))
