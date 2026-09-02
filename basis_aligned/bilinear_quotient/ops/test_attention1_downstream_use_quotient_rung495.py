from types import SimpleNamespace

import torch

import attention1_downstream_use_quotient_rung495 as subject
import mlp0_attention1_finite_path_factorial_rung484 as factor_parent
import mlp1_finite_secant_factor_interchange_rung487 as secant_parent


def _relative_squared(left, right):
    left = left.detach().double()
    right = right.detach().double()
    return float((left - right).square().sum()
                 / right.square().sum().clamp_min(1e-30))


def test_exact_factor_mobius_and_mlp1_polarization():
    generator = torch.Generator().manual_seed(495)
    batch, length = 2, 3
    attention = SimpleNamespace(c_proj=torch.nn.Linear(subject.D, subject.D, bias=False))
    attention.c_proj.weight.data.copy_(torch.randn(
        attention.c_proj.weight.shape, generator=generator) / subject.D ** .5)
    normal = (
        torch.randn(batch, subject.HEADS, length, length, generator=generator),
        torch.randn(batch, subject.HEADS, length, length, generator=generator),
        torch.randn(batch, length, subject.HEADS, subject.HEAD_DIM, generator=generator),
    )
    absent = tuple(value + .1 * torch.randn(value.shape, generator=generator)
                   for value in normal)
    pieces, detail = subject.exact_factor_pieces(attention, normal, absent)

    assert pieces.shape == (batch, length, 63, subject.D)
    assert _relative_squared(
        detail["reconstructed_delta"], detail["factor_delta"]) < 1e-12
    assert _relative_squared(
        detail["normal_write"], factor_parent._attention_write(attention, normal)) < 1e-12
    assert _relative_squared(
        detail["absent_write"], factor_parent._attention_write(attention, absent)) < 1e-12

    hidden = 31
    mlp1 = SimpleNamespace(
        Left=torch.nn.Linear(subject.D, hidden, bias=False),
        Right=torch.nn.Linear(subject.D, hidden, bias=False),
        Down=torch.nn.Linear(hidden, subject.D, bias=False),
        Down_bias=torch.randn(subject.D, generator=generator),
    )
    direct = torch.randn(batch, length, subject.D, generator=generator)
    responses, complete = subject.exact_mlp1_piece_responses(
        mlp1, pieces, direct, detail["normal_write"], detail["absent_write"])
    expected = secant_parent._mlp_write(
        mlp1, direct + detail["normal_write"]) - secant_parent._mlp_write(
            mlp1, direct + detail["absent_write"])
    assert responses.shape == pieces.shape
    assert _relative_squared(responses.sum(2), complete) < 1e-12
    assert _relative_squared(complete, expected) < 1e-9
