from types import SimpleNamespace

import torch

import attention1_downstream_use_quotient_rung495 as subject
import mlp0_attention1_finite_path_factorial_rung484 as factor_parent


def _relative_squared(left, right):
    left = left.detach().double()
    right = right.detach().double()
    return float((left - right).square().sum()
                 / right.square().sum().clamp_min(1e-30))


def test_exact_factor_mobius_and_gradient_contraction():
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

    gradient = torch.randn(batch, length, subject.D, generator=generator)
    per_piece = torch.einsum("btd,btpd->p", gradient, pieces)
    complete = (gradient * detail["factor_delta"]).sum()
    assert float((per_piece.sum() - complete).abs().detach()) <= 2e-4 * max(
        float(complete.abs().detach()), 1.0)


def test_bf16_factor_inputs_use_registered_float32_algebra():
    generator = torch.Generator().manual_seed(49501)
    batch, length = 1, 2
    attention = SimpleNamespace(c_proj=torch.nn.Linear(subject.D, subject.D, bias=False))
    attention.c_proj.weight.data.copy_(torch.randn(
        attention.c_proj.weight.shape, generator=generator) / subject.D ** .5)
    normal = (
        torch.randn(batch, subject.HEADS, length, length, generator=generator).bfloat16(),
        torch.randn(batch, subject.HEADS, length, length, generator=generator).bfloat16(),
        torch.randn(
            batch, length, subject.HEADS, subject.HEAD_DIM,
            generator=generator).bfloat16(),
    )
    absent = tuple(
        (value.float() + .1 * torch.randn(value.shape, generator=generator)).bfloat16()
        for value in normal)
    pieces, detail = subject.exact_factor_pieces(attention, normal, absent)

    assert pieces.dtype == torch.float32
    assert detail["normal_write"].dtype == torch.float32
    assert _relative_squared(
        detail["reconstructed_delta"], detail["factor_delta"]) < 1e-10
    normal32 = tuple(value.float() for value in normal)
    absent32 = tuple(value.float() for value in absent)
    assert _relative_squared(
        detail["normal_write"], factor_parent._attention_write(attention, normal32)) < 1e-10
    assert _relative_squared(
        detail["absent_write"], factor_parent._attention_write(attention, absent32)) < 1e-10


def test_registered_analysis_selects_cross_head_pair_and_position_control():
    tags = [f"tag{index}" for index in range(32)]
    discovery = subject._empty_collection(tags, tuple(range(63)), (0,))
    discovery["counts"].fill_(1)
    base = torch.linspace(-2, 2, len(tags), dtype=torch.float64)
    branch_scale = torch.tensor([1.0, -1.0, 2.0, -2.0], dtype=torch.float64)
    for half in range(2):
        for branch in range(4):
            vector = branch_scale[branch] * base * (1 + .02 * half)
            discovery["sums"][half, branch, 0, :, 0, 0] = vector
            discovery["sums"][half, branch, 0, :, 0, 7] = 1.2 * vector
            discovery["complete_sums"][half, branch, 0] = 2.2 * vector
    preliminary = subject._preliminary_analysis(discovery)
    assert preliminary["selected_indices"] == [0, 7]
    assert preliminary["preliminary_holds"] is True

    controls = subject._empty_collection(tags, (0, 7), (0, *subject.POSITION_SHIFTS))
    controls["counts"].fill_(1)
    for half in range(2):
        for branch in range(4):
            vector = branch_scale[branch] * base * (1 + .02 * half)
            controls["sums"][half, branch, 0, :, 0, 0] = vector
            controls["sums"][half, branch, 0, :, 0, 1] = 1.2 * vector
            # Position shifts are deliberately orthogonal-ish alternating controls.
            for shift_index in range(1, 17):
                controls["sums"][half, branch, 0, :, shift_index, 1] = \
                    1.2 * vector.roll(shift_index)
            controls["complete_sums"][half, branch, 0] = 2.2 * vector
    position = subject._position_and_validation_report(controls, (0, 7))
    assert subject._b_holds(preliminary, position) is True
