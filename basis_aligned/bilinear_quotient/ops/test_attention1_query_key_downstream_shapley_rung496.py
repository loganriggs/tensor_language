from types import SimpleNamespace

import torch

import attention1_query_key_downstream_shapley_rung496 as subject


def _relative_squared(left, right):
    left = left.detach().double()
    right = right.detach().double()
    return float((left - right).square().sum()
                 / right.square().sum().clamp_min(1e-30))


def _synthetic(seed=496):
    generator = torch.Generator().manual_seed(seed)
    batch, length = 2, 3
    attention = SimpleNamespace(c_proj=torch.nn.Linear(subject.D, subject.D, bias=False))
    attention.c_proj.weight.data.copy_(torch.randn(
        attention.c_proj.weight.shape, generator=generator) / subject.D ** .5)
    shape = (batch, length, subject.HEADS, subject.HEAD_DIM)
    absent = tuple(torch.randn(shape, generator=generator) for _ in subject.FACTOR_NAMES)
    normal = tuple(
        value + .1 * torch.randn(shape, generator=generator) for value in absent)
    return generator, attention, normal, absent


def test_five_factor_shapley_and_mobius_close_exactly():
    _, attention, normal, absent = _synthetic()
    views, detail = subject.exact_factor_allocations(attention, normal, absent)

    assert set(views) == {"shapley", "first", "last"}
    for value in views.values():
        assert value.shape == (2, 3, 45, subject.D)
        assert value.dtype == torch.float32
    assert _relative_squared(
        detail["mobius_reconstruction"], detail["factor_delta"]) < 1e-10
    assert _relative_squared(
        detail["shapley_reconstruction"], detail["factor_delta"]) < 1e-10


def test_factor_first_and_last_are_literal_endpoint_marginals():
    _, attention, normal, absent = _synthetic(4961)
    views, _ = subject.exact_factor_allocations(attention, normal, absent)
    head, factor = 4, 2
    piece = head * len(subject.FACTOR_NAMES) + factor

    empty = subject._per_head_factor_writes(attention, absent)
    first_factors = list(absent)
    first_factors[factor] = normal[factor]
    first = subject._per_head_factor_writes(attention, first_factors)
    full = subject._per_head_factor_writes(attention, normal)
    last_factors = list(normal)
    last_factors[factor] = absent[factor]
    last = subject._per_head_factor_writes(attention, last_factors)

    assert _relative_squared(views["first"][:, :, piece], first[:, :, head] - empty[:, :, head]) < 1e-12
    assert _relative_squared(views["last"][:, :, piece], full[:, :, head] - last[:, :, head]) < 1e-12


def test_private_qk_rotation_does_not_change_allocated_raw_writes():
    generator, attention, normal, absent = _synthetic(4962)
    original, _ = subject.exact_factor_allocations(attention, normal, absent)
    matrix = torch.randn(subject.HEAD_DIM, subject.HEAD_DIM, generator=generator)
    rotation = torch.linalg.qr(matrix).Q
    rotated_normal = list(normal)
    rotated_absent = list(absent)
    # Apply the same orthogonal coordinate change to Q1 and K1 of one head.
    # qR dot kR = q dot k, so every complete attention arm is invariant.
    head = 3
    for factors in (rotated_normal, rotated_absent):
        for index in (0, 1):
            changed = factors[index].clone()
            changed[:, :, head] = changed[:, :, head] @ rotation
            factors[index] = changed
    rotated, _ = subject.exact_factor_allocations(
        attention, tuple(rotated_normal), tuple(rotated_absent))

    for name in original:
        assert _relative_squared(rotated[name], original[name]) < 1e-10
