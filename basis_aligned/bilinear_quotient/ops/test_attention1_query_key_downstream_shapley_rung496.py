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


def _passing_collection(tags, piece_indices, shifts):
    collection = subject._empty_collection(tags, piece_indices, shifts)
    collection["counts"].fill_(1)
    generator = torch.Generator().manual_seed(49603 + len(tags))
    base = torch.randn(len(tags), generator=generator, dtype=torch.float64)
    alternative = torch.randn(len(tags), generator=generator, dtype=torch.float64)
    alternative2 = torch.randn(len(tags), generator=generator, dtype=torch.float64)
    branch_scale = (1.0, -1.0, 2.0, -2.0)
    lookup = {global_index: local for local, global_index in enumerate(piece_indices)}
    pair = (0, 7)  # h0.Q1 and h1.Q2: query/query across heads.
    opposites = (subject._opposite(pair[0]), subject._opposite(pair[1]))
    for half in range(2):
        for branch, scale in enumerate(branch_scale):
            target = scale * base * (1 + .01 * half)
            for view, view_scale in enumerate((1.0, .8, 1.1)):
                if pair[0] in lookup:
                    collection["sums"][view, half, branch, 0, :, 0,
                                       lookup[pair[0]]] = view_scale * target
                if pair[1] in lookup:
                    collection["sums"][view, half, branch, 0, :, 0,
                                       lookup[pair[1]]] = 1.2 * view_scale * target
                if opposites[0] in lookup:
                    collection["sums"][view, half, branch, 0, :, 0,
                                       lookup[opposites[0]]] = scale * alternative
                if opposites[1] in lookup:
                    collection["sums"][view, half, branch, 0, :, 0,
                                       lookup[opposites[1]]] = scale * alternative2
            for shift_index in range(1, len(shifts)):
                if pair[1] in lookup:
                    for view, view_scale in enumerate((1.0, .8, 1.1)):
                        collection["sums"][view, half, branch, 0, :, shift_index,
                                           lookup[pair[1]]] = \
                            1.2 * view_scale * target.roll(shift_index)
            collection["complete_sums"][half, branch, 0] = 2.0 * target
            collection["head_sums"][half, branch, 0, :, 0, 0] = scale * alternative
            collection["head_sums"][half, branch, 0, :, 0, 1] = scale * alternative2
    return collection


def test_registered_scorer_selects_shared_query_side_and_rejects_controls():
    discovery_tags = [f"d{index}" for index in range(32)]
    discovery = _passing_collection(
        discovery_tags, tuple(range(len(subject.PIECE_NAMES))), (0,))
    preliminary = subject._preliminary_analysis(discovery)
    assert preliminary["selected_indices"] == [0, 7]
    assert preliminary["preliminary_holds"] is True
    assert preliminary["specificity"]["opposite_margin_holds"] is True

    position = _passing_collection(
        discovery_tags, (0, 7), (0, *subject.POSITION_SHIFTS))
    position_report = subject._position_and_validation_report(position, (0, 7))
    assert subject._b_holds(preliminary, position_report) is True

    validation_tags = [f"v{index}" for index in range(30)]
    validation_indices = (0, 7, subject._opposite(0), subject._opposite(7))
    validation = _passing_collection(
        validation_tags, validation_indices, (0, *subject.POSITION_SHIFTS))
    validation_report = subject._position_and_validation_report(validation, (0, 7))
    validation_specificity = subject._specificity_report(validation, (0, 7))
    assert subject._c_holds(validation_report, preliminary) is True
    assert subject._d_holds(
        preliminary["specificity"], validation_specificity) is True
    assert subject._next_step(True, True, True, True) == \
        "preregister_finite_query_key_input_side_interchange"
    assert subject._next_step(False, False, False, False) == \
        "repair_instrument_only_no_scientific_successor"
