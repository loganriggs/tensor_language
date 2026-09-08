import torch

import entry12_response_shape_router_contract as shape


def states():
    torch.manual_seed(7)
    off = torch.randn(3, 5, 8)
    return off, {"A1": off + torch.randn_like(off), "A2": off + torch.randn_like(off)}, (2, 3, 4)


def test_shape_and_finiteness_across_variable_prefixes():
    off, experts, positions = states()
    value = shape.features(torch, off, experts, positions)
    assert value.shape == (3, 14)
    assert torch.isfinite(value).all()


def test_simultaneous_response_sign_is_invariant():
    off, experts, positions = states()
    changed = {name: off - (state - off) for name, state in experts.items()}
    assert torch.allclose(shape.features(torch, off, experts, positions),
                          shape.features(torch, off, changed, positions), atol=1e-6)


def test_common_hidden_rotation_is_invariant():
    off, experts, positions = states()
    q = torch.linalg.qr(torch.randn(8, 8)).Q
    rotated_off = off @ q
    rotated = {name: value @ q for name, value in experts.items()}
    assert torch.allclose(shape.features(torch, off, experts, positions),
                          shape.features(torch, rotated_off, rotated, positions), atol=2e-6)


def test_bad_shapes_and_positions_fail_closed():
    off, experts, positions = states()
    try:
        shape.features(torch, off, {"A1": experts["A1"]}, positions)
    except shape.ResponseShapeError:
        pass
    else:
        raise AssertionError("missing expert accepted")
    try:
        shape.features(torch, off, experts, (2, 3, 5))
    except shape.ResponseShapeError:
        pass
    else:
        raise AssertionError("bad semantic position accepted")
