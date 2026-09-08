import pytest
import torch

import residual_state_mediation_executor as executor


def test_boundary_order_is_exact_and_complete():
    assert len(executor.BOUNDARIES) == 13
    assert executor.BOUNDARIES[0] == "entry12"
    assert executor.BOUNDARIES[-1] == "post_mlp17"
    assert executor.BOUNDARIES[1:5] == (
        "post_attn12", "post_mlp12", "post_attn13", "post_mlp13")


def test_hybrid_state_uses_semantic_prefix_only():
    off = torch.zeros(1, 3, 2)
    on = torch.ones_like(off)
    expected = torch.tensor([[[1., 1.], [1., 1.], [0., 0.]]])
    assert torch.equal(executor.hybrid_state(off, on, (1,)), expected)


def test_apply_captures_native_and_replaces_absolutely():
    native = torch.zeros(1, 2, 3)
    fixed = torch.ones_like(native)
    captured = {}
    changed = executor._apply("entry12", native, {"entry12": fixed}, captured)
    assert torch.equal(captured["entry12"], native)
    assert torch.equal(changed, fixed)
    with pytest.raises(executor.ResidualStateError):
        executor._apply("entry12", native, {}, captured)
