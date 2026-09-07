import pytest
import torch

from absolute_head_response_clamp_contract import (
    AbsoluteHeadResponseClampError,
    build_absolute_head_response_plan,
)


def spec(layer, heads, base, changed):
    return {
        "layer": layer,
        "selected_heads": heads,
        "base_capture": {"head_output": base},
        "changed_capture": {"head_output": changed},
    }


def test_absolute_plan_uses_changed_tensor_not_delta():
    base = torch.full((2, 3, 9, 4), 100.0)
    changed8 = torch.arange(base.numel()).reshape_as(base)
    changed15 = changed8 + 1000
    cache, support = build_absolute_head_response_plan(
        (spec(8, (1,), base, changed8), spec(15, tuple(range(9)), base, changed15)),
        complete_head_sites={15: tuple(range(9))},
    )
    assert support == ("L8H1", "attn:15")
    assert torch.equal(cache["head_layer:8"], changed8.reshape(2, 3, 36))
    assert torch.equal(cache["attn:15"], changed15.reshape(2, 3, 36))
    assert not torch.equal(cache["head_layer:8"], (changed8 - base).reshape(2, 3, 36))


@pytest.mark.parametrize("bad", ("duplicate", "unordered", "shape", "partial_complete"))
def test_absolute_plan_fails_closed(bad):
    base = torch.zeros(2, 3, 9, 4)
    changed = torch.ones_like(base)
    if bad == "duplicate":
        specs = (spec(8, (1,), base, changed), spec(8, (2,), base, changed))
    elif bad == "unordered":
        specs = (spec(9, (1,), base, changed), spec(8, (1,), base, changed))
    elif bad == "shape":
        specs = (spec(8, (1,), base, changed[:, :, :8]),)
    else:
        specs = (spec(15, (1,), base, changed),)
    with pytest.raises(AbsoluteHeadResponseClampError):
        build_absolute_head_response_plan(
            specs, complete_head_sites={15: tuple(range(9))}
        )
