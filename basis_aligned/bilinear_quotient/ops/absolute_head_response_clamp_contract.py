"""Fail-closed plans for absolute native head-response clamps."""

# BQGATE: LIBRARY
from __future__ import annotations


class AbsoluteHeadResponseClampError(RuntimeError):
    pass


def build_absolute_head_response_plan(specs, *, complete_head_sites=None):
    """Return the cache/support plan for absolute, causally ordered head clamps.

    ``changed_capture['head_output']`` is the absolute tensor to install.  It is never
    interpreted as a delta to add to the live stream.  Complete-head sites may be mapped
    to native attention-module labels such as ``{15: (0, ..., 8)}``.
    """
    specs = tuple(specs)
    complete = {
        int(layer): tuple(int(head) for head in heads)
        for layer, heads in (complete_head_sites or {}).items()
    }
    layers = [int(spec["layer"]) for spec in specs]
    if not specs or layers != sorted(layers) or len(layers) != len(set(layers)):
        raise AbsoluteHeadResponseClampError("specs must name unique increasing layers")

    cache, support = {}, []
    for spec in specs:
        layer = int(spec["layer"])
        heads = tuple(int(head) for head in spec["selected_heads"])
        base = spec["base_capture"]["head_output"]
        changed = spec["changed_capture"]["head_output"]
        if (getattr(base, "ndim", None) != 4 or base.shape != changed.shape
                or not heads or len(heads) != len(set(heads))
                or any(head < 0 or head >= int(base.shape[2]) for head in heads)):
            raise AbsoluteHeadResponseClampError("head response shape or selection is invalid")
        flattened = changed.reshape(changed.shape[0], changed.shape[1], -1)
        if layer in complete:
            if heads != complete[layer] or heads != tuple(range(int(base.shape[2]))):
                raise AbsoluteHeadResponseClampError("complete attention site must select every head")
            cache[f"attn:{layer}"] = flattened
            support.append(f"attn:{layer}")
        else:
            cache[f"head_layer:{layer}"] = flattened
            support.extend(f"L{layer}H{head}" for head in heads)
    if len(support) != len(set(support)):
        raise AbsoluteHeadResponseClampError("support contains duplicate sites")
    return cache, tuple(support)
