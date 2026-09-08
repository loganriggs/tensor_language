"""Manual v15 execution with absolute complete-residual boundary interventions."""

from __future__ import annotations

import module_write_mediation_contract as write_contract


BOUNDARIES = ("entry12",) + tuple(
    f"post_{kind}{layer}" for layer in range(12, 18) for kind in ("attn", "mlp"))


class ResidualStateError(ValueError):
    pass


def hybrid_state(background, source, semantic_positions):
    return write_contract.prefix_hybrid(background, source, semantic_positions)


def _apply(name, state, absolute, captured):
    if name in captured:
        raise ResidualStateError(f"boundary executed twice: {name}")
    captured[name] = state.detach().clone()
    if name not in absolute:
        return state
    changed = absolute[name]
    if changed.shape != state.shape:
        raise ResidualStateError(f"absolute state shape mismatch at {name}")
    return changed.to(state)


def execute(parent, backend, context, counters, bases, *, absolute=None, capture=False):
    """Execute the parent route, optionally replacing complete residual states.

    `absolute` may contain any frozen subset of BOUNDARIES. Captures always record
    the native pre-replacement value, allowing a coherent all-boundary self replay.
    """
    torch, F, model = backend.torch, backend.F, backend.model
    absolute = {} if absolute is None else dict(absolute)
    unknown = set(absolute) - set(BOUNDARIES)
    if unknown:
        raise ResidualStateError(f"unknown residual boundaries: {sorted(unknown)}")
    batch = context["base_batch"]
    counters["differentiable_transformer_forwards"] += 1
    counters["example_evaluations"] += len(batch.row_ids)
    tokens, lengths = backend._tensor_batch(batch)
    n_head = int(model.config.n_head)
    width = int(model.config.n_embd // model.config.n_head)
    selected = bases or {}
    captured = {}
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
        x0, v1 = x, None
        for layer, block in enumerate(model.transformer.h):
            if layer == 12:
                x = _apply("entry12", x, absolute, captured)
            live = block.lambdas[0] * x + block.lambdas[1] * x0
            layer_sites = tuple(
                (head, f"L{layer}H{head}") for head in parent.HEADS_BY_LAYER.get(layer, ())
                if f"L{layer}H{head}" in selected)

            def patch(_module, arguments, layer_sites=layer_sites, layer=layer):
                flattened = arguments[0]
                changed = flattened.clone().view(
                    len(batch.row_ids), flattened.shape[1], n_head, width)
                if layer_sites:
                    base = context["base_cache"][f"head_layer:{layer}"].view_as(changed).to(changed)
                    donor = context["donor_cache"][f"head_layer:{layer}"].view_as(changed).to(changed)
                    raw_heads = {head: selected[site] for head, site in layer_sites}
                    fixed = parent.absolute_projected_head_response(torch, base, donor, raw_heads)
                    for row, stop in enumerate(batch.semantic_positions):
                        for head, _site in layer_sites:
                            changed[row, :int(stop) + 1, head] = fixed[
                                row, :int(stop) + 1, head].to(changed)
                return (changed.reshape_as(flattened),) + tuple(arguments[1:])

            handle = block.attn.c_proj.register_forward_pre_hook(patch) if layer_sites else None
            try:
                attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
            finally:
                if handle is not None:
                    handle.remove()
            x = live + attention
            if 12 <= layer <= 17:
                x = _apply(f"post_attn{layer}", x, absolute, captured)
            x = x + block.mlp(F.rms_norm(x, (model.config.n_embd,)))
            if 12 <= layer <= 17:
                x = _apply(f"post_mlp{layer}", x, absolute, captured)
        index = torch.arange(len(lengths), device=backend.device)
        position = torch.tensor([length - 1 for length in lengths], device=backend.device)
        state = x[index, position].float()
        logits = 30.0 * torch.tanh(
            model.lm_head(F.rms_norm(state, (model.config.n_embd,))) / 30.0)
    if set(captured) != set(BOUNDARIES):
        raise ResidualStateError("not every registered boundary executed exactly once")
    result = {"state": state.detach(), "logits": logits.detach()}
    return (result, captured) if capture else result


def boundary_cells(parent, backend, context, counters, bases, off_output, off_states,
                   on_output, on_states, boundary):
    if boundary not in BOUNDARIES:
        raise ResidualStateError(f"unknown boundary: {boundary}")
    positions = context["base_batch"].semantic_positions
    rescue = hybrid_state(off_states[boundary], on_states[boundary], positions)
    reset = hybrid_state(on_states[boundary], off_states[boundary], positions)
    return {
        "00": off_output,
        "01": execute(parent, backend, context, counters, {}, absolute={boundary: rescue}),
        "10": execute(parent, backend, context, counters, bases, absolute={boundary: reset}),
        "11": on_output,
    }
