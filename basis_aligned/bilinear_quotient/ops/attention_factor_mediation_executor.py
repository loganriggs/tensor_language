"""Absolute Q/K/Q2/K2/V factor interventions under an entry12 state program."""

from __future__ import annotations


FACTORS = ("q", "k", "q2", "k2", "v")


class AttentionFactorError(ValueError):
    pass


def site(layer, head, factor):
    if factor not in FACTORS:
        raise AttentionFactorError("unknown attention factor")
    return f"L{int(layer)}H{int(head)}:{factor}"


def prefix_hybrid(background, source, semantic_positions):
    if background.shape != source.shape or background.ndim != 3:
        raise AttentionFactorError("factor tensors must have equal [row,token,width] shape")
    result = background.clone()
    for row, stop in enumerate(semantic_positions):
        stop = int(stop)
        if not 0 <= stop < background.shape[1]:
            raise AttentionFactorError("semantic position out of range")
        result[row, :stop + 1] = source[row, :stop + 1].to(result)
    return result


def mediator_sites(layer, head=None, factor=None, *, n_head=9):
    if head is None and factor is not None:
        raise AttentionFactorError("factor requires a head")
    heads = range(n_head) if head is None else (int(head),)
    factors = FACTORS if factor is None else (factor,)
    return tuple(site(layer, item, name) for item in heads for name in factors)


def execute(parent, backend, context, counters, *, entry12=None, factor_absolute=None, capture_layers=(12, 13)):
    torch, F, model = backend.torch, backend.F, backend.model
    factor_absolute = {} if factor_absolute is None else dict(factor_absolute)
    valid = {site(layer, head, factor) for layer in capture_layers
             for head in range(model.config.n_head) for factor in FACTORS}
    unknown = set(factor_absolute) - valid
    if unknown:
        raise AttentionFactorError(f"unknown factor sites: {sorted(unknown)}")
    batch = context["base_batch"]
    counters["differentiable_transformer_forwards"] += 1
    counters["example_evaluations"] += len(batch.row_ids)
    tokens, lengths = backend._tensor_batch(batch)
    captured, calls = {}, {name: 0 for name in valid}
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
        x0, v1 = x, None
        for layer, block in enumerate(model.transformer.h):
            if layer == 12 and entry12 is not None:
                if entry12.shape != x.shape:
                    raise AttentionFactorError("entry12 state shape mismatch")
                x = entry12.to(x)
            live = block.lambdas[0] * x + block.lambdas[1] * x0
            handles = []
            if layer in capture_layers:
                for factor in FACTORS:
                    module = getattr(block.attn, "c_" + factor)
                    def hook(_module, _arguments, output, factor=factor, layer=layer):
                        shaped = output.view(output.shape[0], output.shape[1], model.config.n_head, -1)
                        changed = shaped.clone()
                        for head in range(model.config.n_head):
                            name = site(layer, head, factor)
                            calls[name] += 1
                            captured[name] = shaped[:, :, head].detach().clone()
                            if name in factor_absolute:
                                replacement = factor_absolute[name]
                                if replacement.shape != captured[name].shape:
                                    raise AttentionFactorError(f"factor shape mismatch at {name}")
                                changed[:, :, head] = replacement.to(changed)
                        return changed.reshape_as(output)
                    handles.append(module.register_forward_hook(hook))
            try:
                attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
            finally:
                for handle in handles:
                    handle.remove()
            x = live + attention
            x = x + block.mlp(F.rms_norm(x, (model.config.n_embd,)))
        index = torch.arange(len(lengths), device=backend.device)
        position = torch.tensor([length - 1 for length in lengths], device=backend.device)
        final = x[index, position].float()
        logits = 30.0 * torch.tanh(model.lm_head(F.rms_norm(final, (model.config.n_embd,))) / 30.0)
    if set(captured) != valid or any(count != 1 for count in calls.values()):
        raise AttentionFactorError("factor capture coverage/call count changed")
    return {"state": final.detach(), "logits": logits.detach()}, captured


def absolute_for(background, source, names, semantic_positions):
    return {name: prefix_hybrid(background[name], source[name], semantic_positions) for name in names}
