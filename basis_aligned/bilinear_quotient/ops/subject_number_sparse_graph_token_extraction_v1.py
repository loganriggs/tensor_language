#!/usr/bin/env python3
"""Frozen token-to-effect evaluator for the subject-number sparse graph."""
from __future__ import annotations

PORTS = ("embedding_recurrence", "early_writes_0_3", "middle_writes_4_7", "mlp_8", "mlp_10")
MASKS = (8, 1, 4, 2, 24, 12, 16, 9, 20)
TERMS = ("mlp_8", "embedding_recurrence", "middle_writes_4_7", "early_writes_0_3",
         "mlp_8*mlp_10", "middle_writes_4_7*mlp_8", "mlp_10",
         "embedding_recurrence*mlp_8", "middle_writes_4_7*mlp_10")
PREDICTIVE_CORNERS = (0, 1, 2, 4, 8, 9, 12, 16, 20, 24)
LAYER = 11

def _capture(model, initial, torch, F):
    x = initial; x0 = initial; first = None; components = {("embedding", -1): initial.clone()}
    with torch.no_grad():
        for layer, block in enumerate(model.transformer.h):
            x = block.lambdas[0] * x + block.lambdas[1] * x0
            for key in components: components[key] = components[key] * block.lambdas[0]
            components[("embedding", -1)] += block.lambdas[1] * x0
            state = F.rms_norm(x, (x.shape[-1],))
            if layer == LAYER:
                zero = torch.zeros_like(x)
                upstream = sum((v for (kind, idx), v in components.items() if kind == "embedding" or idx <= 7), zero).double()
                embedding = components[("embedding", -1)].double()
                early = sum((v for (kind, idx), v in components.items() if kind != "embedding" and idx in range(0, 4)), zero).double()
                middle_explicit = sum((v for (kind, idx), v in components.items() if kind != "embedding" and idx in range(4, 8)), zero).double()
                middle = upstream - embedding - early
                explicit8 = [upstream]
                explicit8.extend(components[(kind, idx)].double() for idx in range(8, 11) for kind in ("attn", "mlp"))
                mlp10 = x.double() - sum(explicit8[:-1])
                ports = [embedding, early, middle, explicit8[2], mlp10]
                audit = {"aggregation_max_abs_error": float((sum(ports[:3]) - upstream).abs().max()),
                    "middle_gauge_correction_relative_l2": float((middle - middle_explicit).norm() / middle_explicit.norm().clamp_min(1e-30)),
                    "mlp10_gauge_correction_relative_l2": float((mlp10 - explicit8[-1]).norm() / explicit8[-1].norm().clamp_min(1e-30))}
                return x, x0, first, ports, audit
            attention, first = block.attn(state, first); x = x + attention; components[("attn", layer)] = attention
            mlp = block.mlp(F.rms_norm(x, (x.shape[-1],))); x = x + mlp; components[("mlp", layer)] = mlp
    raise RuntimeError("layer 11 not reached")

def _suffix_margin(model, raw, x0, first, positions, answer_pairs, torch, F):
    with torch.no_grad():
        x = raw
        for layer in range(LAYER, len(model.transformer.h)):
            block = model.transformer.h[layer]
            if layer > LAYER: x = block.lambdas[0] * x + block.lambdas[1] * x0
            attention, first = block.attn(F.rms_norm(x, (x.shape[-1],)), first); x = x + attention
            x = x + block.mlp(F.rms_norm(x, (x.shape[-1],)))
        logits = 30. * torch.tanh(model.lm_head(F.rms_norm(x, (x.shape[-1],))) / 30.)
        batch = torch.arange(len(raw), device=raw.device)
        selected = logits[batch[:, None], positions[:, None], answer_pairs]
        return (selected[:, 0] - selected[:, 1]).double()

def native_margin(model, tokens, positions, answer_pairs, torch, F):
    """Independent native full-model margin used only for extraction audits."""
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)).float(); x0 = x; first = None
        for block in model.transformer.h:
            x = block.lambdas[0] * x + block.lambdas[1] * x0
            attention, first = block.attn(F.rms_norm(x, (x.shape[-1],)), first); x = x + attention
            x = x + block.mlp(F.rms_norm(x, (x.shape[-1],)))
        logits = 30. * torch.tanh(model.lm_head(F.rms_norm(x, (x.shape[-1],))) / 30.)
        batch = torch.arange(len(tokens), device=tokens.device)
        selected = logits[batch[:, None], positions[:, None], answer_pairs]
        return (selected[:, 0] - selected[:, 1]).double()

def evaluate(model, tokens, positions, answer_pairs, decoder_axis, threshold, torch, F, include_target=True):
    """Return the frozen nine-edge prediction from token inputs and checkpoint weights."""
    if tokens.ndim != 2 or positions.ndim != 1 or answer_pairs.shape != (len(tokens), 2):
        raise ValueError("token/position/answer shape mismatch")
    batch = torch.arange(len(tokens), device=tokens.device)
    with torch.no_grad(): initial = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)).float()
    axis = decoder_axis.double(); unit = axis / axis.norm(); subject = initial[batch, positions].double()
    projection = subject @ unit; target_projection = float(threshold) / float(axis.norm())
    orthogonal = subject - projection[:, None] * unit
    scale = torch.sqrt((subject.square().sum(1) - target_projection ** 2) / orthogonal.square().sum(1))
    removed_subject = target_projection * unit + scale[:, None] * orthogonal
    removed = initial.clone(); removed[batch, positions] = removed_subject.float()
    raw_b, x0_b, first_b, ports_b, audit_b = _capture(model, initial, torch, F)
    _raw_r, _x0_r, _first_r, ports_r, audit_r = _capture(model, removed, torch, F)
    deltas = [(ports_r[i] - ports_b[i])[batch, positions] for i in range(5)]
    corner_masks = list(PREDICTIVE_CORNERS) + ([31] if include_target else [])
    values = {}
    for mask in corner_masks:
        edited = raw_b.clone()
        delta = sum(deltas[i] for i in range(5) if mask & (1 << i)) if mask else torch.zeros_like(deltas[0])
        edited[batch, positions] += delta.float()
        values[mask] = _suffix_margin(model, edited, x0_b, first_b, positions, answer_pairs, torch, F)
    atoms = []
    for mask in MASKS:
        bits = [1 << i for i in range(5) if mask & (1 << i)]
        if len(bits) == 1: dividend = values[mask] - values[0]
        elif len(bits) == 2: dividend = values[mask] - values[bits[0]] - values[bits[1]] + values[0]
        else: raise ValueError("frozen graph contains a higher-order term")
        atoms.append(-dividend)
    atoms = torch.stack(atoms); prediction = atoms.sum(0)
    result = {"prediction": prediction, "atoms": atoms, "base_margin": values[0],
        "ports": PORTS, "masks": MASKS, "terms": TERMS,
        "audit": {"external_activation_inputs": 0, "internally_derived_ports": len(PORTS),
            "edges": len(MASKS), "predictive_corners": len(PREDICTIVE_CORNERS),
            "partial_forwards": 2, "suffix_forwards": len(corner_masks),
            "aggregation_max_abs_error": max(audit_b["aggregation_max_abs_error"], audit_r["aggregation_max_abs_error"]),
            "gauge_correction_max_relative_l2": max(audit_b["middle_gauge_correction_relative_l2"],
                audit_b["mlp10_gauge_correction_relative_l2"], audit_r["middle_gauge_correction_relative_l2"],
                audit_r["mlp10_gauge_correction_relative_l2"])}}
    if include_target: result["target"] = values[0] - values[31]
    return result
