#!/usr/bin/env python3
# BQGATE: LIBRARY -- shared exact writer fold (review-9 efficiency item): the v82 / v103 body, once.
"""`writer_fold(fw, model, rows, layer, head, comp, reader, positions_of, pattern, partner)`: at block `layer`, for head `head` of component
`comp` (reader direction r = V_h^T v_hat, `reader`), decompose the state the head reads at positions_of(row)[label] into the embedding, every
head of blocks < layer (c_proj column blocks), each block's c_proj bias and every MLP < layer, via the exact lambda recurrence; project on r,
scale by p_h(final, s) x (1 - lamb_h) / rms(live), and pool the oriented contrast over aligned pairs. Returns per-position reports with writer
shares, head / mlp / embedding parts and the recon closure. `pattern[(row_id)][pos]` supplies the native pattern scalars (from head_source_terms)."""
from __future__ import annotations
import aspectual_dod_lib as L


def contributions(tr, pos, upto):
    x0 = tr[("embed", pos)]; C = {"embed": x0.clone()}
    for l in range(upto):
        l0, l1 = tr[f"lambda0:{l:02d}"], tr[f"lambda1:{l:02d}"]
        for k in C:
            C[k] = l0 * C[k]
        C["embed"] = C["embed"] + l1 * x0
        heads = {f"attnhead:{l:02d}:{h}": tr[(f"attnhead:{l:02d}:{h}", pos)].clone() for h in range(9)}
        C.update(heads); C[f"attnbias:{l:02d}"] = tr[(f"attn:{l:02d}", pos)] - sum(heads.values()); C[f"mlp:{l:02d}"] = tr[(f"mlp:{l:02d}", pos)].clone()
    return C


def writers(layer):
    return ["embed"] + [f"attnhead:{l:02d}:{h}" for l in range(layer) for h in range(9)] + [f"attnbias:{l:02d}" for l in range(layer)] + [f"mlp:{l:02d}" for l in range(layer)]


def writer_fold(fw, model, rows, layer, head, reader, positions_of, pattern, partner, batch=32):
    """positions_of(row) -> {label: pos}; pattern[row_id][pos] -> scalar; partner[(construction, group, present)] -> row index."""
    W = writers(layer)
    block = model.transformer.h[layer]; l0_L, l1_L = float(block.lambdas[0]), float(block.lambdas[1]); lamb = float(block.attn.lamb)
    forwards, traces = 0, []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        traces.extend(L.forward_trace_positions(fw, chunk, lambda rw: list(positions_of(rw).values()), upto_layer=layer, head_write_layers=range(layer))); forwards += 1
    closure, per_row = 0.0, []
    for row, tr in zip(rows, traces):
        entry = {}
        for label, pos in positions_of(row).items():
            C = contributions(tr, pos, layer); C = {k: l0_L * v for k, v in C.items()}; C["embed"] = C["embed"] + l1_L * tr[("embed", pos)]
            true = tr[("live", pos)]; recon = sum(C.values()); closure = max(closure, float((recon - true).norm() / true.norm()))
            rms = float(true.pow(2).mean().sqrt()); scale = pattern[row.row_id][pos] * (1 - lamb) / rms
            rr = reader.to(true.device); entry[label] = {w: float(rr @ C[w]) * scale for w in W}
        per_row.append(entry)
    report = {}
    for label in positions_of(rows[0]):
        totals = {w: 0.0 for w in W}; contrast = 0.0
        for i, row in enumerate(rows):
            if not row.present:
                continue
            j = partner[(row.construction, row.group, False)]; a, b = per_row[i][label], per_row[j][label]
            for w in W:
                totals[w] += a[w] - b[w]
            contrast += sum(a.values()) - sum(b.values())
        shares = {w: totals[w] / contrast for w in W}
        heads = {w: s for w, s in shares.items() if w.startswith("attnhead")}; ranked = sorted(heads, key=lambda w: -abs(heads[w]))
        report[label] = {"contrast_total": contrast, "shares": shares, "head_part": sum(heads.values()), "mlp_part": sum(s for w, s in shares.items() if w.startswith("mlp")), "embed_share": shares["embed"],
                         "top_heads": [(w, heads[w]) for w in ranked[:8]], "mlp_by_block": {w: s for w, s in shares.items() if w.startswith("mlp")}}
    return {"closure_max": closure, "positions": report}, forwards
