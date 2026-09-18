#!/usr/bin/env python3
# BQGATE: LIBRARY -- MLP hidden-unit census and unit edits for one bilinear MLP block (review-22 item; the v164 / v165 bodies, once).
"""Two tools for opening an MLP port at unit grain.

`unit_census(backend, fw, rows, layer, reader, positions_of, batch=32)`: forward blocks 0..layer natively, capture the bilinear hidden
h = (Lx)*(Rx) at positions_of(row), and return per-row per-unit terms T_j = (reader . Down[:, j]) h_j together with the closure against
reader . mlp(x). Exact: sum_j T_j + reader . bias = reader . mlp(x). Pool with `pooled_contrast(rows, per_row, partner)`.

`product_unit_census(backend, fw, rows, src_layers, dst_layer, unit, positions_of, batch=32)`: the sign-carrying census (v177 lesson; the v182
body, once): for a bilinear unit u = (L . x)(R . x) of block dst_layer at positions_of(row), and every hidden unit j of each block in src_layers,
the exact leave-one-unit-out change D_j = u(x) - u(x - c_j) with c_j = (prod of lambda0 over blocks src+1..dst) h_j Down[:, j] the unit's write as
it reaches x (rms held). Returns {src: per_row}, closure (linear parts + bias terms vs (L . m)(R . x) + (L . x)(R . m)), forwards. One forward per batch
for all sources.

`forward_margins(backend, fw, rows, layer, edits, readers)`: a plain forward that matches the producer (rms -> blocks -> rms -> 30 tanh)
with the chosen hidden units of block `layer` zeroed at positions_fn(row) (None = every position); returns per-row answer / foil logits and
reader margins at row.final. edits = (units, positions_fn) or None.

Both work on bilin18's ungated MLP (mlp(x) = Down(Left(x) * Right(x)) + Down_bias); the gated variant is handled by `hidden()`."""
from __future__ import annotations
import aspectual_dod_lib as L


def hidden(model, mlp, xin):
    Lx, Rx = mlp.Left(xin), mlp.Right(xin)
    F = __import__("torch").nn.functional
    return (F.silu(Lx) * Rx) if getattr(model.config, "gated", False) else (Lx * Rx)


def unit_census(backend, fw, rows, layer, reader, positions_of, batch=32):
    torch, F, model = backend.torch, backend.F, backend.model
    mlp = model.transformer.h[layer].mlp
    Dw, bias = mlp.Down.weight.detach().float(), mlp.Down_bias.detach().float()
    rD = reader.to(Dw.device).float() @ Dw; r_bias = float(reader.to(bias.device).float() @ bias)
    per_row, closure, forwards = [], 0.0, 0
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]; tokens = fw._tokens(chunk)
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
            for l, block in enumerate(model.transformer.h):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
                x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l == layer:
                    h = hidden(model, mlp, xin); out = mlp(xin)
                    for i, row in enumerate(chunk):
                        entry = {}
                        for label, pos in positions_of(row).items():
                            T = rD * h[i, pos].float()
                            true = float(reader.to(out.device).float() @ out[i, pos].float()); recon = float(T.sum()) + r_bias
                            closure = max(closure, abs(recon - true) / max(abs(true), 1e-6)); entry[label] = T.cpu()
                        per_row.append(entry)
                    break
                x = x + block.mlp(xin)
            forwards += 1
    return per_row, closure, forwards


def product_unit_census(backend, fw, rows, src_layers, dst_layer, unit, positions_of, batch=32):
    torch, F, model = backend.torch, backend.F, backend.model
    blocks = model.transformer.h; dst = blocks[dst_layer].mlp
    Lrow, Rrow = dst.Left.weight.detach().float()[unit], dst.Right.weight.detach().float()[unit]
    srcs = {}
    for s in src_layers:
        m = blocks[s].mlp; Dw, b = m.Down.weight.detach().float(), m.Down_bias.detach().float()
        scale = 1.0
        for l in range(s + 1, dst_layer + 1): scale *= float(blocks[l].lambdas[0])
        srcs[s] = (m, scale, scale * (Lrow @ Dw), scale * (Rrow @ Dw), scale * float(Lrow @ b), scale * float(Rrow @ b), Dw, b)
    per_row, closure, forwards = {s: [] for s in src_layers}, 0.0, 0
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]; tokens = fw._tokens(chunk)
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_, hs = x, None, {}
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
                x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l in srcs: hs[l] = hidden(model, srcs[l][0], xin).float()
                if l == dst_layer: xd = x.float(); break
                x = x + block.mlp(xin)
            forwards += 1
            for i, row in enumerate(chunk):
                for s, (m, scale, lD, rD, lb, rb, Dw, b) in srcs.items():
                    entry = {}
                    for label, pos in positions_of(row).items():
                        xi = xd[i, pos].to(Lrow.device); hi = hs[s][i, pos].to(Lrow.device); rms2 = float(xi.pow(2).mean())
                        Lx, Rx = float(Lrow @ xi), float(Rrow @ xi); lc, rc = lD * hi, rD * hi
                        lin = lc * Rx + Lx * rc; D = (lin - lc * rc) / rms2
                        mw = scale * (Dw @ hi + b); true = float(Lrow @ mw) * Rx + Lx * float(Rrow @ mw); recon = float(lin.sum()) + lb * Rx + Lx * rb
                        closure = max(closure, abs(recon - true) / max(abs(true), 1e-6)); entry[label] = D.cpu()
                    per_row[s].append(entry)
    return per_row, closure, forwards


def pooled_contrast(rows, per_row, partner, label):
    """Oriented (present - partner) sum of per-unit terms over aligned pairs; partner[(construction, group, present)] -> row index."""
    import torch
    acc = None
    for i, row in enumerate(rows):
        if not row.present: continue
        j = partner[(row.construction, row.group, False)]; d = per_row[i][label] - per_row[j][label]
        acc = d.clone() if acc is None else acc + d
    return acc


def forward_margins(backend, fw, rows, layer, edits, readers):
    torch, F, model = backend.torch, backend.F, backend.model
    tokens = fw._tokens(rows); mlp = model.transformer.h[layer].mlp
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for l, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0
            attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
            x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if l == layer and edits is not None:
                units, positions_fn = edits; h = hidden(model, mlp, xin); idx = torch.tensor(list(units), device=h.device)
                for i, row in enumerate(rows):
                    pos = positions_fn(row)
                    if pos is None: h[i, :, idx] = 0
                    else: h[i, pos, idx] = 0
                x = x + mlp.Down(h) + mlp.Down_bias
            else:
                x = x + block.mlp(xin)
        logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
    out = []
    for i, row in enumerate(rows):
        lg = logits[i, row.final].float()
        out.append({"answer": float(lg[row.answer_id]), "foil": float(lg[row.foil_id]), **{name: float(lg[a] - lg[b]) for name, (a, b) in readers.items()}})
    return out
