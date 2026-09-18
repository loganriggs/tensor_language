#!/usr/bin/env python3
# BQGATE: LIBRARY -- MLP hidden-unit census and unit edits for one bilinear MLP block (review-22 item; the v164 / v165 bodies, once).
"""Two tools for opening an MLP port at unit grain.

`unit_census(backend, fw, rows, layer, reader, positions_of, batch=32)`: forward blocks 0..layer natively, capture the bilinear hidden
h = (Lx)*(Rx) at positions_of(row), and return per-row per-unit terms T_j = (reader . Down[:, j]) h_j together with the closure against
reader . mlp(x). Exact: sum_j T_j + reader . bias = reader . mlp(x). Pool with `pooled_contrast(rows, per_row, partner)`.

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
