"""Data for the interactive induction-circuit diagram (atlas artifact).

Picks a clean depth-2-gated example (bigram repeat visible in a short window), then for
attn2-seed0 computes everything the vertical causal view needs:
  - per-token strings for the ribbon, with q (prediction pos), j (earlier occurrence of
    current token), j+1 (induction source) marked
  - baseline P(target) and per-head single-ablation P(target)  (8 heads)
  - per-head top attention sources at q (position, signed score)
  - for layer-1 heads: direct OV-path contribution to the target logit from source j+1,
    split as pattern(q→j+1) × OV(v[j+1]→logit[tgt]) — the sign-AGREEMENT (XNOR) datum
  - aggregate over ~2k induction positions: mean signed pattern×OV product per L1 head
    (the honest replacement for the misleading "negative attention score" story)

Run with TL_CORPUS=tiny. Writes runs_lm/circuit_demo.json.
"""

import json
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, einsum

from lm_eval import load_model
from induction_dynamics import induction_structure
from text_data import N_CTX, RUNS, load_tokenizer, val_windows

DEVICE = "cuda"
TAG = "attn2-seed0"


def head_forward(model, x, ablate=None):
    """Forward with optional (layer, head) zeroed; returns logits and per-layer
    (pattern, v, W_O) for analysis."""
    aux = []
    h = model.embed(x)
    for li, layer in enumerate(model.layers):
        pat = layer.pattern(h)
        v = rearrange(layer.v(layer.norm(h)), "b t (n d) -> b t n d", n=layer.n_head)
        z = einsum(pat, v, "b n q k, b k n d -> b q n d")
        if ablate is not None and ablate[0] == li:
            z = z.clone()
            z[:, :, ablate[1], :] = 0
        aux.append((pat, v))
        zc = rearrange(z, "b q n d -> b q (n d)")
        h = torch.lerp(h, layer.o(zc), layer.scale)
    return model.head(h), aux


def main():
    tok = load_tokenizer()
    data, n_win = val_windows()
    model = load_model(RUNS / TAG, None, DEVICE)
    dh = model.layers[0].d_head
    gated = set(np.load(RUNS / "gated_depth2.npy").tolist())

    # ---- find a display example: gated token, repeat within 60 tokens, clean words
    best = None
    for w in range(400):
        buf = torch.from_numpy(np.array(data[w * N_CTX:w * N_CTX + N_CTX + 1], dtype=np.int64))[None].to(DEVICE)
        j_last, is_ind = induction_structure(buf)
        for q in range(N_CTX - 1, 40, -1):
            flat = w * N_CTX + q
            j = int(j_last[0, q])
            if flat in gated and bool(is_ind[0, q]) and j > 0 and 8 < q - j < 55:
                with torch.no_grad():
                    logits, _ = head_forward(model, buf[:, :-1])
                p = torch.softmax(logits[0, q], -1)[buf[0, q + 1]].item()
                if p > 0.8 and (best is None or p > best[-1]):
                    best = (w, q, j, p)
        if best and best[-1] > 0.9:
            break
    w, q, j, p0 = best
    print(f"example: window {w} q={q} j={j} P={p0:.3f}")

    buf = torch.from_numpy(np.array(data[w * N_CTX:w * N_CTX + N_CTX + 1], dtype=np.int64))[None].to(DEVICE)
    x = buf[:, :-1]
    tgt = int(buf[0, q + 1])
    start = max(0, j - 12)
    ribbon = [tok.decode([int(t)]) for t in x[0, start:q + 1]]

    with torch.no_grad():
        logits, aux = head_forward(model, x)
        base_p = torch.softmax(logits[0, q], -1)[tgt].item()

        heads = {}
        W_U = model.head.weight                       # (V, d_model)
        for li in range(2):
            pat, v = aux[li]
            W_O = model.layers[li].o.weight           # (d, d)
            scale = model.layers[li].scale
            for hi in range(model.layers[li].n_head):
                al, _ = head_forward(model, x, ablate=(li, hi))
                ab_p = torch.softmax(al[0, q], -1)[tgt].item()
                row = pat[0, hi, q]                   # (T,) signed scores from q
                topk = torch.topk(row.abs(), 3).indices.tolist()
                tops = [{"pos": int(s), "tok": tok.decode([int(x[0, s])]),
                         "score": round(float(row[s]), 4)} for s in topk]
                d = {"layer": li, "head": hi, "ablated_p": round(ab_p, 4), "top": tops}
                if li == 1:
                    src = j + 1
                    ov = scale * (W_U[tgt] @ W_O[:, hi * dh:(hi + 1) * dh] @ v[0, src, hi])
                    d["src_score"] = round(float(row[src]), 4)
                    d["src_ov"] = round(float(ov), 4)
                    d["src_product"] = round(float(row[src]) * float(ov), 4)
                heads[f"L{li}H{hi}"] = d

        # ---- aggregate XNOR stat over many induction positions (first 100 windows)
        bufs = torch.from_numpy(np.stack([np.array(data[wi * N_CTX:wi * N_CTX + N_CTX + 1],
                                dtype=np.int64) for wi in range(100)])).to(DEVICE)
        jl, im = induction_structure(bufs)
        xx = bufs[:, :-1]
        logits2, aux2 = head_forward(model, xx)
        pat, v = aux2[1]
        W_O = model.layers[1].o.weight
        scale = model.layers[1].scale
        tgts = bufs[:, 1:]
        agg = {}
        bi, qi = torch.where(im)
        src = (jl[im] + 1)
        keep = src < qi                                # valid sources
        bi, qi, src = bi[keep], qi[keep], src[keep]
        t_ids = tgts[bi, qi]
        for hi in range(4):
            scores = pat[bi, hi, qi, src]                                # signed pattern
            ovs = scale * einsum(W_U[t_ids], W_O[:, hi * dh:(hi + 1) * dh],
                                 v[bi, src, hi], "n d, d h, n h -> n")
            prod = (scores * ovs)
            agg[f"L1H{hi}"] = {"mean_product": round(float(prod.mean()), 4),
                               "frac_positive": round(float((prod > 0).float().mean()), 3),
                               "mean_score": round(float(scores.mean()), 4),
                               "mean_ov": round(float(ovs.mean()), 4)}

    out = {"tag": TAG, "window": w, "q": q, "j": j, "start": start, "ribbon": ribbon,
           "target": tok.decode([tgt]), "base_p": round(base_p, 4), "heads": heads,
           "xnor": agg, "n_agg": int(len(bi))}
    (RUNS / "circuit_demo.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "ribbon"}, indent=1)[:1500])


if __name__ == "__main__":
    main()
