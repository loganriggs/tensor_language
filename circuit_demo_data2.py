"""Position-specific circuit data for the induction diagram (v2).

On the jewelry example (circuit_demo.json), for attn2-seed0:
  - per-L0-head attention AT THE SOURCE position j+1 (prev-token behavior = attends j)
    and at the query position q
  - POSITION-SPECIFIC ablations: zero head h's output only at position p ∈ {j+1, q}
    (vs everywhere) → P(target at q). Connects each head to the text region it serves:
    the prev-token head matters at j+1 (builds the induction key); the induction head
    matters at q (does match-and-copy).
  - K-composition check: how much of L1H2's key at j+1 comes through L0 heads — measured
    by zeroing each L0 head at j+1 and reading L1H2's pattern weight q→j+1.

Writes runs_lm/circuit_demo2.json. Run with TL_CORPUS=tiny.
"""

import json

import numpy as np
import torch
from einops import rearrange, einsum

from lm_eval import load_model
from text_data import N_CTX, RUNS, val_windows

DEVICE = "cuda"


def forward(model, x, zero=None):
    """zero = list of (layer, head, position) — zero that head's output at that position
    only; position=None → everywhere. Returns logits and per-layer patterns."""
    zero = zero or []
    pats = []
    h = model.embed(x)
    for li, layer in enumerate(model.layers):
        pat = layer.pattern(h)
        v = rearrange(layer.v(layer.norm(h)), "b t (n d) -> b t n d", n=layer.n_head)
        z = einsum(pat, v, "b n q k, b k n d -> b q n d")
        for (zl, zh, zp) in zero:
            if zl == li:
                z = z.clone()
                if zp is None:
                    z[:, :, zh, :] = 0
                else:
                    z[:, zp, zh, :] = 0
        pats.append(pat)
        h = torch.lerp(h, layer.o(rearrange(z, "b q n d -> b q (n d)")), layer.scale)
    return model.head(h), pats


def main():
    demo = json.loads((RUNS / "circuit_demo.json").read_text())
    w, q, j = demo["window"], demo["q"], demo["j"]
    src = j + 1
    data, _ = val_windows()
    buf = torch.from_numpy(np.array(data[w * N_CTX:w * N_CTX + N_CTX + 1],
                           dtype=np.int64))[None].to(DEVICE)
    x = buf[:, :-1]
    tgt = int(buf[0, q + 1])
    model = load_model(RUNS / "attn2-seed0", None, DEVICE)

    def ptgt(zero=None):
        logits, _ = forward(model, x, zero)
        return round(torch.softmax(logits[0, q], -1)[tgt].item(), 4)

    with torch.no_grad():
        base = ptgt()
        _, pats = forward(model, x)

        out = {"base_p": base, "q": q, "j": j, "src": src, "heads": {}}
        for li in range(2):
            for hi in range(4):
                name = f"L{li}H{hi}"
                row_src = pats[li][0, hi, src]
                row_q = pats[li][0, hi, q]
                d = {
                    "p_zero_at_src": ptgt([(li, hi, src)]),
                    "p_zero_at_q": ptgt([(li, hi, q)]),
                    "p_zero_everywhere": ptgt([(li, hi, None)]),
                    "attn_at_src": {"to_j": round(float(row_src[j]), 3),
                                    "to_self": round(float(row_src[src]), 3),
                                    "top": int(row_src.abs().argmax())},
                    "attn_at_q": {"to_src": round(float(row_q[src]), 3),
                                  "to_j": round(float(row_q[j]), 3),
                                  "top": int(row_q.abs().argmax())},
                }
                out["heads"][name] = d

        # K-composition: L1H2's pattern weight q->src after zeroing each L0 head at src
        base_w = float(pats[1][0, 2, q, src])
        out["l1h2_qsrc_weight"] = round(base_w, 4)
        comp = {}
        for hi in range(4):
            _, pz = forward(model, x, [(0, hi, src)])
            comp[f"L0H{hi}"] = round(float(pz[1][0, 2, q, src]), 4)
        out["l1h2_qsrc_weight_after_L0_zero_at_src"] = comp
        # both key-relevant L0 heads at once
        _, pz = forward(model, x, [(0, 2, src), (0, 3, src)])
        out["l1h2_qsrc_weight_after_L0H2H3_zero"] = round(float(pz[1][0, 2, q, src]), 4)
        out["p_after_L0H2H3_zero_at_src"] = ptgt([(0, 2, src), (0, 3, src)])

    (RUNS / "circuit_demo2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
