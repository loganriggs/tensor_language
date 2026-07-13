"""Deep induction-circuit accounting for the atlas (v3 data). TL_CORPUS=tiny.

On the jewelry example (attn2-seed0):

1. CAUSAL MAP head × token: for every head and every ribbon position p, zero that head's
   output at p only → ΔP(target at q). Shows which heads matter AT which tokens.
2. OUTPUT DISTRIBUTION: top output tokens + probabilities for the full model and under
   each head's (full) ablation → bar-chart data.
3. WEIGHT-LEVEL COMPOSITION (bilinear analogue of Elhage et al.): for each L0 head A,
   OV_A = W_O[:,A] @ W_V[A,:]. For each L1 head B and each read branch
   W ∈ {Q1, K1, Q2, K2, V}: composition score
       C = ||W_B @ OV_A||_F / (||W_B||_F ||OV_A||_F).
   Bilinear attention has TWO key branches (K1, K2) — both are candidate match paths.
4. CAUSAL VERIFICATION of the composition table: for every (A, B) pair, zero A at the
   source position and record B's match weight pattern_B(q → src) before/after. High
   K-composition entries should collapse; low ones shouldn't.

Writes runs_lm/circuit_demo3.json.
"""

import json

import numpy as np
import torch
from einops import rearrange, einsum

from lm_eval import load_model
from text_data import N_CTX, RUNS, load_tokenizer, val_windows

DEVICE = "cuda"


def forward(model, x, zero=None):
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
    tok = load_tokenizer()
    demo = json.loads((RUNS / "circuit_demo.json").read_text())
    w, q, j, start = demo["window"], demo["q"], demo["j"], demo["start"]
    src = j + 1
    data, _ = val_windows()
    buf = torch.from_numpy(np.array(data[w * N_CTX:w * N_CTX + N_CTX + 1],
                           dtype=np.int64))[None].to(DEVICE)
    x = buf[:, :-1]
    tgt = int(buf[0, q + 1])
    model = load_model(RUNS / "attn2-seed0", None, DEVICE)
    dh = model.layers[0].d_head

    with torch.no_grad():
        logits, pats = forward(model, x)
        probs = torch.softmax(logits[0, q], -1)
        base_p = probs[tgt].item()

        # ---- 1. causal map: head x ribbon-position ----
        n_rib = q - start + 1
        cmap = np.zeros((8, n_rib), dtype=np.float32)
        for li in range(2):
            for hi in range(4):
                for rp in range(n_rib):
                    lg, _ = forward(model, x, [(li, hi, start + rp)])
                    p = torch.softmax(lg[0, q], -1)[tgt].item()
                    cmap[li * 4 + hi, rp] = base_p - p
        print("causal map done", flush=True)

        # ---- 2. output distributions ----
        top_full = torch.topk(probs, 6).indices.tolist()
        token_set = list(dict.fromkeys(top_full))
        dists = {"full": None}
        abl_probs = {}
        for li in range(2):
            for hi in range(4):
                lg, _ = forward(model, x, [(li, hi, None)])
                pr = torch.softmax(lg[0, q], -1)
                abl_probs[f"L{li}H{hi}"] = pr
                for t in torch.topk(pr, 3).indices.tolist():
                    if t not in token_set and len(token_set) < 10:
                        token_set.append(t)
        dists["full"] = [round(probs[t].item(), 4) for t in token_set]
        for k, pr in abl_probs.items():
            dists[k] = [round(pr[t].item(), 4) for t in token_set]
        out_tokens = [tok.decode([t]) for t in token_set]
        print("output dists done", flush=True)

        # ---- 3. weight composition ----
        L0, L1 = model.layers[0], model.layers[1]
        W_O0, W_V0 = L0.o.weight, L0.v.weight
        branches = {"Q1": L1.q1.weight, "K1": L1.k1.weight,
                    "Q2": L1.q2.weight, "K2": L1.k2.weight, "V": L1.v.weight}
        comp = {br: [[0.0] * 4 for _ in range(4)] for br in branches}
        for a in range(4):
            OV = W_O0[:, a * dh:(a + 1) * dh] @ W_V0[a * dh:(a + 1) * dh, :]
            for br, W in branches.items():
                for b in range(4):
                    Wb = W[b * dh:(b + 1) * dh, :]
                    c = (Wb @ OV).norm() / (Wb.norm() * OV.norm())
                    comp[br][a][b] = round(float(c), 4)
        print("weight composition done", flush=True)

        # ---- 4. causal verification: match-weight retention ----
        base_match = [round(float(pats[1][0, b, q, src]), 4) for b in range(4)]
        retain = [[0.0] * 4 for _ in range(4)]
        for a in range(4):
            _, pz = forward(model, x, [(0, a, src)])
            for b in range(4):
                retain[a][b] = round(float(pz[1][0, b, q, src]), 4)

    out = {"base_p": round(base_p, 4), "start": start, "q": q, "j": j, "src": src,
           "target": demo["target"],
           "causal_map": [[round(float(v), 4) for v in row] for row in cmap],
           "out_tokens": out_tokens, "dists": dists,
           "composition": comp, "base_match": base_match, "match_retention": retain}
    (RUNS / "circuit_demo3.json").write_text(json.dumps(out))
    # console summary
    print("\nK1/K2 composition (rows=L0 head, cols=L1 head):")
    for br in ("K1", "K2"):
        print(br, np.round(np.array(comp[br]), 3))
    print("base match weights (L1H0..3):", base_match)
    print("match retention after zeroing L0Ha@src (rows a, cols b):")
    print(np.round(np.array(retain), 3))
    print("\ncausal map row maxima:",
          {f"L{i//4}H{i%4}": round(float(np.abs(cmap[i]).max()), 3) for i in range(8)})


if __name__ == "__main__":
    main()
