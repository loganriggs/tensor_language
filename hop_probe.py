"""Per-hop LINEAR PROBES: hop_residual_track showed intermediate hops aren't decodable via the
OUTPUT head, but they may live in a ROTATED basis. Train a linear probe on each layer's residual
(at the answer position) to predict f^k(e) for each k. If layer L best-decodes f^L (accuracy rising
along the diagonal), the model advances a "current entity" pointer layer-by-layer in a hidden basis
-> chained-retrieval mechanism reverse-engineered. If only the final layer decodes f^3, the
intermediate entities aren't linearly present at all (computed more holistically).

Run: python hop_probe.py
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from hop_data import sample_docs, E, ANS_OFFSET
from hop_circuit import load, fmap_from_bindings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_DOCS = 400


def collect(name="attn4-rms-seed0"):
    m = load(name)
    g = torch.Generator().manual_seed(555)
    reps, chains = [], []
    for _ in range(N_DOCS):
        tokens, qa, qk = sample_docs(1, g)
        tok = tokens[0]
        f = fmap_from_bindings(tok)
        blks = [j for j in range(len(qk[0])) if int(qk[0][j]) == 3]
        if not blks:
            continue
        x = m.embed(tok.unsqueeze(0).to(DEVICE))
        stream = [x]
        for layer in m.layers:
            x = layer(x); stream.append(x)
        for blk in blks:
            base = 2 * E + 4 * blk
            e = int(tok[base + 1]); ans_pos = base + ANS_OFFSET
            chain = [e, f[e], f[f[e]], f[f[f[e]]]]
            reps.append(torch.stack([s[0, ans_pos] for s in stream]).detach().cpu().numpy())
            chains.append(chain)
    return np.array(reps), np.array(chains)                    # (N, n_stream, d), (N, 4)


def run(name="attn4-rms-seed0"):
    reps, chains = collect(name)
    N, S, d = reps.shape
    ntr = int(0.7 * N)
    labels = ["embed", "L0", "L1", "L2", "L3"][:S]
    print(f"{name}: per-hop linear-probe accuracy from each layer's residual (N={N} hop-3 queries)")
    print("          f^0(e)  f^1(e)  f^2(e)  f^3=ANS")
    for Li in range(S):
        Xtr, Xte = reps[:ntr, Li], reps[ntr:, Li]
        accs = []
        for k in range(4):
            ytr, yte = chains[:ntr, k], chains[ntr:, k]
            clf = LogisticRegression(max_iter=200, C=1.0).fit(Xtr, ytr)
            accs.append((clf.predict(Xte) == yte).mean())
        star = int(np.argmax(accs))
        print(f"  {labels[Li]:6s}  " + "  ".join(f"{a:.2f}" for a in accs) +
              f"   -> best f^{star}", flush=True)
    print("\nrising diagonal (L1->f^1, L2->f^2, L3->f^3) = layer-by-layer entity-pointer advance in a "
          "hidden basis; only-L3->f^3 = intermediate entities not linearly present", flush=True)


if __name__ == "__main__":
    run()
