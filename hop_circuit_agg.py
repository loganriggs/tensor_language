"""Robust version of hop_circuit: aggregate over MANY hop-3 docs the attention mass each layer
places (from the answer position) on each chain-entity's binding-value slot (f^0..f^3 of the
query entity). A clean layer->hop progression confirms the layers compose the lookups.

Run: python hop_circuit_agg.py
"""

import torch

from deep_model import DeepModel, SPECS
from hop_data import sample_docs, E, N_CTX, V, ANS_OFFSET
from model import Attention
from hop_circuit import load, fmap_from_bindings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_DOCS = 200


def run(name="attn4-rms-seed0"):
    m = load(name)
    g = torch.Generator().manual_seed(123)
    n_layers = sum(isinstance(l, Attention) for l in m.layers)
    mass = torch.zeros(n_layers, 4)                            # [layer, hop k] attention mass on f^k(e) binding val
    count = 0
    for _ in range(N_DOCS):
        tokens, qa, qk = sample_docs(1, g)
        tok = tokens[0]
        f = fmap_from_bindings(tok)
        vpos = {int(tok[2 * i]): 2 * i for i in range(E)}      # entity -> its binding-pair start
        blks = [j for j in range(len(qk[0])) if int(qk[0][j]) == 3]
        if not blks:
            continue
        # per-layer patterns
        x = m.embed(tok.unsqueeze(0).to(DEVICE))
        pats = []
        for layer in m.layers:
            if isinstance(layer, Attention):
                pats.append(layer.pattern(x)[0].detach())
            x = layer(x)
        for blk in blks:
            base = 2 * E + 4 * blk
            e = int(tok[base + 1]); ans_pos = base + ANS_OFFSET
            chain = [e, f[e], f[f[e]], f[f[f[e]]]]
            valpos = [vpos[chain[k]] + 1 for k in range(4)]    # value slot of f^k(e) binding pair
            for L, p in enumerate(pats):
                row = p[:, ans_pos, :].mean(0)                 # head-avg attention from answer pos
                row = row / (row.abs().sum() + 1e-9)
                for k in range(4):
                    mass[L, k] += float(row[valpos[k]])
            count += 1
    mass /= count
    print(f"{name}: attention mass (from answer pos) on f^k(e)'s binding, averaged over {count} hop-3 queries")
    print("        f^0(e)  f^1(e)  f^2(e)  f^3(e)=ANS")
    for L in range(n_layers):
        print(f"  layer{L}: " + "  ".join(f"{mass[L,k]:+.3f}" for k in range(4)) +
              f"   -> peak hop {int(mass[L].argmax())}", flush=True)
    print("\nif peak-hop increases with layer index (layer L peaks near f^L) => layers COMPOSE the "
          "lookups; final layer peaking on f^3=ANS = it reads out the answer binding", flush=True)


if __name__ == "__main__":
    run()
