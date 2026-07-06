"""Correct circuit tool (per hop_circuit_agg's lesson): the chained lookup likely advances a
"current entity" in the RESIDUAL STREAM, not via attention to fixed positions. Track it directly:
apply the model's unembedding head to each layer's residual at the answer position and see which
entity it most encodes. If the decoded entity advances e -> f(e) -> f^2(e) -> f^3(e) across layers,
the mechanism is a layer-by-layer entity-pointer advance (chained retrieval reverse-engineered).

Run: python hop_residual_track.py
"""

import torch

from deep_model import SPECS
from hop_data import sample_docs, E, N_CTX, V, ANS_OFFSET
from model import Attention
from hop_circuit import load, fmap_from_bindings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_DOCS = 300


def run(name="attn4-rms-seed0"):
    m = load(name)
    g = torch.Generator().manual_seed(321)
    n_layers = len(m.layers)
    # frac[L,k] = fraction of hop-3 queries where layer L's residual decodes to f^k(e)
    frac = torch.zeros(n_layers + 1, 4)                        # +1 for the embedding (layer -1)
    count = 0
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
            x = layer(x)
            stream.append(x)
        for blk in blks:
            base = 2 * E + 4 * blk
            e = int(tok[base + 1]); ans_pos = base + ANS_OFFSET
            chain = [e, f[e], f[f[e]], f[f[f[e]]]]             # f^0..f^3(e)
            for Li, res in enumerate(stream):
                logits = m.head(res[0, ans_pos])               # (V,)
                pred = int(logits[:E].argmax())                # most-encoded entity
                for k in range(4):
                    if pred == chain[k]:
                        frac[Li, k] += 1
            count += 1
    frac /= count
    labels = ["embed"] + [f"L{i}({m.spec[i]})" for i in range(n_layers)]
    print(f"{name}: which f^k(e) does each layer's residual (at answer pos) decode to? "
          f"(frac of {count} hop-3 queries)")
    print("           f^0(e)  f^1(e)  f^2(e)  f^3(e)=ANS")
    for Li, lab in enumerate(labels):
        print(f"  {lab:10s} " + "  ".join(f"{frac[Li,k]:.2f}" for k in range(4)) +
              f"   -> {['f^0','f^1','f^2','f^3=ANS'][int(frac[Li].argmax())]}", flush=True)
    print("\nif the decoded entity advances f^0 -> ... -> f^3=ANS down the layers, the model "
          "runs a layer-by-layer ENTITY-POINTER ADVANCE (chained-retrieval mechanism)", flush=True)


if __name__ == "__main__":
    run()
