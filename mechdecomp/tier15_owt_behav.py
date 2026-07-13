"""TIER 1.5 on OWT models — induction head identified BEHAVIOURALLY (sign-agnostic).

Why this replaces `tier15_owt.py`'s criterion. That script picked the induction-head candidate as the
L1 head with the most NEGATIVE match weight, generalising from tiny attn2 (L1H2, match −0.434). That
is a sign convention, not a definition: a bilinear head copies whenever the attention pattern and the
OV circuit AGREE in sign (the XNOR), so a head can implement induction with a POSITIVE match and a
positive OV. block2-dense's L1H2 has match +0.0669 — the largest magnitude of any head — and the sign
rule threw it away.

Correct criterion: ablate each L1 head and measure the drop in the model's probability of the copied
token at induction sites. That is sign-agnostic and is the thing we actually mean by "induction head".

  GUARD  the same procedure, run on tiny attn2-seed0, must select L1H2 (the causally-verified head).
         If it does not reproduce the known answer, the criterion is wrong and nothing else is read.

Run: TL_CORPUS=owt python -m mechdecomp.tier15_owt_behav <run>       (TL_CORPUS=tiny for the guard)
"""

import sys

import numpy as np
import torch

from lm_eval import load_model
from text_data import N_CTX, RUNS, val_windows

from .tier15_owt import mine_sites

DEV = "cuda"
N_WIN = 64


@torch.no_grad()
def induction_prob(model, toks, sites):
    """mean P(tok[src] | prefix) at the query position of each induction site."""
    logits = model(toks).float()
    ps = []
    for (b, q, _, s) in sites:
        p = torch.softmax(logits[b, q], -1)[toks[b, s]]
        ps.append(p)
    return torch.stack(ps).mean().item()


@torch.no_grad()
def ablate_head_prob(model, toks, sites, layer_idx, head):
    """Zero one head's output columns in W_O, measure induction probability, restore."""
    L = model.layers[layer_idx]
    dh = L.d_head
    sl = slice(head * dh, (head + 1) * dh)
    saved = L.o.weight[:, sl].clone()
    L.o.weight[:, sl] = 0
    p = induction_prob(model, toks, sites)
    L.o.weight[:, sl] = saved
    return p


@torch.no_grad()
def analyse(run, expect=None):
    model = load_model(RUNS / run, None, DEV)
    attn_idx = [i for i, l in enumerate(model.layers) if hasattr(l, "q1")]
    print(f"=== {run} ===  attention layers {attn_idx}", flush=True)
    if len(attn_idx) < 2:
        print("  <2 attention layers"); return None
    i1 = attn_idx[1]

    data, _ = val_windows()
    buf = np.stack([data[w * N_CTX:w * N_CTX + N_CTX] for w in range(N_WIN)]).astype(np.int64)
    toks = torch.from_numpy(buf).to(DEV)
    sites = mine_sites(toks)
    print(f"  mined {len(sites)} induction sites", flush=True)
    if len(sites) < 20:
        print("  too few sites"); return None

    # behavioural baseline: does the model do induction at all?
    p0 = induction_prob(model, toks, sites)
    # control: probability of a random token at the same positions
    g = torch.Generator().manual_seed(0)
    rnd = torch.randint(1, model.embed.weight.shape[0], (len(sites),), generator=g).to(DEV)
    logits = model(toks).float()
    p_rnd = torch.stack([torch.softmax(logits[b, q], -1)[rnd[i]]
                         for i, (b, q, _, _) in enumerate(sites)]).mean().item()
    print(f"  P(copied token) = {p0:.4f}   P(random token) = {p_rnd:.5f}   ratio {p0/max(p_rnd,1e-9):.1f}x", flush=True)

    L1 = model.layers[i1]
    print(f"  ablate each L1 head (sign-agnostic):", flush=True)
    drops = {}
    for hd in range(L1.n_head):
        p = ablate_head_prob(model, toks, sites, i1, hd)
        drops[hd] = p0 - p
        print(f"    L1H{hd}: P {p:.4f}   drop {drops[hd]:+.4f}", flush=True)
    top = max(drops, key=lambda k: drops[k])
    nxt = max(k for k in drops if k != top)
    nxtv = max(drops[k] for k in drops if k != top)
    rel = drops[top] / max(p0, 1e-9)
    print(f"  → induction head by ablation: L1H{top}  (removes {100*rel:.1f}% of the copy probability;"
          f" next head {100*nxtv/max(p0,1e-9):.1f}%)", flush=True)
    if expect is not None:
        ok = top == expect
        print(f"  [GUARD] expected L1H{expect} → {'PASS' if ok else 'FAIL'}", flush=True)
        assert ok, f"criterion does not reproduce the known head on {run}"
    return top, rel, p0, p_rnd


def main():
    if len(sys.argv) > 1:
        analyse(sys.argv[1])
    else:
        analyse("attn2-seed0", expect=2)


if __name__ == "__main__":
    main()
