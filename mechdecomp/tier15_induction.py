"""TIER 1.5 against a GENUINE content-based induction circuit.

Until now the Tier-1.5 gate was validated against attn2-seed0's L0H3→L1H2 edge, which we later
established is a *repeated-bigram* match-and-copy circuit, not induction (it copies arbitrary
repeated tokens at 1.2x chance). `attn2-s30k-mix50-rp-dense-seed0` — trained with TILED
random-period copy bursts — is the real thing: it copies at untrained periods (P=16/32) and at
P=96, a period where a positional copier scores chance.

Its causal ground truth (behavioural head ablation at P=96, base P(copy)=0.7483):
    L0H1  -99.0%     L0H0  -88.8%     L0H3  -44.0%     L0H2  -0.6%
    L1H0  -76.2%     L1H3  -75.3%     (redundant copy pair)
Two L0 heads are required, so this is NOT a single edge. The gate is therefore a RANK test:
the atom-reconstructed write-ablation Δŝ must reproduce the causal ORDER over L0 heads.

Tests
  GUARD  reproduce the causal ablation ranking by exact write removal.
  (F)    FAITHFULNESS: swap each head's write for its k-sparse atom reconstruction -> P(copy) ~ base.
  (G)    GATE: remove the atom-reconstructed write; the induced Δŝ ranking must match the causal one,
         and argmax must be L0H1.

Evaluated at P=96 throughout: positional copying scores chance there, so every number is about
content matching.

Run: TL_CORPUS=owt python -m mechdecomp.tier15_induction
"""

import torch

from lm_eval import load_model
from text_data import RUNS

from .release_d import omp_codes
from .tier15_contraction import head_write, learn_dict

DEV = "cuda"
RUN = "attn2-s30k-mix50-rp-dense-seed0"
V, N_CTX, PERIOD = 5120, 256, 96
NSEQ, KSPARSE, M_ATOMS = 64, 8, 128


def tiled_batch(n=NSEQ, P=PERIOD, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(V, (n, P), generator=g)
    b = w.repeat(1, (N_CTX + 1 + P - 1) // P)[:, :N_CTX + 1]
    return b.to(DEV)


@torch.no_grad()
def copy_prob(model, h1, y, q):
    """P(target) at the query positions, given a (possibly modified) layer-1 input residual."""
    out = model.layers[1](h1)
    logits = model.head(out).float()
    p = torch.softmax(logits, -1)
    return p[:, q, :].gather(2, y[:, q].unsqueeze(-1)).squeeze(-1).mean().item()


@torch.no_grad()
def main():
    model = load_model(RUNS / RUN, None, DEV)
    L0 = model.layers[0]
    b = tiled_batch()
    x, y = b[:, :-1], b[:, 1:]
    q = torch.arange(PERIOD + 2, N_CTX - 2, device=DEV)

    h0 = model.embed(x).float()
    h1 = L0(h0).float()
    base = copy_prob(model, h1, y, q)
    print(f"model {RUN}\nP={PERIOD} (positional copiers score chance here)\nbase P(copy) = {base:.4f}\n", flush=True)
    assert base > 0.5, f"GUARD FAIL: base {base:.4f} — model does not copy, wrong checkpoint?"

    # ---------- GUARD: exact write removal reproduces the causal ranking ----------
    print("GUARD — remove each L0 head's exact write (all positions):", flush=True)
    true_d, OVs, us, ws = {}, {}, {}, {}
    for h in range(L0.n_head):
        w_h, u, OV = head_write(model, h0, h)
        ws[h], us[h], OVs[h] = w_h, u, OV
        true_d[h] = base - copy_prob(model, h1 - w_h, y, q)
        print(f"  L0H{h}: P(copy) {base-true_d[h]:.4f}   drop {true_d[h]:+.4f} ({100*true_d[h]/base:5.1f}%)", flush=True)
    true_rank = sorted(range(L0.n_head), key=lambda h: -true_d[h])
    print(f"  causal ranking: {['L0H%d' % h for h in true_rank]}", flush=True)
    assert true_rank[0] == 1, f"GUARD FAIL: expected L0H1 dominant, got L0H{true_rank[0]}"
    print("  [guard ok] reproduces the behavioural ablation ranking\n", flush=True)

    # ---------- decompose each L0 OV map on the EVALUATION distribution ----------
    n0 = L0.norm(h0).reshape(-1, h0.shape[-1]).float()
    sc = float(n0.shape[-1]) ** 0.5
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(n0.shape[0], generator=g)[:8000]
    X0 = (n0[idx].T / sc).contiguous()
    print(f"decomposing L0 OV maps (OMP k={KSPARSE}, {M_ATOMS} atoms) on tiled activations:", flush=True)
    Ds = {}
    for h in range(L0.n_head):
        Ds[h], r2 = learn_dict(OVs[h], X0, m=M_ATOMS, k=KSPARSE)
        print(f"  L0H{h}-OV  R2 {r2:.4f}", flush=True)

    # ---------- (F) faithfulness and (G) gate ----------
    print("\n  head   true Δ     atom Δŝ    faithful P(copy)   [base %.4f]" % base, flush=True)
    est = {}
    for h in range(L0.n_head):
        OV, D, u = OVs[h], Ds[h], us[h]
        B, T, dm = u.shape
        uu = (u.reshape(-1, dm).T / sc).contiguous()          # (d, B*T)
        C, _ = omp_codes(OV @ D, OV @ uu, KSPARSE)
        w_hat = ((L0.scale * sc) * ((OV @ D) @ C)).T.reshape(B, T, dm)
        est[h] = base - copy_prob(model, h1 - w_hat, y, q)
        faith = copy_prob(model, h1 - ws[h] + w_hat, y, q)
        print(f"  L0H{h}  {true_d[h]:+.4f}   {est[h]:+.4f}     {faith:.4f}", flush=True)

    est_rank = sorted(range(L0.n_head), key=lambda h: -est[h])
    print(f"\n  causal ranking : {['L0H%d' % h for h in true_rank]}", flush=True)
    print(f"  atom   ranking : {['L0H%d' % h for h in est_rank]}", flush=True)
    ok_rank = est_rank == true_rank
    ok_top = est_rank[0] == 1
    print(f"\n  RANK GATE  {'PASS' if ok_rank else 'FAIL'}   |   ARGMAX GATE (L0H1)  {'PASS' if ok_top else 'FAIL'}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
