"""TIER 1.5 — does the decomposition recover the causally-verified L0H3 → L1H2 edge?

Background. `logs/tier15_omp2.log` reported the spec §1.5 per-branch contraction gate as
K1 PASS / K2 FAIL (K2's max was L0H2, not L0H3). But §2.6 established

    Δs = a₁b₂ + a₂b₁ − b₁b₂,   a_i = Q_i x_q·K_i x_k,   b_i = Q_i x_q·K_i v

i.e. each branch is cross-weighted by the OTHER branch's score. A per-branch attribution is
therefore NOT the causal quantity, and demanding L0H3 dominance in each branch separately
presumes a separability the product form denies. That script was an unsaved heredoc; this is the
reproducible replacement.

What is tested here, on the JOINT (data-conditioned) quantity:
  GUARD  reproduce the causal table by direct ablation: zeroing L0H3's write at `src` must
         collapse L1H2's match (≈ −0.434 → −0.031); other L0 heads must leave it intact.
  (F)    FAITHFULNESS: replace head h's write by its k-sparse atom reconstruction. The match
         must be ≈ unchanged, else the atoms do not represent the write at all.
  (G)    GATE: remove the atom-reconstructed write. argmax_h |Δŝ_h| must be L0H3, and Δŝ_h must
         track the true Δs_h.
  (B)    For contrast, the spec's per-branch aggregate |d_kᵀ OV_h d_j|, which is expected to fail.

Run: TL_CORPUS=tiny python -m mechdecomp.tier15_contraction
"""

import torch
import torch.nn.functional as Fn

from lm_eval import load_model
from text_data import RUNS

from .mstep import rowspace_basis
from .release_d import omp_codes
from .tier1_recovery import mstep_gs

DEV = "cuda"
N_SEQ, N_CTX_USE, BLOCK = 96, 40, 8
M_ATOMS, KSPARSE, ROUNDS = 128, 8, 25
H_IND = 2                      # L1H2 is the induction head


def induction_batch(n_vocab, seed=0):
    """Sequences with a repeated block; q attends to src = j+1 where tok[j] == tok[q]."""
    g = torch.Generator().manual_seed(seed)
    toks = torch.randint(1, n_vocab, (N_SEQ, N_CTX_USE), generator=g)
    p1, p2, t = 4, 24, 3
    toks[:, p2:p2 + BLOCK] = toks[:, p1:p1 + BLOCK]     # repeat the block
    q = p2 + t                                          # query: 2nd occurrence of tok[j]
    j = p1 + t                                          # 1st occurrence
    return toks.to(DEV), q, j, j + 1                    # src = j+1 (token to copy)


@torch.no_grad()
def l1_match(model, h1, q, src):
    """L1H2's bilinear match weight s(q, src) from the layer-1 input residual h1."""
    pat = model.layers[1].pattern(h1)                   # (B, n_head, Q, K)
    return pat[:, H_IND, q, src]


@torch.no_grad()
def head_write(model, h0, h):
    """scale · OV_h · u_h(pos), where u_h(pos) = Σ_k pattern0_h[pos,k]·n0[k]."""
    L0 = model.layers[0]
    dh = L0.d_head
    n0 = L0.norm(h0)
    pat0 = L0.pattern(h0)[:, h]                          # (B, Q, K)
    u = torch.einsum("bqk,bkd->bqd", pat0, n0)           # (B, Q, d_model)
    OV = (L0.o.weight[:, h * dh:(h + 1) * dh] @ L0.v.weight[h * dh:(h + 1) * dh, :]).float()
    return L0.scale * (u @ OV.T), u, OV


def learn_dict(W, X, m=M_ATOMS, k=KSPARSE, rounds=ROUNDS, seed=0):
    """OMP + validated Gauss-Seidel M-step (tier1_recovery)."""
    RS = rowspace_basis(W); Wp = torch.linalg.pinv(W)
    Y = W @ X
    g = torch.Generator().manual_seed(seed)
    D = Fn.normalize(torch.randn(X.shape[0], m, generator=g).to(X.device), dim=0)
    for _ in range(rounds):
        C, _ = omp_codes(W @ D, Y, k)
        D, C = mstep_gs(W, D, C, Y, RS, Wp)
    C, _ = omp_codes(W @ D, Y, k)
    r2 = float(1 - (((W @ D) @ C - Y) ** 2).sum() / ((Y - Y.mean(1, keepdim=True)) ** 2).sum())
    return D, r2


@torch.no_grad()
def main():
    model = load_model(RUNS / "attn2-seed0", None, DEV)
    L0, L1 = model.layers[0], model.layers[1]
    dh = L0.d_head
    n_vocab = model.embed.weight.shape[0]
    toks, q, j, src = induction_batch(n_vocab)
    print(f"induction batch: {N_SEQ} seqs, q={q} j={j} src={src}\n", flush=True)

    h0 = model.embed(toks).float()
    h1 = L0(h0).float()
    base = l1_match(model, h1, q, src)
    print(f"BASE L1H2 match s(q,src) = {base.mean():+.4f}\n", flush=True)

    # ---------- GUARD: reproduce the causal table by direct ablation at src ----------
    print("GUARD — zero each L0 head's write at src; L0H3 must collapse L1H2's match:", flush=True)
    true_ds, writes, us, OVs = {}, {}, {}, {}
    for h in range(4):
        w, u, OV = head_write(model, h0, h)
        writes[h], us[h], OVs[h] = w, u, OV
        h1a = h1.clone(); h1a[:, src, :] -= w[:, src, :]
        s = l1_match(model, h1a, q, src)
        true_ds[h] = float((s - base).mean())
        print(f"  zero L0H{h}@src → match {s.mean():+.4f}   Δs {true_ds[h]:+.4f}", flush=True)
    dom = max(range(4), key=lambda h: abs(true_ds[h]))
    assert dom == 3, f"GUARD FAIL: causal ground truth says L0H3, got L0H{dom}"
    print(f"  [guard ok] L0H3 dominant (|Δs| {abs(true_ds[3]):.4f} vs next {max(abs(true_ds[h]) for h in range(3)):.4f})\n", flush=True)

    # ---------- decompose each L0 OV map on the layer-0 input distribution ----------
    n0 = L0.norm(h0).reshape(-1, h0.shape[-1]).float()
    scale = float(n0.shape[-1]) ** 0.5
    Xg = torch.Generator().manual_seed(0)
    idx = torch.randperm(n0.shape[0], generator=Xg)[:10000]
    X0 = (n0[idx].T / scale).contiguous()
    print("decomposing L0 OV maps (OMP + validated M-step):", flush=True)
    Ds = {}
    for h in range(4):
        Ds[h], r2 = learn_dict(OVs[h], X0)
        print(f"  L0H{h}-OV  atoms {M_ATOMS}  k {KSPARSE}  R2 {r2:.4f}", flush=True)

    # ---------- (F) faithfulness + (G) gate on the JOINT quantity ----------
    print("\n(F) FAITHFULNESS: swap head h's write for its atom reconstruction (match must not move)", flush=True)
    print("(G) GATE: remove the atom-reconstructed write; argmax|Δŝ| must be L0H3\n", flush=True)
    print("  head   true Δs    atom Δŝ    faithful match   |Δŝ| rank", flush=True)
    est_ds = {}
    for h in range(4):
        OV, D, u = OVs[h], Ds[h], us[h]
        uu = (u[:, src, :].T / scale).contiguous()        # (d, B) test inputs at src
        Y = OV @ uu
        C, _ = omp_codes(OV @ D, Y, KSPARSE)
        w_hat = (L0.scale * scale) * ((OV @ D) @ C).T     # (B, d) reconstructed write at src

        h1f = h1.clone(); h1f[:, src, :] += (w_hat - writes[h][:, src, :])
        s_f = l1_match(model, h1f, q, src).mean()

        h1r = h1.clone(); h1r[:, src, :] -= w_hat
        est_ds[h] = float((l1_match(model, h1r, q, src) - base).mean())
        print(f"  L0H{h}  {true_ds[h]:+.4f}   {est_ds[h]:+.4f}      {s_f:+.4f}", flush=True)

    dom_est = max(range(4), key=lambda h: abs(est_ds[h]))
    ok = dom_est == 3
    nxt = max(abs(est_ds[h]) for h in range(3))
    print(f"\n  JOINT GATE: argmax|Δŝ| = L0H{dom_est}  → {'PASS' if ok else 'FAIL'}"
          f"   (L0H3 {abs(est_ds[3]):.4f} vs next {nxt:.4f}, {abs(est_ds[3])/max(nxt,1e-9):.2f}x)", flush=True)

    # ---------- (B) the spec's per-branch aggregate, for contrast ----------
    print("\n(B) SPEC §1.5 per-branch contraction |d_kᵀ OV_h d_j| (expected to fail on K2):", flush=True)
    n1 = L1.norm(h1).reshape(-1, h1.shape[-1]).float()
    X1 = (n1[idx].T / scale).contiguous()
    Kmaps = {"K1": L1.k1.weight[H_IND * dh:(H_IND + 1) * dh, :].float(),
             "K2": L1.k2.weight[H_IND * dh:(H_IND + 1) * dh, :].float()}
    for kname, Wk in Kmaps.items():
        Dk, r2k = learn_dict(Wk, X1)
        aggs = []
        for h in range(4):
            G = Dk.T @ OVs[h] @ Ds[h]
            aggs.append(float(G.abs().mean()))
        best = max(range(4), key=lambda h: aggs[h])
        print(f"  {kname} (R2 {r2k:.3f}): " + "  ".join(f"L0H{h} {aggs[h]:.4f}" for h in range(4))
              + f"   → max L0H{best} {'PASS' if best == 3 else 'FAIL'}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
