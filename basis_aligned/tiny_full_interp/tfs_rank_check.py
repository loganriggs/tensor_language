"""Known-answer check on a structural claim about rung 2, run BEFORE the
shared tf_model.py is finalized because it changes what rung 2 has to store.

CLAIM. In this architecture the layer-0 attention score table is not a
single V x V object, and it is also far smaller than one. At layer 0 the
attention input is the embedding alone, so each token t has a FIXED per-head
query/key vector Q[t], K[t] in R^hd (linear map, per-head RMSNorm, both
functions of the token only). Rotary then makes the score depend on the pair
AND the relative offset. Writing the RoPE frequency pairs as m = 1..hd/2:

  s(t_i, t_j, D) = (1/hd) * sum_m [ cos(theta_m D) * (Q1_m[t_i] K1_m[t_j]
                                                    + Q2_m[t_i] K2_m[t_j])
                                  + sin(theta_m D) * (Q2_m[t_i] K1_m[t_j]
                                                    - Q1_m[t_i] K2_m[t_j]) ]

Two consequences, both testable here:

  (A) EXACT RELATIVITY: the score depends on i and j only through D = i - j.
  (B) EXACT LOW RANK: for every fixed D, the V x V matrix s(.,.,D) has rank
      at most hd -- its row space is spanned by the hd columns of K for all
      D at once. So the whole offset-indexed family lives in one fixed
      hd-dimensional subspace, and the offset enters only through hd
      sinusoidal scalars.

Why it matters for the grid: rung 2 says "materialize the V x V per-head-
branch tables". Taken literally with rotary present, that is a table PER
OFFSET -- at V=4096, T=512, one head-branch is 4096^2 * 512 * 4 bytes = 32
GiB, and the scale box's width-256 cells have 16 heads x 2 branches, so the
literal reading is about 1 TiB per layer. If (B)
holds, the same information is exactly (V x hd) + (V x hd) + hd frequencies
per head-branch = 0.5 MB at V=4096, hd=16, and any offset's table is
reconstructible on demand. That is a storage question, not a physics
question: the object is the same, the factored form is just the honest one.

This script asserts both properties against a direct forward computation
using the parent program's own rope_tables_exact / apply_rot, at exact
float64, on random weights (no training needed -- it is a statement about
the architecture, not about a learned solution). CPU only, seconds.
"""
import math
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')

torch.manual_seed(0)
DTYPE = torch.float64
V, HD, T = 512, 16, 64          # small V keeps the dense V x V check exact-cheap


def rope_tables_exact(T, hd):
    """Verbatim qk_tokenline_train.rope_tables_exact, in float64."""
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, dtype=DTYPE) / hd))
    t = torch.arange(T, dtype=DTYPE)
    fr = torch.outer(t, inv)
    return fr.cos(), fr.sin()


def apply_rot(x, c, s):
    """Verbatim qk_tokenline_train.apply_rot."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], -1)


def main():
    # Per-token query/key vectors: whatever the layer-0 path produces, it is
    # a function of the token alone, so random V x HD stands in for it exactly.
    Qt = torch.randn(V, HD, dtype=DTYPE)
    Kt = torch.randn(V, HD, dtype=DTYPE)
    cos, sin = rope_tables_exact(T, HD)

    # direct: rotate by absolute position, then inner product
    def score_direct(ti, tj, i, j):
        q = apply_rot(Qt[ti], cos[i], sin[i])
        k = apply_rot(Kt[tj], cos[j], sin[j])
        return float(q @ k) / HD

    # ---- (A) exact relativity: same offset -> same score ----
    worst_rel = 0.0
    for (i, j) in ((40, 33), (41, 34), (60, 53), (7, 0)):
        base = None
        for (ti, tj) in ((3, 11), (100, 250), (7, 7)):
            v = score_direct(ti, tj, i, j)
            if base is None:
                base = {}
            base[(ti, tj)] = v
        if i == 40:
            ref = base
        else:
            worst_rel = max(worst_rel,
                            max(abs(base[k] - ref[k]) for k in ref))
    print(f"(A) relativity: max |score(D) - score(same D, shifted)| "
          f"= {worst_rel:.3e}")
    assert worst_rel < 1e-12, worst_rel

    # ---- (B) exact low rank of the V x V table at fixed offset ----
    half = HD // 2
    Q1, Q2 = Qt[:, :half], Qt[:, half:]
    K1, K2 = Kt[:, :half], Kt[:, half:]
    inv = 1.0 / (10000 ** (torch.arange(0, HD, 2, dtype=DTYPE) / HD))

    worst_tab, worst_rank_gap = 0.0, 0.0
    for D in (0, 1, 2, 5, 17, 63):
        i = T - 1
        j = i - D
        # dense reference table, built one entry at a time from the direct path
        q_all = apply_rot(Qt, cos[i].expand(V, half), sin[i].expand(V, half))
        k_all = apply_rot(Kt, cos[j].expand(V, half), sin[j].expand(V, half))
        ref = (q_all @ k_all.t()) / HD

        # factored form: hd sinusoidal scalars against fixed rank-1 pieces
        cD, sD = torch.cos(inv * D), torch.sin(inv * D)
        tab = ((Q1 * cD) @ K1.t() + (Q2 * cD) @ K2.t()
               + (Q2 * sD) @ K1.t() - (Q1 * sD) @ K2.t()) / HD
        d_tab = float((tab - ref).abs().max())
        worst_tab = max(worst_tab, d_tab)

        # rank: singular values beyond index HD must be exactly ~0
        sv = torch.linalg.svdvals(ref)
        tail = float(sv[HD:].max())
        worst_rank_gap = max(worst_rank_gap, tail / float(sv[0]))
        print(f"  D={D:3d}: |factored - direct| {d_tab:.3e}; "
              f"sv[0] {float(sv[0]):.4f}, sv[{HD - 1}] {float(sv[HD - 1]):.4f}, "
              f"sv[{HD}] {tail:.3e}")

    print(f"(B) factored reconstruction max abs err {worst_tab:.3e}; "
          f"max relative rank tail {worst_rank_gap:.3e}")
    assert worst_tab < 1e-12, worst_tab
    assert worst_rank_gap < 1e-12, worst_rank_gap

    # ---- storage arithmetic for the actual grid cells ----
    print("\nstorage at V=4096, hd=16, T=512, fp32:")
    Vg, hd, Tg = 4096, 16, 512
    dense_per_branch = Vg * Vg * Tg * 4
    fact_per_branch = (2 * Vg * hd + hd) * 4
    print(f"  dense per head-branch, all offsets: "
          f"{dense_per_branch / 2**30:.0f} GiB")
    print(f"  dense per head-branch, ONE offset:  "
          f"{Vg * Vg * 4 / 2**20:.0f} MiB")
    print(f"  factored per head-branch (exact):   "
          f"{fact_per_branch / 2**10:.0f} KiB")
    for w in (32, 64, 128, 256):
        heads = w // hd
        print(f"  width {w:3d}: {heads:2d} heads x 2 branches -> factored "
              f"{heads * 2 * fact_per_branch / 2**20:.1f} MiB per layer, "
              f"vs {heads * 2 * Vg * Vg * 4 / 2**30:.1f} GiB for one-offset "
              f"dense tables")
    print("\nBOTH PROPERTIES HOLD EXACTLY (float64).")


if __name__ == '__main__':
    main()
