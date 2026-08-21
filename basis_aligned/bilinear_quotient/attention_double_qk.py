"""ATTENTION DOUBLE QK -- is the attention's focality (681) DUE to the
double-QK design? The pattern is pat = s1*s2 where s1=(q.k1)/HD,
s2=(q2.k2)/HD -- a PRODUCT of two independent QK dot-products. Hypothesis:
the product is more focal than either single term, because pat is large
only where BOTH s1 and s2 are large (an intersection/AND of two
selections), which sharpens the pattern relative to one QK.

For each head, compute the effective-keys fraction (participation ratio /
#valid keys) of |pat=s1*s2| vs |s1| alone vs |s2| alone. If the product is
more focal (lower fraction) than the single terms, the double-QK buys
the sharpening.

REGISTERED PREDICTIONS:
  (0) SANITY: reproduce 681 (pat mean fraction ~0.2-0.3);
  (a) PRODUCT SHARPENS: pat = s1*s2 is more focal (lower eff-keys
      fraction) than the mean of s1-alone and s2-alone, across heads --
      the double-QK is the peaking mechanism;
  (b) report mean eff-keys fraction for pat, s1, s2, and how often the
      product is more focal than both terms;
  NULL: report -- if s1 and s2 are ALREADY as focal as their product, the
      double-QK does not sharpen (the focality is in a single QK)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV, NH, HD, D, rope_tables, apply_rot

T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attention_double_qk_results.json'
NFRESH = 16


def frac_eff(w):
    # w: (NH, k) nonneg; participation ratio / k
    s = w.sum(1); s2 = (w ** 2).sum(1)
    pr = (s ** 2) / (s2 + 1e-12)
    return pr / w.shape[1]


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    NL = len(m.transformer.h)
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    QPOS = list(range(16, T, 16))

    acc = {k: np.zeros((NL, NH)) for k in ['pat', 's1', 's2']}
    prod_wins = np.zeros((NL, NH)); n = np.zeros((NL, NH))

    for bi in range(0, NFRESH, 4):
        bb = fresh[bi:bi + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li in range(NL):
            blk = m.transformer.h[li]
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            a = blk.attn
            hcur = F.rms_norm(x, (D,))

            def qk(l):
                z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,))
                return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T, NH, HD)
            if v1 is None:
                v1 = v
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            q, k1_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = (torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD)
            s2 = (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD)
            pat = s1 * s2
            for r in range(B):
                for i in QPOS:
                    fp = frac_eff(pat[r, :, i, :i + 1].abs().cpu().numpy())
                    f1 = frac_eff(s1[r, :, i, :i + 1].abs().cpu().numpy())
                    f2 = frac_eff(s2[r, :, i, :i + 1].abs().cpu().numpy())
                    acc['pat'][li] += fp; acc['s1'][li] += f1; acc['s2'][li] += f2
                    prod_wins[li] += (fp < f1) & (fp < f2)
                    n[li] += 1
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd',
                             pat.masked_fill(~mask, 0.0), v).reshape(B, T, -1))
            xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
            x = x + mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias

    means = {k: float((acc[k] / np.maximum(n, 1)).mean()) for k in acc}
    prod_win_rate = float((prod_wins / np.maximum(n, 1)).mean())
    print(f'mean eff-keys fraction: pat {means["pat"]:.4f}  s1 {means["s1"]:.4f}  '
          f's2 {means["s2"]:.4f}', flush=True)
    print(f'product more focal than BOTH terms: {100*prod_win_rate:.0f}% of cases',
          flush=True)

    p0 = 0.1 < means['pat'] < 0.4
    pa = means['pat'] < 0.5 * (means['s1'] + means['s2'])
    print(f'\n(0) sane: {p0}', flush=True)
    print(f'(a) product sharpens (pat < mean of s1,s2): {pa}', flush=True)
    print(f'    (product-wins-both rate {prod_win_rate:.2f})', flush=True)

    out = {'mean_frac_pat': round(means['pat'], 4), 'mean_frac_s1': round(means['s1'], 4),
           'mean_frac_s2': round(means['s2'], 4), 'product_wins_both_rate': round(prod_win_rate, 4),
           'pred_0': bool(p0), 'pred_a_product_sharpens': bool(pa), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
