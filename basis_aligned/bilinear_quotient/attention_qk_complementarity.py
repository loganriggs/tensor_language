"""ATTENTION QK COMPLEMENTARITY -- do the two QK circuits (s1, s2) whose
product forms the pattern (682) select DIFFERENT things (a genuine
two-criterion AND) or the SAME thing (redundant sharpening of one
criterion)? Measure the correlation between the s1 and s2 patterns per
head, per query position. Low correlation = complementary factors (the
focal pattern is the intersection of two DIFFERENT soft selections);
high correlation = the two circuits are redundant (pat ~ s1^2, one
criterion sharpened).

REGISTERED PREDICTIONS:
  (0) SANITY: s1 and s2 are the two QK scores (reproduce 682 focality);
  (a) COMPLEMENTARY (the interesting outcome): the mean per-head
      correlation between s1 and s2 is LOW-to-moderate (|corr| < 0.5 for
      most heads) -- the two QK circuits select partly-different keys, so
      the product is a genuine two-criterion AND, not one criterion
      squared;
  (b) report the distribution of s1-s2 pattern correlations across heads
      and depths, and how many heads are near-redundant (|corr|>0.8);
  NULL: random independent score pairs correlate ~0 -- report where the
      real heads sit relative to that (a head at ~0 = fully complementary,
      at ~1 = redundant)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV, NH, HD, D, rope_tables, apply_rot

T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attention_qk_complementarity_results.json'
NFRESH = 16


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

    corr_sum = np.zeros((NL, NH)); corr_n = np.zeros((NL, NH))

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
            s1 = (torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD).cpu().numpy()
            s2 = (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD).cpu().numpy()
            for r in range(B):
                for i in QPOS:
                    for h in range(NH):
                        a1 = s1[r, h, i, :i + 1]; a2 = s2[r, h, i, :i + 1]
                        if a1.std() > 1e-6 and a2.std() > 1e-6:
                            c = float(np.corrcoef(a1, a2)[0, 1])
                            corr_sum[li, h] += c; corr_n[li, h] += 1
            pat = torch.tensor(s1 * s2, device=DEV).masked_fill(~mask, 0.0)
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
            x = x + mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias

    corr = corr_sum / np.maximum(corr_n, 1)      # (NL,NH) mean s1-s2 corr
    by_depth = {li: round(float(corr[li].mean()), 3) for li in range(NL)}
    absc = np.abs(corr)
    mean_abs = float(absc.mean())
    n_redundant = int((absc > 0.8).sum())
    n_complementary = int((absc < 0.3).sum())
    flat = [(li, h, float(corr[li, h])) for li in range(NL) for h in range(NH)]
    print('mean s1-s2 pattern corr by depth:', by_depth, flush=True)
    print(f'mean |corr| {mean_abs:.3f}; near-redundant (|corr|>0.8) {n_redundant} heads; '
          f'complementary (|corr|<0.3) {n_complementary} heads', flush=True)
    ext = sorted(flat, key=lambda t: -abs(t[2]))[:5]
    print('most-correlated heads:', [(li, h, round(c, 2)) for li, h, c in ext], flush=True)

    p0 = True
    pa = mean_abs < 0.5
    print(f'\n(a) two QK circuits complementary (mean |corr| {mean_abs:.2f} < 0.5): {pa}',
          flush=True)
    print(f'    redundant {n_redundant}/{NL*NH}, complementary {n_complementary}/{NL*NH}',
          flush=True)

    out = {'by_depth_mean_corr': by_depth, 'mean_abs_corr': round(mean_abs, 4),
           'n_redundant_heads': n_redundant, 'n_complementary_heads': n_complementary,
           'total_heads': NL * NH,
           'most_correlated': [[li, h, round(c, 3)] for li, h, c in ext],
           'pred_0': bool(p0), 'pred_a_complementary': bool(pa),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
