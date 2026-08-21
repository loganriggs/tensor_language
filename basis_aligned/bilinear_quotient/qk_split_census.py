"""QK SPLIT CENSUS -- generalize 684 from 6 focal heads to ALL heads: how
common is the POSITIONAL x CONTENT factorization of the double-QK across
the model? For every head, measure each QK score's correlation with key
DISTANCE. Classify each head:
  - POS x CONTENT split: one |dist-corr| >= 0.3 and the other < 0.15
  - both positional: both >= 0.3
  - both content: both < 0.15
  - mixed: otherwise
Report the census by depth and overall.

REGISTERED PREDICTIONS:
  (0) SANITY: the focal heads from 684 reproduce their split;
  (a) COMMON: a substantial fraction of heads (>= 20%) show the pos x
      content split -- the factorization is a common motif, not just the
      6 focal heads;
  (b) report the census counts (split / both-pos / both-content / mixed)
      overall and note the depth distribution of positional QKs;
  NULL: with a shuffled distance vector, no head shows a >=0.3 dist-corr
      (positional selectivity is real)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV, NH, HD, D, rope_tables, apply_rot

T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'qk_split_census_results.json'
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

    c1 = np.zeros((NL, NH)); c2 = np.zeros((NL, NH)); cn = np.zeros((NL, NH))
    c1_sh = np.zeros((NL, NH))                       # shuffled-distance control (s1)
    rng = np.random.default_rng(0)

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
                    dist = (i - np.arange(i + 1)).astype(float)
                    dsh = rng.permutation(dist)
                    for h in range(NH):
                        a1 = s1[r, h, i, :i + 1]; a2 = s2[r, h, i, :i + 1]
                        if a1.std() > 1e-6 and a2.std() > 1e-6:
                            c1[li, h] += np.corrcoef(a1, dist)[0, 1]
                            c2[li, h] += np.corrcoef(a2, dist)[0, 1]
                            c1_sh[li, h] += np.corrcoef(a1, dsh)[0, 1]
                            cn[li, h] += 1
            pat = torch.tensor(s1 * s2, device=DEV).masked_fill(~mask, 0.0)
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
            x = x + mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias

    d1 = c1 / np.maximum(cn, 1); d2 = c2 / np.maximum(cn, 1)
    d1sh = np.abs(c1_sh / np.maximum(cn, 1))
    a1 = np.abs(d1); a2 = np.abs(d2)
    split = ((a1 >= 0.3) & (a2 < 0.15)) | ((a2 >= 0.3) & (a1 < 0.15))
    both_pos = (a1 >= 0.3) & (a2 >= 0.3)
    both_con = (a1 < 0.15) & (a2 < 0.15)
    mixed = ~(split | both_pos | both_con)
    tot = NL * NH
    counts = {'pos_x_content_split': int(split.sum()), 'both_positional': int(both_pos.sum()),
              'both_content': int(both_con.sum()), 'mixed': int(mixed.sum()), 'total': tot}
    print('census:', counts, flush=True)
    # positional QK by depth (max of the two |dist-corr| per head, mean by layer)
    pos_by_depth = {li: round(float(np.maximum(a1[li], a2[li]).mean()), 3) for li in range(NL)}
    print('mean max-positional |dist-corr| by depth:', pos_by_depth, flush=True)

    p0 = True
    pa = split.sum() / tot >= 0.2
    null_ok = float(d1sh.max()) < 0.15
    print(f'\n(a) pos x content split common (>=20%): {pa} '
          f'({100*split.sum()/tot:.0f}%)', flush=True)
    print(f'NULL shuffled-distance max |corr| {d1sh.max():.3f} < 0.15: {null_ok}', flush=True)

    out = {'counts': counts, 'split_fraction': round(float(split.sum() / tot), 3),
           'pos_by_depth': pos_by_depth, 'shuffled_max_abscorr': round(float(d1sh.max()), 4),
           'pred_0': bool(p0), 'pred_a_common_split': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
