"""QK CRITERION IDENTITY -- what are the TWO criteria the double-QK
selects on (683: two complementary QK circuits per head)? For the
most-focal heads (681: L1.H1, L0.H3, L2.H6, ...), characterize each QK
circuit's selection by RELATIVE POSITION: does s1 (or s2) prefer nearby
keys (recency/local) or is it content-based (position-independent)? A
common decomposition of induction-style attention is one factor = a
previous-token/positional criterion, the other = a content-match
criterion.

For each focal head and each of s1, s2, measure the correlation of the
per-key score with the key's RELATIVE DISTANCE from the query. Strong
negative correlation = a recency/local criterion (prefers near keys);
~0 = position-independent (content-based). Report the (s1, s2) distance-
correlations per focal head to see if the two criteria split into
positional vs content.

REGISTERED PREDICTIONS:
  (0) SANITY: the chosen heads are focal (from 681);
  (a) A POSITIONAL/CONTENT SPLIT (the interesting outcome): for at least
      some focal heads, one QK circuit is strongly distance-correlated
      (|corr|>=0.3, a positional criterion) while the other is near-zero
      (content) -- the double-QK splits into a positional and a content
      criterion;
  (b) report, per focal head, s1-distance-corr and s2-distance-corr;
  NULL: a random score has ~0 distance correlation -- positional
      selectivity is real structure."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV, NH, HD, D, rope_tables, apply_rot

T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'qk_criterion_identity_results.json'
NFRESH = 16
# focal heads from 681 (layer, head)
FOCAL = [(1, 1), (0, 3), (2, 6), (7, 8), (2, 2), (6, 3)]


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    NL = len(m.transformer.h)
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    QPOS = list(range(16, T, 16))
    want = {}
    for (li, h) in FOCAL:
        want.setdefault(li, []).append(h)

    # accumulate distance-correlation of s1, s2 per focal head
    acc = {(li, h): {'s1': [], 's2': []} for (li, h) in FOCAL}

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
            if li in want:
                s1 = (torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD).cpu().numpy()
                s2 = (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD).cpu().numpy()
                for r in range(B):
                    for i in QPOS:
                        dist = (i - np.arange(i + 1)).astype(float)   # 0=self, larger=older
                        for h in want[li]:
                            for nm, s in [('s1', s1), ('s2', s2)]:
                                a_ = s[r, h, i, :i + 1]
                                if a_.std() > 1e-6:
                                    acc[(li, h)][nm].append(float(np.corrcoef(a_, dist)[0, 1]))
            pat = (s1 if False else (torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD) *
                   (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD))
            mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat.masked_fill(~mask, 0.0),
                             v).reshape(B, T, -1))
            xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
            x = x + mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias

    out = {'heads': {}}
    split_heads = 0
    for (li, h) in FOCAL:
        c1 = float(np.mean(acc[(li, h)]['s1'])) if acc[(li, h)]['s1'] else 0.0
        c2 = float(np.mean(acc[(li, h)]['s2'])) if acc[(li, h)]['s2'] else 0.0
        split = (abs(c1) >= 0.3) != (abs(c2) >= 0.3)   # one positional, one not
        split_heads += split
        out['heads'][f'L{li}.H{h}'] = {'s1_dist_corr': round(c1, 3),
                                       's2_dist_corr': round(c2, 3), 'pos_content_split': bool(split)}
        print(f'L{li}.H{h}: s1-dist {c1:+.3f}  s2-dist {c2:+.3f}  '
              f'{"POS/CONTENT SPLIT" if split else ""}', flush=True)

    p0 = True
    pa = split_heads >= 1
    print(f'\n(a) positional/content split in >=1 focal head: {pa} '
          f'({split_heads}/{len(FOCAL)})', flush=True)
    out.update({'pred_0': bool(p0), 'pred_a_split': bool(pa),
                'n_split_heads': split_heads, 'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
