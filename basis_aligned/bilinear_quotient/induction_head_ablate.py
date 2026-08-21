"""INDUCTION HEAD ABLATE -- causally confirm 647's induction heads. 647
found by attention PATTERN that L5.H5 (+ L8.H4, L8.H6, L10.H8) point at
the copy-source. This mean-ablates those heads' attention output and
measures whether in-context copying (P of the copied continuation B)
drops -- the causal test, versus ablating matched RANDOM heads.

Head ablation: at the head's layer, replace that head's attention output
(its HD-slice, over all positions) with its position-mean before c_proj,
removing its position-specific contribution while keeping the mean.

REGISTERED PREDICTIONS:
  (0) SANITY: baseline P(B) reproduces 645 (~0.14);
  (a) INDUCTION HEADS ARE CAUSAL: ablating the top-4 induction heads
      {L5.H5, L8.H4, L8.H6, L10.H8} drops P(B) substantially more than
      ablating 4 matched RANDOM heads;
  (b) L5.H5 CARRIES THE MOST: ablating L5.H5 alone drops P(B) more than
      any single random head;
  (c) report P(B) under each ablation;
  NULL: ablating 4 random non-induction heads barely changes P(B)
      (the copying is specific to the induction heads)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV, NH, HD, D, rope_tables, apply_rot

T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'induction_head_ablate_results.json'
NFRESH = 48

TOP = {5: [5], 8: [4, 6], 10: [8]}                 # induction heads by layer
L5ONLY = {5: [5]}
RANDOM = {3: [0], 7: [3], 11: [6], 15: [2]}        # matched non-induction control


@torch.no_grad()
def pB(fresh, ablate, TB):
    """ablate: dict layer->list of heads to mean-fill. Returns mean P(B)
    over induction positions."""
    NLb = len(m.transformer.h)
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    vals = []
    for bi in range(0, NFRESH, 4):
        bb = fresh[bi:bi + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li in range(NLb):
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
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            ao = torch.einsum('bhqk,bkhd->bqhd', pat, v)     # (B,T,NH,HD)
            if ablate and li in ablate:
                for h in ablate[li]:
                    ao[:, :, h, :] = ao[:, :, h, :].mean(dim=(0, 1), keepdim=True)
            x = x + a.c_proj(ao.reshape(B, T, -1))
            xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
            x = x + mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1)
        for r in range(B):
            for i in range(T):
                if TB[bi + r, i] >= 0:
                    vals.append(float(p[r, i, TB[bi + r, i]]))
    return float(np.mean(vals))


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    toks = fresh[:, :257].numpy()
    TB = np.full((NFRESH, T), -1, np.int64)
    for r in range(NFRESH):
        last = {}
        for i in range(T):
            aa = int(toks[r, i])
            if aa in last and last[aa] <= i - 2:
                TB[r, i] = int(toks[r, last[aa] + 1])
            last[aa] = i
    n = int((TB >= 0).sum()); print(f'{n} induction positions', flush=True)

    base = pB(fresh, None, TB)
    top = pB(fresh, TOP, TB)
    l5 = pB(fresh, L5ONLY, TB)
    rand = pB(fresh, RANDOM, TB)
    print(f'baseline P(B)          {base:.4f}', flush=True)
    print(f'ablate top-4 induction {top:.4f}  (drop {base-top:+.4f})', flush=True)
    print(f'ablate L5.H5 only      {l5:.4f}  (drop {base-l5:+.4f})', flush=True)
    print(f'ablate 4 random heads  {rand:.4f}  (drop {base-rand:+.4f})', flush=True)

    p0 = 0.08 < base < 0.25
    pa = (base - top) > 2 * (base - rand)
    pb = (base - l5) > (base - rand)
    null_ok = (base - rand) < 0.3 * (base - top)
    print(f'\n(0) baseline ~0.14: {p0}', flush=True)
    print(f'(a) top induction heads causal (drop>2x random): {pa}', flush=True)
    print(f'(b) L5.H5 alone > random: {pb}', flush=True)
    print(f'NULL random heads barely matter: {null_ok}', flush=True)

    out = {'n_induction': n, 'baseline_PB': round(base, 5),
           'ablate_top4_PB': round(top, 5), 'ablate_L5H5_PB': round(l5, 5),
           'ablate_random_PB': round(rand, 5),
           'drop_top4': round(base - top, 5), 'drop_L5H5': round(base - l5, 5),
           'drop_random': round(base - rand, 5),
           'pred_0': bool(p0), 'pred_a_causal': bool(pa), 'pred_b_L5H5': bool(pb),
           'null_ok': bool(null_ok), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
