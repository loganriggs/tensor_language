"""INDUCTION HEAD SEARCH -- find the actual heads that implement the
in-context copying circuit (645/646). Unlike the diffuse MLP circuits,
induction heads are often localizable, so this may reach component level.

At an induction position i (current token A, previous occurrence at j,
copy-source s = j+1 = the token that followed A before), an induction
head should point its attention at s -- it matches the current token to
where its earlier copy sits and reads the next token. This reimplements
the model's double-QK bilinear attention (pat = (q.k1)(q2.k2)/HD^2,
causal-masked, NO softmax) and scores every head by how strongly its
raw pattern at induction positions targets the copy-source s.

Metric per head: (1) z-score of pat[i,s] within the head's valid keys
pat[i,0:i+1] (how much it stands out); (2) fraction of induction
positions where s is the head's argmax key.

REGISTERED PREDICTIONS:
  (0) SANITY: >= 200 induction positions with antecedent >= 2 back;
  (a) INDUCTION HEADS EXIST AND ARE LOCALIZABLE: a small number of heads
      have a mean copy-source z-score >= 1.0 (they specifically target
      the copy-source), well above the head population median;
  (b) THEY SIT IN FRONT+MID LAYERS: the top induction heads are in
      layers 0-9 (consistent with 645's front+mid attention
      localization), not the late layers;
  (c) report the top-8 heads by copy-source z-score and argmax fraction;
  NULL: a shuffled control key (a random valid key instead of s) has
      near-zero mean z-score for the same heads -- the targeting is
      specific to the true copy-source."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV, NH, HD, D, rope_tables, apply_rot

T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'induction_head_search_results.json'
NFRESH = 48


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    toks = fresh[:, :257].numpy()

    # induction positions: (row, i, s) with s = j+1, previous occurrence j <= i-2
    ind = [[] for _ in range(NFRESH)]        # per row: list of (i, s)
    n_ind = 0
    for r in range(NFRESH):
        last = {}
        for i in range(T):
            a = int(toks[r, i])
            if a in last and last[a] <= i - 2:
                ind[r].append((i, last[a] + 1))
                n_ind += 1
            last[a] = i
    print(f'{n_ind} induction positions (antecedent >=2 back)', flush=True)

    NL = len(m.transformer.h)
    # accumulators per (layer, head)
    z_sum = np.zeros((NL, NH)); z_ctrl = np.zeros((NL, NH))
    argmax_hit = np.zeros((NL, NH)); cnt = 0
    rng = np.random.default_rng(0)

    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))

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
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, float('nan'))   # (B,NH,T,T)
            patc = pat.cpu().numpy()

            for r in range(B):
                for (i, s) in ind[bi + r]:
                    row = patc[r, :, i, :i + 1]                 # (NH, i+1)
                    mu = np.nanmean(row, axis=1); sd = np.nanstd(row, axis=1) + 1e-9
                    z_sum[li] += (row[:, s] - mu) / sd
                    # control: a random valid key != s
                    kc = int(rng.integers(0, i + 1))
                    if i > 0 and kc == s:
                        kc = (s + 1) % (i + 1)
                    z_ctrl[li] += (row[:, kc] - mu) / sd
                    argmax_hit[li] += (np.nanargmax(row, axis=1) == s)
            # advance residual with the real attention output
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd',
                             (s1 * s2).masked_fill(~mask, 0.0), v).reshape(B, T, -1))
            xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
            x = x + mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias
        cnt += sum(len(ind[bi + r]) for r in range(B))

    z_mean = z_sum / cnt; zc_mean = z_ctrl / cnt; am = argmax_hit / cnt
    flat = [(li, h, float(z_mean[li, h]), float(am[li, h]), float(zc_mean[li, h]))
            for li in range(NL) for h in range(NH)]
    flat.sort(key=lambda t: -t[2])
    top = flat[:8]
    print('top induction heads (layer.head  z(copy-src)  argmax-frac  z(control)):',
          flush=True)
    for li, h, z, a_, zc in top:
        print(f'  L{li}.H{h}   z {z:+.3f}   argmax {a_:.3f}   ctrl-z {zc:+.3f}',
              flush=True)

    p0 = n_ind >= 200
    pa = top[0][2] >= 1.0
    top_layers = [li for li, h, z, a_, zc in top]
    pb = np.median(top_layers) <= 9
    null_ok = abs(np.mean([zc for *_, zc in top])) < 0.3
    print(f'\n(0) enough: {p0}', flush=True)
    print(f'(a) induction heads exist (top z>=1.0): {pa} (top z {top[0][2]:.2f})',
          flush=True)
    print(f'(b) top heads in front+mid (median layer<=9): {pb} '
          f'(layers {sorted(top_layers)})', flush=True)
    print(f'NULL control-key z near 0 for top heads: {null_ok}', flush=True)

    out = {'n_induction_pos': n_ind,
           'top_heads': [{'layer': li, 'head': h, 'z_copysrc': round(z, 4),
                          'argmax_frac': round(a_, 4), 'z_control': round(zc, 4)}
                         for li, h, z, a_, zc in top],
           'pred_0': bool(p0), 'pred_a_heads_exist': bool(pa),
           'pred_b_front_mid': bool(pb), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
