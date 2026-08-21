"""ATTENTION DENSITY -- fresh architecture-specific probe. This model's
attention is a softmax-FREE double-bilinear product: pat = (q.k1)(q2.k2)
/HD^2, causal-masked, NOT normalized to a probability distribution. So
"attention" here can be negative, need not sum to 1, and its structure is
unconstrained. Are the patterns FOCAL (a few keys dominate, like softmax
attention) or DIFFUSE (spread over many keys)?

For each layer/head, compute the pattern pat over real text; per query
position, measure the effective number of keys attended = participation
ratio of |pat[i,:i+1]| ( (sum|w|)^2 / sum(w^2) ), normalized by the number
of valid keys (i+1). Low ratio = focal; near 1 = uniform/diffuse. Report
the distribution across heads and depths.

REGISTERED PREDICTIONS:
  (0) SANITY: patterns are non-degenerate (participation varies across
      heads);
  (a) MIXED, with FOCAL heads: at least some heads are focal (effective-
      keys fraction < 0.2 -- they concentrate on a few keys despite no
      softmax), i.e. the bilinear product can produce peaked patterns;
  (b) report the mean effective-keys fraction per depth and the most-focal
      and most-diffuse heads;
  NULL: a RANDOM pattern (gaussian) has participation fraction ~0.5-1
      (diffuse) -- focal heads are learned structure, not the baseline."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV, NH, HD, D, rope_tables, apply_rot

T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attention_density_results.json'
NFRESH = 16                      # fewer rows: per-head pattern is heavy


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    NL = len(m.transformer.h)
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    # accumulate mean participation fraction per (layer, head)
    part_sum = np.zeros((NL, NH)); part_n = np.zeros((NL, NH))
    QPOS = list(range(16, T, 16))    # sample query positions (enough context)

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
            pat = (s1 * s2).masked_fill(~mask, 0.0)          # (B,NH,T,T)
            aw = pat.abs().cpu().numpy()
            for r in range(B):
                for i in QPOS:
                    w = aw[r, :, i, :i + 1]                   # (NH, i+1)
                    s = w.sum(1); s2_ = (w ** 2).sum(1)
                    pr = (s ** 2) / (s2_ + 1e-12)             # participation ratio (# eff keys)
                    frac = pr / (i + 1)                       # fraction of valid keys
                    part_sum[li] += frac; part_n[li] += 1
            # advance residual
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
            x = x + mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias

    frac = part_sum / np.maximum(part_n, 1)      # (NL,NH) mean eff-keys fraction
    by_depth = {li: round(float(frac[li].mean()), 4) for li in range(NL)}
    flat = [(li, h, float(frac[li, h])) for li in range(NL) for h in range(NH)]
    focal = sorted(flat, key=lambda t: t[2])[:6]
    diffuse = sorted(flat, key=lambda t: -t[2])[:6]
    print('mean eff-keys fraction by depth:', by_depth, flush=True)
    print('most FOCAL heads (layer,head,frac):',
          [(li, h, round(f, 3)) for li, h, f in focal], flush=True)
    print('most DIFFUSE heads:',
          [(li, h, round(f, 3)) for li, h, f in diffuse], flush=True)

    # NULL: random gaussian pattern participation fraction
    g = np.random.default_rng(0)
    rr = np.abs(g.standard_normal((200, 128)))
    s = rr.sum(1); pr = (s ** 2) / (rr ** 2).sum(1); rand_frac = float((pr / 128).mean())

    n_focal = int((frac < 0.2).sum())
    p0 = float(frac.std()) > 0.02
    pa = n_focal >= 1
    null_ok = rand_frac > 0.4
    print(f'\n(0) non-degenerate: {p0}; (a) focal heads exist (frac<0.2): {pa} '
          f'({n_focal} heads); NULL random diffuse (frac {rand_frac:.2f}): {null_ok}',
          flush=True)

    out = {'by_depth_eff_key_frac': by_depth,
           'most_focal': [[li, h, round(f, 4)] for li, h, f in focal],
           'most_diffuse': [[li, h, round(f, 4)] for li, h, f in diffuse],
           'n_focal_heads': n_focal, 'random_frac': round(rand_frac, 4),
           'pred_0': bool(p0), 'pred_a_focal_exist': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
