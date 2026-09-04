"""INDUCTION HEADSET ABLATE -- directly answers "why can't we narrow to
components": single-head ablation under-reads because heads are redundant
(648). The right test is ablating the whole SET of heads that write the
copy signal. Cumulatively mean-ablate the top-K induction heads (647
ranking) and measure how P(B) collapses, versus random K-head sets and
the all-attention ceiling.

If ablating the top-K set (for modest K) collapses copying toward the
all-attention-ablated floor, the circuit IS localized -- to a head-SET,
redundant within it, which is a real component-level answer.

647 induction-head ranking by z(copy-source):
  L5.H5, L8.H4, L8.H6, L10.H8, L13.H2, L2.H1, L14.H0, L14.H7, ...

REGISTERED PREDICTIONS:
  (0) SANITY: baseline P(B) ~ 0.14; all-attention-ablated is the floor;
  (a) SET COLLAPSES COPYING: ablating the top-8 induction heads drops
      P(B) toward the all-attention floor -- markedly more than 8 random
      heads, and enough that the SET (not a single head) is the circuit;
  (b) MONOTONE, SUB-LINEAR: P(B) falls monotonically with K but each
      added head contributes less (redundant within the set);
  (c) report P(B) for K = 1,2,4,8,16 induction heads, K=8 random, and
      all-attention-ablated;
  NULL: random K-head sets drop P(B) far less than the induction K-set
      at every K."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV, NH, HD, D, rope_tables, apply_rot

T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'induction_headset_ablate_results.json'
NFRESH = 48

# 647 ranking (layer, head), strongest first
RANK = [(5, 5), (8, 4), (8, 6), (10, 8), (13, 2), (2, 1), (14, 0), (14, 7),
        (6, 3), (9, 1), (7, 2), (11, 4), (4, 5), (12, 6), (3, 7), (1, 3)]


def to_dict(pairs):
    d = {}
    for (li, h) in pairs:
        d.setdefault(li, []).append(h)
    return d


@torch.no_grad()
def pB(fresh, ablate, TB, all_attn=False):
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
            ao = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            if all_attn:
                ao = ao.mean(dim=(0, 1), keepdim=True).expand_as(ao)
            elif ablate and li in ablate:
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
    print(f'{int((TB>=0).sum())} induction positions', flush=True)

    base = pB(fresh, None, TB)
    floor = pB(fresh, None, TB, all_attn=True)
    print(f'baseline {base:.4f}   all-attention-ablated floor {floor:.4f}', flush=True)

    rng = np.random.default_rng(1)
    allheads = [(li, h) for li in range(len(m.transformer.h)) for h in range(NH)
                if (li, h) not in set(RANK)]
    curve = {}
    for K in [1, 2, 4, 8, 16]:
        pk = pB(fresh, to_dict(RANK[:K]), TB)
        rp = rng.choice(len(allheads), size=K, replace=False)
        randset = [allheads[j] for j in rp]
        pr = pB(fresh, to_dict(randset), TB)
        curve[K] = {'induction': round(pk, 5), 'random': round(pr, 5)}
        frac = (base - pk) / (base - floor + 1e-9)
        print(f'K={K:2d}: induction P(B) {pk:.4f} (drop {base-pk:+.4f}, '
              f'{100*frac:.0f}% of floor gap)  random {pr:.4f}', flush=True)

    p0 = 0.08 < base < 0.25 and floor < base
    k8 = curve[8]['induction']
    pa = (base - k8) > 2 * (base - curve[8]['random'])
    inds = [curve[K]['induction'] for K in [1, 2, 4, 8, 16]]
    pb = all(inds[i] >= inds[i + 1] - 1e-4 for i in range(len(inds) - 1))
    null_ok = all((base - curve[K]['induction']) > 1.5 * (base - curve[K]['random'])
                  for K in [4, 8, 16])
    frac8 = (base - k8) / (base - floor + 1e-9)
    print(f'\n(0) sane: {p0}', flush=True)
    print(f'(a) top-8 set collapses copying (drop>2x random, {100*frac8:.0f}% of '
          f'floor gap): {pa}', flush=True)
    print(f'(b) monotone in K: {pb}', flush=True)
    print(f'NULL induction K-set >> random K-set: {null_ok}', flush=True)

    out = {'baseline': round(base, 5), 'all_attn_floor': round(floor, 5),
           'curve': {str(k): v for k, v in curve.items()},
           'top8_frac_of_floor_gap': round(frac8, 4),
           'pred_0': bool(p0), 'pred_a_set_collapses': bool(pa),
           'pred_b_monotone': bool(pb), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
