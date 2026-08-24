# swiglu_crowd: is the crowd law family-general? Screen all 162 swiglu18 heads under the
# prose W=64 read-mask, greedy-select a 12-head core, and measure random-k scaling — the
# §1222-23 design on the softmax sibling.
#
# bilin18's answer: a nameable copy/induction core (12 heads = 43%, first picks = the §1207
# matchers) + a 150-head synergistic collective (random-k ~k^1.2); core ≈ tail in total value.
#
# Registered predictions:
#   pred_a ITS STATIONS TOP THE SCREEN: >= 2 of swiglu's copy heads {4.4, 5.2, 8.0, 8.8}
#          appear in the screening top-6 (prose read value concentrates on the copy front
#          end, family-general — bilin18: 2.5/3.8/5.5 were #1/#3/#2).
#   pred_b NO COMPACT CORE HERE EITHER: greedy-12 <= 0.5 x all18 (bilin18: 0.43).
#   pred_c SYNERGY FAMILY-GENERAL: per-head cost at k=64 >= 1.2 x per-head at k=8 (random
#          draws from non-core heads; bilin18: 1.56x).
# Controls: sanity base = model CE via same custom forward unmasked (structural); all18 near
# swiglu's 0.2231 (±0.03, §1188).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'swiglu_crowd_results.json'
DEV = 'cuda'
mdl, cfg = load_elriggs('swiglu18'); mdl = mdl.to(DEV).eval()
D = cfg['n_embd']; T = 256; NR = 24; WIN = 64; QSTART = 128; TOPK_SCREEN = 40; ROUNDS = 12
are = sys.modules[type(mdl.transformer.h[0].attn).__module__].apply_rotary_emb

MASK_W = None
FULL = None
ALLH = set(range(9))


@torch.no_grad()
def forward_headmasked(idx, spec):
    x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(mdl.transformer.h):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        q = F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,))
        k = F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        q = are(q, cos, sin); k = are(k, cos, sin)
        scores = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / (128 ** 0.5)
        heads = spec.get(L, None)
        if heads is None:
            scores = scores.masked_fill(~FULL, float('-inf'))
        else:
            msk = torch.stack([MASK_W if h in heads else FULL for h in range(9)], 0)
            scores = scores.masked_fill(~msk.unsqueeze(0), float('-inf'))
        pat = F.softmax(scores, dim=-1)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(mdl.lm_head(F.rms_norm(x, (D,))) / 30.0)




def spec_of(heads):
    sp = {}
    for L, h in heads:
        sp.setdefault(L, set()).add(h)
    return sp


@torch.no_grad()
def ce_of(spec, ROWS, qp):
    tot = 0.0; n = 0
    for i in range(0, NR, 8):
        bb = ROWS[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo = forward_headmasked(idx, spec).float()
        tot += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                     tgt[:, qp].reshape(-1), reduction='sum'))
        n += idx.shape[0] * len(qp)
    return tot / n


@torch.no_grad()
def main():
    global MASK_W, FULL
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ar = torch.arange(T, device=DEV)
    vis = ((ar[:, None] - ar[None, :]) < WIN) | (ar[None, :] == 0)
    FULL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MASK_W = FULL & vis
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    qp = torch.arange(QSTART, T, device=DEV)
    base = ce_of({}, ROWS, qp)
    allheads = [(L, h) for L in range(18) for h in range(9)]
    all18 = ce_of(spec_of(allheads), ROWS, qp) - base
    print(f"base {base:.4f} | all18 {all18:.4f}", flush=True)
    singles = {}
    for L in range(18):
        for h in range(9):
            singles[(L, h)] = round(ce_of({L: {h}}, ROWS, qp) - base, 5)
        print(f"screened L{L}", flush=True)
    top = sorted(singles.items(), key=lambda kv: -kv[1])[:TOPK_SCREEN]
    cand = [k for k, _ in top]
    chosen = []; curve = []; cur = 0.0
    for r in range(ROUNDS):
        best = None; bc = cur
        for c in cand:
            if c in chosen:
                continue
            cost = ce_of(spec_of(chosen + [c]), ROWS, qp) - base
            if cost > bc:
                bc = cost; best = c
        if best is None:
            break
        chosen.append(best); cur = bc
        curve.append({'head': f'{best[0]}.{best[1]}', 'joint_cost': round(cur, 4)})
        print(f"round {r + 1}: +{best[0]}.{best[1]} -> {cur:.4f} ({cur / all18:.0%})", flush=True)
    pool = [x for x in allheads if x not in chosen]
    g = torch.Generator().manual_seed(11)
    rk = {}
    for k in (8, 64):
        vals = []
        for s in range(3):
            perm = torch.randperm(len(pool), generator=g).tolist()
            vals.append(ce_of(spec_of([pool[j] for j in perm[:k]]), ROWS, qp) - base)
        rk[k] = round(sum(vals) / 3, 5)
    per8 = rk[8] / 8; per64 = rk[64] / 64
    top6 = [f'{L}.{h}' for (L, h), _ in top[:6]]
    stations = {'4.4', '5.2', '8.0', '8.8'}
    n_st = len(stations & set(top6))
    out = {'model': 'swiglu18', 'n_rows': NR, 'W': WIN, 'base': round(base, 4),
           'all18_cost': round(all18, 4),
           'screen_top12': [[f'{L}.{h}', v] for (L, h), v in top[:12]],
           'greedy_curve': curve, 'chosen': [f'{L}.{h}' for L, h in chosen],
           'random_k_mean': rk, 'per_head': {'k8': round(per8, 6), 'k64': round(per64, 6)},
           'all18_near_1188': bool(abs(all18 - 0.2231) <= 0.03),
           'pred_a_stations_top': bool(n_st >= 2), 'stations_in_top6': n_st,
           'pred_b_no_compact_core': bool(curve and curve[-1]['joint_cost'] <= 0.5 * all18),
           'pred_c_synergy': bool(per64 >= 1.2 * per8),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"top6 {top6} (stations {n_st}) | greedy12 {curve[-1]['joint_cost'] if curve else 0} of {all18:.4f} | per-head k8 {per8:.6f} k64 {per64:.6f}")
    print(f"pred_a stations {out['pred_a_stations_top']} | pred_b nocore {out['pred_b_no_compact_core']} | pred_c synergy {out['pred_c_synergy']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
