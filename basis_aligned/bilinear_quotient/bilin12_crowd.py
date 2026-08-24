# bilin12_crowd: does the crowd's scaling exponent track the SCORE FUNCTION? (§1224 test.)
# bilin18 (bilinear, unnormalized): synergistic, per-head value x1.56 from k=8 to 64.
# swiglu18 (softmax): coverage-like, x0.84. bilin12 = bilinear scores + row normalization —
# the same deciding position that settled the matcher mechanism (§1218: bilinearity won).
#
# Design (prose, natural rows, W=64 read-mask, pos-0 visible, scored t>=128): screen all
# 72 heads singly; greedy-12 core; random-k scaling k in {8, 48} x 3 draws from non-core
# heads (72-head model: k=48 plays the role of 64/162).
#
# Registered predictions:
#   pred_a SCORE FUNCTION DECIDES: per-head cost at k=48 >= 1.2 x per-head at k=8
#          (bilinear -> synergy, despite normalization — the §1218 pattern).
#   pred_b STATIONS TOP THE SCREEN: >= 2 of its copy heads {2.1, 2.3, 5.1, 5.5} in the
#          screening top-6.
#   pred_c NO COMPACT CORE: greedy-12 <= 0.55 x all12 (family law; bar loosened to 0.55
#          since 12 of 72 heads is proportionally more of this model).
# Controls: base = custom unmasked forward, sanity vs reference_forward (±0.005).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bilin12_crowd_results.json'
DEV = 'cuda'
mdl, cfg = load_elriggs('bilin12', device=DEV, dtype=torch.float32); mdl.eval()
D = 768; NH = 6; HD = 128; NL = 12; T = 256; NR = 24; WIN = 64; QSTART = 128
TOPK_SCREEN = 30; ROUNDS = 12
V12 = int(mdl.lm_head.weight.shape[0])

MASK_W = None
FULL = None


@torch.no_grad()
def forward_headmasked(idx, spec):
    dt = mdl.transformer.wte.weight.dtype
    x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    for L, blk in enumerate(mdl.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (D,))

        def qk(lin):
            z = lin(h).view(B, T, NH, HD)
            return apply_rot(F.rms_norm(z, (HD,)), cos, sin)

        v = a.c_v(h).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k = qk(a.c_q), qk(a.c_k)
        s = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / HD
        heads = spec.get(L, None)
        if heads is None:
            msk = FULL.expand(NH, T, T)
        else:
            msk = torch.stack([MASK_W if h in heads else FULL for h in range(NH)], 0)
        pat = s.square().masked_fill(~msk.unsqueeze(0), 0.0)
        pat = pat / pat.sum(-1, keepdim=True).clamp_min(1e-9)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(v.dtype), v).reshape(B, T, D)
        x = x + a.c_proj(y)
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
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clamp_max(V12 - 1)
    qp = torch.arange(QSTART, T, device=DEV)
    base = ce_of({}, ROWS, qp)
    tot = 0.0; n = 0
    for i in range(0, NR, 8):
        bb = ROWS[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lt = reference_forward(mdl, idx).float()
        tot += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        n += idx.shape[0] * len(qp)
    ce_true = tot / n
    allheads = [(L, h) for L in range(NL) for h in range(NH)]
    all12 = ce_of(spec_of(allheads), ROWS, qp) - base
    print(f"base {base:.4f} true {ce_true:.4f} | all12 {all12:.4f}", flush=True)
    singles = {}
    for L in range(NL):
        for h in range(NH):
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
        print(f"round {r + 1}: +{best[0]}.{best[1]} -> {cur:.4f} ({cur / all12:.0%})", flush=True)
    pool = [x for x in allheads if x not in chosen]
    g = torch.Generator().manual_seed(11)
    rk = {}
    for k in (8, 48):
        vals = []
        for s in range(3):
            perm = torch.randperm(len(pool), generator=g).tolist()
            vals.append(ce_of(spec_of([pool[j] for j in perm[:k]]), ROWS, qp) - base)
        rk[k] = round(sum(vals) / 3, 5)
    per8 = rk[8] / 8; per48 = rk[48] / 48
    top6 = [f'{L}.{h}' for (L, h), _ in top[:6]]
    stations = {'2.1', '2.3', '5.1', '5.5'}
    n_st = len(stations & set(top6))
    out = {'model': 'bilin12', 'n_rows': NR, 'W': WIN, 'base': round(base, 4),
           'all12_cost': round(all12, 4),
           'screen_top12': [[f'{L}.{h}', v] for (L, h), v in top[:12]],
           'greedy_curve': curve, 'chosen': [f'{L}.{h}' for L, h in chosen],
           'random_k_mean': rk, 'per_head': {'k8': round(per8, 6), 'k48': round(per48, 6)},
           'sanity': bool(abs(base - ce_true) <= 0.005),
           'pred_a_synergy': bool(per48 >= 1.2 * per8),
           'pred_b_stations_top': bool(n_st >= 2), 'stations_in_top6': n_st,
           'pred_c_no_compact_core': bool(curve and curve[-1]['joint_cost'] <= 0.55 * all12),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"top6 {top6} (stations {n_st}) | greedy12 {curve[-1]['joint_cost'] if curve else 0} of {all12:.4f} | per-head k8 {per8:.6f} k48 {per48:.6f}")
    print(f"sanity {out['sanity']} | pred_a synergy {out['pred_a_synergy']} | pred_b stations {out['pred_b_stations_top']} | pred_c nocore {out['pred_c_no_compact_core']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
