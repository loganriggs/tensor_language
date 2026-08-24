# seed_ensemble: WHICH heads carry the prose pooling crowd? (FINDINGS Open A, attacked with
# the arc-validated read-mask instrument.)
#
# The prose long-range budget is 0.176 @W64 (§1186, replicated 3x this arc), carried
# collectively: every single band-layer <= 0.009 (§1187), quad copy stations = 22% (§1211).
# Open A asks for the ENSEMBLE: the minimal head set whose joint read-mask reproduces the
# crowd's value. Method: (1) screen all 162 heads singly (W=64 read-mask, pos-0 visible,
# natural rows, scored t>=128); (2) GREEDY forward selection from the top-40 screened heads —
# at each round add the head whose joint mask costs most — for 12 rounds; (3) all18 anchor.
#
# Registered predictions:
#   pred_a CONCENTRATION AT ENSEMBLE GRAIN: some <= 12-head set reaches >= 50% of all18's
#          cost (the crowd has a core, even though singles are tiny).
#   pred_b THE CORE IS MID-HEAVY: >= 60% of the selected heads sit in L4-L9 (the §1186
#          band geography at head grain).
#   pred_c SUPER-ADDITIVE CORE: the final set's joint cost >= 2x the sum of its members'
#          single costs (crowd signature §1187, now at the named-ensemble level).
# Controls: sanity base = true model (±0.005); all18 near 0.176 (±0.03); the copy quad's
# singles must match §1211-era screening (2.5/3.8 among top screened heads — consistency).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'seed_ensemble_results.json'
NR = 24; WIN = 64; QSTART = 128; TOPK_SCREEN = 40; ROUNDS = 12
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb

MASK_W = None
FULL_TRIL = None


def make_mask():
    ar = torch.arange(T, device=DEV)
    vis = ((ar[:, None] - ar[None, :]) < WIN) | (ar[None, :] == 0)
    return (torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)) & vis)


@torch.no_grad()
def forward_headmasked(idx, spec):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
        q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
        heads = spec.get(L, None)
        if heads is None:
            pat = pat.masked_fill(~FULL_TRIL, 0.0)
        else:
            msk = torch.stack([MASK_W if h in heads else FULL_TRIL for h in range(9)], 0)
            pat = pat.masked_fill(~msk.unsqueeze(0), 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


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
    global MASK_W, FULL_TRIL
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    MASK_W = make_mask()
    FULL_TRIL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    qp = torch.arange(QSTART, T, device=DEV)

    base = ce_of({}, ROWS, qp)
    # true-model sanity
    tot = 0.0; n = 0
    for i in range(0, NR, 8):
        bb = ROWS[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        tot += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        n += idx.shape[0] * len(qp)
    ce_true = tot / n
    all18 = ce_of({L: set(range(9)) for L in range(18)}, ROWS, qp) - base
    print(f"base {base:.4f} (true {ce_true:.4f}) | all18 cost {all18:.4f}", flush=True)

    # 1) screen all 162
    singles = {}
    for L in range(18):
        for h in range(9):
            singles[(L, h)] = round(ce_of({L: {h}}, ROWS, qp) - base, 5)
        print(f"screened L{L}: top {max((v, f'{L}.{k}') for (l2, k), v in singles.items() if l2 == L)}", flush=True)
    top = sorted(singles.items(), key=lambda kv: -kv[1])[:TOPK_SCREEN]
    cand = [k for k, _ in top]

    # 2) greedy forward selection
    chosen = []; curve = []
    cur_cost = 0.0
    for r in range(ROUNDS):
        best = None; best_cost = cur_cost
        for c in cand:
            if c in chosen:
                continue
            cost = ce_of(spec_of(chosen + [c]), ROWS, qp) - base
            if cost > best_cost:
                best_cost = cost; best = c
        if best is None:
            break
        chosen.append(best); cur_cost = best_cost
        curve.append({'head': f'{best[0]}.{best[1]}', 'joint_cost': round(cur_cost, 4)})
        print(f"round {r + 1}: +{best[0]}.{best[1]} -> joint {cur_cost:.4f} ({cur_cost / all18:.0%} of all18)", flush=True)

    sum_singles = sum(singles[c] for c in chosen)
    n_mid = sum(1 for (L, h) in chosen if 4 <= L <= 9)
    reach50 = next((i + 1 for i, p in enumerate(curve) if p['joint_cost'] >= 0.5 * all18), None)
    out = {'n_rows': NR, 'W': WIN, 'base': round(base, 4), 'true_model': round(ce_true, 4),
           'all18_cost': round(all18, 4),
           'screen_top20': [[f'{L}.{h}', v] for (L, h), v in top[:20]],
           'greedy_curve': curve, 'chosen': [f'{L}.{h}' for L, h in chosen],
           'sum_singles_of_chosen': round(sum_singles, 4),
           'sanity': bool(abs(base - ce_true) <= 0.005),
           'all18_near_1186': bool(abs(all18 - 0.1759) <= 0.03),
           'pred_a_concentration': bool(reach50 is not None and reach50 <= 12),
           'reach50_at': reach50,
           'pred_b_mid_heavy': bool(len(chosen) > 0 and n_mid / len(chosen) >= 0.6),
           'pred_c_superadditive': bool(len(curve) > 0 and curve[-1]['joint_cost'] >= 2 * sum_singles),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"chosen {out['chosen']} | joint {curve[-1]['joint_cost'] if curve else 0} vs singles-sum {out['sum_singles_of_chosen']}")
    print(f"sanity {out['sanity']} | near1186 {out['all18_near_1186']} | pred_a conc {out['pred_a_concentration']} (50% at {reach50}) | pred_b mid {out['pred_b_mid_heavy']} | pred_c super {out['pred_c_superadditive']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
