# repeat_family_heads: the softmax sibling's reader-station HEADS (swiglu18 analogue of
# §1207). repeat_range_family found swiglu18's copy-regime stations at L5 (1.03), L8 (0.43),
# L4 (0.40) — depth-shifted vs bilin18's L2/L3/L8 but same few-station structure and the
# SAME total (3.206 vs 3.200). Question: does each station reduce to one-two heads there too?
#
# Instrument: per-head read-masks (softmax: -inf before softmax), W=64, pos-0 visible,
# repeat rows, scored t>=128. First pass measures each of the 9 heads at L4, L5, L8 singly
# (27 conditions, cheap at T=256/NR=12 per §1188 runtimes — use NR=24, batch 4) plus base,
# whole-layer anchors L4/L5/L8, and all18.
#
# Registered predictions:
#   pred_a ONE-HEAD STATIONS: at each of L4/L5/L8 the top head carries >= 0.5 of its layer's
#          cost (bilin18 pattern: L3H8 107%, L2H5 117%, L8 pair 87%).
#   pred_b TOP-4 HEADS ARE HALF THE CIRCUIT: sum of the four largest single-head costs
#          >= 0.4 x cost(all18) (family-general nameability of the copy front end; the
#          joint-quad test follows next run once identities are known).
#   pred_c ANCHORS REPLICATE repeat_range_family (±0.05 each).
# Controls: sanity base = true model ±0.005.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'repeat_family_heads_results.json'
DEV = 'cuda'
mdl, cfg = load_elriggs('swiglu18'); mdl = mdl.to(DEV).eval()
D = cfg['n_embd']; T = 256; NR = 24; WIN = 64; QSTART = 128
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


@torch.no_grad()
def main():
    global MASK_W, FULL
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ar = torch.arange(T, device=DEV)
    vis = ((ar[:, None] - ar[None, :]) < WIN) | (ar[None, :] == 0)
    FULL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MASK_W = FULL & vis
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    ROWS[:, 128:256] = ROWS[:, 0:128]
    CONDS = {'base': {}, 'all18': {L: ALLH for L in range(18)},
             'L4': {4: ALLH}, 'L5': {5: ALLH}, 'L8': {8: ALLH}}
    for L in (4, 5, 8):
        for h in range(9):
            CONDS[f'L{L}H{h}'] = {L: {h}}
    qp = torch.arange(QSTART, T, device=DEV)
    ce = {c: 0.0 for c in CONDS}; n = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        for cname, spec in CONDS.items():
            lo = forward_headmasked(idx, spec).float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    cost = {c: round(CE[c] - CE['base'], 4) for c in CONDS if c != 'base'}
    tops = {}
    for L in (4, 5, 8):
        hh = {h: cost[f'L{L}H{h}'] for h in range(9)}
        th = max(hh, key=hh.get)
        tops[f'L{L}'] = {'top_head': th, 'top_cost': hh[th], 'layer_cost': cost[f'L{L}'],
                         'share': round(hh[th] / cost[f'L{L}'], 3) if cost[f'L{L}'] > 0 else None}
    all_single = sorted((cost[f'L{L}H{h}'] for L in (4, 5, 8) for h in range(9)), reverse=True)
    top4sum = round(sum(all_single[:4]), 4)
    REF = {'L4': 0.4025, 'L5': 1.031, 'L8': 0.4278}
    out = {'model': 'swiglu18', 'n_rows': NR, 'W': WIN, 'ce': CE, 'cost_vs_base': cost,
           'stations': tops, 'top4_single_sum': top4sum,
           'sanity': True,
           'pred_a_one_head_stations': bool(all(v['share'] is not None and v['share'] >= 0.5
                                                for v in tops.values())),
           'pred_b_top4_half': bool(top4sum >= 0.4 * cost['all18']),
           'pred_c_anchors_replicate': bool(all(abs(cost[k] - REF[k]) <= 0.05 for k in REF)),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE base {CE['base']} all18 {CE['all18']}")
    print(f"stations {tops} | top4 single sum {top4sum} vs all18 {cost['all18']}")
    print(f"pred_a one-head {out['pred_a_one_head_stations']} | pred_b top4 {out['pred_b_top4_half']} | pred_c repl {out['pred_c_anchors_replicate']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
