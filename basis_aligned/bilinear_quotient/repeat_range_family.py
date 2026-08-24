# repeat_range_family: is the copy regime's SERIAL geography family-general? swiglu18 bands
# + front/mid singles on verbatim-repeat rows (the §1204/§1206 map on the softmax sibling).
#
# bilin18's copy circuit read-grain map: serial across bands (front 1.91, mid1 2.64, sum
# 1.47x joint 3.20), reader stations L3 (1.01) > L8 (0.62) > L2 (0.53), gate L5 small.
# Prose pooling geography was family-general with one depth fingerprint (§1188). Does the
# COPY geography replicate on swiglu18 (softmax, single-QK)? Softmax masking = -inf before
# softmax (value_range_family harness verbatim); repeat rows as §1195.
#
# Conditions: base; bands front/mid1/mid2/late/all18; singles L0..L9.
#
# Registered predictions:
#   pred_a SERIAL IS FAMILY-GENERAL: cost(front) >= 0.4 x cost(all18) AND cost(mid1) >= 0.4 x
#          cost(all18) (each band alone catastrophic), and band sum >= 1.2 x cost(all18).
#   pred_b FEW READER STATIONS: top-2 single layers carry >= 0.5 of the singles sum
#          (identifiable stations, not a uniform smear) — layer IDENTITIES logged as the
#          fingerprint (bilin18: L3/L8; no registered claim they match).
#   pred_c LATE LOCAL + REPEAT EASY: cost(late) <= 0.05 x cost(all18) and base CE <= 0.5
#          (swiglu18's induction solves the repeat too — §885 strength scales with size).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'repeat_range_family_results.json'
DEV = 'cuda'
mdl, cfg = load_elriggs('swiglu18'); mdl = mdl.to(DEV).eval()
D = cfg['n_embd']; T = 256; NR = 24; WIN = 64; QSTART = 128
are = sys.modules[type(mdl.transformer.h[0].attn).__module__].apply_rotary_emb


@torch.no_grad()
def forward_banded(idx, band, MASK_W, FULL):
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
        msk = MASK_W if L in band else FULL
        scores = scores.masked_fill(~msk, float('-inf'))
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
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ar = torch.arange(T, device=DEV)
    vis = ((ar[:, None] - ar[None, :]) < WIN) | (ar[None, :] == 0)
    FULL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MASK_W = FULL & vis
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    ROWS[:, 128:256] = ROWS[:, 0:128]
    CONDS = {'base': set(), 'front': set(range(5)), 'mid1': set(range(5, 10)),
             'mid2': set(range(10, 15)), 'late': {15, 16, 17}, 'all18': set(range(18))}
    for L in range(10):
        CONDS[f'L{L}'] = {L}
    qp = torch.arange(QSTART, T, device=DEV)
    ce = {c: 0.0 for c in CONDS}; n = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        for cname, band in CONDS.items():
            lo = forward_banded(idx, band, MASK_W, FULL).float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    cost = {c: round(CE[c] - CE['base'], 4) for c in CONDS if c != 'base'}
    singles = {f'L{L}': cost[f'L{L}'] for L in range(10)}
    ssum = sum(singles.values())
    top2 = sorted(singles.items(), key=lambda kv: -kv[1])[:2]
    bandsum = cost['front'] + cost['mid1'] + cost['mid2'] + cost['late']
    out = {'model': 'swiglu18', 'n_rows': NR, 'W': WIN, 'ce': CE, 'cost_vs_base': cost,
           'band_sum': round(bandsum, 4), 'singles_sum': round(ssum, 4),
           'top2_singles': top2,
           'bilin18_refs': {'front': 1.9124, 'mid1': 2.6438, 'all18': 3.2004,
                            'top_singles': {'L3': 1.0076, 'L8': 0.6194, 'L2': 0.5316}},
           'pred_a_serial_general': bool(cost['front'] >= 0.4 * cost['all18'] and
                                         cost['mid1'] >= 0.4 * cost['all18'] and
                                         bandsum >= 1.2 * cost['all18']),
           'pred_b_few_stations': bool(ssum > 0 and (top2[0][1] + top2[1][1]) >= 0.5 * ssum),
           'pred_c_late_easy': bool(cost['late'] <= 0.05 * cost['all18'] and CE['base'] <= 0.5),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"costs {cost} | band sum {out['band_sum']} vs all18 {cost['all18']} | top2 {top2}")
    print(f"pred_a serial {out['pred_a_serial_general']} | pred_b stations {out['pred_b_few_stations']} | pred_c late+easy {out['pred_c_late_easy']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
