# repeat_range_bilin12: the THIRD family member's copy-regime read map (completes §1204-10).
#
# bilin18 and swiglu18 pay the SAME whole-model copy-regime read cost (3.200 / 3.206 @W64)
# with few-station structure. bilin12 (12L, D=768, 6 heads, single-QK squared ROW-NORMALIZED
# attention) has markedly WEAKER induction (synthetic 4.3 vs 11.8, §885) — the family test:
# is the few-station law general, and does the total PRICE track induction strength (weaker
# copier = less long-range value at stake) rather than being a constant?
#
# Instrument: read-mask W=64, pos-0 visible, verbatim-repeat rows (128=first 128), scored
# t>=128. Row-normalized attention masked like softmax: window-zero the SQUARED scores, then
# renormalize over surviving keys. Bands for 12L: front L0-3, mid1 L4-7, mid2 L8-9, late
# L10-11, all12; singles L0..L7.
#
# Registered predictions:
#   pred_a FEW STATIONS (family law): top-2 singles >= 0.5 x singles sum.
#   pred_b PRICE TRACKS INDUCTION, NOT A CONSTANT: cost(all12) <= 0.7 x 3.20 — the 18L
#          constant does NOT extend to the weak-induction sibling.
#   pred_c LATE LOCAL + REPEAT STILL EASYish: cost(late) <= 0.1 x cost(all12); base CE <= 1.2
#          (§885's weaker induction still nearly solves verbatim repeat).
# Controls: sanity = custom unmasked forward vs model's own forward (±0.005).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'repeat_range_bilin12_results.json'
DEV = 'cuda'
mdl, cfg = load_elriggs('bilin12', device=DEV, dtype=torch.float32); mdl.eval()
D = 768; NH = 6; HD = 128; NL = 12; T = 256; NR = 24; WIN = 64; QSTART = 128
V12 = int(mdl.lm_head.weight.shape[0])


@torch.no_grad()
def forward_banded(idx, band, MASK_W, FULL):
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
        msk = MASK_W if L in band else FULL
        pat = s.square().masked_fill(~msk, 0.0)
        pat = pat / pat.sum(-1, keepdim=True).clamp_min(1e-9)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(v.dtype), v).reshape(B, T, D)
        x = x + a.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(mdl.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ar = torch.arange(T, device=DEV)
    vis = ((ar[:, None] - ar[None, :]) < WIN) | (ar[None, :] == 0)
    FULL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MASK_W = FULL & vis
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone().clamp_max(V12 - 1)
    ROWS[:, 128:256] = ROWS[:, 0:128]
    CONDS = {'base': set(), 'front': set(range(4)), 'mid1': set(range(4, 8)),
             'mid2': {8, 9}, 'late': {10, 11}, 'all12': set(range(NL))}
    for L in range(8):
        CONDS[f'L{L}'] = {L}
    qp = torch.arange(QSTART, T, device=DEV)
    ce = {c: 0.0 for c in CONDS}; ce_true = 0.0; n = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        from tier2_model import reference_forward
        lt = reference_forward(mdl, idx).float()
        ce_true += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        for cname, band in CONDS.items():
            lo = forward_banded(idx, band, MASK_W, FULL).float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / n, 4)
    cost = {c: round(CE[c] - CE['base'], 4) for c in CONDS if c != 'base'}
    singles = {f'L{L}': cost[f'L{L}'] for L in range(8)}
    ssum = sum(singles.values())
    top2 = sorted(singles.items(), key=lambda kv: -kv[1])[:2]
    out = {'model': 'bilin12', 'n_rows': NR, 'W': WIN, 'ce': CE, 'cost_vs_base': cost,
           'singles_sum': round(ssum, 4), 'top2_singles': top2,
           'family_refs': {'bilin18_all18': 3.2004, 'swiglu18_all18': 3.2059},
           'sanity': bool(abs(CE['base'] - CE['true_model']) <= 0.005),
           'pred_a_few_stations': bool(ssum > 0 and (top2[0][1] + top2[1][1]) >= 0.5 * ssum),
           'pred_b_price_tracks': bool(cost['all12'] <= 0.7 * 3.2004),
           'pred_c_late_easy': bool(cost['late'] <= 0.1 * cost['all12'] and CE['base'] <= 1.2),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"costs {cost} | top2 {top2} | singles sum {round(ssum,4)}")
    print(f"sanity {out['sanity']} | pred_a stations {out['pred_a_few_stations']} | pred_b price {out['pred_b_price_tracks']} | pred_c late+easy {out['pred_c_late_easy']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
