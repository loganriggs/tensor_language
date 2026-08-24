# bilin12_interference: locate the ensemble-scale interference. §1225: masking bilin12's
# 12-head core costs 0.2996 — MORE than masking all 72 heads (0.2478). Some unmasked heads'
# long-range reads are net harmful once the core is blind. WHICH?
#
# Conditions (prose, W=64 read-mask, pos-0 visible): base; core12 (§1225's chosen set,
# anchor); core12 + remaining-front (L0-3 non-core heads also masked); core12 + remaining-mid
# (L4-7); core12 + remaining-late (L8-11); all72.
#
# Registered predictions:
#   pred_a CONCENTRATED: the best single added band recovers >= 60% of the full recovery
#          (core12 − all72 = 0.052).
#   pred_b MONOTONE: adding ANY band to core12 reduces cost (no band's masking hurts).
#   pred_c ANCHORS: core12 within ±0.01 of 0.2996; all72 within ±0.01 of 0.2478.
# Control: sanity base = reference_forward (±0.005).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward, rope_tables, apply_rot
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bilin12_interference_results.json'
DEV = 'cuda'
mdl, cfg = load_elriggs('bilin12', device=DEV, dtype=torch.float32); mdl.eval()
D = 768; NH = 6; HD = 128; NL = 12; T = 256; NR = 24; WIN = 64; QSTART = 128
V12 = int(mdl.lm_head.weight.shape[0])
CORE = [(2,1),(5,5),(5,1),(7,0),(7,5),(10,0),(11,3),(10,2),(11,2),(2,3),(8,0),(5,0)]

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
    rest = [x for x in allheads if x not in CORE]
    bands = {'front': [x for x in rest if x[0] <= 3],
             'mid': [x for x in rest if 4 <= x[0] <= 7],
             'late': [x for x in rest if x[0] >= 8]}
    cost = {}
    cost['core12'] = round(ce_of(spec_of(CORE), ROWS, qp) - base, 4)
    cost['all72'] = round(ce_of(spec_of(allheads), ROWS, qp) - base, 4)
    for bn, bh in bands.items():
        cost[f'core12+{bn}'] = round(ce_of(spec_of(CORE + bh), ROWS, qp) - base, 4)
    rec_full = cost['core12'] - cost['all72']
    recs = {bn: round(cost['core12'] - cost[f'core12+{bn}'], 4) for bn in bands}
    best = max(recs, key=recs.get)
    out = {'model': 'bilin12', 'n_rows': NR, 'W': WIN, 'base': round(base, 4),
           'cost': cost, 'recovery_full': round(rec_full, 4), 'band_recoveries': recs,
           'best_band': best,
           'sanity': bool(abs(base - ce_true) <= 0.005),
           'pred_a_concentrated': bool(rec_full > 0 and recs[best] >= 0.6 * rec_full),
           'pred_b_monotone': bool(all(v >= -0.005 for v in recs.values())),
           'pred_c_anchors': bool(abs(cost['core12'] - 0.2996) <= 0.01 and
                                  abs(cost['all72'] - 0.2478) <= 0.01),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"cost {cost} | recoveries {recs} | best {best} of full {rec_full}")
    print(f"sanity {out['sanity']} | pred_a conc {out['pred_a_concentrated']} | pred_b mono {out['pred_b_monotone']} | pred_c anchors {out['pred_c_anchors']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
