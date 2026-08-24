# bilin12_interference2: per-layer split of the TOXIC FRONT. §1226: adding the non-core
# front heads (L0-3) to the core mask recovers 0.102. Which front layer's heads are the
# toxic readers? Conditions: base; core12; core12 + each single front layer's non-core
# heads (L0/L1/L2/L3); core12+front (anchor); all72 (anchor).
# Registered predictions:
#   pred_a ONE LAYER DOMINATES: best single front layer >= 0.6 x the 0.102 front recovery.
#   pred_b IT IS L2 OR L3 (the copy-station layers — siblings of the §1213 pattern where
#          the auxiliary sits NEXT TO the matcher): the dominant layer hosts a core station.
#   pred_c ANCHORS replicate §1226 (±0.01).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward, rope_tables, apply_rot
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bilin12_interference2_results.json'
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
    bands = {'front': [x for x in rest if x[0] <= 3]}
    for L in range(4):
        bands[f'L{L}'] = [x for x in rest if x[0] == L]
    cost = {}
    cost['core12'] = round(ce_of(spec_of(CORE), ROWS, qp) - base, 4)
    cost['all72'] = round(ce_of(spec_of(allheads), ROWS, qp) - base, 4)
    for bn, bh in bands.items():
        cost[f'core12+{bn}'] = round(ce_of(spec_of(CORE + bh), ROWS, qp) - base, 4)
    rec_front = cost['core12'] - cost['core12+front']
    recs = {bn: round(cost['core12'] - cost[f'core12+{bn}'], 4) for bn in bands if bn != 'front'}
    best = max(recs, key=recs.get)
    out = {'model': 'bilin12', 'n_rows': NR, 'W': WIN, 'base': round(base, 4),
           'cost': cost, 'front_recovery': round(rec_front, 4), 'layer_recoveries': recs,
           'best_layer': best,
           'sanity': bool(abs(base - ce_true) <= 0.005),
           'pred_a_one_layer': bool(rec_front > 0 and recs[best] >= 0.6 * rec_front),
           'pred_b_station_layer': bool(best in ('L2',)),
           'pred_c_anchors': bool(abs(cost['core12'] - 0.2996) <= 0.01 and
                                  abs(cost['core12+front'] - 0.1978) <= 0.01),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"cost {cost} | layer recoveries {recs} | best {best} of front {rec_front}")
    print(f"sanity {out['sanity']} | pred_a one-layer {out['pred_a_one_layer']} | pred_b station {out['pred_b_station_layer']} | pred_c anchors {out['pred_c_anchors']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
