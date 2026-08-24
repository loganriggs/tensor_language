# repeat_bilin12_late: which late layers/heads carry bilin12's NON-local readout-zone
# copy reading? (§1212 deviation: late band L10-11 costs 0.5575 = 17% of all12 — both 18L
# siblings' late bands were <1%.) Singles L8..L11 + late anchor locate the reader(s); if one
# layer dominates, its 6 heads are masked singly too.
#
# Instrument: as repeat_range_bilin12 (W=64 read-mask, pos-0 visible, repeat rows, t>=128).
# Conditions: base; L8..L11 singles; late (L10-11) anchor; per-head masks at L10 and L11
# (12 conditions) — head grain included up front since the layer set is tiny.
#
# Registered predictions:
#   pred_a ONE LATE STATION: max single late layer >= 0.6 x cost(late).
#   pred_b HEAD-CONCENTRATED: the top head at that layer >= 0.5 x its layer cost (family
#          pattern: stations reduce to 1-2 heads).
#   pred_c ANCHOR REPLICATES §1212 late 0.5575 (±0.05).
# Control: sanity base = true model ±0.005.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'repeat_bilin12_late_results.json'
DEV = 'cuda'
mdl, cfg = load_elriggs('bilin12', device=DEV, dtype=torch.float32); mdl.eval()
D = 768; NH = 6; HD = 128; NL = 12; T = 256; NR = 24; WIN = 64; QSTART = 128
V12 = int(mdl.lm_head.weight.shape[0])


@torch.no_grad()
def forward_banded(idx, spec, MASK_W, FULL):
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


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ar = torch.arange(T, device=DEV)
    vis = ((ar[:, None] - ar[None, :]) < WIN) | (ar[None, :] == 0)
    FULL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MASK_W = FULL & vis
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone().clamp_max(V12 - 1)
    ROWS[:, 128:256] = ROWS[:, 0:128]
    AH = set(range(NH))
    CONDS = {'base': {}, 'late': {10: AH, 11: AH}}
    for L in (8, 9, 10, 11):
        CONDS[f'L{L}'] = {L: AH}
    for L in (10, 11):
        for h in range(NH):
            CONDS[f'L{L}H{h}'] = {L: {h}}
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
    lsing = {L: cost[f'L{L}'] for L in (10, 11)}
    topL = max(lsing, key=lsing.get)
    hh = {h: cost[f'L{topL}H{h}'] for h in range(NH)}
    toph = max(hh, key=hh.get)
    out = {'model': 'bilin12', 'n_rows': NR, 'W': WIN, 'ce': CE, 'cost_vs_base': cost,
           'top_late_layer': topL, 'top_head': toph,
           'sanity': bool(abs(CE['base'] - CE['true_model']) <= 0.005),
           'pred_a_one_station': bool(lsing[topL] >= 0.6 * cost['late']),
           'pred_b_head_concentrated': bool(lsing[topL] > 0 and hh[toph] >= 0.5 * lsing[topL]),
           'pred_c_replicates': bool(abs(cost['late'] - 0.5575) <= 0.05),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"costs {cost} | top late L{topL} | top head H{toph}")
    print(f"sanity {out['sanity']} | pred_a one-station {out['pred_a_one_station']} | pred_b head {out['pred_b_head_concentrated']} | pred_c repl {out['pred_c_replicates']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
