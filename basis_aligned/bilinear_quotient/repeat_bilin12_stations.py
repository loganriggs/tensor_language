# repeat_bilin12_stations: bilin12's mid stations at head grain + WHICH MECHANISM —
# the normalization test. bilin18 (unnormalized bilinear patterns) matches the SOURCE by
# direct long-range reads (front stations at offset 128, §1215); swiglu18 (softmax) has
# NO source-matcher — all stations are successor-fetchers at o=127, textbook induction
# via local key-composition (§1217). bilin12 is the deciding case: bilinear-squared like
# bilin18, but ROW-NORMALIZED like softmax. If normalization decides the mechanism, its
# far-reads peak at 127 (fetch); if bilinearity decides, a 128 source-matcher appears.
#
# Part 1 (masks): per-head W=64 read-masks at L2 and L5 on repeat rows (6 heads each) —
# does the least-modular sibling (§1214 late crowd) have sharp mid stations or crowds?
# Part 2 (offsets): pattern-mass-by-offset shares (normalized patterns, key 0 excluded,
# queries t>=160) for all heads at L2/L5.
#
# Registered predictions:
#   pred_a MID STATIONS ARE CROWDS TOO: at both L2 and L5 the top head < 0.5 x layer cost
#          (the §1214 "least modular" reading extends to its mid stations).
#   pred_b NORMALIZATION DECIDES: the top far-reading head at each of L2/L5 peaks at
#          o=127 (successor), not 128 — bilin12 sides with softmax despite bilinear scores.
#   pred_c ANCHORS REPLICATE §1212 (L2 1.3052, L5 1.7022, ±0.06).
# Control: sanity base = reference_forward CE (±0.005).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'repeat_bilin12_stations_results.json'
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
    CONDS = {'base': {}}
    for L in (2, 5):
        CONDS[f'L{L}'] = {L: AH}
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
    # Part 2: offset shares (normalized |pattern|) at L2/L5
    QS = 160
    accs = {L: torch.zeros(NH, 4, dtype=torch.float64) for L in (2, 5)}
    dt = mdl.transformer.wte.weight.dtype
    from tier2_model import rope_tables as rt2, apply_rot as ar2
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous()
        for LT in (2, 5):
            x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
            B = idx.shape[0]
            cos, sin = rt2(T, HD, idx.device, dt, 'bf16')
            cos, sin = cos[None, :, None, :], sin[None, :, None, :]
            pat_t = None
            for L, blk in enumerate(mdl.transformer.h):
                x = blk.lambdas[0] * x + blk.lambdas[1] * x0
                a = blk.attn
                h = F.rms_norm(x, (D,))
                q = ar2(F.rms_norm(a.c_q(h).view(B, T, NH, HD), (HD,)), cos, sin)
                k = ar2(F.rms_norm(a.c_k(h).view(B, T, NH, HD), (HD,)), cos, sin)
                s = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / HD
                tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
                pat = s.square().masked_fill(~tril, 0.0)
                pat = pat / pat.sum(-1, keepdim=True).clamp_min(1e-9)
                if L == LT:
                    pat_t = pat
                    break
                v = a.c_v(h).view(B, T, NH, HD)
                if v1 is None:
                    v1 = v
                vv = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
                y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
                x = x + a.c_proj(y)
                x = x + blk.mlp(F.rms_norm(x, (D,)))
            qsl = torch.arange(QS, T, device=DEV)
            p = pat_t[:, :, qsl, 1:]
            keys = torch.arange(1, T, device=DEV)
            off = qsl[:, None] - keys[None, :]
            far = off > 64
            m127 = (p * ((off == 127) & far)).sum((0, 2, 3)); m128 = (p * ((off == 128) & far)).sum((0, 2, 3))
            mband = (p * ((off >= 126) & (off <= 130) & far)).sum((0, 2, 3)); mfar = (p * far).sum((0, 2, 3))
            accs[LT] += torch.stack([m127, m128, mband, mfar], 1).double().cpu()
    shares = {}
    for L in (2, 5):
        a = accs[L]
        for h in range(NH):
            tot = float(a[h, 3])
            shares[f'{L}.{h}'] = {'share_127': round(float(a[h, 0]) / tot, 4) if tot > 0 else None,
                                  'share_128': round(float(a[h, 1]) / tot, 4) if tot > 0 else None,
                                  'share_band': round(float(a[h, 2]) / tot, 4) if tot > 0 else None,
                                  'far_mass': round(tot, 2)}
    picks = {}
    ok_b = True
    for L in (2, 5):
        hh = {h: cost[f'L{L}H{h}'] for h in range(NH)}
        toph = max(hh, key=hh.get)
        sh = shares[f'{L}.{toph}']
        peak = '127' if (sh['share_127'] or 0) >= (sh['share_128'] or 0) else '128'
        picks[f'L{L}'] = {'top_head': toph, 'head_cost': hh[toph], 'layer_cost': cost[f'L{L}'],
                          'share_of_layer': round(hh[toph] / cost[f'L{L}'], 3) if cost[f'L{L}'] > 0 else None,
                          'peak': peak}
        if peak != '127':
            ok_b = False
    pa = all(v['share_of_layer'] is not None and v['share_of_layer'] < 0.5 for v in picks.values())
    pc = abs(cost['L2'] - 1.3052) <= 0.06 and abs(cost['L5'] - 1.7022) <= 0.06
    out = {'model': 'bilin12', 'n_rows': NR, 'W': WIN, 'ce': CE, 'cost_vs_base': cost,
           'offset_shares': shares, 'stations': picks,
           'sanity': bool(abs(CE['base'] - CE['true_model']) <= 0.005),
           'pred_a_crowds': bool(pa), 'pred_b_normalization_decides': bool(ok_b),
           'pred_c_replicates': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"costs {cost}")
    print(f"stations {json.dumps(picks)}")
    print(f"sanity {out['sanity']} | pred_a crowds {out['pred_a_crowds']} | pred_b norm {out['pred_b_normalization_decides']} | pred_c repl {out['pred_c_replicates']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
