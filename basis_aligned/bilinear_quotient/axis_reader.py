# axis_reader: WHO consumes the match-verdict axis? §1249 removed it at all block entries
# 4-8 (cost 1.00). Band-restricted removal: {4-5}, {6-8}, {9-12}, {13-17}, plus the full 4-8
# anchor. If the fetch band (L8) is the reader, 6-8 carries most and 9+ nothing; if the
# readout reads the axis directly, late removal also costs.
#
# Registered predictions:
#   pred_a CONSUMED BY 8: removal at 9-12 and 13-17 each <= 0.15 nats.
#   pred_b THE READER IS LATE-FRONT/MID: cost(6-8) >= 0.5 x cost(4-8 anchor).
#   pred_c ANCHOR: 4-8 replicates §1249's 1.001 (±0.1); sanity base = true model.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'axis_reader_results.json'
NFIT = 12; NR = 24; QSTART = 128; QFIT = 160
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
MATCHERS = {2: [5], 3: [8]}


@torch.no_grad()
def fit_axis(rows):
    DL = []
    for i in range(0, NFIT, 4):
        idx = rows[i:i + 4, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]
        wsum = torch.zeros(B, T, D, device=DEV)
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
            tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
            pat = pat.masked_fill(~tril, 0.0)
            v = at.c_v(xin).view(B, T, 9, 128)
            if v1 is None:
                v1 = v
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            if L in MATCHERS:
                for h in MATCHERS[L]:
                    yh = torch.zeros_like(y)
                    yh[:, :, h, :] = y[:, :, h, :]
                    wsum = wsum + at.c_proj(yh.reshape(B, T, D)).float()
            x = xm + at.c_proj(y.reshape(B, T, D))
            if L == 3:
                full = blk.mlp(F.rms_norm(x, (D,)))
                blind = blk.mlp(F.rms_norm(x - wsum.to(x.dtype), (D,)))
                DL.append((full - blind).float()[:, QFIT:].reshape(-1, D).cpu())
                break
            x = x + blk.mlp(F.rms_norm(x, (D,)))
    DL = torch.cat(DL)
    Dc = DL - DL.mean(0)
    _, _, V = torch.pca_lowrank(Dc, q=4)
    return V[:, 0].to(DEV)




@torch.no_grad()
def forward_rm_band(idx, d, band):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for L, blk in enumerate(m.transformer.h):
        if d is not None and L in band:
            x = x - (x * d).sum(-1, keepdim=True) * d
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def ce_of(rows, d, band):
    qp = torch.arange(QSTART, T, device=DEV)
    tot = 0.0; n = 0
    for i in range(0, rows.shape[0], 4):
        bb = rows[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo = forward_rm_band(idx, d, band).float()
        tot += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                     tgt[:, qp].reshape(-1), reduction='sum'))
        n += idx.shape[0] * len(qp)
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    FIT = cl.fineweb_rows(NFIT)[:, :T + 1].contiguous().clone()
    FIT[:, 128:256] = FIT[:, 0:128]
    REP = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    REP[:, 128:256] = REP[:, 0:128]
    d = fit_axis(FIT)
    BANDS = {'base': None, 'b45': {4, 5}, 'b68': {6, 7, 8}, 'b912': {9, 10, 11, 12},
             'b1317': {13, 14, 15, 16, 17}, 'b48': {4, 5, 6, 7, 8}}
    CE = {}
    for name, band in BANDS.items():
        CE[name] = round(ce_of(REP, None if band is None else d, band or set()), 4)
        print(f"{name}: {CE[name]}", flush=True)
    cost = {k: round(v - CE['base'], 4) for k, v in CE.items() if k != 'base'}
    out = {'n_rows': NR, 'ce': CE, 'cost': cost,
           'pred_a_consumed_by_8': bool(cost['b912'] <= 0.15 and cost['b1317'] <= 0.15),
           'pred_b_reader_68': bool(cost['b68'] >= 0.5 * cost['b48']),
           'pred_c_anchor': bool(abs(cost['b48'] - 1.001) <= 0.1),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"costs {cost}")
    print(f"pred_a consumed {out['pred_a_consumed_by_8']} | pred_b 68 {out['pred_b_reader_68']} | pred_c anchor {out['pred_c_anchor']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
