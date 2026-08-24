# reencode_axis: causal test of the §1248 stable axis. mlp3's re-encoding has one
# split-half-stable dominant direction (PC1 |cos| 0.972). If that direction IS the match
# evidence's carrier, projecting it out of the residual stream (after blocks 4-8, each
# block entry) should selectively damage copying while barely touching prose.
#
# Direction: PC1 of the §1248 deltas, recomputed in-script on a FIT half (12 rows); all
# evaluations on DISJOINT rows (24 repeat + 24 prose).
#
# Conditions: base; rm_axis (x -= (x·d)d at entries of blocks 4..8); rm_rand (same with a
# random unit direction — parameter-matched null); rm_axis_prose / rm_rand_prose (same on
# natural rows, CE split rare/freq).
#
# Registered predictions:
#   pred_a MATCH-SPECIFIC CHANNEL: repeat-CE rise under rm_axis >= 0.3 nats AND >= 10x the
#          rm_rand rise.
#   pred_b PROSE BARELY TOUCHED: prose all-CE rise under rm_axis <= 0.15 x its repeat rise
#          (the §1219/§1248 half-strength prose activity predicts SOME prose cost — bar
#          scaled, not zero).
#   pred_c SANITY: base = true model ±0.005; rm_rand repeat rise <= 0.03.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'reencode_axis_results.json'
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
def forward_rm(idx, d):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for L, blk in enumerate(m.transformer.h):
        if d is not None and 4 <= L <= 8:
            x = x - (x * d).sum(-1, keepdim=True) * d
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def ce_split(rows, d, is_freq):
    qp = torch.arange(QSTART, T, device=DEV)
    tots = [0.0, 0.0, 0.0]; ns = [0, 0, 0]
    for i in range(0, rows.shape[0], 4):
        bb = rows[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo = forward_rm(idx, d).float()
        lse = F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                              tgt[:, qp].reshape(-1), reduction='none')
        tq = tgt[:, qp].reshape(-1); fr = is_freq[tq]
        tots[0] += float(lse.sum()); ns[0] += len(lse)
        tots[1] += float(lse[~fr].sum()); ns[1] += int((~fr).sum())
        tots[2] += float(lse[fr].sum()); ns[2] += int(fr.sum())
    return tuple(t / max(n, 1) for t, n in zip(tots, ns))


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    FIT = cl.fineweb_rows(NFIT)[:, :T + 1].contiguous().clone()
    FIT[:, 128:256] = FIT[:, 0:128]
    REP = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    REP[:, 128:256] = REP[:, 0:128]
    PROSE = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    V = int(m.lm_head.weight.shape[0])
    cnts = torch.bincount(PROSE[:, :T].reshape(-1), minlength=V)
    is_freq = torch.zeros(V, dtype=torch.bool, device=DEV)
    is_freq[torch.topk(cnts, 128).indices.to(DEV)] = True

    d = fit_axis(FIT)
    g = torch.Generator(device=DEV).manual_seed(8)
    dr = torch.randn(D, device=DEV, generator=g); dr = dr / dr.norm()

    res = {}
    for name, dd, rows in (('base_rep', None, REP), ('axis_rep', d, REP), ('rand_rep', dr, REP),
                           ('base_pr', None, PROSE), ('axis_pr', d, PROSE), ('rand_pr', dr, PROSE)):
        a, r, f = ce_split(rows, dd, is_freq)
        res[name] = {'all': round(a, 4), 'rare': round(r, 4), 'freq': round(f, 4)}
        print(f"{name}: all {res[name]['all']} rare {res[name]['rare']} freq {res[name]['freq']}", flush=True)

    d_rep = res['axis_rep']['all'] - res['base_rep']['all']
    d_rep_r = res['rand_rep']['all'] - res['base_rep']['all']
    d_pr = res['axis_pr']['all'] - res['base_pr']['all']
    out = {'n_fit': NFIT, 'n_rows': NR, 'results': res,
           'deltas': {'axis_repeat': round(d_rep, 4), 'rand_repeat': round(d_rep_r, 4),
                      'axis_prose': round(d_pr, 4)},
           'pred_a_match_channel': bool(d_rep >= 0.3 and d_rep >= 10 * max(d_rep_r, 1e-6)),
           'pred_b_prose_spared': bool(d_pr <= 0.15 * max(d_rep, 1e-6)),
           'pred_c_rand_null': bool(d_rep_r <= 0.03),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"deltas {out['deltas']}")
    print(f"pred_a channel {out['pred_a_match_channel']} | pred_b prose {out['pred_b_prose_spared']} | pred_c rand {out['pred_c_rand_null']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
