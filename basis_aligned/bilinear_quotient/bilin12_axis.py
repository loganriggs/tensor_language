# bilin12_axis: third family member's verdict axis (completes §1249-51). bilin12's match
# station = L2 pair H1+H3 (§1218); fit PC1 of mlp2's re-encoding delta w.r.t. their joint
# write; project out at block entries 3-7; repeat + prose CE, random null.
#
# Registered predictions:
#   pred_a VERDICT AXIS FAMILY-LAW (3rd member): removal costs >= 0.3 nats on repeat.
#   pred_b SELECTIVE: prose rise <= 0.15 x repeat rise; random null <= 0.03.
#   pred_c AXIS EXISTS: split-half PC1 |cos| >= 0.5.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bilin12_axis_results.json'
DEV = 'cuda'
mdl, cfg = load_elriggs('bilin12', device=DEV, dtype=torch.float32); mdl.eval()
D = 768; NH = 6; HD = 128; T = 256; NFIT = 12; NR = 24; QSTART = 128; QFIT = 160
V12 = int(mdl.lm_head.weight.shape[0])
STL = 2; STHS = [1, 3]


@torch.no_grad()
def block_pass(x, x0, v1, blk, capture_heads=None):
    at = blk.attn
    xn = blk.lambdas[0] * x + blk.lambdas[1] * x0
    h = F.rms_norm(xn, (D,))
    B, Tn = x.shape[0], x.shape[1]
    dt = mdl.transformer.wte.weight.dtype
    cos, sin = rope_tables(Tn, HD, DEV, dt, 'bf16')
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    q = apply_rot(F.rms_norm(at.c_q(h).view(B, Tn, NH, HD), (HD,)), cos, sin)
    k = apply_rot(F.rms_norm(at.c_k(h).view(B, Tn, NH, HD), (HD,)), cos, sin)
    s = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / HD
    tril = torch.tril(torch.ones(Tn, Tn, device=DEV, dtype=torch.bool))
    pat = s.square().masked_fill(~tril, 0.0)
    pat = pat / pat.sum(-1, keepdim=True).clamp_min(1e-9)
    v = at.c_v(h).view(B, Tn, NH, HD)
    if v1 is None:
        v1 = v
    vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
    y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
    w = None
    if capture_heads is not None:
        yh = torch.zeros_like(y)
        for hh in capture_heads:
            yh[:, :, hh, :] = y[:, :, hh, :]
        w = at.c_proj(yh.reshape(B, Tn, D)).float()
    x = xn + at.c_proj(y.reshape(B, Tn, D))
    return x, v1, w


@torch.no_grad()
def fit_axis(rows):
    DL = []
    for i in range(0, NFIT, 4):
        idx = rows[i:i + 4, :-1].to(DEV).contiguous()
        x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for L, blk in enumerate(mdl.transformer.h):
            cap = STHS if L == STL else None
            x, v1, w = block_pass(x, x0, v1, blk, cap)
            if L == STL:
                full = blk.mlp(F.rms_norm(x, (D,)))
                blind = blk.mlp(F.rms_norm(x - w.to(x.dtype), (D,)))
                DL.append((full - blind).float()[:, QFIT:].reshape(-1, D).cpu())
                break
            x = x + blk.mlp(F.rms_norm(x, (D,)))
    DL = torch.cat(DL)
    Dc = DL - DL.mean(0)
    half = Dc.shape[0] // 2
    _, _, Va = torch.pca_lowrank(Dc[:half], q=4)
    _, _, Vb = torch.pca_lowrank(Dc[half:], q=4)
    stab = float(F.cosine_similarity(Va[:, 0], Vb[:, 0], dim=0).abs())
    _, _, V = torch.pca_lowrank(Dc, q=4)
    return V[:, 0].to(DEV), stab


@torch.no_grad()
def forward_rm(idx, d):
    x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for L, blk in enumerate(mdl.transformer.h):
        if d is not None and 3 <= L <= 7:
            x = x - (x * d).sum(-1, keepdim=True) * d
        x, v1, _ = block_pass(x, x0, v1, blk)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(mdl.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def ce_of(rows, d):
    qp = torch.arange(QSTART, T, device=DEV)
    tot = 0.0; n = 0
    for i in range(0, rows.shape[0], 4):
        bb = rows[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo = forward_rm(idx, d).float()
        tot += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                     tgt[:, qp].reshape(-1), reduction='sum'))
        n += idx.shape[0] * len(qp)
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    FIT = cl.fineweb_rows(NFIT)[:, :T + 1].contiguous().clone().clamp_max(V12 - 1)
    FIT[:, 128:256] = FIT[:, 0:128]
    REP = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone().clamp_max(V12 - 1)
    REP[:, 128:256] = REP[:, 0:128]
    PROSE = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clamp_max(V12 - 1)
    d, stab = fit_axis(FIT)
    g = torch.Generator(device=DEV).manual_seed(8)
    dr = torch.randn(D, device=DEV, generator=g); dr = dr / dr.norm()
    CE = {'base_rep': round(ce_of(REP, None), 4), 'axis_rep': round(ce_of(REP, d), 4),
          'rand_rep': round(ce_of(REP, dr), 4),
          'base_pr': round(ce_of(PROSE, None), 4), 'axis_pr': round(ce_of(PROSE, d), 4)}
    d_rep = CE['axis_rep'] - CE['base_rep']; d_rand = CE['rand_rep'] - CE['base_rep']
    d_pr = CE['axis_pr'] - CE['base_pr']
    out = {'model': 'bilin12', 'ce': CE, 'splithalf_pc1': round(stab, 4),
           'deltas': {'axis_repeat': round(d_rep, 4), 'rand_repeat': round(d_rand, 4),
                      'axis_prose': round(d_pr, 4)},
           'family_refs': {'bilin18': 1.001, 'swiglu18': 1.1763},
           'pred_a_axis_law': bool(d_rep >= 0.3),
           'pred_b_selective': bool(d_pr <= 0.15 * max(d_rep, 1e-6) and d_rand <= 0.03),
           'pred_c_exists': bool(stab >= 0.5),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE} | stab {stab:.3f} | deltas {out['deltas']}")
    print(f"pred_a law {out['pred_a_axis_law']} | pred_b sel {out['pred_b_selective']} | pred_c exists {out['pred_c_exists']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
