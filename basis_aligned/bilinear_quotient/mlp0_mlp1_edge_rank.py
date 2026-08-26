# mlp0_mlp1_edge_rank: RANK TRUNCATION OF THE mlp0->mlp1 EDGE (S1458 pool: the edge
# is dense with Gaussian-level entry mass but sv_top128 share .205 = ~4x Gaussian —
# dense-low-rank should beat sparse for such objects). Truncate Cl and Cr to rank r
# in the h0-whitened metric: SVD of C @ diag(rms(h0)), keep top r, unwhiten. Cost:
# 2 * r * (4608 + 4608) floats per side vs sparse 2k * 4608 entries. Same exact causal
# harness. rms from 960 rows.
#
# Registered predictions:
#   pred_a rank-32 >= .60 recovery (sparse k32 got .49).
#   pred_b rank-128 >= .85 recovery (sparse k128 got .77).
#   pred_c rank-8 >= .35 recovery.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_mlp1_edge_rank_results.json'
NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h


@torch.no_grad()
def block_attn(blk, xin, B, v1):
    at = blk.attn
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
    return at.c_proj(y.reshape(B, T, D)), v1


@torch.no_grad()
def fwd_arm(idx, dCl, dCr):
    """dCl/dCr: [HD, HD] DELTA matrices (Cfull - Ctopk) to SUBTRACT from the direct
    mlp0->mlp1 edge, or None for clean."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]; h0 = None
    for L, blk in enumerate(H):
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        ao, v1 = block_attn(blk, xin, B, v1)
        x = xm + ao
        z = F.rms_norm(x, (D,))
        if L == 0:
            h0 = (blk.mlp.Left(z).float() * blk.mlp.Right(z).float())
            x = x + blk.mlp(z)
        elif L == 1 and dCl is not None:
            lam0 = float(blk.lambdas[0])
            g = (D ** 0.5) / x.float().norm(dim=-1, keepdim=True)   # z = x * g
            pl = blk.mlp.Left(z).float()
            pr = blk.mlp.Right(z).float()
            corr = (lam0 * g)
            pl = pl - corr * (h0 @ dCl.T)
            pr = pr - corr * (h0 @ dCr.T)
            mo = blk.mlp.Down((pl * pr).to(blk.mlp.Down.weight.dtype)) \
                + blk.mlp.Down_bias
            x = x + mo
        else:
            x = x + blk.mlp(z)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()

    L1 = H[1].mlp.Left.weight.float().to(DEV)
    R1 = H[1].mlp.Right.weight.float().to(DEV)
    Wd0 = H[0].mlp.Down.weight.float().to(DEV)
    Cl = L1 @ Wd0; Cr = R1 @ Wd0                          # [HD, HD]

    # ---- h0 rms on 960 rows ----
    FR = cl.fineweb_rows(960, skip=80)[:, :T + 1].contiguous()
    a1 = torch.zeros(HD, device=DEV); a2 = torch.zeros(HD, device=DEV); n0 = 0
    for i in range(0, 960, 8):
        idx = FR[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
        blk = H[0]
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        ao, _ = block_attn(blk, xin, idx.shape[0], None)
        xx = xm + ao
        z = F.rms_norm(xx, (D,))
        h = (blk.mlp.Left(z).float() * blk.mlp.Right(z).float()).reshape(-1, HD)
        a1 += h.sum(0); a2 += (h * h).sum(0); n0 += h.shape[0]
    mu = a1 / n0
    rms = (a2 / n0).clamp_min(1e-12).sqrt()
    print("rms(h0) measured", flush=True)

    def rank_delta(M, r):
        Mw = M * rms.unsqueeze(0)
        U, S, Vt = torch.linalg.svd(Mw, full_matrices=False)
        Mr = (U[:, :r] * S[:r]) @ Vt[:r]
        return M - Mr / rms.unsqueeze(0)          # delta = full - lowrank (unwhitened)

    def ce_run(dCl, dCr):
        s_ = 0.0; n_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, dCl, dCr).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        return s_ / max(n_, 1)

    res = {'clean': round(ce_run(None, None), 4)}
    res['k0'] = round(ce_run(Cl.clone(), Cr.clone()), 4)
    print(f"clean {res['clean']} k0 {res['k0']}", flush=True)
    for r in (8, 32, 128):
        res[f'rank{r}'] = round(ce_run(rank_delta(Cl, r), rank_delta(Cr, r)), 4)
        print(f"rank{r}: {res[f'rank{r}']}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    gap = res['k0'] - res['clean']
    rec = lambda a: (res['k0'] - res[a]) / max(gap, 1e-6)
    recs = {f'rank{r}': round(rec(f'rank{r}'), 4) for r in (8, 32, 128)}
    pa = rec('rank32') >= 0.60
    pb = rec('rank128') >= 0.85
    pc = rec('rank8') >= 0.35
    out = {'ce': res, 'edge_gap_k0': round(gap, 4), 'recovery': recs,
           'sparse_ref': {'k32': 0.491, 'k128': 0.7683},
           'pred_a_rank32_60': bool(pa), 'pred_b_rank128_85': bool(pb),
           'pred_c_rank8_35': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"recs {recs}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
