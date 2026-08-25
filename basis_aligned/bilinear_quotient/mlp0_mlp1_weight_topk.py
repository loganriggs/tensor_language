# mlp0_mlp1_weight_topk: READ THE mlp0->mlp1 EDGE FROM THE WEIGHTS AND TOP-K IT
# (user directive 2026-08-26: compose mlp0's Down matrix with downstream readers —
# the bilinear pair needs BOTH Left and Right composed — then top-k the composed
# matrices; cross-terms with attn1/embedding at the junction are knowingly missed).
#
# Weight objects: Cl = Left1 @ Down0, Cr = Right1 @ Down0 (both [4608, 4608], mlp0
# hidden unit -> mlp1 hidden unit). mlp1 unit u's mlp0-only input term is
# (Cl_u . h0)(Cr_u . h0) — a rank-1 quadratic in mlp0's hidden h0, so top-k pairs
# factor into (top entries of Cl_u) x (top entries of Cr_u).
# The rms_norm between them contributes a per-position SCALAR g and block-1 lambda
# mixing a constant lam0 — both kept EXACT in the causal arms (captured), so only
# the weight edge is sparsified. Exactness self-test: the k=full arm must reproduce
# clean CE (S1426 lesson: assert the ledger).
#
# Stage 1 (weights only): per-row top-k |mass| shares of Cl, Cr vs a Gaussian control
# of the same shape; effective-rank spectrum.
# Stage 2 (causal, held-out NR=960): mlp1's output recomputed with the DIRECT
# mlp0->mlp1 path routed through row-wise top-k sparsified Cl/Cr (k in
# {0, 8, 32, 128, full}); everything else exact. k=0 = direct edge cut; the k0->clean
# gap is the edge's causal size (reported even if small).
#
# Registered predictions:
#   pred_a median top-32 row-mass share of |Cl| >= 3x the Gaussian control's.
#   pred_b |CE(k=full) - clean| <= .002 (the decomposition algebra is exact).
#   pred_c k=128 (2.8% of units) recovers >= .70 of the k0 -> clean gap.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_mlp1_weight_topk_results.json'
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

    # ---- Stage 1: concentration stats vs Gaussian control ----
    g_ = torch.Generator(device='cpu').manual_seed(5)
    CTRL = torch.randn(HD, HD, generator=g_).to(DEV)
    def share(M, k):
        a = M.abs()
        tk = a.topk(k, dim=1).values.sum(1)
        return (tk / a.sum(1).clamp_min(1e-9))
    stats = {}
    for nm, M in (('Cl', Cl), ('Cr', Cr), ('ctrl', CTRL)):
        stats[nm] = {f'top{k}': round(float(share(M, k).median()), 4)
                     for k in (8, 32, 128)}
    sv = torch.linalg.svdvals(Cl)
    stats['Cl_sv_top128_share'] = round(float(sv[:128].sum() / sv.sum()), 4)
    print("stage1", json.dumps(stats), flush=True)

    # ---- Stage 2: causal top-k ----
    def topk_delta(M, k):
        if k == 0:
            return M.clone()                              # subtract everything
        if k >= HD:
            return torch.zeros_like(M)                    # subtract nothing
        thr = M.abs().topk(k, dim=1).values[:, -1:]
        keep = M.abs() >= thr
        return M * (~keep)

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
    print(f"clean {res['clean']}", flush=True)
    for k in ('full', 0, 8, 32, 128):
        kk = HD if k == 'full' else k
        res[f'k{k}'] = round(ce_run(topk_delta(Cl, kk), topk_delta(Cr, kk)), 4)
        print(f"k{k}: {res[f'k{k}']}", flush=True)
        json.dump({'partial': True, 'res': res, 'stats': stats},
                  open(OUT, 'w'), indent=1)

    gap = res['k0'] - res['clean']
    rec = lambda k: (res['k0'] - res[k]) / max(gap, 1e-6)
    pa = stats['Cl']['top32'] >= 3 * stats['ctrl']['top32']
    pb = abs(res['kfull'] - res['clean']) <= 0.002
    pc = rec('k128') >= 0.70
    out = {'ce': res, 'weight_stats': stats, 'edge_gap_k0': round(gap, 4),
           'recovery': {f'k{k}': round(rec(f'k{k}'), 4) for k in (8, 32, 128)},
           'pred_a_conc_3x': bool(pa), 'pred_b_exact_ledger': bool(pb),
           'pred_c_k128_70': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"gap {gap:.4f} rec {out['recovery']}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
