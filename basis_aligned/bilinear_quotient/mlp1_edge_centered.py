# mlp1_edge_centered: SIZE THE SIGNAL OF THE mlp0->mlp1 EDGE (S1468 pool: the .221
# CE cut-to-zero conflates mean transport with signal — block-1 attention edges were
# ~95% mean. Same centered cut here: subtract corr*((h0-mu0)@C.T) so the mean flows).
# Arms: uncentered cut (ref, .221), centered cut, centered rank-32 residual kept
# (= centered cut of everything BEYOND the top-32 whitened signal directions).
# Original header: READ THE mlp0->mlp1 EDGE FROM THE WEIGHTS AND TOP-K IT
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
#   pred_a centered cut <= .110 (at most half of the .221 was signal).
#   pred_b centered cut >= .02 (the signal is real, unlike block-1 attn's .09 split
#          this edge carries more information).
#   pred_c centered-beyond-rank32 <= .30 x the centered cut (the signal is low-rank).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp1_edge_centered_results.json'
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


MU = {'mu': None, 'center': False}


@torch.no_grad()
def fwd_arm(idx, dCl, dCr):
    """dCl/dCr: [HD, HD] DELTA matrices to SUBTRACT from the direct mlp0->mlp1 edge,
    or None for clean. MU['center']: subtract deltas of (h0 - mu0) instead of h0."""
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
            hh = (h0 - MU['mu']) if MU['center'] else h0
            pl = pl - corr * (hh @ dCl.T)
            pr = pr - corr * (hh @ dCr.T)
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

    # mu0, rms on 960 rows
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
    MU['mu'] = a1 / n0
    rms = (a2 / n0).clamp_min(1e-12).sqrt()
    print("mu, rms measured", flush=True)

    def rank_delta(M, r):
        Mw = M * rms.unsqueeze(0)
        U, S, Vt = torch.linalg.svd(Mw, full_matrices=False)
        Mr = (U[:, :r] * S[:r]) @ Vt[:r]
        return M - Mr / rms.unsqueeze(0)

    def ce_run(dCl, dCr, center):
        MU['center'] = center
        s_ = 0.0; n_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_arm(idx, dCl, dCr).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
        MU['center'] = False
        return s_ / max(n_, 1)

    res = {'clean': round(ce_run(None, None, False), 4)}
    res['cut'] = round(ce_run(Cl.clone(), Cr.clone(), False), 4)
    res['cut_centered'] = round(ce_run(Cl.clone(), Cr.clone(), True), 4)
    dl32 = rank_delta(Cl, 32); dr32 = rank_delta(Cr, 32)
    res['centered_beyond_r32'] = round(ce_run(dl32, dr32, True), 4)
    print(res, flush=True)

    g_unc = res['cut'] - res['clean']
    g_cen = res['cut_centered'] - res['clean']
    g_b32 = res['centered_beyond_r32'] - res['clean']
    pa = g_cen <= 0.110
    pb = g_cen >= 0.02
    pc = g_b32 <= 0.30 * max(g_cen, 1e-6)
    out = {'ce': res, 'gaps': {'uncentered': round(g_unc, 4),
                               'centered': round(g_cen, 4),
                               'centered_beyond_r32': round(g_b32, 4)},
           'pred_a_cen_le_110': bool(pa), 'pred_b_cen_ge_02': bool(pb),
           'pred_c_beyond32_le_30pct': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"gaps {out['gaps']}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
