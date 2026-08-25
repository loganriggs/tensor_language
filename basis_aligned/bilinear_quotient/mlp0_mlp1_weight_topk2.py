# mlp0_mlp1_weight_topk2: VARIANCE-WEIGHTED TOP-K (S1455 follow-up: plain |w| top-k
# recovers .75 of the edge at k=128 even though entry mass is GAUSSIAN-dense vs
# control — the causal concentration must come from h0's own statistics, i.e. a few
# high-variance mlp0 hidden units. So rank edges by IMPORTANCE = |w_ij| * std(h0_j)
# instead of |w_ij| alone; std(h0) measured on 96 fit rows, priced 4608 floats.)
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
#   pred_a var-weighted k=32 beats plain k=32 by >= .10 recovery (.40 -> .50).
#   pred_b var-weighted k=128 recovers >= .85 (plain got .746).
#   pred_c var-weighted k=8 >= plain k=32's recovery (.40) — importance ordering,
#          not weight mass, is the real sparse object.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_mlp1_weight_topk2_results.json'
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

    # ---- measure std(h0) on fit rows (priced: 4608 floats) ----
    FR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    acc1 = torch.zeros(HD, device=DEV); acc2 = torch.zeros(HD, device=DEV); n0 = 0
    for i in range(0, 96, 8):
        idx = FR[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
        blk = H[0]
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        ao, _ = block_attn(blk, xin, idx.shape[0], None)
        xx = xm + ao
        z = F.rms_norm(xx, (D,))
        h = (blk.mlp.Left(z).float() * blk.mlp.Right(z).float()).reshape(-1, HD)
        acc1 += h.sum(0); acc2 += (h * h).sum(0); n0 += h.shape[0]
    mu = acc1 / n0
    sd = (acc2 / n0 - mu * mu).clamp_min(0).sqrt()
    print("h0 std measured", flush=True)

    IMP_l = Cl.abs() * sd.unsqueeze(0)      # importance per edge
    IMP_r = Cr.abs() * sd.unsqueeze(0)

    def topk_delta_by(M, IMP, k):
        if k == 0:
            return M.clone()
        if k >= HD:
            return torch.zeros_like(M)
        thr = IMP.topk(k, dim=1).values[:, -1:]
        keep = IMP >= thr
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
    res['k0'] = round(ce_run(topk_delta_by(Cl, IMP_l, 0),
                             topk_delta_by(Cr, IMP_r, 0)), 4)
    print(f"clean {res['clean']} k0 {res['k0']}", flush=True)
    for k in (8, 32, 128):
        res[f'vark{k}'] = round(ce_run(topk_delta_by(Cl, IMP_l, k),
                                       topk_delta_by(Cr, IMP_r, k)), 4)
        print(f"vark{k}: {res[f'vark{k}']}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    prior = json.load(open(PT + 'mlp0_mlp1_weight_topk_results.json'))
    plain_rec = prior['recovery']
    gap = res['k0'] - res['clean']
    rec = lambda k: (res['k0'] - res[f'vark{k}']) / max(gap, 1e-6)
    recs = {f'vark{k}': round(rec(k), 4) for k in (8, 32, 128)}
    pa = (rec(32) - plain_rec['k32']) >= 0.10
    pb = rec(128) >= 0.85
    pc = rec(8) >= plain_rec['k32']
    out = {'ce': res, 'edge_gap_k0': round(gap, 4), 'recovery': recs,
           'plain_recovery_ref': plain_rec,
           'pred_a_var32_beats_plain32_by10': bool(pa),
           'pred_b_var128_85': bool(pb), 'pred_c_var8_ge_plain32': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"recs {recs} vs plain {plain_rec}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
