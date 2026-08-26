# mlp0_mlp1_weight_topk3: TWO USER DIRECTIVES (2026-08-26). (1) DATA CHECK: h0 stats
# were measured on 96 rows — re-measure mean/std on 960 rows and score the top-128
# importance-set overlap (if rankings move, S1458's numbers were undertrained).
# (2) JOINT L&R SELECTION ("you need a match in both"): mlp1 unit u's mlp0 term is
# (Cl_u.h0)(Cr_u.h0) — a product, so select ONE shared set S_u of mlp0 units per mlp1
# unit and zero BOTH Cl and Cr outside it: halves the index bits (k indices + 2k
# values vs 2k indices + 2k values) and matches the product structure. Also upgrade
# std -> rms = sqrt(mu^2 + sigma^2): the product has a large mean-driven component,
# std underrates bias-like high-mean units. Scores: additive (|Cl|+|Cr|)*rms and
# product sqrt(|Cl|*|Cr|)*rms (match-in-both). Same exact causal harness (kfull==clean
# ledger held in S1455). The bilinear-attention analog (double QK = match-in-FOUR) is
# pooled behind this result.
#
# Registered predictions:
#   pred_a 96-row vs 960-row importance rankings agree: median top-128 set overlap
#          >= .90 (the S1458 stats were NOT data-starved).
#   pred_b rms-ranked independent k=128 >= .78 recovery (std got .7683).
#   pred_c best SHARED-set k=128 >= .74 recovery at half the index bits.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_mlp1_weight_topk3_results.json'
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

    # ---- h0 stats at two data sizes ----
    def h0_stats(nrows):
        FR = cl.fineweb_rows(nrows, skip=80)[:, :T + 1].contiguous()
        a1 = torch.zeros(HD, device=DEV); a2 = torch.zeros(HD, device=DEV); n0 = 0
        for i in range(0, nrows, 8):
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
        var = (a2 / n0 - mu * mu).clamp_min(0)
        return mu, var.sqrt(), (mu * mu + var).sqrt()
    mu96, sd96, rms96 = h0_stats(96)
    mu960, sd960, rms960 = h0_stats(960)
    print("h0 stats measured (96 and 960 rows)", flush=True)

    # ranking-stability: per-row top-128 sets under |Cl|*sd, 96 vs 960 rows
    I96 = (Cl.abs() * sd96.unsqueeze(0)).topk(128, dim=1).indices
    I960 = (Cl.abs() * sd960.unsqueeze(0)).topk(128, dim=1).indices
    ov = []
    for i in range(0, HD, 512):
        a = I96[i:i + 512]; b = I960[i:i + 512]
        eq = (a.unsqueeze(2) == b.unsqueeze(1)).any(2).float().mean(1)
        ov.append(eq)
    overlap = float(torch.cat(ov).median())
    print(f"top128 overlap 96v960: {overlap:.4f}", flush=True)

    IMP_l = Cl.abs() * rms960.unsqueeze(0)
    IMP_r = Cr.abs() * rms960.unsqueeze(0)
    SH_ADD = IMP_l + IMP_r
    SH_PROD = (Cl.abs() * Cr.abs()).sqrt() * rms960.unsqueeze(0)

    def delta_indep(M, IMP, k):
        thr = IMP.topk(k, dim=1).values[:, -1:]
        return M * (~(IMP >= thr))

    def delta_shared(M, SCORE, k):
        thr = SCORE.topk(k, dim=1).values[:, -1:]
        return M * (~(SCORE >= thr))

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
    ARMS = [('rms32', delta_indep(Cl, IMP_l, 32), delta_indep(Cr, IMP_r, 32)),
            ('rms128', delta_indep(Cl, IMP_l, 128), delta_indep(Cr, IMP_r, 128)),
            ('shadd32', delta_shared(Cl, SH_ADD, 32), delta_shared(Cr, SH_ADD, 32)),
            ('shadd128', delta_shared(Cl, SH_ADD, 128), delta_shared(Cr, SH_ADD, 128)),
            ('shprod128', delta_shared(Cl, SH_PROD, 128), delta_shared(Cr, SH_PROD, 128))]
    for nm, dl, dr in ARMS:
        res[nm] = round(ce_run(dl, dr), 4)
        print(f"{nm}: {res[nm]}", flush=True)
        json.dump({'partial': True, 'res': res, 'overlap': overlap},
                  open(OUT, 'w'), indent=1)

    gap = res['k0'] - res['clean']
    rec = lambda a: (res['k0'] - res[a]) / max(gap, 1e-6)
    recs = {nm: round(rec(nm), 4) for nm, _, _ in ARMS}
    best_shared = max(recs['shadd128'], recs['shprod128'])
    pa = overlap >= 0.90
    pb = rec('rms128') >= 0.78
    pc = best_shared >= 0.74
    out = {'ce': res, 'edge_gap_k0': round(gap, 4), 'recovery': recs,
           'overlap_96v960_top128': round(overlap, 4),
           'std_ref_recovery': {'k32': 0.491, 'k128': 0.7683},
           'pred_a_overlap_90': bool(pa), 'pred_b_rms128_78': bool(pb),
           'pred_c_shared128_74': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"recs {recs} overlap {overlap:.3f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
