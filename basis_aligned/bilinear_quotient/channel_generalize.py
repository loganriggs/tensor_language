# channel_generalize: THE THIRD PROPERTY AT CHANNEL GRAIN (S1486 measured extraction
# and removal on skip=7000; generalization = the same effects on FRESH skip=2000
# rows, same bases, no refit). Arms: ex_w8, ex_pca8, rm_w8, rm_pca8 (+ clean, mean).
# Also scores whether the SELECTIVITY finding (weight 3.8x vs PCA 1.2x) replicates.
# Original header: CIRCUITS AT CHANNEL GRAIN + RED-TEAM BASELINES (S1484: single
# axes carry ~.002 CE — too small. The causal objects are 8-32-direction channels.
# User directive: red-team WHICH structure helps). Basis under test: top-k whitened
# right-vectors of the STACKED composed block-1 edge [pat(4); val; mlp Left; mlp
# Right] — the weight-derived channel. Baselines: activation-PCA of h0 (same k, NO
# weight structure) and a random k-subspace. Arms at mlp0's output:
#   extraction: mlp0_out = mean + subspace component (k in {1, 8, 32} for the weight
#               channel; k=8 for PCA baseline).
#   removal:    mlp0_out = full - subspace component (k=8: weight / PCA / random).
# Metrics: global + frag-class + det-class CE, skip=7000.
# Original header: THE FIRST CIRCUIT THROUGH THE COMPRESSION (user directive:
# use the compression to find circuits with the 3 properties — generalizing,
# extraction, removal). Pilot: the D2 'capitalized name-fragment' axis at mlp0's
# OUTPUT (not just block-1's reads): the axis's output image is the rank-1 vector
# u = Down0(d * rms), scaled per-position by s = ((h0-mu)/rms).d.
#   REMOVAL:    mlp0_out' = mlp0_out - s*u             (axis cut, everywhere).
#   EXTRACTION: mlp0_out' = mean_out + s*u             (ONLY the axis + mean kept).
# Scored globally AND on name-fragment-following positions, on TWO row sets
# (skip=7000 and fresh skip=2000) — GENERALIZATION = the class-conditional effects
# replicate. d = the shared top-2 subspace's D2 direction (pattern-edge dir1, S1470).
# Original header: CAUSAL TEST OF THE NAMED AXES (S1470 named D1 'determiner/NP-
# start' and D2 'capitalized name-fragment' from the composed-edge SVDs; naming is
# only real if cutting the axis hurts the PREDICTIONS the name implies). Cuts are
# rank-1 mean-preserving: remove the whitened component of (h0 - mu0) along each
# edge's own dir (pattern dir0+values dir0 = D1; pattern dir1+values dir1 = D2;
# random whitened unit vector = control), applied at block-1's pattern AND values
# reads. Scored globally AND class-conditionally: CE at positions whose PREVIOUS
# token is in the axis's top-token set (determiners for D1, name-fragments for D2).
# Original header: MEAN-PRESERVING EDGE CUTS (S1466: the values edge cut costs
# 1.149 CE — MORE than mlp0's whole delta_opt .908 — because cut-to-zero removes the
# MEAN transport too, which the optimal constant would keep. Here each edge is cut
# CENTERED: subtract corr * ((h0 - mu0) @ C.T) — the mean flows, the data-dependent
# signal is removed. This sizes the true per-edge information content. mu0 from 960
# rows. Arms: centered cuts of the values edge, the pattern edge (all four QK maps),
# and both applied together (block-1 attn reads only mlp0's mean).
# Original header: COMPLETE BLOCK-1'S READ OF mlp0 (S1464: the PATTERN edge is
# .655 CE and rank-8 recovers 98%; mlp1 edge .221 and rank-32 .81). Remaining pathway:
# the VALUES edge Av = c_v1@Down0 — linear, no double-QK. Same exact harness (delta
# subtracted from c_v(xin) before head split; patterns live).
# Original header: THE ATTENTION ANALOG OF THE EDGE-RANK RESULT (user directive:
# "the same applies for the bilinear attn, to a larger degree for the double QK").
# Composed matrices Aq = c_q1@Down0, Ak = c_k1@Down0, Aq2 = c_q2_1@Down0,
# Ak2 = c_k2_1@Down0 (each [1152, 4608]): how block-1's PATTERN computation reads
# mlp0's hidden units. The pattern is (q.k)(q2.k2) — a product of two bilinear scores,
# so an mlp0 direction must register in BOTH factors to move it. S1463 showed rank
# beats sparse at the mlp1 edge, so the arms here are rank truncations (h0-whitened)
# of ALL FOUR matrices simultaneously; values path (c_v) stays live/exact — this
# isolates the mlp0 -> attn1-PATTERN edge. Exactness: delta subtracted from c_*(xin)
# BEFORE the per-head rms_norm and rotary, so the ledger stays exact by construction
# (k=cut is the edge size; no kfull arm needed since delta=0 is identically clean).
#
# Registered predictions:
#   pred_a ex_w8 frag-keep on skip=2000 >= .261 (half the .522 measured on 7000).
#   pred_b the selectivity ordering replicates: w8 frag/global keep ratio >= 2x the
#          pca8 ratio on the fresh rows.
#   pred_c rm_w8 global rise >= .01 on fresh rows (removal causality generalizes).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'channel_generalize_results.json'
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


MU0 = {'mu': None}


@torch.no_grad()
def fwd_arm(idx, mode, CV, CP):
    """mode None=clean; 'v'/'pat'/'both' = centered cut of that edge at block 1.
    CV: values composition [1152,4608]; CP: dict of 4 pattern compositions."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]; h0 = None
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        if L == 1 and mode is not None:
            lam0 = float(blk.lambdas[0])
            g = (D ** 0.5) / xm.float().norm(dim=-1, keepdim=True)
            corr = lam0 * g
            hc = h0 - MU0['mu']
            if mode in ('pat', 'both'):
                pre = {}
                for nm, mod_ in (('q', at.c_q), ('k', at.c_k),
                                 ('q2', at.c_q2), ('k2', at.c_k2)):
                    p = mod_(xin).float() - corr * (hc @ CP[nm].T)
                    pre[nm] = p.view(B, T, 9, 128)
            else:
                pre = {'q': at.c_q(xin).view(B, T, 9, 128).float(),
                       'k': at.c_k(xin).view(B, T, 9, 128).float(),
                       'q2': at.c_q2(xin).view(B, T, 9, 128).float(),
                       'k2': at.c_k2(xin).view(B, T, 9, 128).float()}
            cos, sin = at.rotary(pre['q'])
            q = are(F.rms_norm(pre['q'], (128,)), cos, sin)
            k = are(F.rms_norm(pre['k'], (128,)), cos, sin)
            q2 = are(F.rms_norm(pre['q2'], (128,)), cos, sin)
            k2 = are(F.rms_norm(pre['k2'], (128,)), cos, sin)
            pat = (torch.einsum('bqhd,bkhd->bhqk', q, k) / 128.0) \
                * (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / 128.0)
            tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
            pat = pat.masked_fill(~tril, 0.0)
            if mode in ('v', 'both'):
                vpre = at.c_v(xin).float() - corr * (hc @ CV.T)
            else:
                vpre = at.c_v(xin).float()
            v = vpre.view(B, T, 9, 128)
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v).float()
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            ao = at.c_proj(y.reshape(B, T, D).to(at.c_proj.weight.dtype))
        else:
            ao, v1n = block_attn(blk, xin, B, v1)
            if v1 is None:
                v1 = v1n
        x = xm + ao
        z = F.rms_norm(x, (D,))
        if L == 0:
            h0 = (blk.mlp.Left(z).float() * blk.mlp.Right(z).float())
        x = x + blk.mlp(z)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()

    at1 = H[1].attn
    Wd0 = H[0].mlp.Down.weight.float().to(DEV)
    CV = at1.c_v.weight.float().to(DEV) @ Wd0
    CP = {'q': at1.c_q.weight.float().to(DEV) @ Wd0,
          'k': at1.c_k.weight.float().to(DEV) @ Wd0,
          'q2': at1.c_q2.weight.float().to(DEV) @ Wd0,
          'k2': at1.c_k2.weight.float().to(DEV) @ Wd0}

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
    MU0['mu'] = a1 / n0
    rms = (a2 / n0).clamp_min(1e-12).sqrt()
    print("mu, rms measured", flush=True)

    PATC = torch.cat([CP[n] for n in ('q', 'k', 'q2', 'k2')], 0)
    Wd0 = H[0].mlp.Down.weight.float().to(DEV)
    CM = torch.cat([H[1].mlp.Left.weight.float().to(DEV) @ Wd0,
                    H[1].mlp.Right.weight.float().to(DEV) @ Wd0], 0)
    STACK = torch.cat([PATC, CV, CM], 0)                 # all block-1 reads
    _, _, Vt = torch.linalg.svd(STACK * rms.unsqueeze(0), full_matrices=False)
    W8 = Vt[:8]; W1 = Vt[:1]; W32 = Vt[:32]              # weight channel bases

    # activation-PCA baseline: PCA of centered h0 (whitened space), same data
    FR2 = cl.fineweb_rows(240, skip=80)[:, :T + 1].contiguous()
    hs = []
    for i in range(0, 240, 8):
        idx = FR2[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
        blk = H[0]
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        ao, _ = block_attn(blk, xin, idx.shape[0], None)
        z = F.rms_norm(xm + ao, (D,))
        h = (blk.mlp.Left(z).float() * blk.mlp.Right(z).float()).reshape(-1, HD)
        hs.append(((h - MU0['mu']) / rms)[::7].cpu())     # subsample
    HS = torch.cat(hs).to(DEV)
    _, _, Vp = torch.svd_lowrank(HS, q=16, niter=4)
    P8 = Vp[:, :8].T                                     # PCA basis (whitened)
    gr = torch.Generator().manual_seed(77)
    R8 = torch.linalg.qr(torch.randn(HD, 8, generator=gr))[0].to(DEV).T
    print("bases built", flush=True)

    import tiktoken
    ENC = tiktoken.get_encoding('gpt2')
    FRAG = [' Ch', ' Pl', ' Sh', ' H', ' G', ' Br', ' Th', ' B', ' T', ' W', ' M',
            ' D', ' L', ' R', ' Fl', ' Bl', ' Sp', ' Z', ' K', ' F', ' S', ' Howard']
    DET = [' the', ' our', ' his', ' a', ' your', ' their', ' an', ' my', ' this']
    FRAG_IDS = torch.tensor([ENC.encode(t)[0] for t in FRAG])
    DET_IDS = torch.tensor([ENC.encode(t)[0] for t in DET])

    mo_acc = torch.zeros(D, device=DEV); nmo = 0
    for i in range(0, 240, 8):
        idx = FR2[i:i + 8, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
        blk = H[0]
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        ao, _ = block_attn(blk, xin, idx.shape[0], None)
        z = F.rms_norm(xm + ao, (D,))
        mo = blk.mlp(z).float().reshape(-1, D)
        mo_acc += mo.sum(0); nmo += mo.shape[0]
    MEAN_OUT = mo_acc / nmo
    print("mean out measured", flush=True)

    @torch.no_grad()
    def fwd_circ(idx, mode, BASIS):
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]
        for L, blk in enumerate(H):
            at = blk.attn
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            ao, v1n = block_attn(blk, xin, B, v1)
            if v1 is None:
                v1 = v1n
            x = xm + ao
            z = F.rms_norm(x, (D,))
            if L == 0 and mode is not None:
                h0 = (blk.mlp.Left(z).float() * blk.mlp.Right(z).float())
                hw = (h0 - MU0['mu']) / rms
                S = hw @ BASIS.T                          # [B, T, k]
                comp_h = (S @ BASIS) * rms                # back to h0 space
                axis_out = comp_h @ Wd0.T                 # output image
                mo_full = blk.mlp(z).float()
                if mode == 'removal':
                    mo = mo_full - axis_out
                else:
                    mo = MEAN_OUT.expand(B, T, D) + axis_out
                x = x + mo.to(x.dtype)
            else:
                x = x + blk.mlp(z)
        return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)

    def ce_run2(mode, BASIS):
        s_ = 0.0; n_ = 0; sf = 0.0; nf = 0; sd = 0.0; nd = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_circ(idx, mode, BASIS).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            s_ += float(ce[mk].sum()); n_ += int(mk.sum())
            isf = torch.isin(idx.cpu(), FRAG_IDS).to(DEV) & mk
            isd = torch.isin(idx.cpu(), DET_IDS).to(DEV) & mk
            sf += float(ce[isf].sum()); nf += int(isf.sum())
            sd += float(ce[isd].sum()); nd += int(isd.sum())
        return {'global': s_ / max(n_, 1), 'frag': sf / max(nf, 1),
                'det': sd / max(nd, 1)}

    EVR = cl.fineweb_rows(NR, skip=2000)[:, :T + 1].contiguous()
    res = {}
    res['clean'] = {k: round(v, 4) for k, v in ce_run2(None, W8).items()}
    ARMS = [('mean', 'extract', torch.zeros(1, HD, device=DEV)),
            ('ex_w8', 'extract', W8), ('ex_pca8', 'extract', P8),
            ('rm_w8', 'removal', W8), ('rm_pca8', 'removal', P8)]
    for nm, md, B_ in ARMS:
        res[nm] = {k: round(v, 4) for k, v in ce_run2(md, B_).items()}
        print(nm, res[nm], flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    c = res['clean']; mo = res['mean']
    eff = {k: mo[k] - c[k] for k in c}
    keep = lambda nm, k: (mo[k] - res[nm][k]) / max(eff[k], 1e-6)
    keeps = {nm: {k: round(keep(nm, k), 4) for k in c}
             for nm in ('ex_w8', 'ex_pca8')}
    rises = {nm: {k: round(res[nm][k] - c[k], 4) for k in c}
             for nm in ('rm_w8', 'rm_pca8')}
    ratio_w = keeps['ex_w8']['frag'] / max(keeps['ex_w8']['global'], 1e-6)
    ratio_p = keeps['ex_pca8']['frag'] / max(keeps['ex_pca8']['global'], 1e-6)
    pa = keeps['ex_w8']['frag'] >= 0.261
    pb = ratio_w >= 2 * ratio_p
    pc = rises['rm_w8']['global'] >= 0.01
    out = {'res': res, 'effect': {k: round(v, 4) for k, v in eff.items()},
           'keeps': keeps, 'rises': rises,
           'selectivity': {'w8': round(ratio_w, 3), 'pca8': round(ratio_p, 3)},
           'ref_7000': {'ex_w8_frag': 0.522, 'ratio_w': 3.84, 'ratio_p': 1.19},
           'pred_a_frag_keep_generalizes': bool(pa),
           'pred_b_selectivity_replicates': bool(pb),
           'pred_c_removal_generalizes': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"keeps {keeps}")
    print(f"rises {rises}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
