# handle_site_mlp2: THE SITE-SELECTIVITY MAP CONTINUES (S1491: selectivity is a
# site property — mlp0's channel 4.3x, mlp1's ~1). Site #3 = mlp2 (front-stack,
# fid .944, lin2-structured): basis = top-8 whitened right-vectors of block-3's
# stacked composed reads of h2; baselines PCA-8 / random-8; det + name-fragment
# classes (front sites may still carry mlp0-flavored classes). Both row sets.
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
#   pred_a w8 frag-keep >= .25 at mlp2 (7000).
#   pred_b w8 selectivity >= 2x PCA at mlp2 (does front-stack selectivity persist
#          past layer 0?).
#   pred_c generalization holds on skip=2000 (sign + half magnitude).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; HD = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'handle_site_mlp2_results.json'
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

    # h2 stats (mlp2 hidden), block-3 composed reads
    Wd1 = H[2].mlp.Down.weight.float().to(DEV)
    at2 = H[3].attn
    STACK = torch.cat([at2.c_q.weight.float().to(DEV) @ Wd1,
                       at2.c_k.weight.float().to(DEV) @ Wd1,
                       at2.c_q2.weight.float().to(DEV) @ Wd1,
                       at2.c_k2.weight.float().to(DEV) @ Wd1,
                       at2.c_v.weight.float().to(DEV) @ Wd1,
                       H[3].mlp.Left.weight.float().to(DEV) @ Wd1,
                       H[3].mlp.Right.weight.float().to(DEV) @ Wd1], 0)

    @torch.no_grad()
    def run_to_h1(idx):
        """Return h2 (mlp2 hidden)."""
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]
        for L in (0, 1, 2):
            blk = H[L]
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            ao, v1n = block_attn(blk, xin, B, v1)
            if v1 is None:
                v1 = v1n
            x = xm + ao
            z = F.rms_norm(x, (D,))
            if L == 2:
                h1 = blk.mlp.Left(z).float() * blk.mlp.Right(z).float()
                return h1
            x = x + blk.mlp(z)

    FR2 = cl.fineweb_rows(480, skip=80)[:, :T + 1].contiguous()
    a1s = torch.zeros(HD, device=DEV); a2s = torch.zeros(HD, device=DEV)
    mo_acc = torch.zeros(D, device=DEV); n0 = 0
    hs = []
    for i in range(0, 480, 8):
        idx = FR2[i:i + 8, :-1].to(DEV).contiguous()
        h1 = run_to_h1(idx).reshape(-1, HD)
        a1s += h1.sum(0); a2s += (h1 * h1).sum(0); n0 += h1.shape[0]
        mo = (h1.to(H[2].mlp.Down.weight.dtype) @ H[2].mlp.Down.weight.T).float() \
            + H[2].mlp.Down_bias.float()
        mo_acc += mo.sum(0)
        if i < 240:
            hs.append(h1[::7].cpu())
    MU1 = a1s / n0
    RMS1 = (a2s / n0).clamp_min(1e-12).sqrt()
    MEAN_OUT = mo_acc / n0
    _, _, Vt = torch.linalg.svd(STACK * RMS1.unsqueeze(0), full_matrices=False)
    W8 = Vt[:8]
    HS = torch.cat(hs).to(DEV)
    HSw = (HS - MU1) / RMS1
    _, _, Vp = torch.svd_lowrank(HSw, q=16, niter=4)
    P8 = Vp[:, :8].T
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
            if L == 2 and mode is not None:
                h1 = blk.mlp.Left(z).float() * blk.mlp.Right(z).float()
                hw = (h1 - MU1) / RMS1
                S = hw @ BASIS.T
                comp_h = (S @ BASIS) * RMS1
                axis_out = comp_h @ Wd1.T
                mo_full = blk.mlp(z).float()
                if mode == 'removal':
                    mo = mo_full - axis_out
                else:
                    mo = MEAN_OUT.expand(B, T, D) + axis_out
                x = x + mo.to(x.dtype)
            else:
                x = x + blk.mlp(z)
        return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)

    def ce_run2(rows, mode, BASIS):
        s_ = 0.0; n_ = 0; sf = 0.0; nf = 0; sd = 0.0; nd = 0
        for i in range(0, NR, 8):
            bb = rows[i:i + 8].to(DEV)
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

    EV2 = cl.fineweb_rows(NR, skip=2000)[:, :T + 1].contiguous()
    res = {}
    for setname, rows in (('s7000', EVR), ('s2000', EV2)):
        res[setname] = {}
        for nm, md, B_ in (('clean', None, W8),
                           ('mean', 'extract', torch.zeros(1, HD, device=DEV)),
                           ('ex_w8', 'extract', W8), ('ex_pca8', 'extract', P8),
                           ('rm_w8', 'removal', W8), ('rm_rand8', 'removal', R8)):
            res[setname][nm] = {k: round(v, 4)
                                for k, v in ce_run2(rows, md, B_).items()}
            print(setname, nm, res[setname][nm], flush=True)
            json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    def scores(sn):
        c = res[sn]['clean']; mo = res[sn]['mean']
        eff = {k: mo[k] - c[k] for k in c}
        keep = lambda nm, k: (mo[k] - res[sn][nm][k]) / max(eff[k], 1e-6)
        kw = {k: keep('ex_w8', k) for k in c}
        kp = {k: keep('ex_pca8', k) for k in c}
        rw = {k: res[sn]['rm_w8'][k] - c[k] for k in c}
        rr = {k: res[sn]['rm_rand8'][k] - c[k] for k in c}
        return eff, kw, kp, rw, rr
    eff7, kw7, kp7, rw7, rr7 = scores('s7000')
    eff2, kw2, kp2, rw2, rr2 = scores('s2000')
    ratio_w7 = kw7['frag'] / max(kw7['global'], 1e-6)
    ratio_p7 = kp7['frag'] / max(kp7['global'], 1e-6)

    pa = kw7['frag'] >= 0.25
    pb = ratio_w7 >= 2 * ratio_p7
    pc = ((kw2['frag'] > 0) == (kw7['frag'] > 0)
          and abs(kw2['frag']) >= 0.5 * abs(kw7['frag'])
          and (rw2['global'] > 0) == (rw7['global'] > 0)
          and abs(rw2['global']) >= 0.5 * abs(rw7['global']))
    out = {'res': res,
           'keeps_7000': {'w8': {k: round(v, 4) for k, v in kw7.items()},
                          'pca8': {k: round(v, 4) for k, v in kp7.items()}},
           'rises_7000': {'w8': {k: round(v, 4) for k, v in rw7.items()},
                          'rand8': {k: round(v, 4) for k, v in rr7.items()}},
           'keeps_2000_w8': {k: round(v, 4) for k, v in kw2.items()},
           'selectivity_7000': {'w8': round(ratio_w7, 3), 'pca8': round(ratio_p7, 3)},
           'pred_a_frag_keep_25': bool(pa), 'pred_b_selectivity_2x': bool(pb),
           'pred_c_generalizes': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"kw7 {kw7} sel {ratio_w7:.2f} vs {ratio_p7:.2f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
